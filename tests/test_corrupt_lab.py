import io
import unittest

import numpy as np

try:
    from PIL import Image
    from PIL import ImageFilter
except ImportError:
    Image = None

from corrupt_lab import CORRUPTIONS, SEVERITIES, convolve, corruption_error, gaussian_kernel, hed_to_rgb, jpeg_roundtrip, macenko_normalise, macenko_vectors, psnr, rgb_to_hed, robustness_table, stain_jitter, stain_shift
from corrupt_lab.core import HED


def picture(h=96, w=128, seed=0):
    """A photo-like test image: smooth gradients, shapes and fine texture."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    a = np.zeros((h, w, 3))
    a[..., 0] = 120 + 80 * np.sin(x / 11)
    a[..., 1] = 120 + 80 * np.cos(y / 7)
    a[..., 2] = 120 + 60 * np.sin((x + y) / 9)
    for _ in range(12):
        cy, cx, r = rng.integers(0, h), rng.integers(0, w), rng.integers(5, 16)
        mask = (y - cy) ** 2 + (x - cx) ** 2 < r ** 2
        a[mask] = rng.integers(0, 255, 3)
    return np.clip(a + rng.normal(0, 4, a.shape), 0, 255).astype(np.uint8)


def tissue(h=128, w=128, seed=0, stains=None, strength=1.0):
    """Synthetic H&E: blobs of nuclei in eosin-coloured tissue, mixed with known stain vectors."""
    rng = np.random.default_rng(seed)
    stains = HED[:2] if stains is None else stains
    y, x = np.mgrid[0:h, 0:w]
    H, E = np.zeros((h, w)), np.full((h, w), 0.35) + rng.normal(0, 0.03, (h, w))
    for _ in range(40):
        cy, cx, r = rng.integers(0, h), rng.integers(0, w), rng.integers(4, 9)
        H[(y - cy) ** 2 + (x - cx) ** 2 < r ** 2] = rng.uniform(0.5, 0.9)
    E[H > 0] *= 0.4
    E[:h // 6, :] = 0
    H[:h // 6, :] = 0
    od = (np.stack([H, E], axis=2) * strength) @ stains
    return np.clip(256 * 10 ** (-od) - 1, 0, 255).astype(np.uint8)


class TestBasics(unittest.TestCase):
    def test_shape_type_and_repeatability(self):
        a = picture()
        for name, fn in CORRUPTIONS.items():
            for s in SEVERITIES:
                x = fn(a, s, np.random.default_rng(1))
                y = fn(a, s, np.random.default_rng(1))
                self.assertEqual(x.shape, a.shape, name)
                self.assertEqual(x.dtype, np.uint8, name)
                self.assertTrue(np.array_equal(x, y), name)

    def test_input_is_not_changed(self):
        a = picture()
        before = a.copy()
        for fn in CORRUPTIONS.values():
            fn(a, 5, np.random.default_rng(0))
        self.assertTrue(np.array_equal(a, before))

    def test_higher_severity_does_more_damage(self):
        a = picture()
        for name, fn in CORRUPTIONS.items():
            scores = [np.mean([psnr(a, fn(a, s, np.random.default_rng(i))) for i in range(3)]) for s in SEVERITIES]
            self.assertLess(scores[-1], scores[0], name)
            self.assertTrue(all(q <= p + 0.6 for p, q in zip(scores, scores[1:])), (name, scores))

    def test_bad_input(self):
        a = picture()
        with self.assertRaises(ValueError):
            CORRUPTIONS["fog"](a, 6)
        with self.assertRaises(ValueError):
            CORRUPTIONS["fog"](a.astype(float), 1)
        with self.assertRaises(ValueError):
            CORRUPTIONS["fog"](np.zeros((4, 4, 2), np.uint8), 1)

    def test_grey_images_are_accepted(self):
        self.assertEqual(CORRUPTIONS["darken"](np.zeros((9, 9), np.uint8), 1).shape, (9, 9, 3))

    def test_psnr(self):
        a = np.zeros((4, 4, 3), np.uint8)
        self.assertEqual(psnr(a, a), float("inf"))
        self.assertAlmostEqual(psnr(a, a + 255), 0)


class TestConvolution(unittest.TestCase):
    def test_identity(self):
        a = picture().astype(float)
        self.assertTrue(np.allclose(convolve(a, np.array([[1.0]])), a))

    def test_against_direct_sum(self):
        a = np.random.default_rng(1).random((20, 24, 3))
        k = gaussian_kernel(1.0)
        r = k.shape[0] // 2
        p = np.pad(a, ((r, r), (r, r), (0, 0)), mode="reflect")
        want = np.zeros_like(a)
        for y in range(a.shape[0]):
            for x in range(a.shape[1]):
                want[y, x] = (p[y:y + k.shape[0], x:x + k.shape[1]] * k[:, :, None]).sum(axis=(0, 1))
        self.assertTrue(np.allclose(convolve(a, k), want, atol=1e-9))

    def test_kernel_sums_to_one(self):
        self.assertAlmostEqual(gaussian_kernel(2.0).sum(), 1)

    def test_blur_keeps_mean(self):
        a = np.full((32, 32, 3), 100, np.uint8)
        self.assertTrue(np.all(np.abs(convolve(a, gaussian_kernel(3)) - 100) < 1e-6))

    @unittest.skipIf(Image is None, "Pillow is not installed")
    def test_gaussian_blur_is_close_to_pillow(self):
        a = picture()
        ours = np.clip(np.rint(convolve(a, gaussian_kernel(2.0))), 0, 255).astype(np.uint8)
        theirs = np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(2.0)))
        self.assertGreater(psnr(ours[8:-8, 8:-8], theirs[8:-8, 8:-8]), 38)


class TestJpeg(unittest.TestCase):
    def test_quality_order(self):
        a = picture()
        scores = [psnr(a, jpeg_roundtrip(a, q)) for q in (95, 75, 50, 25, 10)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_odd_sizes(self):
        self.assertEqual(jpeg_roundtrip(picture(37, 53), 50).shape, (37, 53, 3))

    def test_flat_image_survives(self):
        a = np.full((32, 32, 3), 128, np.uint8)
        self.assertLess(np.abs(jpeg_roundtrip(a, 10).astype(int) - 128).max(), 2)

    @unittest.skipIf(Image is None, "Pillow is not installed")
    def test_close_to_a_real_jpeg_encoder(self):
        a = picture()
        for q in (90, 50, 20):
            buf = io.BytesIO()
            Image.fromarray(a).save(buf, "JPEG", quality=q, subsampling=2)
            real = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
            self.assertLess(abs(psnr(a, jpeg_roundtrip(a, q)) - psnr(a, real)), 1.5, q)


class TestStain(unittest.TestCase):
    def test_hed_round_trip(self):
        a = tissue()
        self.assertLessEqual(np.abs(hed_to_rgb(rgb_to_hed(a)).astype(int) - a).max(), 2)

    def test_nuclei_appear_in_haematoxylin(self):
        a = tissue()
        h = rgb_to_hed(a)[..., 0]
        dark = a.sum(axis=2) < np.percentile(a.sum(axis=2), 15)
        self.assertGreater(h[dark].mean(), 3 * h[~dark].mean())

    def test_jitter_changes_colour_and_keeps_structure(self):
        a = tissue()
        b = stain_jitter(a, 0.2, 0.05, np.random.default_rng(1))
        self.assertFalse(np.array_equal(a, b))
        self.assertGreater(np.corrcoef(a.mean(axis=2).ravel(), b.mean(axis=2).ravel())[0, 1], 0.9)

    def test_background_stays_pale(self):
        a = tissue()
        b = stain_jitter(a, 0.3, 0.1, np.random.default_rng(2))
        self.assertGreater(b[:10].mean(), 240)

    def test_shift_severity(self):
        a = tissue()
        s = [np.mean([psnr(a, stain_shift(a, k, np.random.default_rng(i))) for i in range(8)]) for k in (1, 5)]
        self.assertGreater(s[0], s[1])

    def test_macenko_recovers_known_stains(self):
        for seed, angle in ((1, 0.0), (2, 0.15)):
            c, s = np.cos(angle), np.sin(angle)
            stains = HED[:2] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            stains /= np.linalg.norm(stains, axis=1, keepdims=True)
            est = macenko_vectors(tissue(256, 256, seed, stains))
            self.assertGreater(float(est[0] @ stains[0]), 0.98)
            self.assertGreater(float(est[1] @ stains[1]), 0.98)

    def test_normalisation_brings_a_shifted_lab_closer(self):
        target = tissue(seed=1)
        c, s = np.cos(0.2), np.sin(0.2)
        other = HED[:2] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        other /= np.linalg.norm(other, axis=1, keepdims=True)
        source = tissue(seed=1, stains=other, strength=1.3)
        fixed = macenko_normalise(source, target)
        self.assertGreater(psnr(fixed, target), psnr(source, target) + 3)

    def test_blank_slide_is_refused(self):
        with self.assertRaises(ValueError):
            macenko_vectors(np.full((32, 32, 3), 255, np.uint8))


class TestBenchmark(unittest.TestCase):
    def test_corruption_error(self):
        ce, mce = corruption_error({"a": [0.2, 0.4], "b": [0.1, 0.1]}, {"a": [0.4, 0.4], "b": [0.1, 0.1]})
        self.assertAlmostEqual(ce["a"], 0.75)
        self.assertAlmostEqual(mce, 0.875)

    def test_table(self):
        images = [picture(32, 32, i) for i in range(6)]
        labels = [0, 1, 0, 1, 0, 1]

        def model(batch):
            return np.array([0 if i % 2 == 0 else 1 for i in range(len(batch))])
        r = robustness_table(model, images, labels, ["darken", "fog"], (1, 3))
        self.assertEqual(r["clean"], 1)
        self.assertEqual(set(r["by_corruption"]), {"darken", "fog"})
        self.assertEqual(len(r["by_corruption"]["fog"]), 2)


if __name__ == "__main__":
    unittest.main()
