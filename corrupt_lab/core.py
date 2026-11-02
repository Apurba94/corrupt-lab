"""Image corruptions at set severities, and stain augmentation for histology, in NumPy.

Images are uint8 arrays of shape H×W×3. Severity runs from 1 to 5, in the style of the
ImageNet-C benchmark (Hendrycks and Dietterich 2019). Every function takes an optional
NumPy generator, so that a run can be repeated exactly.
"""
import numpy as np

SEVERITIES = (1, 2, 3, 4, 5)


def _rng(rng):
    return rng if rng is not None else np.random.default_rng(0)


def _check(image):
    a = np.asarray(image)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=2)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("expected an H×W×3 image")
    if a.dtype != np.uint8:
        raise ValueError("expected uint8 pixels")
    return a


def _pick(values, severity):
    if severity not in SEVERITIES:
        raise ValueError("severity must be 1 to 5")
    return values[severity - 1]


def _to_uint8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def convolve(image, kernel):
    """Convolve each channel with a 2-D kernel, reflecting at the edges. Uses the FFT, so large kernels are fine."""
    a = np.asarray(image, float)
    k = np.asarray(kernel, float)
    ph, pw = k.shape[0] // 2, k.shape[1] // 2
    padded = np.pad(a, ((ph, ph), (pw, pw), (0, 0)), mode="reflect")
    shape = (padded.shape[0] + k.shape[0] - 1, padded.shape[1] + k.shape[1] - 1)
    fk = np.fft.rfft2(k, shape)
    out = np.fft.irfft2(np.fft.rfft2(padded, shape, axes=(0, 1)) * fk[:, :, None], shape, axes=(0, 1))
    y, x = k.shape[0] - 1, k.shape[1] - 1
    return out[y:y + a.shape[0], x:x + a.shape[1]]


def gaussian_kernel(sigma):
    r = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-r, r + 1)
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


def gaussian_noise(image, severity=1, rng=None):
    a = _check(image)
    s = _pick((0.04, 0.06, 0.08, 0.09, 0.10), severity) * 255
    return _to_uint8(a + _rng(rng).normal(0, s, a.shape))


def shot_noise(image, severity=1, rng=None):
    a = _check(image)
    c = _pick((60, 25, 12, 5, 3), severity)
    return _to_uint8(_rng(rng).poisson(a / 255.0 * c) / c * 255)


def impulse_noise(image, severity=1, rng=None):
    a = _check(image).copy()
    p = _pick((0.01, 0.02, 0.03, 0.05, 0.07), severity)
    r = _rng(rng)
    mask = r.random(a.shape[:2])
    a[mask < p / 2] = 0
    a[mask > 1 - p / 2] = 255
    return a


def gaussian_blur(image, severity=1, rng=None):
    a = _check(image)
    return _to_uint8(convolve(a, gaussian_kernel(_pick((0.6, 1.0, 1.5, 2.0, 3.0), severity))))


def defocus_blur(image, severity=1, rng=None):
    a = _check(image)
    r = _pick((1.5, 2.5, 3.5, 5, 7), severity)
    n = int(np.ceil(r))
    y, x = np.mgrid[-n:n + 1, -n:n + 1]
    disc = (x ** 2 + y ** 2 <= r ** 2).astype(float)
    return _to_uint8(convolve(a, disc / disc.sum()))


def motion_blur(image, severity=1, rng=None):
    a = _check(image)
    length = _pick((5, 9, 13, 19, 25), severity)
    angle = _rng(rng).uniform(0, np.pi)
    n = length | 1
    k = np.zeros((n, n))
    c = n // 2
    for t in np.linspace(-c, c, n * 3):
        k[int(round(c + t * np.sin(angle))), int(round(c + t * np.cos(angle)))] = 1
    return _to_uint8(convolve(a, k / k.sum()))


def brightness(image, severity=1, rng=None):
    a = _check(image)
    return _to_uint8(a + _pick((0.1, 0.2, 0.3, 0.4, 0.5), severity) * 255)


def darken(image, severity=1, rng=None):
    a = _check(image)
    return _to_uint8(a * _pick((0.85, 0.7, 0.55, 0.4, 0.25), severity))


def contrast(image, severity=1, rng=None):
    a = _check(image).astype(float)
    c = _pick((0.75, 0.6, 0.45, 0.3, 0.2), severity)
    m = a.mean(axis=(0, 1), keepdims=True)
    return _to_uint8((a - m) * c + m)


def pixelate(image, severity=1, rng=None):
    a = _check(image)
    f = _pick((2, 3, 4, 6, 8), severity)
    h, w = a.shape[:2]
    out = a.astype(float).copy()
    for y in range(0, h, f):
        for x in range(0, w, f):
            out[y:y + f, x:x + f] = out[y:y + f, x:x + f].mean(axis=(0, 1))
    return _to_uint8(out)


def fog(image, severity=1, rng=None):
    a = _check(image).astype(float)
    r = _rng(rng)
    amount = _pick((0.25, 0.4, 0.55, 0.7, 0.8), severity)
    h, w = a.shape[:2]
    cloud = np.zeros((h, w))
    for scale in (4, 8, 16, 32):
        small = r.random((h // scale + 2, w // scale + 2))
        ys = np.linspace(0, small.shape[0] - 1.001, h)
        xs = np.linspace(0, small.shape[1] - 1.001, w)
        y0, x0 = ys.astype(int), xs.astype(int)
        fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
        cloud += (small[np.ix_(y0, x0)] * (1 - fy) * (1 - fx) + small[np.ix_(y0 + 1, x0)] * fy * (1 - fx) + small[np.ix_(y0, x0 + 1)] * (1 - fy) * fx + small[np.ix_(y0 + 1, x0 + 1)] * fy * fx) * scale
    cloud = (cloud - cloud.min()) / max(np.ptp(cloud), 1e-9)
    alpha = (amount * (0.6 + 0.4 * cloud))[:, :, None]
    return _to_uint8(a * (1 - alpha) + 255 * alpha)


JPEG_LUMA = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92], [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], float)
JPEG_CHROMA = np.array([[17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99], [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99], [99] * 8, [99] * 8, [99] * 8, [99] * 8], float)
_DCT = np.array([[np.sqrt((1 if u == 0 else 2) / 8) * np.cos((2 * x + 1) * u * np.pi / 16) for x in range(8)] for u in range(8)])


def _scale(table, quality):
    s = 5000 / quality if quality < 50 else 200 - 2 * quality
    return np.clip(np.floor((table * s + 50) / 100), 1, 255)


def jpeg_roundtrip(image, quality=75):
    """What JPEG does to pixels: colour conversion, 4:2:0 chroma, 8×8 DCT and quantisation."""
    a = _check(image).astype(float)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128
    h, w = Y.shape

    def blocks(ch, table):
        H, W = ch.shape
        ph, pw = (-H) % 8, (-W) % 8
        c = np.pad(ch, ((0, ph), (0, pw)), mode="edge") - 128
        tiled = c.reshape(c.shape[0] // 8, 8, c.shape[1] // 8, 8).transpose(0, 2, 1, 3)
        coef = _DCT @ tiled @ _DCT.T
        q = np.rint(coef / table) * table
        back = _DCT.T @ q @ _DCT
        return (back.transpose(0, 2, 1, 3).reshape(c.shape) + 128)[:H, :W]

    def down(ch):
        H, W = ch.shape
        p = np.pad(ch, ((0, H % 2), (0, W % 2)), mode="edge")
        return p.reshape(p.shape[0] // 2, 2, p.shape[1] // 2, 2).mean(axis=(1, 3))

    def up(ch):
        return np.repeat(np.repeat(ch, 2, axis=0), 2, axis=1)[:h, :w]
    ly = _scale(JPEG_LUMA, quality)
    lc = _scale(JPEG_CHROMA, quality)
    y2 = blocks(Y, ly)
    cb2 = up(blocks(down(Cb), lc))
    cr2 = up(blocks(down(Cr), lc))
    r = y2 + 1.402 * (cr2 - 128)
    g = y2 - 0.344136 * (cb2 - 128) - 0.714136 * (cr2 - 128)
    b = y2 + 1.772 * (cb2 - 128)
    return _to_uint8(np.stack([r, g, b], axis=2))


def jpeg(image, severity=1, rng=None):
    return jpeg_roundtrip(image, _pick((60, 40, 25, 15, 8), severity))


CORRUPTIONS = {"gaussian_noise": gaussian_noise, "shot_noise": shot_noise, "impulse_noise": impulse_noise, "gaussian_blur": gaussian_blur, "defocus_blur": defocus_blur, "motion_blur": motion_blur,
               "brightness": brightness, "darken": darken, "contrast": contrast, "pixelate": pixelate, "fog": fog, "jpeg": jpeg}

# Stain separation (Ruifrok and Johnston 2001): optical density is a linear mix of stains.
HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
HED = HED / np.linalg.norm(HED, axis=1, keepdims=True)
HED_INV = np.linalg.inv(HED)


def to_optical_density(image):
    return -np.log10((np.asarray(image, float) + 1) / 256)


def from_optical_density(od):
    return _to_uint8(256 * 10 ** (-od) - 1)


def rgb_to_hed(image):
    """Separate an image into haematoxylin, eosin and residual channels."""
    return to_optical_density(_check(image)) @ HED_INV


def hed_to_rgb(hed):
    return from_optical_density(np.asarray(hed, float) @ HED)


def stain_jitter(image, sigma=0.05, bias=0.05, rng=None):
    """Tellez-style augmentation: scale and shift each stain channel by a small random amount."""
    r = _rng(rng)
    h = rgb_to_hed(image)
    alpha = r.uniform(1 - sigma, 1 + sigma, 3)
    beta = r.uniform(-bias, bias, 3)
    return hed_to_rgb(h * alpha + beta * (h.sum(axis=2, keepdims=True) > 0.02))


def stain_shift(image, severity=1, rng=None):
    """A strong stain change, to simulate a different laboratory."""
    s = _pick((0.05, 0.1, 0.2, 0.3, 0.4), severity)
    return stain_jitter(image, s, s / 3, rng)


def macenko_vectors(image, beta=0.15, alpha=1.0):
    """Estimate an image's two stain vectors (Macenko et al. 2009)."""
    od = to_optical_density(_check(image)).reshape(-1, 3)
    od = od[(od > beta).any(axis=1)]
    if len(od) < 10:
        raise ValueError("the image has too little tissue")
    _, vecs = np.linalg.eigh(np.cov(od, rowvar=False))
    plane = vecs[:, 1:3]
    if plane[:, 0].sum() < 0:
        plane[:, 0] *= -1
    if plane[:, 1].sum() < 0:
        plane[:, 1] *= -1
    t = od @ plane
    phi = np.arctan2(t[:, 1], t[:, 0])
    lo, hi = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)
    a = plane @ np.array([np.cos(lo), np.sin(lo)])
    b = plane @ np.array([np.cos(hi), np.sin(hi)])
    stains = np.stack([a, b] if a[0] > b[0] else [b, a])        # haematoxylin first: more red absorbed
    return stains / np.linalg.norm(stains, axis=1, keepdims=True)


def macenko_normalise(image, target):
    """Recolour `image` to match the stain look of `target`, keeping its structure."""
    src_v, tgt_v = macenko_vectors(image), macenko_vectors(target)
    src_od = to_optical_density(_check(image)).reshape(-1, 3)
    tgt_od = to_optical_density(_check(target)).reshape(-1, 3)
    c_src = np.linalg.lstsq(src_v.T, src_od.T, rcond=None)[0].T
    c_tgt = np.linalg.lstsq(tgt_v.T, tgt_od.T, rcond=None)[0].T
    scale = np.percentile(c_tgt, 99, axis=0) / np.maximum(np.percentile(c_src, 99, axis=0), 1e-6)
    return from_optical_density((c_src * scale @ tgt_v).reshape(np.asarray(image).shape))


def psnr(a, b):
    mse = np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def corruption_error(errors, baseline):
    """errors and baseline: {corruption: [error at severity 1..5]}. mCE compares with a baseline
    model, and the relative form subtracts clean error (pass `clean` to `robustness_table`)."""
    ce = {}
    for name, values in errors.items():
        ce[name] = sum(values) / max(sum(baseline[name]), 1e-12)
    return ce, sum(ce.values()) / len(ce)


def robustness_table(model, images, labels, corruptions=None, severities=SEVERITIES, seed=0):
    """Run model(batch) -> predictions on each corruption. Returns clean accuracy and accuracy by severity."""
    labels = np.asarray(labels)
    clean = float(np.mean(np.asarray(model(np.stack(images))) == labels))
    table = {}
    for name in corruptions or CORRUPTIONS:
        fn = CORRUPTIONS[name]
        row = []
        for s in severities:
            rng = np.random.default_rng([seed, s])
            batch = np.stack([fn(im, s, rng) for im in images])
            row.append(float(np.mean(np.asarray(model(batch)) == labels)))
        table[name] = row
    return {"clean": clean, "by_corruption": table, "mean": float(np.mean([v for r in table.values() for v in r]))}
