# Corrupt Lab

How does your image model behave when the camera is blurred, the light is poor, or the slide came from a different laboratory? Corrupt Lab applies 12 standard corruptions at five severities, plus stain changes for histology, in plain NumPy.

```python
import numpy as np
from corrupt_lab import CORRUPTIONS, robustness_table, stain_jitter

blurred = CORRUPTIONS["defocus_blur"](image, severity=3, rng=np.random.default_rng(0))
augmented = stain_jitter(slide, sigma=0.1, rng=np.random.default_rng(0))

result = robustness_table(my_model, images, labels)      # accuracy by corruption and severity
print(result["clean"], result["mean"])
```

To see what each one does to an image (this needs Pillow):

```
python -m corrupt_lab photo.png out 3
```

## What is included

| Family | Corruptions |
|---|---|
| Noise | Gaussian, shot, impulse |
| Blur | Gaussian, defocus, motion |
| Light | Brightness, darkening, contrast, fog |
| Digital | Pixelate, JPEG |

**Stain tools for histology and blood smears:**

- Colour deconvolution into haematoxylin, eosin and residual channels (Ruifrok and Johnston)
- `stain_jitter`, the augmentation that makes models robust across laboratories (in the style of Tellez et al.)
- `stain_shift`, to simulate a new laboratory at a chosen severity
- Macenko stain vector estimation and normalisation

**Benchmarks:** `robustness_table` and the corruption error from the ImageNet-C paper.

The JPEG corruption is a real JPEG pipeline without the file format: colour conversion, 4:2:0 chroma, 8×8 DCT and the standard quantisation tables.

## How it is tested

- Every corruption keeps the image shape and type, repeats exactly with the same seed, leaves the input untouched, and does more damage at higher severity.
- The fast FFT convolution matches a direct sum.
- With Pillow installed, the blur is compared with Pillow's, and the JPEG with Pillow's real encoder, whose quality must be within 1.5 dB.
- Macenko estimation recovers known stain vectors from synthetic slides, and normalisation raises PSNR against the target by at least 3 dB.

## Limits

Severity scales are similar in spirit to ImageNet-C but are not the same values, so scores cannot be compared with published ImageNet-C numbers. Macenko is tested on synthetic slides here, so check it on real images before relying on it. Macenko assumes two stains. Motion blur and fog use the seeded generator, so keep the seed if you need matching images.

## Tests

```
pip install -r requirements.txt
python -m unittest discover -s tests
```

MIT licence.
