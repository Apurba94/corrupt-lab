"""Write every corruption of an image into a folder, to see what they look like.

Usage: python -m corrupt_lab image.png out_folder [severity]
Needs Pillow to read and write image files.
"""
import os
import sys

import numpy as np

from . import CORRUPTIONS, psnr, stain_shift


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    from PIL import Image
    image = np.array(Image.open(argv[1]).convert("RGB"))
    severity = int(argv[3]) if len(argv) > 3 else 3
    os.makedirs(argv[2], exist_ok=True)
    for name, fn in {**CORRUPTIONS, "stain_shift": stain_shift}.items():
        out = fn(image, severity, np.random.default_rng(0))
        Image.fromarray(out).save(os.path.join(argv[2], f"{name}_{severity}.png"))
        print(f"{name:<16} PSNR {psnr(image, out):5.1f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
