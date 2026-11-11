"""Deterministic mild photometric perturbations for the advisory-stability test.

Geometry is never changed, so objects can be matched between the clean and the
perturbed image by box IoU. Each perturbation is mild enough that a human would
give the same advisory.

    dark      gamma 1.5 (darker)
    bright    gamma 0.67 (brighter)
    contrast  contrast scaled by 0.7 around the mean
    blur      Gaussian blur, sigma 1.5 px
    noise     additive Gaussian noise, sigma 8 (seeded by image id)
    jpeg      JPEG re-compression at quality 35
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "camera_ready" / "perturbed"
PERTURBATIONS = ("dark", "bright", "contrast", "blur", "noise", "jpeg")


def gamma(image, value):
    table = np.clip(((np.arange(256) / 255.0) ** value) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


def perturb(image, name, image_id):
    if name == "dark":
        return gamma(image, 1.5)
    if name == "bright":
        return gamma(image, 0.67)
    if name == "contrast":
        mean = image.reshape(-1, 3).mean(axis=0)
        return np.clip((image.astype(np.float32) - mean) * 0.7 + mean, 0, 255).astype(np.uint8)
    if name == "blur":
        return cv2.GaussianBlur(image, (0, 0), 1.5)
    if name == "noise":
        seed = int(hashlib.sha256(image_id.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 8.0, image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if name == "jpeg":
        return image  # the quality-35 encoding happens when the file is written
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-list", required=True)
    args = parser.parse_args()
    paths = [Path(line.strip()) for line in Path(args.image_list).read_text(encoding="utf-8").splitlines() if line.strip()]
    for name in PERTURBATIONS:
        directory = OUT / name
        directory.mkdir(parents=True, exist_ok=True)
        listing = []
        for path in paths:
            image = cv2.imread(str(path))
            output = perturb(image, name, path.stem)
            if name == "jpeg":
                target = directory / f"{path.stem}__{name}.jpg"
                cv2.imwrite(str(target), output, [cv2.IMWRITE_JPEG_QUALITY, 35])
            else:
                target = directory / f"{path.stem}__{name}.png"
                cv2.imwrite(str(target), output)
            listing.append(str(target.resolve()))
        (OUT / f"{name}.txt").write_text("\n".join(listing) + "\n", encoding="utf-8")
        print(f"{name}: {len(listing)} images")


if __name__ == "__main__":
    main()
