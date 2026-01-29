import random
import shutil
from pathlib import Path

import config

TRAIN_DIR = config.TRAIN_DIR
VALID_DIR = config.VALID_DIR
TEST_DIR = config.TEST_DIR

# Fraction of validation images to move to test (per class)
TEST_FRACTION = 0.2  # 20%

# Image extensions to consider
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix in IMAGE_EXTS


def count_images_per_split():
    def count_in(root: Path):
        total = 0
        per_class = {}
        if not root.exists():
            return total, per_class
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            imgs = [p for p in class_dir.iterdir() if is_image(p)]
            per_class[class_dir.name] = len(imgs)
            total += len(imgs)
        return total, per_class

    train_total, _ = count_in(TRAIN_DIR)
    valid_total, valid_per_class = count_in(VALID_DIR)
    test_total, test_per_class = count_in(TEST_DIR)
    return (train_total, valid_total, test_total,
            valid_per_class, test_per_class)


def main():
    random.seed(42)  # reproducibility

    if not VALID_DIR.exists():
        raise SystemExit(f"Validation directory not found: {VALID_DIR}")

    TEST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using TEST_FRACTION = {TEST_FRACTION:.2f}")
    print(f"VALID_DIR = {VALID_DIR}")
    print(f"TEST_DIR  = {TEST_DIR}")

    # Pre‑split counts
    train_total, valid_total, test_total, _, _ = count_images_per_split()
    print("\nBefore split:")
    print(f"  Train images: {train_total}")
    print(f"  Valid images: {valid_total}")
    print(f"  Test  images: {test_total}")

    # Loop over each class folder in data/valid
    for class_dir in sorted(VALID_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")

        # Collect image files for this class
        images = [p for p in class_dir.iterdir() if is_image(p)]
        n_total = len(images)
        if n_total == 0:
            print("  No images found, skipping.")
            continue

        # How many to move
        n_test = max(1, int(round(n_total * TEST_FRACTION)))
        # Guard: if n_total == 1, don't move it (keep at least 1 in valid)
        if n_total == 1:
            n_test = 0

        print(f"  Total images in valid: {n_total}, moving to test: {n_test}")
        if n_test == 0:
            print("  Not moving any images for this class.")
            continue

        # Randomly pick images for test
        random.shuffle(images)
        test_images = images[:n_test]

        # Ensure class folder exists in data/test
        test_class_dir = TEST_DIR / class_name
        test_class_dir.mkdir(parents=True, exist_ok=True)

        # Move images
        for img_path in test_images:
            dest_path = test_class_dir / img_path.name
            shutil.move(str(img_path), str(dest_path))

        print(f"  Moved {n_test} images to {test_class_dir}")

    # Post‑split counts
    (train_total, valid_total, test_total, valid_per_class, test_per_class) = count_images_per_split()

    print("\nAfter split:")
    print(f"  Train images: {train_total}")
    print(f"  Valid images: {valid_total}")
    print(f"  Test  images: {test_total}")

    # Optional: print a small summary for a few classes
    print("\nPer‑class summary (first 10 classes):")
    for class_name in sorted(valid_per_class.keys())[:10]:
        v = valid_per_class.get(class_name, 0)
        t = test_per_class.get(class_name, 0)
        print(f"  {class_name}: valid={v}, test={t}")

    print("\nDone. You now have a labeled data/test/ split.")


if __name__ == "__main__":
    main()