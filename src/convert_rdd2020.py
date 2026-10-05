from pathlib import Path
from shutil import copy2
from collections import Counter
import random
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "data" / "train"
DEST_DIR = PROJECT_ROOT / "data" / "yolo"
CLASS_MAPPING = {
    "D00": 0,
    "D10": 1,
    "D20": 2,
    "D40": 3
}

random.seed(42)


def convert_voc_to_yolo(voc_path, yolo_path, stats):
    try:
        tree = ET.parse(voc_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML {voc_path}: {e}")
        with open(yolo_path, "w") as f:
            f.write("")
        return

    size_elem = root.find("size")
    if size_elem is not None:
        w_elem = size_elem.find("width")
        h_elem = size_elem.find("height")
        width = int(w_elem.text) if w_elem is not None and w_elem.text else 0
        height = int(h_elem.text) if h_elem is not None and h_elem.text else 0
    else:
        width, height = 0, 0

    objects = root.findall("object")
    if len(objects) == 0:
        stats["xml_no_objects"] += 1

    yolo_annotations = []

    for obj in objects:
        name_elem = obj.find("name")
        class_name = name_elem.text.strip() if name_elem is not None and name_elem.text else ""

        if class_name not in CLASS_MAPPING:
            stats["objects_unknown_class"] += 1
            stats["unknown_classes"][class_name] += 1
            continue

        class_id = CLASS_MAPPING[class_name]
        stats["class_counts"][class_name] += 1
        stats["total_converted_annotations"] += 1

        bbox = obj.find("bndbox")
        if bbox is None or width == 0 or height == 0:
            continue

        xmin_el = bbox.find("xmin")
        ymin_el = bbox.find("ymin")
        xmax_el = bbox.find("xmax")
        ymax_el = bbox.find("ymax")

        if xmin_el is None or ymin_el is None or xmax_el is None or ymax_el is None:
            continue
        if xmin_el.text is None or ymin_el.text is None or xmax_el.text is None or ymax_el.text is None:
            continue

        xmin = float(xmin_el.text)
        ymin = float(ymin_el.text)
        xmax = float(xmax_el.text)
        ymax = float(ymax_el.text)

        # Clip coordinates within image bounds
        xmin = max(0.0, min(float(width), xmin))
        xmax = max(0.0, min(float(width), xmax))
        ymin = max(0.0, min(float(height), ymin))
        ymax = max(0.0, min(float(height), ymax))

        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0

        box_width = xmax - xmin
        box_height = ymax - ymin

        x_center /= width
        y_center /= height
        box_width /= width
        box_height /= height

        yolo_annotations.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

    if len(yolo_annotations) == 0:
        stats["empty_labels"] += 1

    with open(yolo_path, "w") as f:
        f.write("\n".join(yolo_annotations))


def split_images(images, val_ratio=0.2):
    images = list(images)
    random.shuffle(images)
    val_size = int(len(images) * val_ratio)
    val_images = images[:val_size]
    train_images = images[val_size:]
    return train_images, val_images


def process_country(country, images, split, stats):
    country_dir = SOURCE_DIR / country
    annotations_dir = country_dir / "annotations" / "xmls"

    for image_path in images:
        stats["images_processed"] += 1
        stats[f"images_{split}"] += 1

        xml_path = annotations_dir / f"{image_path.stem}.xml"

        if not xml_path.exists():
            stats["missing_xml"] += 1
            print(f"Missing annotation: {xml_path}")
            continue

        image_dest = DEST_DIR / split / "images" / image_path.name
        label_dest = DEST_DIR / split / "labels" / f"{image_path.stem}.txt"

        copy2(image_path, image_dest)
        convert_voc_to_yolo(xml_path, label_dest, stats)

    print(f"{country}: {split} -> {len(images)} images")


def create_data_yaml():
    yaml_path = DEST_DIR / "data.yaml"
    yaml_content = (
        f"path: {DEST_DIR.as_posix()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"\n"
        f"names:\n"
        f"  0: D00\n"
        f"  1: D10\n"
        f"  2: D20\n"
        f"  3: D40\n"
    )
    with open(yaml_path, "w") as f:
        f.write(yaml_content)


def main():
    for split in ["train", "val"]:
        (DEST_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DEST_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {
        "images_processed": 0,
        "images_train": 0,
        "images_val": 0,
        "total_converted_annotations": 0,
        "xml_no_objects": 0,
        "objects_unknown_class": 0,
        "unknown_classes": Counter(),
        "class_counts": Counter(),
        "missing_xml": 0,
        "empty_labels": 0,
    }

    countries = ["Czech", "India", "Japan"]

    for country in countries:
        images_dir = SOURCE_DIR / country / "images"
        if not images_dir.exists():
            print(f"Warning: Directory not found {images_dir}")
            continue

        images = [
            image
            for image in sorted(images_dir.iterdir())
            if image.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        train_images, val_images = split_images(images)

        process_country(country, train_images, "train", stats)
        process_country(country, val_images, "val", stats)

    create_data_yaml()

if __name__ == "__main__":
    main()