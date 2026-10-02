from pathlib import Path
from shutil import copy2
import random
import xml.etree.ElementTree as ET

SOURCE_DIR = Path("data/train")
DEST_DIR = Path("data/yolo")

CLASS_MAPPING = {
    "D00": 0,
    "D10": 1,
    "D20": 2,
    "D40": 3
}

random.seed(42)


for split in ["train", "val"]:
    (DEST_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (DEST_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

def convert_voc_to_yolo(voc_path, yolo_path):

    tree = ET.parse(voc_path)
    root = tree.getroot()

    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    yolo_annotations = []

    for obj in root.findall("object"):

        class_name = obj.find("name").text

        if class_name not in CLASS_MAPPING:
            continue

        class_id = CLASS_MAPPING[class_name]

        bbox = obj.find("bndbox")

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2

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

    with open(yolo_path, "w") as f:
        f.write("\n".join(yolo_annotations))


def split_images(images, val_ratio=0.2):

    images = list(images)

    random.shuffle(images)

    val_size = int(len(images) * val_ratio)

    val_images = images[:val_size]
    train_images = images[val_size:]

    return train_images, val_images

def process_country(country, images, split):

    country_dir = SOURCE_DIR / country

    annotations_dir = country_dir / "annotations" / "xmls"

    for image_path in images:

        xml_path = annotations_dir / f"{image_path.stem}.xml"

        if not xml_path.exists():
            print(f"Missing annotation: {xml_path}")
            continue

        image_dest = DEST_DIR / split / "images" / image_path.name

        label_dest = DEST_DIR / split / "labels" / f"{image_path.stem}.txt"

        # Copy image
        copy2(image_path, image_dest)

        # Convert XML annotation
        convert_voc_to_yolo(xml_path, label_dest)

    print(f"{country}: {split} → {len(images)} images")

def main():

    countries = ["Czech", "India", "Japan"]

    for country in countries:

        images_dir = SOURCE_DIR / country / "images"

        images = [
            image
            for image in images_dir.iterdir()
            if image.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        train_images, val_images = split_images(images)

        process_country(
            country,
            train_images,
            "train"
        )

        process_country(
            country,
            val_images,
            "val"
        )


if __name__ == "__main__":
    main()