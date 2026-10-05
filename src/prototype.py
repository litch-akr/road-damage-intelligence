from pathlib import Path

import gradio as gr
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "yolo26n_rdd2020_768_best.pt"

model = YOLO(str(MODEL_PATH))

def detect_damage(image):
    if image is None:
        return None, "Please upload a road image."

    results = model.predict(
        source=image,
        imgsz=768,
        conf=0.25,
        verbose=False,
    )

    result = results[0]
    annotated = result.plot()
    annotated = Image.fromarray(annotated[:, :, ::-1])

    if result.boxes is None or len(result.boxes) == 0:
        return annotated, "No road damage detected."

    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        detections.append(
            f"- **{class_name}** — {confidence:.1%} confidence"
        )

    summary = (
        f"### Detected damage: {len(detections)}\n\n"
        + "\n".join(detections)
    )

    return annotated, summary


with gr.Blocks(title="Road Damage Intelligence") as app:

    gr.Markdown(
        """
        # Road Damage Intelligence

        **YOLO26n — 768×768**

        Upload a road image to detect road damage.
        """
    )

    with gr.Row():

        with gr.Column():
            input_image = gr.Image(
                type="pil",
                label="Road Image"
            )

            detect_button = gr.Button(
                "Detect Road Damage",
                variant="primary"
            )

        with gr.Column():
            output_image = gr.Image(
                label="Detection Result"
            )

            output_text = gr.Markdown(
                "Upload an image and click **Detect Road Damage**."
            )

    detect_button.click(
        fn=detect_damage,
        inputs=input_image,
        outputs=[output_image, output_text],
    )

    gr.Markdown(
        """
        ### Classes

        - **D00** — longitudinal crack
        - **D10** — transverse crack
        - **D20** — alligator crack
        - **D40** — pothole
        """
    )

if __name__ == "__main__":
    app.launch()