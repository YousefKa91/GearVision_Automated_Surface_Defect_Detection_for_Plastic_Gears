import cv2
import os
import sys
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
IMAGE_FOLDER = os.path.join(BASE_DIR, "test_images")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "results")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        sys.exit(1)

    model = YOLO(MODEL_PATH)

    images = [
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not images:
        print(f"No images found in {IMAGE_FOLDER}")
        sys.exit(1)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Running on {len(images)} images...\n")

    for img_name in images:
        img_path = os.path.join(IMAGE_FOLDER, img_name)
        results = model.predict(source=img_path, conf=0.25, verbose=False)

        for r in results:
            if len(r.boxes) == 0:
                print(f"{img_name}: no defects detected")
            else:
                detections = [
                    f"{model.names[int(box.cls)]} ({float(box.conf):.0%})"
                    for box in r.boxes
                ]
                print(f"{img_name}: {', '.join(detections)}")

            # Save annotated image to results folder
            out_path = os.path.join(OUTPUT_FOLDER, img_name)
            cv2.imwrite(out_path, r.plot())

    print(f"\nDone. Results saved to {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
