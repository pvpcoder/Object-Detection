import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import cv2

# Open Images V4, Faster R-CNN + Inception-ResNet: 600-class vocabulary, much
# higher accuracy than the old SSD MobileNetV2 model, but far too slow on CPU
# for live video (can take well over a minute per frame). So instead of running
# it continuously, this script shows an instant live preview and only runs the
# model once, on a single frozen snapshot, when you press SPACE.
model = hub.load("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1")
detect = model.signatures["default"]

SCORE_THRESHOLD = 0.3
INFER_SIZE = 800  # snapshot mode only runs once per press, so detail > speed
FONT = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)


def letterbox(frame, size):
    """Resize preserving aspect ratio and pad to a square, instead of
    stretching, since squishing distorts object shape and hurts accuracy."""
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top


def run_detection(frame):
    frame_h, frame_w = frame.shape[:2]
    canvas, scale, pad_left, pad_top = letterbox(frame, INFER_SIZE)
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor([canvas_rgb], dtype=tf.float32) / 255.0

    result = detect(input_tensor)
    result = {key: value.numpy() for key, value in result.items()}

    raw = []
    for i in range(result["detection_boxes"].shape[0]):
        score = float(result["detection_scores"][i])
        label = result["detection_class_entities"][i].decode("utf-8")
        ymin, xmin, ymax, xmax = result["detection_boxes"][i]
        # map normalized canvas coords back to normalized original-frame coords
        x1 = (xmin * INFER_SIZE - pad_left) / scale / frame_w
        y1 = (ymin * INFER_SIZE - pad_top) / scale / frame_h
        x2 = (xmax * INFER_SIZE - pad_left) / scale / frame_w
        y2 = (ymax * INFER_SIZE - pad_top) / scale / frame_h
        raw.append(((y1, x1, y2, x2), label, score))

    # Diagnostic: show what the model actually sees, regardless of threshold,
    # so a missing class is visible instead of silent. Person/face/clothing
    # tend to dominate the top of the ranking, so this prints further down
    # the list than just the top handful.
    ranked = sorted(raw, key=lambda d: -d[2])[:15]
    print("Top detections:", ", ".join(f"{l} {s:.2f}" for _, l, s in ranked))

    return [d for d in raw if d[2] >= SCORE_THRESHOLD]


def draw_detections(frame, detections):
    h, w, _ = frame.shape
    for box, label, score in detections:
        ymin, xmin, ymax, xmax = box
        left, top, right, bottom = (
            int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
        )
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        text_y = top - 10 if top - 10 > 10 else top + 20
        cv2.putText(frame, f"{label} ({score:.2f})", (left, text_y),
            FONT, 0.6, (0, 255, 0), 2)


while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to read frame from webcam.")
        break

    preview = frame.copy()
    cv2.putText(preview, "SPACE: detect   Q: quit", (10, 30),
        FONT, 0.7, (0, 255, 255), 2)
    cv2.imshow("Detections", preview)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord(' '):
        busy = frame.copy()
        cv2.putText(busy, "Detecting... this can take a while on CPU", (10, 30),
            FONT, 0.7, (0, 0, 255), 2)
        cv2.imshow("Detections", busy)
        cv2.waitKey(1)  # force the "Detecting..." frame to actually render first

        detections = run_detection(frame)

        result_frame = frame.copy()
        draw_detections(result_frame, detections)
        cv2.putText(result_frame, "Press any key to resume live view",
            (10, result_frame.shape[0] - 10), FONT, 0.6, (0, 255, 255), 2)
        cv2.imshow("Detections", result_frame)
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
