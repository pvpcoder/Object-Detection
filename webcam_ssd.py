import tensorflow_hub as hub
import tensorflow as tf
import cv2

# Open Images V4 model: 600 classes, broader household-object coverage than COCO.
# It returns human-readable class names directly, so no manual label list needed.
model = hub.load("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1")

SCORE_THRESHOLD = 0.3

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to read frame from webcam.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor([frame_rgb], dtype=tf.float32) / 255.0

    result = model(input_tensor)
    result = {key: value.numpy() for key, value in result.items()}

    h, w, _ = frame.shape

    for i in range(result["detection_boxes"].shape[0]):
        score = result["detection_scores"][i]
        if score < SCORE_THRESHOLD:
            continue

        box = result["detection_boxes"][i]
        label = result["detection_class_entities"][i].decode("utf-8")

        ymin, xmin, ymax, xmax = box
        left, top, right, bottom = (
            int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
        )

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        text_y = top - 10 if top - 10 > 10 else top + 20
        cv2.putText(frame, f"{label} ({score:.2f})", (left, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Detections", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
