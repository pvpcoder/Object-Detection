import tensorflow_hub as hub
import tensorflow as tf
import cv2
import numpy as np

model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

labels = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack',
    'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror',
    'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

image_path = r"" ##Insert image path in these qoutations

image = cv2.imread(image_path)
if image is None:
    print("ERROR: Image not found or path is incorrect.")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_tensor = tf.convert_to_tensor([image_rgb], dtype=tf.uint8)

result = model(input_tensor)
result = {key: value.numpy() for key, value in result.items()}

h, w, _ = image.shape

for i in range(result["detection_boxes"].shape[1]):2
    score = result["detection_scores"][0][i]
    if score > 0.5:  
        box = result["detection_boxes"][0][i]
        class_id = int(result["detection_classes"][0][i])
        label = labels[class_id] if class_id < len(labels) else f"unknown ({class_id})"

        ymin, xmin, ymax, xmax = box
        (left, top, right, bottom) = (
            int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
        )

        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        text_y = top - 10 if top - 10 > 10 else top + 20
        cv2.putText(image, f"{label} ({score:.2f})", (left, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)  # reezize windows
cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
