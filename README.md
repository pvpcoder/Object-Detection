# Object Detection with Faster R-CNN (Open Images V4)

This project uses TensorFlow Hub's pre-trained **Faster R-CNN + Inception-ResNet** model
(trained on Open Images V4, 600 object classes) to detect objects from your webcam.

Since this model is accurate but too slow to run on every video frame on a normal CPU,
`webcam_ssd.py` shows an instant live camera preview and only runs detection once, on a
single frozen snapshot, when you press SPACE.

## Features

- Live webcam preview with no detection overhead
- Press SPACE to run detection on the current frame; results stay on screen until you
  press another key
- Draws bounding boxes, labels, and confidence scores for detected objects
- Prints the top 15 raw candidate detections to the console every time you detect, so you
  can see what the model considered even if nothing cleared the confidence threshold

## Requirements

- Python 3.9–3.11 (TensorFlow does not yet support 3.12+ on all platforms)
- A webcam

## Setup (any device)

1. **Clone the repo and open this folder:**

   ```bash
   git clone <this-repo-url>
   cd Object-Detection
   ```

2. **Create and activate a virtual environment** (strongly recommended — installing into
   your system Python can collide with other packages, which is what caused several of the
   errors this script originally ran into):

   ```bash
   python -m venv venv
   ```

   Windows (PowerShell):
   ```powershell
   venv\Scripts\activate
   ```

   macOS / Linux:
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies from `requirements.txt`:**

   ```bash
   pip install -r requirements.txt
   ```

   This pins `numpy<2`, which matters: newer TensorFlow builds pull in `jaxlib`, which
   isn't built for NumPy 2.x yet and will crash with an `_ARRAY_API not found` error if a
   newer NumPy is already installed.

4. **Run it:**

   ```bash
   python webcam_ssd.py
   ```

   The first run downloads the model from TensorFlow Hub (several hundred MB), so it may
   take a few minutes and needs an internet connection. It's cached locally after that.

## Usage

- A live preview window opens immediately.
- Press **SPACE** to freeze the current frame and run detection on it. This can take
  anywhere from several seconds to over a minute on CPU — the window shows a red
  "Detecting..." message while it works.
- Press **any key** to dismiss the result and return to the live preview.
- Press **Q** (from the live preview) to quit.

## Tuning

A few constants near the top of `webcam_ssd.py` are worth adjusting per device/use case:

- `SCORE_THRESHOLD` (default `0.3`) — lower it to see lower-confidence detections; raise it
  to reduce false positives.
- `INFER_SIZE` (default `800`) — the resolution the frame is resized/letterboxed to before
  being fed to the model. Higher values preserve more detail (helpful for small or
  far-away objects) at the cost of slower detection; lower it on a slow machine if a
  single detection is taking too long.

## Troubleshooting

- **`ModuleNotFoundError`** — you're likely running the system Python instead of the venv
  the dependencies were installed into. Make sure the venv is activated, or point your
  IDE/terminal at `venv/Scripts/python.exe` (Windows) / `venv/bin/python` (macOS/Linux).
- **`_ARRAY_API not found` / NumPy 2.x crash** — reinstall with
  `pip install -r requirements.txt`, which pins a compatible NumPy version.
- **Webcam won't open** — another application may be using it, or you may need to change
  `cv2.VideoCapture(0)` to a different index (`1`, `2`, ...) if you have multiple cameras.
- **An object isn't being detected** — check the console's "Top detections" printout after
  pressing SPACE. If the object doesn't appear at all, try better/more even lighting
  (avoid strong backlight), fill more of the frame with the object, and make sure you
  aren't also in frame — a visible person tends to dominate the top of the ranking.
