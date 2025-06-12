# 🎯 YOLO Distance-Based Object Tracking

This project implements a **custom object tracking algorithm** using **YOLOv8** for object detection and a **distance-between-two-points algorithm** for tracking the target across frames. It supports visualizing the tracking process on a sample video and is designed to be light and effective.

## 📂 Project Files

```
yolo_distance_based_object_tracking/
│
├── tracking.py              # Main script: detection + tracking logic
├── yolov8s.pt               # Pre-trained YOLOv8s model weights
├── vidyolov8.mp4            # Sample video for object detection and tracking
├── coco.txt                 # COCO class labels
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## 🧠 Algorithm Overview

* **Detection**: YOLOv8s detects objects in each frame.
* **Tracking**: The script uses a simple but effective approach — it tracks an object across frames by calculating the **Euclidean distance** between detections in consecutive frames.
* **Selection**: The nearest bounding box to the previous target is selected as the new target.

This lightweight algorithm avoids the complexity of deep tracking models (like DeepSORT or ByteTrack) and is ideal for resource-constrained scenarios.

## 🛠️ Setup Instructions

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

> Make sure your Python version is 3.8 or higher.

### 2. Run the tracking script

```bash
python tracking.py
```

The script will:

* Load `vidyolov8.mp4`
* Detect and track an object using YOLOv8
* Display the video with real-time tracking

## 📦 Requirements

Core libraries include:

* `ultralytics`
* `opencv-python`
* `numpy`
* `torch`
  (Full list in `requirements.txt`)

## 🧪 Model

We use **YOLOv8s** — the small and fast version of YOLOv8 from Ultralytics. You can download alternative weights (like `yolov8n.pt` or `yolov8m.pt`) from the [Ultralytics GitHub](https://github.com/ultralytics/ultralytics) if needed.

## 📊 Example Output

> When you run the script, the window shows:

* The original video
* The detected object (with a bounding box)
* A tracking line from frame to frame

You can easily integrate this with gimbal controllers or robotics systems.

## 🔧 Customization

* Replace `vidyolov8.mp4` with your own video.
* Change detection classes in `tracking.py` if you want to focus on specific objects.
* Integrate keyboard input or external control by modifying the logic near the tracking section.

## 📝 License

This project is open for educational and research use.
If you use this in your work, please consider giving credit.

