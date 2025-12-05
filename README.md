# 🎓 Smart Classroom Attendance System

Real-time face recognition and action detection system for classroom attendance monitoring.

![Python](https://img.shields.io/badge/python-3.8--3.11-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.x%20%7C%2013.x-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## ✨ Features

- 🔍 **Real-time Face Recognition** - Detect and recognize 40-50 students simultaneously
- 🎯 **Action Detection** - Monitor student activities (sleeping, eating, raising hand, writing)
- ⚡ **GPU Acceleration** - 30-60 FPS with NVIDIA GPU
- 📊 **Attendance Logging** - Automatic check-in and JSON export
- 🎨 **Visual Feedback** - Color-coded boxes (Green = Attentive, Red = Sleeping, etc.)
- 💾 **Persistent Storage** - Student database with face embeddings

## 🖼️ Screenshots

```
┌─────────────────────────────────────┐
│ Smart Classroom Attendance          │
│ Present: 35/45 | FPS: 42.3          │
│ Attentive: 28 | Writing: 5 | Sleep: 2│
└─────────────────────────────────────┘

[Green Box] John Doe (ID: 20240001)
           Attentive

[Yellow Box] Jane Smith (ID: 20240002)
            Writing
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 - 3.11
- NVIDIA GPU (optional, for better performance)
- CUDA Toolkit 12.x or 13.x (for GPU)
- Webcam or IP camera (RTSP)

### Installation

#### Windows

```bash
# Clone repository
git clone https://github.com/yourusername/smart-class-cam.git
cd smart-class-cam

# Run setup script
setup.bat
```

#### Linux/Mac

```bash
# Clone repository
git clone https://github.com/yourusername/smart-class-cam.git
cd smart-class-cam

# Run setup script
bash setup.sh
```

### Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install PyTorch with CUDA (for GPU support)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage

### 1. Register Students

```bash
cd src
python main.py
```

Choose option `1` and register students:

```
Student name: John Doe
Student ID: 20240001
Photo path: photos/john.jpg
```

### 2. Start Attendance

Choose option `3` to start monitoring.

### 3. Keyboard Controls

- `Q` or `ESC` - Quit and save log
- `L` - List checked-in students
- `R` - Reset attendance
- `S` - Save attendance log

## ⚙️ Configuration

Edit settings in `src/main.py`:

```python
# Camera
RTSP_URL = "rtsp://username:password@192.168.1.14:554/stream1"

# Performance
FRAME_SKIP = 10  # Process every Nth frame (higher = faster)
PROCESS_WIDTH = 480  # Frame width for processing (smaller = faster)
ENABLE_POSE_DETECTION = False  # Disable for 2-3x speed boost

# Recognition
RECOGNITION_THRESHOLD = 0.4  # Lower = stricter (0.3-0.5)
CONFIDENCE_VOTES = 3  # Votes needed before showing name (reduce flashing)
```

## 📊 Performance

| Hardware        | FPS   | Students | Latency |
| --------------- | ----- | -------- | ------- |
| NVIDIA RTX 3060 | 30-60 | 40-50    | <100ms  |
| NVIDIA GTX 1650 | 20-35 | 30-40    | <200ms  |
| CPU only        | 5-15  | 20-30    | ~1s     |

## 📁 Project Structure

```
smart-class-cam/
├── data/
│   ├── known_faces/         # Student database
│   │   ├── students.json    # Student info
│   │   ├── embeddings.npy   # Face encodings
│   │   └── photos/          # Reference photos
│   ├── unknown_faces/       # Unknown faces captured
│   └── logs/                # Attendance logs
├── src/
│   ├── camera/
│   │   └── tapo_stream.py   # Camera interface
│   ├── face_recognition/
│   │   └── detector.py      # Face detector
│   ├── utils/
│   │   └── __init__.py
│   └── main.py              # Main program
├── photos/                  # Place student photos here
├── requirements.txt
├── setup.bat                # Windows setup script
├── setup.sh                 # Linux/Mac setup script
└── README.md
```

## 🔧 Troubleshooting

### Low FPS (< 10)

1. **Enable GPU acceleration**

   ```bash
   # Check if GPU is detected
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Increase FRAME_SKIP**

   ```python
   FRAME_SKIP = 15  # In main.py
   ```

3. **Disable action detection**
   ```python
   ENABLE_POSE_DETECTION = False
   ```

### Camera Lag/Slow Motion

Already fixed in code! If still occurring:

```python
FRAME_SKIP = 15  # Increase this value
```

### Face Recognition Flashing

Adjust smoothing settings:

```python
CONFIDENCE_VOTES = 5  # More stable (slower recognition)
RECOGNITION_THRESHOLD = 0.45  # Stricter matching
```

### GPU Not Working

```bash
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Reinstall onnxruntime-gpu
pip uninstall onnxruntime-gpu -y
pip install onnxruntime-gpu

# Restart computer
```

## 📝 Output Format

Attendance logs are saved as JSON in `data/logs/`:

```json
{
  "datetime": "2025-01-22T14:30:00",
  "total_students": 45,
  "present": 35,
  "absent": 10,
  "attendance": [
    {
      "name": "John Doe",
      "id": "20240001",
      "check_in_time": "2025-01-22T14:25:00",
      "confidence": 0.85,
      "primary_action": "attentive",
      "action_history": ["attentive", "writing", "attentive"]
    }
  ]
}
```

## 🎯 Action Detection

The system detects these student activities:

- 🟢 **Attentive** - Normal posture, paying attention
- 🟡 **Writing** - Hand down, taking notes
- 🟣 **Raising Hand** - Hand above shoulder
- 🔴 **Sleeping** - Head down significantly
- 🟠 **Eating** - Hand near mouth

## 🛠️ Tech Stack

- **Face Recognition**: InsightFace (buffalo_l model)
- **Action Detection**: MediaPipe Pose
- **Deep Learning**: ONNX Runtime (GPU), PyTorch
- **Computer Vision**: OpenCV
- **Language**: Python 3.8+

## 📈 Optimization Tips

**For maximum speed:**

```python
FRAME_SKIP = 15
PROCESS_WIDTH = 320
DETECTION_SIZE = (240, 240)
ENABLE_POSE_DETECTION = False
DISPLAY_SCALE = 0.3
```

**For best accuracy:**

```python
FRAME_SKIP = 3
PROCESS_WIDTH = 640
DETECTION_SIZE = (640, 640)
ENABLE_POSE_DETECTION = True
DISPLAY_SCALE = 0.7
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) - Face recognition
- [MediaPipe](https://google.github.io/mediapipe/) - Pose detection
- [ONNX Runtime](https://onnxruntime.ai/) - GPU acceleration

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

⭐ If you find this project useful, please consider giving it a star!
