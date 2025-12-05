"""
Smart Classroom Attendance System
- Face Recognition (InsightFace - FAST!)
- Action Detection (MediaPipe)
- Real-time monitoring of 40-50 students
"""

import cv2
import numpy as np
import json
import mediapipe as mp
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

# Import torch first to load CUDA libraries
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("⚠ PyTorch not installed - GPU may not work")
    pass

import insightface
from insightface.app import FaceAnalysis

# from camera.tapo_stream import TapoCamera  # Not needed, using OpenCV directly

# ============= CONFIGURATION =============
RTSP_URL = "rtsp://Camera1:123456A@a@192.168.1.14:554/stream1"

# Face Recognition Settings
RECOGNITION_THRESHOLD = 0.4  # Lower = stricter (0.3-0.5 recommended)
DETECTION_SIZE = (640, 640)  # Detection input size (smaller = faster)

# Anti-Flashing Settings
TRACKING_DISTANCE = 100  # pixels - faces closer than this are same person
CONFIDENCE_VOTES = 3     # Need N consistent recognitions before showing name (reduce flashing)

# Performance Settings
FRAME_SKIP = 2  # Process every Nth frame (higher = faster but less smooth detection)
DISPLAY_SCALE = 0.5  # Display window size (smaller = faster rendering)
ENABLE_POSE_DETECTION = False  # Set to False for 2-3x speed boost (no action detection)
RESIZE_BEFORE_PROCESS = True  # Resize frame before processing
PROCESS_WIDTH = 480  # Width to resize to (smaller = faster)

# Paths
DATA_DIR = Path("data")
KNOWN_FACES_DIR = DATA_DIR / "known_faces"
UNKNOWN_FACES_DIR = DATA_DIR / "unknown_faces"
LOGS_DIR = DATA_DIR / "logs"

# Create directories
KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
UNKNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ============= ACTION DETECTOR =============
class ActionDetector:
    """Detect student actions using MediaPipe pose detection"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # 0=Lite (FASTEST), 1=Full, 2=Heavy
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.enabled = True  # Can disable pose detection for speed
    
    def detect_action(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """
        Detect action from pose landmarks
        bbox: (x, y, w, h)
        Returns: action label
        """
        if not self.enabled:
            return "attentive"  # Skip pose detection if disabled
        
        x, y, w, h = bbox
        
        # Add padding to bbox
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        # Crop person region
        person_img = frame[y1:y2, x1:x2]
        if person_img.size == 0:
            return "unknown"
        
        # Convert to RGB
        rgb_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        
        # Detect pose
        results = self.pose.process(rgb_img)
        
        if not results.pose_landmarks:
            return "no_pose"
        
        # Get key landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Extract key points
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_elbow = landmarks[13]
        right_elbow = landmarks[14]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        
        # Action detection logic
        action = self._classify_action(
            nose, left_shoulder, right_shoulder,
            left_elbow, right_elbow,
            left_wrist, right_wrist
        )
        
        return action
    
    def _classify_action(self, nose, l_shoulder, r_shoulder, 
                        l_elbow, r_elbow, l_wrist, r_wrist) -> str:
        """Classify action based on pose landmarks"""
        
        # Calculate shoulder midpoint
        shoulder_mid_y = (l_shoulder.y + r_shoulder.y) / 2
        
        # 1. Detect SLEEPING (head down, slouching)
        if nose.y > shoulder_mid_y + 0.15:
            return "sleeping"
        
        # 2. Detect RAISING HAND (hand above shoulder)
        if l_wrist.y < l_shoulder.y - 0.2 or r_wrist.y < r_shoulder.y - 0.2:
            return "raising_hand"
        
        # 3. Detect EATING/DRINKING (hand near mouth)
        if (abs(l_wrist.y - nose.y) < 0.15 and abs(l_wrist.x - nose.x) < 0.15) or \
           (abs(r_wrist.y - nose.y) < 0.15 and abs(r_wrist.x - nose.x) < 0.15):
            return "eating"
        
        # 4. Detect WRITING (elbow bent, hand down)
        l_elbow_bent = abs(l_elbow.y - l_wrist.y) < 0.2
        r_elbow_bent = abs(r_elbow.y - r_wrist.y) < 0.2
        if l_elbow_bent or r_elbow_bent:
            return "writing"
        
        # Default: normal/attentive
        return "attentive"
    
    def close(self):
        self.pose.close()


# ============= ATTENDANCE SYSTEM =============
class SmartAttendanceSystem:
    """Complete attendance system with face recognition and action detection"""
    
    def __init__(self):
        print("Initializing InsightFace...")
        
        try:
            # Initialize InsightFace with GPU
            print("Attempting to use GPU...")
            self.face_app = FaceAnalysis(
                name='buffalo_l',  # Model pack: buffalo_l, buffalo_m, buffalo_s
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']  # Try GPU first
            )
            self.face_app.prepare(ctx_id=0, det_size=DETECTION_SIZE)
            
            # Check which provider is actually being used
            print(f"✓ Using provider: {self.face_app.det_model.session.get_providers()[0]}")
            
            if 'CUDAExecutionProvider' in str(self.face_app.det_model.session.get_providers()):
                print("✓ GPU acceleration enabled!")
            else:
                print("⚠ Warning: Falling back to CPU (GPU not available)")
                
        except Exception as e:
            print(f"⚠ GPU initialization failed: {e}")
            print("Falling back to CPU mode...")
            self.face_app = FaceAnalysis(
                name='buffalo_l',
                providers=['CPUExecutionProvider']
            )
            self.face_app.prepare(ctx_id=0, det_size=DETECTION_SIZE)
            print("✓ Face recognition initialized (CPU mode)")
        
        # Student database
        self.students_db = {}  # {name: {id, embedding, photo_path, ...}}
        self.checked_in = {}   # {name: {time, actions}}
        
        # Face tracking to prevent flashing
        self.face_history = {}  # {face_id: {name: str, count: int, last_seen: int}}
        self.next_face_id = 0
        self.tracking_threshold = TRACKING_DISTANCE
        self.confidence_votes = CONFIDENCE_VOTES
        
        # Action tracking
        self.action_detector = ActionDetector()
        self.action_detector.enabled = ENABLE_POSE_DETECTION
        self.action_history = {}  # {name: [actions]}
        
        # Tracking
        self.frame_count = 0
        
        # Files
        self.db_file = KNOWN_FACES_DIR / "students.json"
        self.embeddings_file = KNOWN_FACES_DIR / "embeddings.npy"
        
        # Load database
        self.load_database()
        
        print("✓ System initialized")
    
    def load_database(self):
        """Load student database"""
        # Load student info
        if self.db_file.exists():
            with open(self.db_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert list back to dict
                self.students_db = {item['name']: item for item in data}
            
            # Load embeddings
            if self.embeddings_file.exists():
                embeddings_data = np.load(self.embeddings_file, allow_pickle=True).item()
                for name in self.students_db:
                    if name in embeddings_data:
                        self.students_db[name]['embedding'] = embeddings_data[name]
            
            print(f"✓ Loaded {len(self.students_db)} students")
        else:
            print("! No students registered yet")
    
    def save_database(self):
        """Save student database"""
        # Save student info (without embeddings)
        students_list = []
        embeddings_dict = {}
        
        for name, info in self.students_db.items():
            # Save info without embedding
            info_copy = {k: v for k, v in info.items() if k != 'embedding'}
            students_list.append(info_copy)
            
            # Save embedding separately
            if 'embedding' in info:
                embeddings_dict[name] = info['embedding']
        
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump(students_list, f, ensure_ascii=False, indent=2)
        
        # Save embeddings
        np.save(self.embeddings_file, embeddings_dict)
        
        print(f"✓ Saved {len(self.students_db)} students to database")
    
    def register_student(self, name: str, student_id: str, photo_path: str) -> bool:
        """Register a new student"""
        print(f"\n📸 Registering: {name} (ID: {student_id})")
        
        # Check if photo exists
        if not Path(photo_path).exists():
            print(f"✗ Photo not found: {photo_path}")
            return False
        
        try:
            # Load image
            img = cv2.imread(photo_path)
            if img is None:
                print(f"✗ Could not load image")
                return False
            
            # Detect face
            faces = self.face_app.get(img)
            
            if len(faces) == 0:
                print("✗ No face detected in photo!")
                return False
            
            if len(faces) > 1:
                print("⚠ Multiple faces detected, using the largest one")
            
            # Get the face with largest area
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            # Get embedding
            embedding = face.normed_embedding
            
            # Copy photo to known_faces directory
            import shutil
            ref_path = KNOWN_FACES_DIR / f"{student_id}_{name}.jpg"
            shutil.copy2(photo_path, ref_path)
            
            # Add to database
            self.students_db[name] = {
                "id": student_id,
                "name": name,
                "photo": str(ref_path),
                "embedding": embedding,
                "registered": datetime.now().isoformat()
            }
            
            print(f"✓ Successfully registered: {name}")
            return True
            
        except Exception as e:
            print(f"✗ Registration failed: {e}")
            return False
    
    def recognize_face(self, face_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize a face using embedding comparison
        Returns: (student_name, similarity) or (None, 0)
        """
        if len(self.students_db) == 0:
            return None, 0.0
        
        best_match = None
        best_similarity = 0.0
        
        for name, info in self.students_db.items():
            if 'embedding' not in info:
                continue
            
            # Calculate cosine similarity
            similarity = np.dot(face_embedding, info['embedding'])
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        # Use higher threshold to reduce false positives
        if best_similarity > RECOGNITION_THRESHOLD:
            return best_match, best_similarity
        
        return None, best_similarity
    
    def find_or_create_face_track(self, bbox: Tuple[int, int, int, int]) -> int:
        """
        Find existing face track or create new one
        Returns: face_id
        """
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Remove old tracks (not seen in 30 frames)
        to_remove = []
        for face_id, track in self.face_history.items():
            if self.frame_count - track['last_seen'] > 30:
                to_remove.append(face_id)
        for face_id in to_remove:
            del self.face_history[face_id]
        
        # Find matching track
        for face_id, track in self.face_history.items():
            track_x, track_y = track['center']
            distance = np.sqrt((center_x - track_x)**2 + (center_y - track_y)**2)
            
            if distance < self.tracking_threshold:
                # Update track
                track['center'] = (center_x, center_y)
                track['last_seen'] = self.frame_count
                return face_id
        
        # Create new track
        face_id = self.next_face_id
        self.next_face_id += 1
        self.face_history[face_id] = {
            'center': (center_x, center_y),
            'last_seen': self.frame_count,
            'votes': {},  # {name: count}
            'confirmed_name': None
        }
        return face_id
    
    def vote_for_identity(self, face_id: int, name: Optional[str]):
        """
        Vote for face identity to stabilize recognition
        """
        if face_id not in self.face_history:
            return None
        
        track = self.face_history[face_id]
        
        # Add vote
        if name:
            track['votes'][name] = track['votes'].get(name, 0) + 1
        
        # Check if we have enough votes
        if track['confirmed_name'] is None and track['votes']:
            max_votes = max(track['votes'].values())
            if max_votes >= self.confidence_votes:
                # Find name with most votes
                for candidate_name, votes in track['votes'].items():
                    if votes == max_votes:
                        track['confirmed_name'] = candidate_name
                        break
        
        return track['confirmed_name']
    
    def detect_and_recognize(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces, recognize students, and detect actions
        Returns: List of {name, bbox, action, confidence}
        """
        results = []
        
        try:
            # Resize frame for faster processing
            if RESIZE_BEFORE_PROCESS:
                h, w = frame.shape[:2]
                scale = PROCESS_WIDTH / w
                if scale < 1.0:
                    process_frame = cv2.resize(frame, (PROCESS_WIDTH, int(h * scale)))
                else:
                    process_frame = frame
                    scale = 1.0
            else:
                process_frame = frame
                scale = 1.0
            
            # Detect faces using InsightFace
            faces = self.face_app.get(process_frame)
            
            for face in faces:
                # Get face bbox (scale back to original size)
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                if RESIZE_BEFORE_PROCESS and scale < 1.0:
                    x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)
                
                w = x2 - x1
                h = y2 - y1
                
                # Track this face
                face_id = self.find_or_create_face_track((x1, y1, w, h))
                
                # Recognize student
                raw_name, similarity = self.recognize_face(face.normed_embedding)
                
                # Vote for identity (smoothing)
                name = self.vote_for_identity(face_id, raw_name)
                
                # Detect action
                action = "unknown"
                if name:
                    action = self.action_detector.detect_action(frame, (x1, y1, w, h))
                    
                    # Track action
                    if name not in self.action_history:
                        self.action_history[name] = []
                    self.action_history[name].append(action)
                    
                    # Keep only last 30 actions
                    if len(self.action_history[name]) > 30:
                        self.action_history[name].pop(0)
                
                # Check in student
                if name and name not in self.checked_in:
                    self.checked_in[name] = {
                        "time": datetime.now().isoformat(),
                        "confidence": float(similarity)
                    }
                    info = self.students_db[name]
                    print(f"✓ CHECKED IN: {name} (ID: {info['id']}) - Similarity: {similarity:.2f}")
                
                results.append({
                    "name": name or "Unknown",
                    "bbox": (x1, y1, w, h),
                    "action": action,
                    "confidence": float(similarity) if name else 0.0,
                    "face_id": face_id  # For debugging
                })
        
        except Exception as e:
            print(f"Detection error: {e}")
        
        return results
    
    def get_dominant_action(self, name: str) -> str:
        """Get most frequent recent action for a student"""
        if name not in self.action_history or len(self.action_history[name]) == 0:
            return "unknown"
        
        # Count recent actions
        recent = self.action_history[name][-10:]  # Last 10 actions
        action_counts = {}
        for action in recent:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Return most common
        return max(action_counts, key=action_counts.get)
    
    def draw_results(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes, names, and actions on frame"""
        
        # Action colors
        action_colors = {
            "attentive": (0, 255, 0),      # Green
            "writing": (0, 255, 255),      # Yellow
            "raising_hand": (255, 0, 255),  # Magenta
            "sleeping": (0, 0, 255),        # Red
            "eating": (0, 165, 255),        # Orange
            "unknown": (128, 128, 128),     # Gray
            "no_pose": (128, 128, 128)
        }
        
        for det in detections:
            name = det["name"]
            x, y, w, h = det["bbox"]
            action = det["action"]
            
            # Get dominant action for known students
            if name != "Unknown":
                action = self.get_dominant_action(name)
            
            # Choose color based on action
            color = action_colors.get(action, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw label background
            if name != "Unknown":
                info = self.students_db[name]
                label = f"{name} ({info['id']})"
                action_label = f"{action.replace('_', ' ').title()}"
            else:
                label = "Unknown"
                action_label = ""
            
            # Calculate label size
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(frame, (x, y - 60), (x + max(label_size[0], 150), y), color, -1)
            
            # Draw labels
            cv2.putText(frame, label, (x + 5, y - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if action_label:
                cv2.putText(frame, action_label, (x + 5, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_ui(self, frame: np.ndarray, fps: float = 0) -> np.ndarray:
        """Draw UI overlay with statistics"""
        h, w = frame.shape[:2]
        
        # Header background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Smart Classroom Attendance", (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Stats
        present = len(self.checked_in)
        total = len(self.students_db)
        stats = f"Present: {present}/{total} | FPS: {fps:.1f}"
        cv2.putText(frame, stats, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Action statistics
        action_counts = {}
        for name in self.checked_in.keys():
            action = self.get_dominant_action(name)
            action_counts[action] = action_counts.get(action, 0) + 1
        
        action_text = " | ".join([f"{k.title()}: {v}" for k, v in action_counts.items()])
        cv2.putText(frame, action_text, (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Footer
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 35), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "Q: Quit | L: List | R: Reset | S: Save Log", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def save_attendance_log(self):
        """Save detailed attendance log with actions"""
        if len(self.checked_in) == 0:
            print("No attendance to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"attendance_{timestamp}.json"
        
        # Prepare data
        attendance_data = []
        for name, check_info in self.checked_in.items():
            info = self.students_db[name]
            action = self.get_dominant_action(name)
            
            attendance_data.append({
                "name": name,
                "id": info["id"],
                "check_in_time": check_info["time"],
                "confidence": check_info["confidence"],
                "primary_action": action,
                "action_history": self.action_history.get(name, [])
            })
        
        data = {
            "datetime": datetime.now().isoformat(),
            "total_students": len(self.students_db),
            "present": len(self.checked_in),
            "absent": len(self.students_db) - len(self.checked_in),
            "attendance": attendance_data
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Attendance log saved: {log_file}")
    
    def print_attendance(self):
        """Print attendance summary"""
        print("\n" + "="*60)
        print(f"ATTENDANCE SUMMARY - {len(self.checked_in)}/{len(self.students_db)} present")
        print("="*60)
        
        if len(self.checked_in) == 0:
            print("No students checked in yet")
        else:
            for name in sorted(self.checked_in.keys()):
                info = self.students_db[name]
                action = self.get_dominant_action(name)
                print(f"  ✓ {name} (ID: {info['id']}) - {action.title()}")
        print()
    
    def run(self):
        """Main attendance monitoring loop"""
        print("\n" + "="*60)
        print("SMART CLASSROOM ATTENDANCE SYSTEM")
        print("="*60)
        print(f"Students registered: {len(self.students_db)}")
        print("Connecting to camera...")
        
        # Open camera with error handling
        try:
            # Use OpenCV directly (works from test!)
            print("Connecting with OpenCV...")
            cam = cv2.VideoCapture(RTSP_URL)
            
            if not cam.isOpened():
                print(f"✗ Failed to connect to camera")
                print(f"Check RTSP URL: {RTSP_URL}")
                return
            
            print("✓ Camera connected via OpenCV")
            
            # Wrapper class to provide consistent interface
            class CameraWrapper:
                def __init__(self, cap):
                    self.cap = cap
                    # Set buffer size to 1 (read latest frame only)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                def get_frame(self):
                    # Grab multiple frames to get the latest one
                    for _ in range(5):  # Skip buffered frames
                        self.cap.grab()
                    ret, frame = self.cap.retrieve()
                    return frame if ret else None
                
                def close(self):
                    self.cap.release()
            
            cam = CameraWrapper(cam)
            
        except Exception as e:
            print(f"✗ Camera error: {e}")
            return
        
        print("\nMonitoring started...")
        print("-"*60)
        
        # FPS calculation
        import time
        fps_start = time.time()
        fps_counter = 0
        fps = 0
        
        try:
            while True:
                try:
                    frame = cam.get_frame()
                    if frame is None:
                        print("⚠ No frame received")
                        continue
                    
                    self.frame_count += 1
                    fps_counter += 1
                    
                    # Calculate FPS
                    if fps_counter >= 30:
                        fps = fps_counter / (time.time() - fps_start)
                        fps_start = time.time()
                        fps_counter = 0
                    
                    # Process every Nth frame (recognition)
                    detections = []
                    if self.frame_count % FRAME_SKIP == 0:
                        detections = self.detect_and_recognize(frame)
                    
                    # Always draw boxes (even on skipped frames) for smooth display
                    if detections:
                        self.last_detections = detections
                    if hasattr(self, 'last_detections'):
                        frame = self.draw_results(frame, self.last_detections)
                    
                    # Draw UI
                    frame = self.draw_ui(frame, fps)
                    
                    # Scale for display
                    display_frame = cv2.resize(frame, (0, 0), 
                                              fx=DISPLAY_SCALE, 
                                              fy=DISPLAY_SCALE)
                    
                    # Display
                    cv2.imshow("Smart Classroom Attendance", display_frame)
                    
                    # Keyboard controls
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # Q or ESC
                        break
                    elif key == ord('l'):
                        self.print_attendance()
                    elif key == ord('r'):
                        self.checked_in.clear()
                        self.action_history.clear()
                        print("\n✓ Attendance reset\n")
                    elif key == ord('s'):
                        self.save_attendance_log()
                
                except Exception as frame_error:
                    print(f"Frame processing error: {frame_error}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        except KeyboardInterrupt:
            print("\n\nStopped by user")
        
        finally:
            # Cleanup
            print("\n" + "="*60)
            print("FINAL ATTENDANCE")
            print("="*60)
            self.print_attendance()
            self.save_attendance_log()
            self.action_detector.close()
            cam.close()
            cv2.destroyAllWindows()


# ============= REGISTRATION MODE =============
def register_students():
    """Interactive student registration"""
    print("\n" + "="*60)
    print("STUDENT REGISTRATION")
    print("="*60)
    
    system = SmartAttendanceSystem()
    
    while True:
        print("\n1. Register new student")
        print("2. List registered students")
        print("3. Start attendance monitoring")
        print("4. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == "1":
            name = input("Student name: ").strip()
            student_id = input("Student ID: ").strip()
            photo_path = input("Photo path: ").strip()
            
            if system.register_student(name, student_id, photo_path):
                system.save_database()
        
        elif choice == "2":
            print(f"\n📋 Registered students: {len(system.students_db)}")
            for i, (name, info) in enumerate(system.students_db.items(), 1):
                print(f"  {i}. {name} (ID: {info['id']})")
        
        elif choice == "3":
            system.run()
            break
        
        elif choice == "4":
            break


# ============= MAIN =============
def main():
    """Main entry point"""
    system = SmartAttendanceSystem()
    
    if len(system.students_db) == 0:
        print("\n⚠ No students registered!")
        print("\n1. Register students")
        print("2. Exit")
        choice = input("\nChoice: ").strip()
        
        if choice == "1":
            register_students()
    else:
        system.run()


if __name__ == "__main__":
    # Choose mode:
    
    # Mode 1: Register students first
    # register_students()
    
    # Mode 2: Run attendance directly (if already registered)
    main()