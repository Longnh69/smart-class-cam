"""
Simple camera test script
Run this to diagnose camera issues
"""

import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

RTSP_URL = "rtsp://Camera1:123456A@a@192.168.1.14:554/stream1"

print("="*60)
print("CAMERA TEST SCRIPT")
print("="*60)

# Test 1: TapoCamera
print("\n[Test 1] Testing TapoCamera...")
try:
    from src.camera.tapo_stream import TapoCamera
    cam = TapoCamera(RTSP_URL)
    cam.open()
    print("✓ TapoCamera initialized")
    
    # Try to get a frame
    for i in range(5):
        frame = cam.get_frame()
        if frame is not None:
            print(f"✓ Frame {i+1} received: {frame.shape}")
        else:
            print(f"✗ Frame {i+1} is None")
    
    cam.close()
    print("✓ TapoCamera works!\n")
    use_tapo = True
    
except Exception as e:
    print(f"✗ TapoCamera failed: {e}\n")
    use_tapo = False

# Test 2: Direct OpenCV
print("[Test 2] Testing direct OpenCV...")
try:
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("✗ Could not open RTSP stream")
    else:
        print("✓ OpenCV VideoCapture opened")
        
        # Try to read frames
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                print(f"✓ Frame {i+1} received: {frame.shape}")
            else:
                print(f"✗ Frame {i+1} failed")
        
        cap.release()
        print("✓ Direct OpenCV works!\n")
        use_opencv = True

except Exception as e:
    print(f"✗ OpenCV failed: {e}\n")
    use_opencv = False

# Test 3: InsightFace
print("[Test 3] Testing InsightFace...")
try:
    from insightface.app import FaceAnalysis
    import numpy as np
    
    print("Initializing InsightFace...")
    app = FaceAnalysis(
        name='buffalo_l',
        providers=['CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(320, 320))
    print("✓ InsightFace initialized")
    
    # Test with dummy image
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = app.get(dummy_img)
    print(f"✓ InsightFace works! (found {len(faces)} faces in dummy image)\n")
    
except Exception as e:
    print(f"✗ InsightFace failed: {e}\n")
    import traceback
    traceback.print_exc()

# Test 4: MediaPipe
print("[Test 4] Testing MediaPipe...")
try:
    import mediapipe as mp
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5
    )
    print("✓ MediaPipe initialized")
    
    # Test with dummy image
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    results = pose.process(dummy_img)
    print("✓ MediaPipe works!\n")
    pose.close()
    
except Exception as e:
    print(f"✗ MediaPipe failed: {e}\n")

# Test 5: OpenCV GUI
print("[Test 5] Testing OpenCV GUI...")
try:
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_img, "Test Window - Press Q to close", (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("Test Window", test_img)
    print("✓ Window displayed - Press Q to close")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("✓ OpenCV GUI works!\n")
    
except Exception as e:
    print(f"✗ OpenCV GUI failed: {e}")
    print("Try: pip uninstall opencv-python-headless -y")
    print("     pip install opencv-python\n")

# Summary
print("="*60)
print("TEST SUMMARY")
print("="*60)
if use_tapo if 'use_tapo' in locals() else False:
    print("✓ Camera: TapoCamera works")
elif use_opencv if 'use_opencv' in locals() else False:
    print("⚠ Camera: Use OpenCV fallback")
else:
    print("✗ Camera: Connection failed - check RTSP URL")

print("\nNext steps:")
print("1. If all tests pass → Run main.py")
print("2. If camera fails → Check RTSP_URL")
print("3. If InsightFace fails → Check installation")
print("4. If GUI fails → Reinstall opencv-python")