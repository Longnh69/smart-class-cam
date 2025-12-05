"""
Nhận diện khuôn mặt REAL-TIME qua CAMERA
Sử dụng: InsightFace + OpenCV + GPU
FIX: Normalize embedding + Tăng ngưỡng + Check detection score
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis
import time
import os

# ========== CẤU HÌNH ==========
USE_GPU = True  # True = dùng GPU, False = dùng CPU
CAMERA_ID = "rtsp://Camera1:123456A@a@192.168.1.14:554/stream1"   # 0 = camera mặc định

# ⚠️ QUAN TRỌNG: Các ngưỡng đã được điều chỉnh
SIMILARITY_THRESHOLD = 0.55  # Tăng từ 0.4 lên 0.55 (0.5-0.6 là hợp lý)
MIN_DETECTION_SCORE = 0.5    # Chỉ nhận diện khuôn mặt có độ tin cậy > 50%
MIN_FACE_SIZE = 100           # Kích thước tối thiểu của khuôn mặt (pixel)

print("=" * 60)
print("🎥 NHẬN DIỆN KHUÔN MẶT REAL-TIME QUA CAMERA")
print("=" * 60)

# ========== KHỞI TẠO INSIGHTFACE ==========
print("\n🚀 Đang load model InsightFace...")
app = FaceAnalysis(
    name='buffalo_l',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if USE_GPU else ['CPUExecutionProvider']
)
app.prepare(ctx_id=0 if USE_GPU else -1, det_size=(640, 640))
print(f"✅ Model đã sẵn sàng! Sử dụng: {'GPU (CUDA)' if USE_GPU else 'CPU'}")

# ========== DATABASE KHUÔN MẶT ==========
known_faces = {}  # {tên: embedding_vector (đã normalize)}


def normalize_embedding(embedding):
    """
    Normalize embedding vector về unit vector (norm = 1)
    Đây là bước QUAN TRỌNG để cosine similarity hoạt động đúng
    """
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def cosine_similarity(emb1, emb2):
    """
    Tính cosine similarity giữa 2 embedding đã normalize
    Kết quả từ -1 đến 1, càng gần 1 càng giống
    """
    # Normalize để đảm bảo
    emb1_norm = normalize_embedding(emb1)
    emb2_norm = normalize_embedding(emb2)
    return np.dot(emb1_norm, emb2_norm)


def is_valid_face(face):
    """
    Kiểm tra khuôn mặt có đủ chất lượng để nhận diện không
    """
    # Check detection score
    if hasattr(face, 'det_score') and face.det_score < MIN_DETECTION_SCORE:
        return False, f"Detection score quá thấp: {face.det_score:.2f}"
    
    # Check kích thước khuôn mặt
    bbox = face.bbox.astype(int)
    face_width = bbox[2] - bbox[0]
    face_height = bbox[3] - bbox[1]
    
    if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
        return False, f"Khuôn mặt quá nhỏ: {face_width}x{face_height}"
    
    return True, "OK"


def register_face_from_image(name, image_path):
    """Đăng ký khuôn mặt từ file ảnh"""
    if not os.path.exists(image_path):
        print(f"❌ File không tồn tại: {image_path}")
        return False
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Không đọc được ảnh: {image_path}")
        return False
    
    faces = app.get(img)
    if len(faces) == 0:
        print(f"❌ Không tìm thấy khuôn mặt trong ảnh!")
        return False
    
    # Nếu có nhiều người, lấy khuôn mặt lớn nhất
    if len(faces) > 1:
        print(f"⚠️  Phát hiện {len(faces)} khuôn mặt, lấy khuôn mặt lớn nhất")
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
    
    face = faces[0]
    
    # Kiểm tra chất lượng
    is_valid, msg = is_valid_face(face)
    if not is_valid:
        print(f"❌ Khuôn mặt không đạt chất lượng: {msg}")
        return False
    
    # ⚠️ QUAN TRỌNG: Normalize embedding trước khi lưu
    known_faces[name] = normalize_embedding(face.embedding)
    
    det_score = face.det_score if hasattr(face, 'det_score') else 'N/A'
    print(f"✅ Đã đăng ký: {name} (detection score: {det_score})")
    return True


def register_face_from_camera(name, frame):
    """Đăng ký khuôn mặt từ frame camera hiện tại"""
    faces = app.get(frame)
    if len(faces) == 0:
        print(f"❌ Không tìm thấy khuôn mặt!")
        return False
    
    # Nếu có nhiều người, lấy khuôn mặt lớn nhất
    if len(faces) > 1:
        print(f"⚠️  Phát hiện {len(faces)} khuôn mặt, lấy khuôn mặt lớn nhất")
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
    
    face = faces[0]
    
    # Kiểm tra chất lượng
    is_valid, msg = is_valid_face(face)
    if not is_valid:
        print(f"❌ Khuôn mặt không đạt chất lượng: {msg}")
        print("💡 Hãy đưa mặt gần camera hơn và đảm bảo ánh sáng tốt")
        return False
    
    # ⚠️ QUAN TRỌNG: Normalize embedding trước khi lưu
    known_faces[name] = normalize_embedding(face.embedding)
    
    # Lưu ảnh để backup
    timestamp = int(time.time())
    filename = f"registered_{name}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    
    det_score = face.det_score if hasattr(face, 'det_score') else 'N/A'
    print(f"✅ Đã đăng ký: {name} (detection score: {det_score}, lưu tại {filename})")
    return True


def find_match(face_embedding):
    """
    Tìm khuôn mặt khớp trong database
    Trả về: (tên, similarity_score)
    """
    if len(known_faces) == 0:
        return "Unknown", 0.0
    
    # ⚠️ QUAN TRỌNG: Normalize embedding đầu vào
    query_embedding = normalize_embedding(face_embedding)
    
    best_name = "Unknown"
    best_score = -1.0  # Cosine similarity có thể âm
    
    for name, known_embedding in known_faces.items():
        # Tính cosine similarity (known_embedding đã được normalize khi đăng ký)
        similarity = cosine_similarity(query_embedding, known_embedding)
        
        if similarity > best_score:
            best_score = similarity
            best_name = name
    
    # Chỉ trả về tên nếu vượt ngưỡng
    if best_score >= SIMILARITY_THRESHOLD:
        return best_name, best_score
    else:
        return "Unknown", best_score


# ========== ĐĂNG KÝ KHUÔN MẶT TỪ ẢNH (NẾU CÓ) ==========
print("\n📝 Đăng ký khuôn mặt từ ảnh...")
print("-" * 50)

# TODO: Thêm khuôn mặt của bạn tại đây
# Ví dụ:
# register_face_from_image("Nguyen Van A", "C:\\Users\\dowif\\Pictures\\Binh.jpg")
# register_face_from_image("Tran Thi B", "photos/person2.jpg")

# Uncomment dòng dưới và sửa đường dẫn để test
# register_face_from_image("Binh", "C:\\Users\\dowif\\Pictures\\Binh.jpg")

if len(known_faces) == 0:
    print("⚠️  Chưa có khuôn mặt nào được đăng ký từ ảnh!")
    print("💡 Bạn có thể đăng ký khuôn mặt trực tiếp từ camera:")
    print("   - Nhấn 'r' để bắt đầu đăng ký")
    print("   - Nhập tên và Enter")

# ========== MỞ CAMERA ==========
print("\n📹 Đang mở camera...")
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("❌ Không thể mở camera!")
    exit()

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("✅ Camera đã sẵn sàng!")
print("\n" + "=" * 60)
print("HƯỚNG DẪN SỬ DỤNG:")
print("=" * 60)
print("📌 'q' hoặc 'ESC' : Thoát chương trình")
print("📌 'r'            : Đăng ký khuôn mặt hiện tại")
print("📌 's'            : Lưu ảnh snapshot")
print("📌 'd'            : Xóa người vừa nhận diện")
print("📌 'l'            : Hiển thị danh sách người đã đăng ký")
print("📌 '+'/'-'        : Tăng/giảm ngưỡng nhận diện")
print("=" * 60)
print(f"\n⚙️  Cấu hình hiện tại:")
print(f"   - Ngưỡng similarity: {SIMILARITY_THRESHOLD}")
print(f"   - Min detection score: {MIN_DETECTION_SCORE}")
print(f"   - Min face size: {MIN_FACE_SIZE}px")

# ========== BIẾN ĐẾM FPS ==========
fps = 0
fps_counter = 0
fps_start_time = time.time()

# Chế độ đăng ký
registering_mode = False
last_recognized_name = None
current_threshold = SIMILARITY_THRESHOLD

# ========== VÒNG LẶP CHÍNH ==========
print("\n🎬 Bắt đầu nhận diện...\n")

while True:
    # Đọc frame từ camera
    ret, frame = cap.read()
    if not ret:
        print("❌ Không đọc được frame từ camera!")
        break
    
    # Tạo bản sao để vẽ
    display_frame = frame.copy()
    
    # ===== NHẬN DIỆN KHUÔN MẶT =====
    faces = app.get(frame)
    
    # Xử lý từng khuôn mặt tìm được
    for face in faces:
        # Kiểm tra chất lượng khuôn mặt
        is_valid, msg = is_valid_face(face)
        
        # Lấy tọa độ khung mặt
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # Lấy detection score
        det_score = face.det_score if hasattr(face, 'det_score') else 0
        
        if not is_valid:
            # Vẽ khung xám cho khuôn mặt không đạt chất lượng
            color = (128, 128, 128)  # Xám
            label = f"Low Quality ({det_score:.2f})"
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(display_frame, label, 
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            continue
        
        # Nhận diện khuôn mặt
        name, similarity = find_match(face.embedding)
        last_recognized_name = name if name != "Unknown" else None
        
        # Chọn màu dựa trên kết quả
        if name == "Unknown":
            color = (0, 0, 255)  # Đỏ - chưa biết
            label = f"Unknown ({similarity:.2f})"
        else:
            # Màu xanh đậm hơn khi similarity cao hơn
            green_intensity = int(155 + 100 * similarity)
            color = (0, min(255, green_intensity), 0)
            label = f"{name} ({similarity:.2f})"
        
        # Vẽ khung mặt
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        
        # Vẽ background cho text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(display_frame, 
                     (x1, y1 - 30), 
                     (x1 + text_size[0] + 10, y1), 
                     color, -1)
        
        # Vẽ text tên
        cv2.putText(display_frame, label, 
                   (x1 + 5, y1 - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Vẽ thông tin bổ sung
        info_y = y2 + 20
        
        # Detection score
        score_text = f"Det: {det_score:.2f}"
        cv2.putText(display_frame, score_text,
                   (x1, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        info_y += 15
        
        if hasattr(face, 'gender'):
            gender_text = f"{'Male' if face.gender == 1 else 'Female'}"
            cv2.putText(display_frame, gender_text,
                       (x1, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Vẽ điểm landmark (5 điểm đặc trưng trên mặt)
        if hasattr(face, 'kps'):
            for kp in face.kps:
                cv2.circle(display_frame, tuple(kp.astype(int)), 2, (255, 255, 0), -1)
    
    # ===== TÍNH FPS =====
    fps_counter += 1
    if time.time() - fps_start_time >= 1.0:
        fps = fps_counter
        fps_counter = 0
        fps_start_time = time.time()
    
    # ===== VẼ THÔNG TIN HỆ THỐNG =====
    # Background cho info panel
    cv2.rectangle(display_frame, (5, 5), (320, 170), (0, 0, 0), -1)
    cv2.rectangle(display_frame, (5, 5), (320, 170), (255, 255, 255), 2)
    
    # Thông tin
    cv2.putText(display_frame, f"FPS: {fps}", 
               (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Faces Detected: {len(faces)}", 
               (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display_frame, f"Registered: {len(known_faces)}", 
               (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display_frame, f"GPU: {'ON' if USE_GPU else 'OFF'}", 
               (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(display_frame, f"Threshold: {current_threshold:.2f} (+/-)", 
               (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
    cv2.putText(display_frame, f"Min Det Score: {MIN_DETECTION_SCORE}", 
               (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Hiển thị chế độ đăng ký
    if registering_mode:
        cv2.putText(display_frame, "MODE: REGISTERING", 
                   (display_frame.shape[1] - 300, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Hiển thị frame
    cv2.imshow('Face Recognition - Press Q to Quit', display_frame)
    
    # ===== XỬ LÝ PHÍM BẤM =====
    key = cv2.waitKey(1) & 0xFF
    
    # Thoát (Q hoặc ESC)
    if key == ord('q') or key == 27:
        print("\n👋 Đang thoát...")
        break
    
    # Đăng ký khuôn mặt (R)
    elif key == ord('r'):
        if len(faces) == 0:
            print("❌ Không có khuôn mặt nào để đăng ký!")
        else:
            registering_mode = True
            print("\n" + "="*50)
            print("📝 CHẾ ĐỘ ĐĂNG KÝ KHUÔN MẶT")
            print("="*50)
            
            # Lưu frame hiện tại
            register_frame = frame.copy()
            
            # Nhập tên
            name = input("👤 Nhập tên người (hoặc Enter để hủy): ").strip()
            
            if name:
                if name in known_faces:
                    overwrite = input(f"⚠️  '{name}' đã tồn tại. Ghi đè? (y/n): ").strip().lower()
                    if overwrite != 'y':
                        print("⚠️  Đã hủy đăng ký!")
                        registering_mode = False
                        continue
                
                if register_face_from_camera(name, register_frame):
                    print(f"✅ Đã đăng ký thành công: {name}")
                else:
                    print("❌ Đăng ký thất bại!")
            else:
                print("⚠️  Đã hủy đăng ký!")
            
            registering_mode = False
            print("="*50 + "\n")
    
    # Lưu ảnh (S)
    elif key == ord('s'):
        timestamp = int(time.time())
        filename = f"snapshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"💾 Đã lưu ảnh: {filename}")
    
    # Xóa người vừa nhận diện (D)
    elif key == ord('d'):
        if last_recognized_name and last_recognized_name in known_faces:
            del known_faces[last_recognized_name]
            print(f"🗑️  Đã xóa: {last_recognized_name}")
            last_recognized_name = None
        else:
            print("⚠️  Không có người nào để xóa!")
    
    # Hiển thị danh sách (L)
    elif key == ord('l'):
        print("\n" + "="*50)
        print("📋 DANH SÁCH NGƯỜI ĐÃ ĐĂNG KÝ")
        print("="*50)
        if len(known_faces) == 0:
            print("  (Chưa có ai được đăng ký)")
        else:
            for i, name in enumerate(known_faces.keys(), 1):
                print(f"  {i}. {name}")
        print("="*50 + "\n")
    
    # Tăng ngưỡng (+)
    elif key == ord('+') or key == ord('='):
        current_threshold = min(0.9, current_threshold + 0.05)
        print(f"📈 Ngưỡng mới: {current_threshold:.2f}")
    
    # Giảm ngưỡng (-)
    elif key == ord('-'):
        current_threshold = max(0.3, current_threshold - 0.05)
        print(f"📉 Ngưỡng mới: {current_threshold:.2f}")

# ========== DỌN DẸP ==========
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("✅ ĐÃ ĐÓNG CAMERA - Cảm ơn bạn đã sử dụng!")
print("="*60)

# Hiển thị tổng kết
print(f"\n📊 Tổng kết:")
print(f"   - Số người đã đăng ký: {len(known_faces)}")
if len(known_faces) > 0:
    print(f"   - Danh sách: {', '.join(known_faces.keys())}")