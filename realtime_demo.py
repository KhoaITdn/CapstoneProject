import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. CẤU HÌNH
# Đường dẫn đến file model bạn đã tải về
MODEL_PATH = 'best_model.keras' 
# Các nhãn cảm xúc tương ứng với lúc train
EMOTIONS = ['Gian du', 'Kinh tom', 'So hai', 'Hanh phuc', 'Binh thuong', 'Buon', 'Ngac nhien']

# 2. LOAD MODEL
print("Dang tai model... Vui long cho...")
try:
    model = load_model(MODEL_PATH)
    print("Tai model thanh cong!")
except Exception as e:
    print(f"Loi khong the tai model: {e}")
    exit()

# 3. KHOI TAO CAMERA VA FACE DETECTION
# Sử dụng Haar Cascade có sẵn của OpenCV để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mở webcam (0 là thường là camera mặc định)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Khong the mo Camera")
    exit()

print("He thong da san sang! Nhan 'q' de thoat.")

while True:
    # Đọc từng khung hình từ camera
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyen anh sang xam (Grayscale) vi model train bang anh xam
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phat hien khuon mat
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Lay vung anh khuon mat (Region of Interest - ROI)
        roi_gray = gray_frame[y:y+h, x:x+w]
        
        try:
            # --- TIEN XU LY ANH (Giong het luc train) ---
            # 1. Resize ve 48x48
            roi_gray = cv2.resize(roi_gray, (48, 48))
            # 2. Chuan hoa ve khoang [0, 1]
            roi_gray = roi_gray.astype('float') / 255.0
            # 3. Expand dimensions de phu hop input model (1, 48, 48, 1)
            roi_gray = np.expand_dims(roi_gray, axis=0) # Them batch dimension
            roi_gray = np.expand_dims(roi_gray, axis=-1) # Them channel dimension

            # --- DU DOAN ---
            prediction = model.predict(roi_gray, verbose=0)
            max_index = int(np.argmax(prediction))
            predicted_emotion = EMOTIONS[max_index]
            confidence = prediction[0][max_index] * 100

            # --- HIEN THI KET QUA ---
            # Ve khung chu nhat quanh mat
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Viet ten cam xuc va do tin cay
            text = f"{predicted_emotion} ({confidence:.1f}%)"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        except Exception as e:
            print(f"Loi xu ly: {e}")

    # Hien thi khung hình
    cv2.imshow('Emotion Recognition System (Method 2)', frame)

    # Nhan 'q' de thoat
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giai phong tai nguyen
cap.release()
cv2.destroyAllWindows()