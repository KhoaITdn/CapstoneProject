import cv2
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import layers, models, regularizers

# ============================================================
# CUSTOM LAYERS - Copy chính xác từ Optimize__method2 notebook
# ============================================================

@keras.saving.register_keras_serializable()
class ChannelAttention(layers.Layer):
    """Channel Attention: dùng cả AvgPool + MaxPool"""
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        ch = input_shape[-1]
        self.dense1 = layers.Dense(ch // self.ratio, activation='relu',
                                   kernel_initializer='he_normal')
        self.dense2 = layers.Dense(ch, kernel_initializer='he_normal')
        self.gap = layers.GlobalAveragePooling2D()
        self.gmp = layers.GlobalMaxPooling2D()
        super().build(input_shape)

    def call(self, x):
        ch = x.shape[-1]
        avg = self.dense2(self.dense1(self.gap(x)))
        mx  = self.dense2(self.dense1(self.gmp(x)))
        att = tf.sigmoid(avg + mx)
        return x * tf.reshape(att, (-1, 1, 1, ch))

    def get_config(self):
        config = super().get_config()
        config.update({'ratio': self.ratio})
        return config


@keras.saving.register_keras_serializable()
class SpatialAttention(layers.Layer):
    """Spatial Attention: focus vào vùng quan trọng (mắt, miệng...)"""
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(1, self.kernel_size, padding='same',
                                  activation='sigmoid',
                                  kernel_initializer='he_normal')
        super().build(input_shape)

    def call(self, x):
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        mx  = tf.reduce_max (x, axis=-1, keepdims=True)
        att = self.conv(tf.concat([avg, mx], axis=-1))
        return x * att

    def get_config(self):
        config = super().get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


@keras.saving.register_keras_serializable()
class CBAMBlock(layers.Layer):
    """CBAM = Channel Attention -> Spatial Attention"""
    def __init__(self, ratio=8, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.ca = ChannelAttention(ratio=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def call(self, x):
        return self.sa(self.ca(x))

    def get_config(self):
        config = super().get_config()
        config.update({'ratio': self.ratio, 'kernel_size': self.kernel_size})
        return config


# ============================================================
# BUILD MODEL - Chính xác architecture từ notebook training
# ============================================================
def build_cbam_cnn(input_shape=(48, 48, 1), num_classes=7):
    inputs = layers.Input(shape=input_shape)

    # Block 1: 64 filters — 48x48 -> 24x24
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = CBAMBlock(ratio=8)(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2: 128 filters — 24x24 -> 12x12
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = CBAMBlock(ratio=8)(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3: 256 filters — 12x12 -> 6x6
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = CBAMBlock(ratio=16)(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)

    # Block 4: 512 filters — 6x6 -> 3x3
    x = layers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = CBAMBlock(ratio=16)(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)

    # Classifier Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs)


# ============================================================
# 1. CẤU HÌNH
# ============================================================
WEIGHTS_PATH = 'D:/New folder/CapstoneProject/model_weights.weights.h5'
EMOTIONS = ['Gian du', 'Kinh tom', 'So hai', 'Hanh phuc', 'Binh thuong', 'Buon', 'Ngac nhien']

# ============================================================
# 2. BUILD MODEL + LOAD WEIGHTS
# ============================================================
print("Dang xay dung model...")
model = build_cbam_cnn()

# Build model bang cach chay 1 lan voi dummy input
dummy = np.zeros((1, 48, 48, 1))
_ = model.predict(dummy, verbose=0)

print("Dang tai weights...")
try:
    model.load_weights(WEIGHTS_PATH)
    print(f"Tai weights thanh cong! ({model.count_params():,} params)")
except Exception as e:
    print(f"Loi khong the tai weights: {e}")
    exit()

# ============================================================
# 3. KHOI TAO CAMERA VA FACE DETECTION
# ============================================================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Khong the mo Camera")
    exit()

print("He thong da san sang! Nhan 'q' de thoat.")

# ============================================================
# 4. VONG LAP CHINH
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]

        try:
            # Model train voi GRAYSCALE (48, 48, 1)
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype('float') / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=0)   # Batch
            roi_gray = np.expand_dims(roi_gray, axis=-1)   # Channel

            prediction = model.predict(roi_gray, verbose=0)
            max_index = int(np.argmax(prediction))
            predicted_emotion = EMOTIONS[max_index]
            confidence = prediction[0][max_index] * 100

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{predicted_emotion} ({confidence:.1f}%)"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        except Exception as e:
            print(f"Loi xu ly: {e}")

    cv2.imshow('Emotion Recognition System (CBAM)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()