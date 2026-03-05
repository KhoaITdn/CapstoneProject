"""
model.py - Module chứa model EfficientNetB0 + CBAM cho nhận diện cảm xúc
Đã cập nhật cho Method 5 v2: EfficientNetB0 + CBAM + RAF-DB
  - Input: 224x224 RGB
  - Preprocessing: efficientnet.preprocess_input
  - Classes: Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral
"""
import os
import json
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# ============================================================
# CUSTOM LAYERS - CBAM Attention Mechanism
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
        ch = tf.shape(x)[-1]
        avg = self.dense2(self.dense1(self.gap(x)))
        mx  = self.dense2(self.dense1(self.gmp(x)))
        att = tf.sigmoid(avg + mx)
        return x * tf.reshape(att, (-1, 1, 1, tf.shape(att)[-1]))

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
        mx  = tf.reduce_max(x, axis=-1, keepdims=True)
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


@keras.saving.register_keras_serializable()
class FocalLoss(keras.losses.Loss):
    """Focal Loss for handling class imbalance."""
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        if self.label_smoothing > 0:
            n_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / n_classes
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.pow(1.0 - y_pred, self.gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'gamma': self.gamma, 'alpha': self.alpha,
                    'label_smoothing': self.label_smoothing})
        return cfg


# ============================================================
# CONSTANTS - RAF-DB classes (folder 1-7)
# ============================================================
EMOTIONS    = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
EMOTIONS_VI = ['Ngạc nhiên', 'Sợ hãi', 'Kinh tởm', 'Hạnh phúc', 'Buồn', 'Giận dữ', 'Bình thường']
EMOTION_EMOJIS = ['😲', '😨', '🤢', '😊', '😢', '😠', '😐']

IMG_SIZE = 224
COLOR_MODE = 'rgb'

# Engagement weights: positive=1.0, neutral=0.5, negative=0.0
ENGAGEMENT_WEIGHTS = {
    'Surprise': 0.8, 'Fear': 0.1, 'Disgust': 0.0,
    'Happiness': 1.0, 'Sadness': 0.1,
    'Anger': 0.0, 'Neutral': 0.5
}


# ============================================================
# LOAD MODEL
# ============================================================
def load_emotion_model(model_path):
    """Load the full EfficientNetB0+CBAM model from .keras file.
    Returns model or None on failure.
    """
    print(f"[MODEL] Loading EfficientNetB0+CBAM from: {model_path}")

    if not os.path.exists(model_path):
        print(f"[MODEL] FAILED: File not found: {model_path}")
        return None

    try:
        custom_objects = {
            'ChannelAttention': ChannelAttention,
            'SpatialAttention': SpatialAttention,
            'CBAMBlock': CBAMBlock,
            'FocalLoss': FocalLoss,
        }
        model = keras.models.load_model(model_path, custom_objects=custom_objects)

        # Warm up
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        dummy = efficientnet_preprocess(dummy)
        _ = model.predict(dummy, verbose=0)

        print(f"[MODEL] OK! {model.count_params():,} params loaded.")
        print(f"[MODEL] Input: {IMG_SIZE}x{IMG_SIZE} RGB")
        print(f"[MODEL] Preprocessing: efficientnet.preprocess_input")
        print(f"[MODEL] Classes: {EMOTIONS}")
        return model
    except Exception as e:
        print(f"[MODEL] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def preprocess_face(face_bgr):
    """Preprocess a face ROI (BGR, any size) for the model.
    Input:  face_bgr - OpenCV BGR image (h, w, 3)
    Output: preprocessed tensor (1, 224, 224, 3)
    """
    # Resize to 224x224
    face_resized = tf.image.resize(face_bgr, (IMG_SIZE, IMG_SIZE)).numpy()
    # Convert BGR -> RGB
    face_rgb = face_resized[:, :, ::-1].copy()
    # Expand dims -> (1, 224, 224, 3)
    face_batch = np.expand_dims(face_rgb, axis=0).astype(np.float32)
    # Apply EfficientNet preprocessing (NOT /255!)
    face_batch = efficientnet_preprocess(face_batch)
    return face_batch
