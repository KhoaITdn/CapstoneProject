
import json
import os

# --- BASE CELL DEFINITIONS ---

# CELL 1: GPU
c1_source = [
    "!nvidia-smi\n",
    "\n",
    "import tensorflow as tf\n",
    "\n",
    "gpus = tf.config.list_physical_devices('GPU')\n",
    "print(f\"TensorFlow: {tf.__version__}\")\n",
    "print(f\"GPUs: {gpus}\")\n",
    "\n",
    "if gpus:\n",
    "    for gpu in gpus:\n",
    "        tf.config.experimental.set_memory_growth(gpu, True)\n",
    "    print(\"✅ GPU ENABLED!\")\n",
    "else:\n",
    "    print(\"❌ NO GPU! Go to Runtime → Change runtime type → GPU\")\n"
]

# CELL 2: IMPORTS
c2_source = [
    "import os\n",
    "import shutil\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from PIL import Image\n",
    "import warnings\n",
    "import pickle\n",
    "import json\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import layers, models, regularizers\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "from tensorflow.keras.applications import MobileNetV2\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "from sklearn.utils.class_weight import compute_class_weight\n",
    "\n",
    "print(\"✅ Libraries imported!\")\n"
]

# CELL 3: MOUNT
c3_source = [
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "print(\"✅ Drive mounted!\")\n"
]

# CELL 3B: ZIP CHECK
c3b_source = [
    "import os\n",
    "\n",
    "DATASET_PATH = '/content/drive/MyDrive/CaptoneProject/camera'\n",
    "ZIP_PATH = '/content/drive/MyDrive/CaptoneProject/camera.zip'\n",
    "\n",
    "if os.path.exists(ZIP_PATH):\n",
    "    print(f\"✅ ZIP found at {ZIP_PATH}\")\n",
    "else:\n",
    "    print(\"⚠️ ZIP not found! Please run the specific zip creation cell if needed.\")\n"
]

# CELL 4: UNZIP & PATHS
c4_source = [
    "import os\n",
    "import shutil\n",
    "\n",
    "ZIP_PATH = '/content/drive/MyDrive/CaptoneProject/camera.zip'\n",
    "LOCAL_PATH = '/content/dataset'\n",
    "\n",
    "if not os.path.exists(LOCAL_PATH):\n",
    "    if os.path.exists(ZIP_PATH):\n",
    "        print(\"📦 Unzipping... (this might take a moment)\")\n",
    "        !unzip -q -o \"{ZIP_PATH}\" -d /content/\n",
    "        \n",
    "        # Handle directory structure\n",
    "        if os.path.exists('/content/camera'):\n",
    "            !mv /content/camera \"{LOCAL_PATH}\"\n",
    "        elif os.path.exists('/content/train') and os.path.exists('/content/test'):\n",
    "            os.makedirs(LOCAL_PATH, exist_ok=True)\n",
    "            !mv /content/train \"{LOCAL_PATH}/train\"\n",
    "            !mv /content/test \"{LOCAL_PATH}/test\"\n",
    "        print(\"✅ Dataset ready at /content/dataset\")\n",
    "    else:\n",
    "        print(\"❌ ZIP file not found in Drive!\")\n",
    "else:\n",
    "    print(\"✅ Dataset already exists locally!\")\n",
    "\n",
    "TRAIN_DIR = os.path.join(LOCAL_PATH, 'train')\n",
    "TEST_DIR = os.path.join(LOCAL_PATH, 'test')\n"
]

# CELL 7: CLASS WEIGHTS
c7_source = [
    "train_labels = train_generator.classes\n",
    "class_weights_array = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)\n",
    "class_weights = dict(enumerate(class_weights_array))\n",
    "\n",
    "print(\"📊 Class Weights calculated!\")\n"
]

# CELL 9: CALLBACKS
def get_c9_source(method_name):
    return [
        "class SaveHistoryCallback(keras.callbacks.Callback):\n",
        "    def __init__(self, history_path, checkpoint_path):\n",
        "        super().__init__()\n",
        "        self.history_path = history_path\n",
        "        self.checkpoint_path = checkpoint_path\n",
        "        self.history_data = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': [], 'lr': []}\n",
        "\n",
        "    def on_epoch_end(self, epoch, logs=None):\n",
        "        self.history_data['accuracy'].append(logs.get('accuracy'))\n",
        "        self.history_data['val_accuracy'].append(logs.get('val_accuracy'))\n",
        "        self.history_data['loss'].append(logs.get('loss'))\n",
        "        self.history_data['val_loss'].append(logs.get('val_loss'))\n",
        "        self.history_data['lr'].append(float(self.model.optimizer.learning_rate.numpy()))\n",
        "\n",
        "        with open(self.history_path, 'wb') as f:\n",
        "            pickle.dump(self.history_data, f)\n",
        "        # Save checkpoint periodically or on best is handled by ModelCheckpoint, \n",
        "        # but we can save here too if needed. \n",
        "        # self.model.save(self.checkpoint_path)\n",
        "\n",
        "callbacks = [\n",
        "    ModelCheckpoint(filepath=BEST_MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),\n",
        "    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),\n",
        "    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1),\n",
        "    SaveHistoryCallback(HISTORY_PATH, MODEL_CHECKPOINT_PATH)\n",
        "]\n",
        "print(\"✅ Callbacks configured!\")\n"
    ]

# CELL 10: TRAIN
c10_source = [
    "print(\"🚀 Starting Training...\")\n",
    "history = model.fit(\n",
    "    train_generator,\n",
    "    epochs=EPOCHS,\n",
    "    validation_data=validation_generator,\n",
    "    callbacks=callbacks,\n",
    "    class_weight=class_weights,\n",
    "    verbose=1\n",
    ")\n",
    "print(\"✅ Training Completed!\")\n"
]

# CELL 12: EVALUATE & CELL 13: CONFUSION MATRIX
c_eval_source = [
    "# Load best model\n",
    "best_model = keras.models.load_model(BEST_MODEL_PATH)\n",
    "\n",
    "test_generator.reset()\n",
    "test_loss, test_acc = best_model.evaluate(test_generator)\n",
    "print(f\"\\n🎯 TEST ACCURACY: {test_acc*100:.2f}%\")\n",
    "print(f\"   TEST LOSS: {test_loss:.4f}\")\n",
    "\n",
    "# Predictions\n",
    "predictions = best_model.predict(test_generator, verbose=1)\n",
    "y_pred = np.argmax(predictions, axis=1)\n",
    "y_true = test_generator.classes\n",
    "\n",
    "print(classification_report(y_true, y_pred, target_names=EMOTIONS, digits=4))\n",
    "\n",
    "# Confusion Matrix\n",
    "cm = confusion_matrix(y_true, y_pred)\n",
    "plt.figure(figsize=(8,6))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS)\n",
    "plt.ylabel('True Label')\n",
    "plt.xlabel('Predicted Label')\n",
    "plt.show()\n"
]

# --- METHOD SPECIFIC DEFINITIONS ---

# METHOD 1: Enhanced Augmentation
m1_config = [
    "# ========================================\n",
    "# CONFIG: METHOD 1 - ENHANCED AUGMENTATION\n",
    "# ========================================\n",
    "IMG_SIZE = 48\n",
    "BATCH_SIZE = 64\n",
    "EPOCHS = 50\n",
    "LEARNING_RATE = 0.0005\n",
    "NUM_CLASSES = 7\n",
    "EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']\n",
    "SEED = 42\n",
    "np.random.seed(SEED); tf.random.set_seed(SEED)\n",
    "\n",
    "CHECKPOINT_DIR = '/content/drive/MyDrive/CaptoneProject/checkpoints/method1_aug'\n",
    "os.makedirs(CHECKPOINT_DIR, exist_ok=True)\n",
    "\n",
    "MODEL_CHECKPOINT_PATH = f'{CHECKPOINT_DIR}/model_checkpoint.keras'\n",
    "BEST_MODEL_PATH = f'{CHECKPOINT_DIR}/best_model.keras'\n",
    "HISTORY_PATH = f'{CHECKPOINT_DIR}/training_history.pkl'\n",
    "print(f\"📁 Checkpoints Saving to: {CHECKPOINT_DIR}\")\n"
]

m1_data = [
    "# Enhanced Augmentation\n",
    "train_datagen = ImageDataGenerator(\n",
    "    rescale=1./255,\n",
    "    rotation_range=25,\n",
    "    width_shift_range=0.2,\n",
    "    height_shift_range=0.2,\n",
    "    shear_range=0.2,\n",
    "    zoom_range=0.2,\n",
    "    horizontal_flip=True,\n",
    "    brightness_range=[0.7, 1.3],\n",
    "    fill_mode='nearest',\n",
    "    validation_split=0.2\n",
    ")\n",
    "test_datagen = ImageDataGenerator(rescale=1./255)\n",
    "\n",
    "train_generator = train_datagen.flow_from_directory(\n",
    "    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale', batch_size=BATCH_SIZE, class_mode='categorical', subset='training', shuffle=True, seed=SEED)\n",
    "validation_generator = train_datagen.flow_from_directory(\n",
    "    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale', batch_size=BATCH_SIZE, class_mode='categorical', subset='validation', shuffle=False, seed=SEED)\n",
    "test_generator = test_datagen.flow_from_directory(\n",
    "    TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale', batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)\n"
]

m1_model = [
    "# Standard Custom CNN (Same as original)\n",
    "def build_cnn(input_shape=(48, 48, 1), num_classes=7):\n",
    "    model = models.Sequential([\n",
    "        layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape), layers.BatchNormalization(), layers.Activation('relu'),\n",
    "        layers.Conv2D(64, (3, 3), padding='same'), layers.BatchNormalization(), layers.Activation('relu'),\n",
    "        layers.MaxPooling2D((2, 2)), layers.Dropout(0.25),\n",
    "        \n",
    "        layers.Conv2D(128, (3, 3), padding='same'), layers.BatchNormalization(), layers.Activation('relu'),\n",
    "        layers.Conv2D(128, (3, 3), padding='same'), layers.BatchNormalization(), layers.Activation('relu'),\n",
    "        layers.MaxPooling2D((2, 2)), layers.Dropout(0.25),\n",
    "        \n",
    "        layers.Conv2D(256, (3, 3), padding='same'), layers.BatchNormalization(), layers.Activation('relu'),\n",
    "        layers.Conv2D(256, (3, 3), padding='same'), layers.BatchNormalization(), layers.Activation('relu'),\n",
    "        layers.MaxPooling2D((2, 2)), layers.Dropout(0.25),\n",
    "        \n",
    "        layers.Conv2D(512, (3, 3), padding='same'), layers.BatchNormalization(), layers.Activation('relu'),\n",
    "        layers.Conv2D(512, (3, 3), padding='same'), layers.BatchNormalization(), layers.Activation('relu'),\n",
    "        layers.MaxPooling2D((2, 2)), layers.Dropout(0.25),\n",
    "        \n",
    "        layers.Flatten(),\n",
    "        layers.Dense(512, kernel_regularizer=regularizers.l2(0.001)), layers.BatchNormalization(), layers.Activation('relu'), layers.Dropout(0.4),\n",
    "        layers.Dense(256, kernel_regularizer=regularizers.l2(0.001)), layers.BatchNormalization(), layers.Activation('relu'), layers.Dropout(0.4),\n",
    "        layers.Dense(num_classes, activation='softmax')\n",
    "    ])\n",
    "    return model\n",
    "\n",
    "model = build_cnn()\n",
    "model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])\n",
    "model.summary()\n"
]

# METHOD 2: SE-Attention
m2_config = [
    "# ========================================\n",
    "# CONFIG: METHOD 2 - SE ATTENTION CNN\n",
    "# ========================================\n",
    "IMG_SIZE = 48\n",
    "BATCH_SIZE = 64\n",
    "EPOCHS = 50\n",
    "LEARNING_RATE = 0.0005\n",
    "NUM_CLASSES = 7\n",
    "EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']\n",
    "SEED = 42\n",
    "np.random.seed(SEED); tf.random.set_seed(SEED)\n",
    "\n",
    "CHECKPOINT_DIR = '/content/drive/MyDrive/CaptoneProject/checkpoints/method2_se'\n",
    "os.makedirs(CHECKPOINT_DIR, exist_ok=True)\n",
    "MODEL_CHECKPOINT_PATH = f'{CHECKPOINT_DIR}/checkpoint.keras'\n",
    "BEST_MODEL_PATH = f'{CHECKPOINT_DIR}/best_model.keras'\n",
    "HISTORY_PATH = f'{CHECKPOINT_DIR}/history.pkl'\n"
]

m2_data = [
    "# Standard Augmentation\n",
    "train_datagen = ImageDataGenerator(\n",
    "    rescale=1./255, rotation_range=15, width_shift_range=0.15, height_shift_range=0.15,\n",
    "    shear_range=0.15, zoom_range=0.15, horizontal_flip=True, fill_mode='nearest', validation_split=0.2\n",
    ")\n",
    "test_datagen = ImageDataGenerator(rescale=1./255)\n",
    "\n",
    "train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(48,48), color_mode='grayscale', batch_size=BATCH_SIZE, class_mode='categorical', subset='training', seed=SEED)\n",
    "validation_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(48,48), color_mode='grayscale', batch_size=BATCH_SIZE, class_mode='categorical', subset='validation', seed=SEED)\n",
    "test_generator = test_datagen.flow_from_directory(TEST_DIR, target_size=(48,48), color_mode='grayscale', batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)\n"
]

m2_model = [
    "def squeeze_excitation_block(input_tensor, ratio=16):\n",
    "    channels = input_tensor.shape[-1]\n",
    "    se = layers.GlobalAveragePooling2D()(input_tensor)\n",
    "    se = layers.Dense(channels // ratio, activation='relu')(se)\n",
    "    se = layers.Dense(channels, activation='sigmoid')(se)\n",
    "    se = layers.Reshape((1, 1, channels))(se)\n",
    "    return layers.Multiply()([input_tensor, se])\n",
    "\n",
    "def build_se_cnn(input_shape=(48, 48, 1), num_classes=7):\n",
    "    inputs = layers.Input(shape=input_shape)\n",
    "    \n",
    "    # Block 1\n",
    "    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)\n",
    "    x = layers.BatchNormalization()(x)\n",
    "    x = layers.Activation('relu')(x)\n",
    "    x = layers.Conv2D(64, (3, 3), padding='same')(x)\n",
    "    x = layers.BatchNormalization()(x)\n",
    "    x = layers.Activation('relu')(x)\n",
    "    x = squeeze_excitation_block(x, ratio=8)\n",
    "    x = layers.MaxPooling2D((2, 2))(x)\n",
    "    x = layers.Dropout(0.25)(x)\n",
    "\n",
    "    # Block 2\n",
    "    x = layers.Conv2D(128, (3, 3), padding='same')(x)\n",
    "    x = layers.BatchNormalization()(x)\n",
    "    x = layers.Activation('relu')(x)\n",
    "    x = layers.Conv2D(128, (3, 3), padding='same')(x)\n",
    "    x = layers.BatchNormalization()(x)\n",
    "    x = layers.Activation('relu')(x)\n",
    "    x = squeeze_excitation_block(x, ratio=8)\n",
    "    x = layers.MaxPooling2D((2, 2))(x)\n",
    "    x = layers.Dropout(0.25)(x)\n",
    "\n",
    "    # Block 3\n",
    "    x = layers.Conv2D(256, (3, 3), padding='same')(x)\n",
    "    x = layers.BatchNormalization()(x)\n",
    "    x = layers.Activation('relu')(x)\n",
    "    x = layers.Conv2D(256, (3, 3), padding='same')(x)\n",
    "    x = layers.BatchNormalization()(x)\n",
    "    x = layers.Activation('relu')(x)\n",
    "    x = squeeze_excitation_block(x, ratio=16)\n",
    "    x = layers.MaxPooling2D((2, 2))(x)\n",
    "    x = layers.Dropout(0.25)(x)\n",
    "\n",
    "    # Block 4\n",
    "    x = layers.Conv2D(512, (3, 3), padding='same')(x)\n",
    "    x = layers.BatchNormalization()(x)\n",
    "    x = layers.Activation('relu')(x)\n",
    "    x = layers.Conv2D(512, (3, 3), padding='same')(x)\n",
    "    x = layers.BatchNormalization()(x)\n",
    "    x = layers.Activation('relu')(x)\n",
    "    x = squeeze_excitation_block(x, ratio=16)\n",
    "    x = layers.MaxPooling2D((2, 2))(x)\n",
    "    x = layers.Dropout(0.25)(x)\n",
    "\n",
    "    # Classifier using GAP\n",
    "    x = layers.GlobalAveragePooling2D()(x)\n",
    "    x = layers.Dense(512, kernel_regularizer=regularizers.l2(0.001))(x)\n",
    "    x = layers.BatchNormalization()(x)\n",
    "    x = layers.Activation('relu')(x)\n",
    "    x = layers.Dropout(0.5)(x)\n",
    "    \n",
    "    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.001))(x)\n",
    "    x = layers.BatchNormalization()(x)\n",
    "    x = layers.Activation('relu')(x)\n",
    "    x = layers.Dropout(0.5)(x)\n",
    "    \n",
    "    outputs = layers.Dense(num_classes, activation='softmax')(x)\n",
    "    \n",
    "    model = models.Model(inputs=inputs, outputs=outputs)\n",
    "    return model\n",
    "\n",
    "model = build_se_cnn()\n",
    "model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])\n",
    "model.summary()\n"
]

# METHOD 3: MobileNetV2
m3_config = [
    "# ========================================\n",
    "# CONFIG: METHOD 3 - MOBILENETV2 (RGB)\n",
    "# ========================================\n",
    "IMG_SIZE = 48\n",
    "BATCH_SIZE = 64\n",
    "EPOCHS = 50\n",
    "LEARNING_RATE = 0.0001 # Lower LR for Fine-tuning\n",
    "NUM_CLASSES = 7\n",
    "EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']\n",
    "SEED = 42\n",
    "np.random.seed(SEED); tf.random.set_seed(SEED)\n",
    "\n",
    "CHECKPOINT_DIR = '/content/drive/MyDrive/CaptoneProject/checkpoints/method3_mobilenet'\n",
    "os.makedirs(CHECKPOINT_DIR, exist_ok=True)\n",
    "MODEL_CHECKPOINT_PATH = f'{CHECKPOINT_DIR}/checkpoint.keras'\n",
    "BEST_MODEL_PATH = f'{CHECKPOINT_DIR}/best_model.keras'\n",
    "HISTORY_PATH = f'{CHECKPOINT_DIR}/history.pkl'\n"
]

m3_data = [
    "# RGB COLOR MODE FOR TRANSFER LEARNING\n",
    "train_datagen = ImageDataGenerator(\n",
    "    rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,\n",
    "    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, brightness_range=[0.8, 1.2], fill_mode='nearest', validation_split=0.2\n",
    ")\n",
    "test_datagen = ImageDataGenerator(rescale=1./255)\n",
    "\n",
    "train_generator = train_datagen.flow_from_directory(\n",
    "    TRAIN_DIR, target_size=(48,48), color_mode='rgb', batch_size=BATCH_SIZE, class_mode='categorical', subset='training', seed=SEED)\n",
    "validation_generator = train_datagen.flow_from_directory(\n",
    "    TRAIN_DIR, target_size=(48,48), color_mode='rgb', batch_size=BATCH_SIZE, class_mode='categorical', subset='validation', seed=SEED)\n",
    "test_generator = test_datagen.flow_from_directory(\n",
    "    TEST_DIR, target_size=(48,48), color_mode='rgb', batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)\n"
]

m3_model = [
    "def build_mobilenet(input_shape=(48, 48, 3), num_classes=7):\n",
    "    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet', alpha=0.35)\n",
    "    \n",
    "    # Fine-tine: Freeze early layers, unfreeze late\n",
    "    for layer in base_model.layers[:-30]:\n",
    "        layer.trainable = False\n",
    "        \n",
    "    inputs = layers.Input(shape=input_shape)\n",
    "    x = base_model(inputs, training=False)\n",
    "    x = layers.GlobalAveragePooling2D()(x)\n",
    "    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.001))(x)\n",
    "    x = layers.BatchNormalization()(x)\n",
    "    x = layers.Activation('relu')(x)\n",
    "    x = layers.Dropout(0.5)(x)\n",
    "    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001))(x)\n",
    "    x = layers.BatchNormalization()(x)\n",
    "    x = layers.Activation('relu')(x)\n",
    "    x = layers.Dropout(0.5)(x)\n",
    "    \n",
    "    outputs = layers.Dense(num_classes, activation='softmax')(x)\n",
    "    \n",
    "    model = models.Model(inputs=inputs, outputs=outputs)\n",
    "    return model\n",
    "\n",
    "model = build_mobilenet()\n",
    "model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])\n",
    "model.summary()\n"
]

def create_notebook(filename, config_src, data_src, model_src, method_name):
    cells = []
    # Add cells orderly
    cells.append({"cell_type": "markdown", "source": [f"# {method_name} - FER2013 Training"]})
    cells.append({"cell_type": "code", "source": c1_source, "outputs": [], "metadata": {}})
    cells.append({"cell_type": "code", "source": c2_source, "outputs": [], "metadata": {}})
    cells.append({"cell_type": "code", "source": c3_source, "outputs": [], "metadata": {}})
    cells.append({"cell_type": "code", "source": c3b_source, "outputs": [], "metadata": {}})
    cells.append({"cell_type": "code", "source": c4_source, "outputs": [], "metadata": {}})
    
    # Method specific
    cells.append({"cell_type": "code", "source": config_src, "outputs": [], "metadata": {}})
    cells.append({"cell_type": "code", "source": data_src, "outputs": [], "metadata": {}})
    cells.append({"cell_type": "code", "source": c7_source, "outputs": [], "metadata": {}})
    cells.append({"cell_type": "code", "source": model_src, "outputs": [], "metadata": {}})
    cells.append({"cell_type": "code", "source": get_c9_source(method_name), "outputs": [], "metadata": {}})
    cells.append({"cell_type": "code", "source": c10_source, "outputs": [], "metadata": {}})
    cells.append({"cell_type": "code", "source": c_eval_source, "outputs": [], "metadata": {}})
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.8.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
        print(f"Created key file: {filename}")

if __name__ == "__main__":
    create_notebook("d:/capstoneProject/method1_enhanced_aug.ipynb", m1_config, m1_data, m1_model, "Method 1: Enhanced Augmentation")
    create_notebook("d:/capstoneProject/method2_se_attention.ipynb", m2_config, m2_data, m2_model, "Method 2: SE-Attention")
    create_notebook("d:/capstoneProject/method3_mobilenet.ipynb", m3_config, m3_data, m3_model, "Method 3: MobileNetV2 Transfer Learning")
