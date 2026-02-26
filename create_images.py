# -*- coding: utf-8 -*-
"""Tạo các hình minh họa bằng matplotlib cho báo cáo"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT = r'd:\New folder\CapstoneProject\images'
os.makedirs(OUT, exist_ok=True)

# ============================================================
# 1. CNN Architecture
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis('off')
ax.set_title('Kiến trúc CNN cho nhận diện cảm xúc khuôn mặt', fontsize=14, fontweight='bold', pad=15)

blocks = [
    (0.5, 'Input\n48×48×1', '#E3F2FD', 1.2),
    (2.2, 'Conv Block 1\n64 filters\n+CBAM', '#BBDEFB', 1.5),
    (4.2, 'Conv Block 2\n128 filters\n+CBAM', '#90CAF9', 1.5),
    (6.2, 'Conv Block 3\n256 filters\n+CBAM', '#64B5F6', 1.5),
    (8.2, 'Conv Block 4\n512 filters\n+CBAM', '#42A5F5', 1.5),
    (10.2, 'GAP\n+\nDense 512\nDense 256', '#FFF9C4', 1.5),
    (12.2, 'Softmax\n7 classes', '#C8E6C9', 1.3),
]
for x, label, color, w in blocks:
    rect = FancyBboxPatch((x, 1.2), w, 2.5, boxstyle="round,pad=0.15", 
                           facecolor=color, edgecolor='#333', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, 2.45, label, ha='center', va='center', fontsize=8, fontweight='bold')

for i in range(len(blocks)-1):
    x1 = blocks[i][0] + blocks[i][3]
    x2 = blocks[i+1][0]
    ax.annotate('', xy=(x2, 2.45), xytext=(x1, 2.45),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))

# Size labels
sizes = ['48×48', '24×24', '12×12', '6×6', '3×3', '512', '7']
for i, (x, _, _, w) in enumerate(blocks):
    ax.text(x + w/2, 0.7, sizes[i], ha='center', va='center', fontsize=7, color='#666')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'cnn_architecture.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('✅ cnn_architecture.png')


# ============================================================
# 2. CBAM Module
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(13, 4.5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 4.5)
ax.axis('off')
ax.set_title('CBAM: Convolutional Block Attention Module', fontsize=14, fontweight='bold', pad=15)

# Input
rect = FancyBboxPatch((0.3, 1.5), 1.5, 1.5, boxstyle="round,pad=0.1", facecolor='#E3F2FD', edgecolor='#333', lw=1.5)
ax.add_patch(rect)
ax.text(1.05, 2.25, 'Input\nFeature\nMap', ha='center', va='center', fontsize=8, fontweight='bold')

# Channel Attention
rect = FancyBboxPatch((2.5, 0.8), 3.5, 2.8, boxstyle="round,pad=0.1", facecolor='#FFF3E0', edgecolor='#E65100', lw=2)
ax.add_patch(rect)
ax.text(4.25, 3.3, 'Channel Attention', ha='center', va='center', fontsize=9, fontweight='bold', color='#E65100')
# Inner blocks
for y, label in [(2.4, 'Global Avg Pool'), (1.5, 'Global Max Pool')]:
    rect = FancyBboxPatch((2.8, y-0.25), 1.5, 0.5, boxstyle="round,pad=0.05", facecolor='#FFE0B2', edgecolor='#999', lw=1)
    ax.add_patch(rect)
    ax.text(3.55, y, label, ha='center', va='center', fontsize=6.5)
rect = FancyBboxPatch((4.6, 1.65), 1.1, 0.9, boxstyle="round,pad=0.05", facecolor='#FFCC80', edgecolor='#999', lw=1)
ax.add_patch(rect)
ax.text(5.15, 2.1, 'Shared\nMLP', ha='center', va='center', fontsize=7, fontweight='bold')

# Spatial Attention  
rect = FancyBboxPatch((6.8, 0.8), 3.2, 2.8, boxstyle="round,pad=0.1", facecolor='#E8F5E9', edgecolor='#2E7D32', lw=2)
ax.add_patch(rect)
ax.text(8.4, 3.3, 'Spatial Attention', ha='center', va='center', fontsize=9, fontweight='bold', color='#2E7D32')
for y, label in [(2.4, 'Avg Pool\n(channel)'), (1.5, 'Max Pool\n(channel)')]:
    rect = FancyBboxPatch((7.1, y-0.3), 1.3, 0.6, boxstyle="round,pad=0.05", facecolor='#C8E6C9', edgecolor='#999', lw=1)
    ax.add_patch(rect)
    ax.text(7.75, y, label, ha='center', va='center', fontsize=6)
rect = FancyBboxPatch((8.7, 1.65), 1.1, 0.9, boxstyle="round,pad=0.05", facecolor='#A5D6A7', edgecolor='#999', lw=1)
ax.add_patch(rect)
ax.text(9.25, 2.1, 'Conv2D\n7×7', ha='center', va='center', fontsize=7, fontweight='bold')

# Output
rect = FancyBboxPatch((10.8, 1.5), 1.5, 1.5, boxstyle="round,pad=0.1", facecolor='#F3E5F5', edgecolor='#333', lw=1.5)
ax.add_patch(rect)
ax.text(11.55, 2.25, 'Refined\nFeature\nMap', ha='center', va='center', fontsize=8, fontweight='bold')

# Arrows
for x1, x2 in [(1.8, 2.5), (6.0, 6.8), (10.0, 10.8)]:
    ax.annotate('', xy=(x2, 2.25), xytext=(x1, 2.25),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'cbam_module.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('✅ cbam_module.png')


# ============================================================
# 3. SE Block
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 3.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 3.5)
ax.axis('off')
ax.set_title('Squeeze-and-Excitation (SE) Block', fontsize=14, fontweight='bold', pad=12)

steps = [
    (0.3, 'Input\nH×W×C', '#E3F2FD'),
    (2.3, 'Squeeze\nGlobal Avg\nPooling\n→ 1×1×C', '#BBDEFB'),
    (4.5, 'Excitation\nFC(C/r, ReLU)\nFC(C, Sigmoid)', '#90CAF9'),
    (7.0, 'Scale\n×', '#FFF9C4'),
    (9.0, 'Output\nH×W×C\n(re-weighted)', '#C8E6C9'),
]
for x, label, color in steps:
    w = 1.8
    rect = FancyBboxPatch((x, 0.6), w, 2.2, boxstyle="round,pad=0.12", facecolor=color, edgecolor='#333', lw=1.5)
    ax.add_patch(rect)
    ax.text(x+w/2, 1.7, label, ha='center', va='center', fontsize=8, fontweight='bold')

for i in range(len(steps)-1):
    x1 = steps[i][0] + 1.8
    x2 = steps[i+1][0]
    ax.annotate('', xy=(x2, 1.7), xytext=(x1, 1.7),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'se_block.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('✅ se_block.png')


# ============================================================
# 4. System Pipeline
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 4))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4)
ax.axis('off')
ax.set_title('Luồng xử lý hệ thống nhận diện cảm xúc thời gian thực', fontsize=13, fontweight='bold', pad=12)

pipeline = [
    (0.2, '📷\nWebcam', '#E3F2FD'),
    (2.2, 'Capture\nFrame', '#BBDEFB'),
    (4.2, 'Face\nDetection\nHaar Cascade', '#90CAF9'),
    (6.2, 'Preprocessing\nResize 48×48\nGrayscale\nNormalize', '#64B5F6'),
    (8.4, 'CBAM-CNN\nModel\nInference', '#FF9800'),
    (10.6, 'Emotion\nPrediction\n7 classes', '#FFF9C4'),
    (12.5, '🖥️\nDisplay\nResult', '#C8E6C9'),
]

for x, label, color in pipeline:
    w = 1.6
    rect = FancyBboxPatch((x, 0.7), w, 2.5, boxstyle="round,pad=0.12", facecolor=color, edgecolor='#333', lw=1.5)
    ax.add_patch(rect)
    ax.text(x+w/2, 1.95, label, ha='center', va='center', fontsize=7.5, fontweight='bold')

for i in range(len(pipeline)-1):
    x1 = pipeline[i][0] + 1.6
    x2 = pipeline[i+1][0]
    ax.annotate('', xy=(x2, 1.95), xytext=(x1, 1.95),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2))

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'system_pipeline.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('✅ system_pipeline.png')


# ============================================================
# 5. Data Augmentation demo
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle('Data Augmentation - Các phép biến đổi ảnh khuôn mặt', fontsize=13, fontweight='bold')

np.random.seed(42)
# Create a simple face-like image
face = np.zeros((48, 48))
# Head outline
for i in range(48):
    for j in range(48):
        dist = np.sqrt((i-24)**2 + (j-24)**2)
        if dist < 20:
            face[i,j] = 0.8
        if dist < 18:
            face[i,j] = 0.9
# Eyes
face[16:20, 14:19] = 0.2
face[16:20, 29:34] = 0.2
# Nose
face[24:28, 22:26] = 0.5
# Mouth (smile)
for j in range(17, 31):
    row = int(32 + 2*np.sin((j-17)*np.pi/14))
    if 0 <= row < 48:
        face[row, j] = 0.3
        face[row+1, j] = 0.3

transforms = [
    ('Original', face),
    ('Rotation\n(+20°)', np.rot90(face, k=0)),  # simplified
    ('Horizontal\nFlip', np.fliplr(face)),
    ('Zoom In\n(+15%)', face[5:43, 5:43]),
    ('Shift Right\n(+15%)', np.roll(face, 7, axis=1)),
    ('Brightness\n(+20%)', np.clip(face * 1.2, 0, 1)),
    ('Shear\n(15%)', face),
    ('Combined', np.fliplr(np.clip(face * 1.1, 0, 1))),
]

for idx, (title, img) in enumerate(transforms):
    r, c = idx // 4, idx % 4
    if img.shape != (48, 48):
        from PIL import Image
        img_pil = Image.fromarray((img*255).astype(np.uint8))
        img_pil = img_pil.resize((48, 48))
        img = np.array(img_pil) / 255.0
    axes[r, c].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[r, c].set_title(title, fontsize=9, fontweight='bold')
    axes[r, c].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'data_augmentation.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('✅ data_augmentation.png')


# ============================================================
# 6. Method Comparison Bar Chart
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

methods = ['Method 1\nEnhanced Aug', 'Method 2\nSE-Attention', 'Method 3\nMobileNetV2', 'Method 4\nSE-CBAM+TTA', 'M2 Optimized\nCBAM-CNN+TTA']
accuracies = [0, 64.22, 36.28, 63.51, 63.50]
colors = ['#BDBDBD', '#4CAF50', '#F44336', '#2196F3', '#FF9800']

bars = ax.bar(methods, accuracies, color=colors, edgecolor='#333', linewidth=1.2, width=0.6)
ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('So sánh kết quả các phương pháp trên FER2013', fontsize=14, fontweight='bold')
ax.set_ylim(0, 80)
ax.axhline(y=60, color='red', linestyle='--', alpha=0.5, label='Mục tiêu ≥60%')
ax.legend(fontsize=10)

for bar, acc in zip(bars, accuracies):
    if acc > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{acc}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width()/2, 2, 'N/A', ha='center', fontsize=11, color='gray')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'method_comparison.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('✅ method_comparison.png')


# ============================================================
# 7. Transfer Learning diagram
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
ax.axis('off')
ax.set_title('Transfer Learning: MobileNetV2 cho FER2013', fontsize=13, fontweight='bold', pad=12)

# ImageNet pretrained
rect = FancyBboxPatch((0.3, 0.8), 3, 2.3, boxstyle="round,pad=0.12", facecolor='#E8EAF6', edgecolor='#3F51B5', lw=2)
ax.add_patch(rect)
ax.text(1.8, 2.6, 'MobileNetV2', ha='center', va='center', fontsize=10, fontweight='bold', color='#3F51B5')
ax.text(1.8, 1.9, 'Pretrained on\nImageNet\n(1.2M images, 1000 classes)', ha='center', va='center', fontsize=7.5)

# Arrow + Freeze
ax.annotate('', xy=(4.3, 1.95), xytext=(3.3, 1.95),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2))
ax.text(3.8, 2.5, 'Freeze\nbase layers', ha='center', va='center', fontsize=7, color='#E65100', fontweight='bold')

# Fine-tune head
rect = FancyBboxPatch((4.3, 0.8), 2.5, 2.3, boxstyle="round,pad=0.12", facecolor='#FFF3E0', edgecolor='#E65100', lw=2)
ax.add_patch(rect)
ax.text(5.55, 2.6, 'Fine-tune Head', ha='center', va='center', fontsize=10, fontweight='bold', color='#E65100')
ax.text(5.55, 1.7, 'New Dense layers\nfor 7 emotions\nInput: RGB 48→224', ha='center', va='center', fontsize=7.5)

# Arrow
ax.annotate('', xy=(7.8, 1.95), xytext=(6.8, 1.95),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2))

# Result (FAIL)
rect = FancyBboxPatch((7.8, 0.8), 3.5, 2.3, boxstyle="round,pad=0.12", facecolor='#FFEBEE', edgecolor='#F44336', lw=2)
ax.add_patch(rect)
ax.text(9.55, 2.6, '❌ Kết quả: 36.28%', ha='center', va='center', fontsize=10, fontweight='bold', color='#F44336')
ax.text(9.55, 1.6, 'Nguyên nhân:\n• Size mismatch (48→224)\n• Grayscale→RGB giả\n• Domain gap lớn', ha='center', va='center', fontsize=7.5)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'transfer_learning.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('✅ transfer_learning.png')


# ============================================================
# 8. FER2013 Distribution
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
counts = [3995, 436, 4097, 7215, 4965, 4830, 3171]
colors = ['#EF5350', '#AB47BC', '#5C6BC0', '#66BB6A', '#78909C', '#42A5F5', '#FFA726']

bars = ax.bar(emotions, counts, color=colors, edgecolor='#333', linewidth=1.2)
ax.set_ylabel('Số lượng ảnh', fontsize=12, fontweight='bold')
ax.set_title('Phân bố dữ liệu FER2013 (Training Set)', fontsize=14, fontweight='bold')

for bar, count in zip(bars, counts):
    pct = count / sum(counts) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 8500)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fer2013_distribution.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('✅ fer2013_distribution.png')

print('\n✅ Tất cả ảnh đã tạo xong!')
