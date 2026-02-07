# BÁO CÁO NGHIÊN CỨU ĐỀ TÀI (Bổ sung và Chỉnh sửa)

Dưới đây là nội dung đề xuất cho các chương bạn yêu cầu, bao gồm **sơ đồ khối phân tích thiết kế phần train model**. Bạn có thể sao chép nội dung này vào file Word báo cáo chính thức của mình.

---

## Chương 1: Cơ sở lý thuyết

### 1.1. Tổng quan về Nhận diện cảm xúc khuôn mặt (Facial Expression Recognition)
Nhận diện cảm xúc khuôn mặt là một bài toán quan trọng trong lĩnh vực Thị giác máy tính (Computer Vision) và Trí tuệ nhân tạo. Mục tiêu là phân loại cảm xúc của con người dựa trên hình ảnh khuôn mặt vào các nhóm cơ bản như: **Vui, Buồn, Giận dữ, Ngạc nhiên, Sợ hãi, Ghê tởm và Bình thường**.

### 1.2. Mạng Nơ-ron Tích chập (Convolutional Neural Networks - CNN)
CNN là kiến trúc mạng nơ-ron chuyên biệt cho việc xử lý dữ liệu dạng lưới như hình ảnh.
- **Lớp tích chập (Convolutional Layer):** Trích xuất các đặc trưng từ ảnh (cạnh, góc, texture).
- **Lớp gộp (Pooling Layer):** Giảm chiều dữ liệu, giữ lại các đặc trưng quan trọng và giảm chi phí tính toán.
- **Lớp kết nối đầy đủ (Fully Connected Layer):** Thực hiện phân loại dựa trên các đặc trưng đã trích xuất.

### 1.3. Transfer Learning và MobileNetV2
- **Transfer Learning (Học chuyển giao):** Là kỹ thuật sử dụng một mô hình đã được huấn luyện trên tập dữ liệu lớn (như ImageNet) để giải quyết bài toán khác tương tự. Phương pháp này giúp tiết kiệm thời gian huấn luyện và cải thiện độ chính xác khi dữ liệu hạn chế.
- **MobileNetV2:** Là một kiến trúc CNN nhẹ, được tối ưu hóa cho các thiết bị di động và nhúng. MobileNetV2 sử dụng các block "Inverted Residuals" và "Linear Bottlenecks" để giảm số lượng tham số nhưng vẫn duy trì hiệu năng cao.

### 1.4. Các thư viện sử dụng
- **TensorFlow/Keras:** Framework Deep Learning dùng để xây dựng và huấn luyện mô hình.
- **OpenCV:** Thư viện mã nguồn mở hàng đầu về xử lý ảnh, hỗ trợ các tác vụ như đọc camera, phát hiện khuôn mặt (Haar Cascade).

---

## Chương 2: Phương pháp đề xuất

### 2.1. Mô tả bài toán
Xây dựng hệ thống nhận diện 7 loại cảm xúc cơ bản từ webcam thời gian thực.

### 2.2. Quy trình xử lý dữ liệu
1.  **Dữ liệu đầu vào:** Ảnh khuôn mặt được gán nhãn thuộc 7 lớp cảm xúc.
2.  **Tiền xử lý:**
    -   Resize ảnh về kích thước **48x48 pixel** (3 kênh màu RGB) để phù hợp với đầu vào mô hình MobileNetV2.
    -   Chuẩn hóa giá trị pixel về khoảng **[0, 1]** (chia cho 255).
    -   Sử dụng **Data Augmentation** (Tăng cường dữ liệu): Xoay, dịch chuyển, lật ngang, thay đổi độ sáng để giúp mô hình học tốt hơn và tránh Overfitting.
3.  **Cân bằng dữ liệu:** Sử dụng kỹ thuật `class_weight` để xử lý vấn đề mất cân bằng giữa các lớp dữ liệu (ví dụ: lớp 'Hạnh phúc' nhiều ảnh hơn 'Ghê tởm').

### 2.3. Kiến trúc mô hình đề xuất
Chúng tôi sử dụng phương pháp Transfer Learning với kiến trúc **MobileNetV2**.
- **Backbone:** MobileNetV2 (đã bỏ lớp top), sử dụng trọng số 'imagenet'. Các lớp đầu được đóng băng (frozen) để giữ lại các đặc trưng cơ bản, các lớp sau được mở (unfrozen) để fine-tune.
- **Classification Head (Phần phân loại thêm vào):**
    -   `GlobalAveragePooling2D`: Giảm chiều dữ liệu vector đặc trưng.
    -   `Dense` (256 units, ReLU, L2 Regularization): Lớp ẩn thứ nhất trích xuất đặc trưng cấp cao.
    -   `BatchNormalization` & `Dropout` (0.5): Kỹ thuật chống overfitting mạnh mẽ.
    -   `Dense` (128 units, ReLU, L2 Regularization): Lớp ẩn thứ hai.
    -   `Dense` (7 units, Softmax): Lớp đầu ra cho xác suất của 7 cảm xúc.

### 2.4. Cấu hình huấn luyện
- **Loss Function:** Categorical Crossentropy (có Label Smoothing để giảm label noise).
- **Optimizer:** Adam với learning rate thấp (0.0001) để phù hợp với quá trình fine-tuning.
- **Callbacks:** EarlyStopping (dừng sớm), ReduceLROnPlateau (giảm learning rate tự động), ModelCheckpoint (lưu model tốt nhất).

---

## Chương 3: Phân tích thiết kế hệ thống

### 3.1. Sơ đồ khối quy trình huấn luyện (Training Pipeline Diagram)
Dưới đây là thiết kế chi tiết cho thành phần huấn luyện mô hình (Training Component), phân tích luồng dữ liệu từ đầu vào đến khi ra được mô hình tối ưu.

```mermaid
graph TD
    %% Định nghĩa các node và style
    Dat[<b>Bộ dữ liệu ảnh</b><br/>(Dataset Images)]
    
    subgraph Preprocessing [TIỀN XỬ LÝ & TĂNG CƯỜNG DỮ LIỆU]
        Resize[Resize về <b>48x48x3</b>]
        Norm[Chuẩn hóa pixel <b>[0, 1]</b>]
        Split[Chia tập <b>Train/Val</b>]
        Aug[<b>Data Augmentation</b><br/>Rotate, Shift, Flip, Brightness]
        Weight[Tính <b>Class Weights</b>]
    end

    subgraph ModelDesign [KIẾN TRÚC MÔ HÌNH (MobileNetV2)]
        mob[<b>MobileNetV2 Base</b><br/>ImageNet Weights<br/><i>(Frozen low layers)</i>]
        glob[<b>Global Average Pooling</b>]
        fc1[<b>Dense 256</b> + L2 + ReLU]
        bn1[<b>BatchNormalization</b> + Dropout 0.5]
        fc2[<b>Dense 128</b> + L2 + ReLU]
        bn2[<b>BatchNormalization</b> + Dropout 0.5]
        out[<b>Output Layer</b><br/>Softmax (7 units)]
    end

    subgraph Training [QUÁ TRÌNH HUẤN LUYỆN]
        compile[<b>Compile Model</b><br/>Adam (lr=1e-4)<br/>Cat. Crossentropy]
        fit[<b>Model.fit()</b><br/>Training Loop]
        eval[<b>Callbacks</b><br/>EarlyStopping, Checkpoint]
    end
    
    Result((<b>Mô hình tối ưu</b><br/>best_model.keras))

    %% Luồng kết nối
    Dat --> Resize --> Norm --> Split
    Split -- Train Set --> Aug --> Weight
    Split -- Validation Set --> mob
    Weight --> fit
    
    Aug --> mob
    mob --> glob --> fc1 --> bn1 --> fc2 --> bn2 --> out
    
    out --> compile --> fit
    fit <--> eval
    fit -->|Hoàn thành| Result
```

**(Cần vẽ lại sơ đồ này vào báo cáo Word bằng Visio/Draw.io hoặc chèn ảnh chụp màn hình sơ đồ trên)**

**Phân tích các khối chức năng trong sơ đồ:**
1.  **Khối Dữ liệu (Data Block):** Đầu vào là ảnh thô. Thông qua quá trình Normalize và Resize, ảnh được chuẩn hoá về dạng ma trận số học phù hợp cho tính toán. Việc chia tập Train/Val đảm bảo đánh giá khách quan.
2.  **Khối Tăng cường (Augmentation Block):** Đóng vai trò quan trọng trong việc tạo ra các biến thể dữ liệu (nghiêng, tối, lật), giúp mô hình học được các đặc trưng bất biến của khuôn mặt thay vì học vẹt trên ảnh gốc.
3.  **Khối Mô hình (Model Architecture Block):** Thiết kế dạng phễu. MobileNetV2 trích xuất hàng nghìn đặc trưng. Các lớp Dense phía sau cô đọng lại các đặc trưng này, loại bỏ nhiễu (nhờ Dropout/L2) để đưa ra quyết định cuối cùng chính xác nhất.
4.  **Khối Huấn luyện (Training Block):** Là "bộ não" điều khiển việc học. Nó liên tục so sánh dự đoán của model với kết quả thực tế (thông qua hàm Loss) và điều chỉnh trọng số (thông qua Optimizer). Callbacks giám sát quá trình này để đảm bảo lưu lại phiên bản tốt nhất chứ không phải phiên bản cuối cùng.

### 3.2. Sơ đồ luồng hoạt động thời gian thực (Real-time Flow)
`Camera` --> `Lấy Frame` --> `Phát hiện khuôn mặt (Haar Cascade)` --> `Cắt vùng mặt (ROI)` --> `Resize/Chuẩn hóa (theo chuẩn train)` --> `Mô hình AI dự đoán` --> `Hiển thị Kết quả`.

---

## Chương 4: Kết quả và Đánh giá (Tên đề tài: Nhận diện cảm xúc khuôn mặt sử dụng Deep Learning)

### 4.1. Kết quả huấn luyện
-   Hệ thống đã được huấn luyện qua 50 epochs.
-   Độ chính xác trên tập kiểm thử (Test Accuracy) đạt khoảng **36.28%**.
-   Mặc dù độ chính xác chưa cao tuyệt đối (do dữ liệu khó và tính chất nhập nhằng của cảm xúc), mô hình đã học được các đặc trưng cơ bản để phân biệt các cảm xúc rõ ràng như "Hạnh phúc" hay "Ngạc nhiên".

### 4.2. Đánh giá chi tiết (Classification Report)
-   **Độ chính xác cao:** Lớp 'Happy' (Hạnh phúc) và 'Surprise' (Ngạc nhiên) có kết quả nhận diện tốt nhất (F1-score cao).
-   **Độ chính xác thấp:** Các lớp như 'Fear' (Sợ hãi) hay 'Disgust' (Ghê tởm) dễ bị nhầm lẫn do số lượng mẫu ít và biểu cảm khuôn mặt tương đồng với các lớp khác (ví dụ: Sợ hãi dễ nhầm với Ngạc nhiên).

### 4.3. Thử nghiệm thực tế
Hệ thống hoạt động ổn định trên Webcam laptop, tốc độ xử lý nhanh (real-time), có thể bắt được khuôn mặt và đưa ra dự đoán ngay lập tức. Tính năng nhận diện khuôn mặt bằng Haar Cascade hoạt động nhẹ nhàng và hiệu quả trên CPU.

---

## Kết luận

### Kết luận chung
Đồ án đã xây dựng thành công quy trình khép kín: thu thập dữ liệu -> thiết kế pipeline huấn luyện chuẩn chỉnh với Data Augmentation/Transfer Learning -> triển khai ứng dụng thực tế. 

### Hướng phát triển
-   Cải thiện bộ dữ liệu (Dataset Quality).
-   Tinh chỉnh kiến trúc model (Fine-tuning sâu hơn).
-   Sử dụng Face Mesh để lấy đặc trưng hình học thay vì chỉ dùng ảnh pixel.

---

## Tài liệu tham khảo
1.  Simonyan, K. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*.
2.  Sandler, M. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*.
3.  OpenCV Documentation: https://docs.opencv.org/
4.  TensorFlow Core API: https://www.tensorflow.org/api_docs
