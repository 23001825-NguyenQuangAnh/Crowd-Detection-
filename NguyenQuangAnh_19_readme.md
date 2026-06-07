# Thông tin về thành viên nhóm, công việc của mỗi thành viên.

|    Họ Và Tên     | Mã Sinh Viên |                                                                                                                                                                         Công Việc                                                                                                                                                                         |
|:----------------:|:------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Nguyễn Quang Anh |   23001825   |                           Viết pipeline về các vấn đề trích xuất biên và vùng cho việc phát hiện đối tương, khảo sát tập dữ liệu NWPU-Crowd, thiết kế hàm sinh bản đồ mật độ dựa trên bộ lọc Gauss.Xây dựng kiến trúc kết hợp EfficientNet-B0 và khối tích chập hồi quy mật độ (Density Map Regressor), viết báo cáo, làm slide                           |
| Phạm Dương Hoàng |   23001880   |                              Tiền xử lý ảnh với kênh RGB, nghiên cứu hướng tiếp cận đếm đám đông dựa trên bài toán phát hiện đối tượng (Object Detection), huấn luyện mạng hai giai đoạn Faster R-CNN kết hợp backbone tối ưu phần cứng MobileNetV2, đánh giá các chỉ số Precision/Recall/F1-score. viết báo cáo, làm slide                               |
|  Hồ Trọng Hiếu   |   23001874   | Tiền xử lý ảnh trên kênh HSV, làm pipeline phát hiện đối tượng dựa trên đặc trưng cổ điển (HOG), trích xuất đặc trưng HOG (Histogram of Oriented Gradients) và huấn luyện mô hình phân loại/hồi quy tuyến tính HOG + LinearSVM kết hợp Image Pyramid và Orientation Region Proposals, đánh giá kết quả dựa trên chỉ số MAE và MSE viết báo cáo, làm slide |

## 📂 Cấu trúc thư mục (Project Structure)

Dưới đây là sơ đồ tổ chức các thành phần trong dự án:

```text
NWPU-Crowd/
├── .venv/                          # Môi trường ảo Python (Virtual Environment)
├── images_part1/                   # Tập dữ liệu ảnh gốc - Phần 1 (.jpg)
├── images_part2/                   # Tập dữ liệu ảnh gốc - Phần 2 (.jpg)
├── images_part3/                   # Tập dữ liệu ảnh gốc - Phần 3 (.jpg)
├── images_part4/                   # Tập dữ liệu ảnh gốc - Phần 4 (.jpg)
├── images_part5/                   # Tập dữ liệu ảnh gốc - Phần 5 (.jpg)
├── jsons/                          # Chứa các file chứa tọa độ điểm đầu người (.json)
├── mats/                           # Chứa các file Ground-Truth định dạng Matlab (.mat)
│
├── generated_density/              # [Tự động tạo] Thư mục lưu bản đồ mật độ chuẩn dạng (.npy)
├── test_results/                   # [Tự động tạo] Kết quả trực quan hóa trên tập dữ liệu Test
│   ├── ket_qua_test.txt            # File tổng hợp ID ảnh và số lượng người dự đoán tương ứng
│   └── result_xxxx.jpg             # Ảnh kết quả đã được phủ bản đồ nhiệt (Heatmap) và chèn text Count
│
├── train.txt                       # File danh sách ID các ảnh dùng cho việc Huấn luyện (Train)
├── val.txt                         # File danh sách ID các ảnh dùng cho việc Đánh giá (Validation)
├── test.txt                        # File danh sách ID các ảnh dùng cho việc Thử nghiệm (Test)
│
├── best_crowd_model.pth            # [Tự động tạo] File lưu trọng số (weights) tối ưu nhất của mô hình
├── Bao_Cao_Validation.txt          # [Tự động tạo] Bảng thống kê chi tiết lỗi MAE/RMSE trên tập Val
│
├── NguyenQuangAnh_19_crowd_density_HOG_SVM1.ipynb           # [Notebook] Thử nghiệm hướng tiếp cận HOG + SVM
├── NguyenQuangAnh_19_crowd_density_mobilenetv2_fasterrcnn.ipynb # [Notebook] Thử nghiệm hướng tiếp cận Faster R-CNN
│
├── NguyenQuangAnh_19_train_model_crowd_detection_heatmap.py # [Script 1] Tiền xử lý, sinh mật độ Gauss và Huấn luyện/Fine-tuning
├── NguyenQuangAnh_19_crowd_detection_heatmap.py          # [Script 2] Đánh giá mô hình, tính chỉ số MAE, RMSE trên tập Validation
└── NguyenQuangAnh_19_crowd_detection_predict_test.py        # [Script 3] Dự đoán phân bố, đếm số lượng và xuất ảnh Heatmap tập Test
```
## Hướng dẫn tải bộ dữ liệu
Thực hiện tải bộ dữ liệu từ link sau:
1. https://gjy3035.github.io/NWPU-Crowd-Sample-Code/ (Tải bằng link Onedrive rồi giải nén toàn bộ các file .zip có trong dataset)
2. https://drive.google.com/drive/folders/134hSeIHbMdl7SOPOqe37ruexqn9mwKum (Tải thư mục NWPU-Crowd cho dataset của bài)

💻 Hướng Dẫn Chi Tiết Cách Chạy Các Script (Cho phương pháp 2)
🛠️ 0. Chuẩn bị môi trường
Hãy đảm bảo bạn đã kích hoạt môi trường ảo .venv và cài đặt đầy đủ các thư viện phụ thuộc bằng lệnh:

```Bash
pip install torch torchvision numpy opencv-python pillow scipy tqdm
```
🏋️ Bước 1: Huấn luyện hoặc Tối ưu hóa mô hình (Train / Fine-tuning)
Chạy script huấn luyện để mô hình học cách phân bổ mật độ từ các điểm tọa độ:

```Bash
python NguyenQuangAnh_19_train_model_crowd_detection_heatmap.py
```
Cơ chế hoạt động: 
1. Script tự động đọc các file tọa độ .json trong thư mục jsons/, áp dụng bộ lọc mờ Gaussian (gaussian_filter) để chuyển các điểm tọa độ rời rạc thành bản đồ mật độ liên tục, lưu dưới dạng mảng NumPy trong thư mục generated_density/.
2. Chương trình tự động kiểm tra xem file trọng số best_crowd_model.pth đã tồn tại chưa.

Nếu CHƯA tồn tại: Tiến hành huấn luyện toàn bộ mạng từ đầu.

Nếu ĐÃ tồn tại: Tự động chuyển sang chế độ Fine-tuning (Đóng băng các tầng đặc trưng EfficientNet để giữ lại tri thức cốt lõi, chỉ tinh chỉnh khối hồi quy với Learning Rate siêu nhỏ 1e-5).

Cơ chế Early Stopping tích hợp sẵn sẽ tự động ngắt tiến trình nếu chỉ số Validation MAE không cải thiện sau 10 Epochs liên tiếp để tránh hiện tượng quá khớp (Overfitting).

📊 Bước 2: Đánh giá độ chính xác trên tập Validation (Evaluation)
Sau khi có file trọng số tối ưu, thực hiện đánh giá sai số của mô hình:

```Bash
python NguyenQuangAnh_19_crowd_detection_heatmap.py
```
Cơ chế hoạt động: Script quét danh sách các ảnh trong file val.txt, đưa qua mô hình AI để dự đoán bản đồ mật độ, sau đó tính tích phân (tổng giá trị pixel) để quy đổi ra tổng số người trong bức ảnh.

Số lượng dự đoán sẽ được đối chiếu trực tiếp với số lượng điểm chính xác trong các file cấu hình .json.

Đầu ra: Kết quả đánh giá sẽ được ghi nhận chi tiết vào file Bao_Cao_Validation.txt và in ra màn hình bảng tổng kết gồm 2 chỉ số:

MAE (Sai số tuyệt đối trung bình): Trung bình mô hình đoán lệch bao nhiêu người trên một bức ảnh.

RMSE (Sai số bình phương trung bình tối thiểu): Đánh giá độ ổn định và mức độ ảnh hưởng của các điểm lỗi lớn.

🎨 Bước 3: Dự đoán thực tế và Trực quan hóa Heatmap (Inference / Predict)
Sử dụng mô hình để đếm người và sinh bản đồ nhiệt trên tập ảnh thực tế (mặc định cấu hình trỏ tới thư mục ảnh images_part5):

```Bash
python NguyenQuangAnh_19_crowd_detection_predict_test.py
```
Cơ chế hoạt động:

Script nạp trọng số từ file best_crowd_model.pth, thực hiện quét qua toàn bộ ảnh .jpg của tập kiểm thử được chỉ định.

Bản đồ mật độ đầu ra từ mô hình AI được chuẩn hóa và áp bộ lọc màu sinh động cv2.COLORMAP_JET (Vùng tập trung đông người hiển thị màu đỏ/ấm; vùng thưa thớt hiển thị màu xanh/lạnh).

Kết quả được hòa trộn (cv2.addWeighted) với ảnh gốc theo tỷ lệ 50-50 để tạo hiệu ứng xuyên thấu trực quan, đồng thời ghi đè thông tin số lượng người Count: X lên góc trái ảnh.

Đầu ra: Toàn bộ ảnh kết quả trực quan được lưu tại thư mục test_results/, kèm theo file tổng hợp số liệu ket_qua_test.txt.

⚙️ Cấu Hình Kiến Trúc Mô Hình (Model Architecture)
Mô hình được xây dựng trên nền tảng mạng kết hợp:

- Feature Extractor: EfficientNet-B0 (Pre-trained trên tập ImageNet) loại bỏ các tầng phân lớp fully-connected phía sau, giữ lại khối trích xuất đặc trưng không gian sâu với kích thước đầu ra gồm 1280 kênh màu (channels).

- Feature Interpolation: Áp dụng hàm nội suy song tuyến tính (F.interpolate) phóng đại bản đồ đặc trưng lên gấp 8 lần để khôi phục lại độ phân giải không gian đã mất mát qua các tầng Pooling/Stride.

- Density Map Regressor: Chuỗi gồm 4 tầng tích chập (Conv2d) hạ số kênh từ 1280 ──► 512 ──► 256 ──► 64 ──► 1 kết hợp hàm kích hoạt ReLU, ánh xạ thành công các vector đặc trưng trừu tượng thành một bản đồ mật độ đơn kênh duy nhất. Tổng số lượng người được tính bằng tổng toàn bộ giá trị trên bản đồ mật độ này.



