import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFile
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
import os
import json
import math
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ==============================================================================
# 1. KHAI BÁO CẤU TRÚC MẠNG
# ==============================================================================
class EfficientNetCrowdCounter(nn.Module):
    def __init__(self):
        super(EfficientNetCrowdCounter, self).__init__()
        efficientnet = models.efficientnet_b0(weights='DEFAULT')
        self.features = efficientnet.features
        self.regressor = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        return self.regressor(x)


# ==============================================================================
# 2. HÀM QUÉT ẢNH (Tương tự lúc Train)
# ==============================================================================
def build_image_path_map(base_dir):
    img_map = {}
    for i in range(1, 6):
        part_folder = os.path.join(base_dir, f"images_part{i}")
        if os.path.exists(part_folder):
            for filename in os.listdir(part_folder):
                if filename.endswith(".jpg"):
                    img_id = filename.split('.')[0]
                    img_map[img_id] = os.path.join(part_folder, filename)
    return img_map


# ==============================================================================
# 3. HÀM CHẠY ĐÁNH GIÁ (EVALUATION)
# ==============================================================================
def evaluate_model(base_dir, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Đang khởi động đánh giá trên thiết bị: {device}")

    # Nạp mô hình
    model = EfficientNetCrowdCounter().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Đã nạp thành công bộ não AI!")
    else:
        print("❌ Không tìm thấy file trọng số.")
        return
    model.eval()

    # Khởi tạo đường dẫn
    val_txt = os.path.join(base_dir, "val.txt")
    json_dir = os.path.join(base_dir, "jsons")
    img_map = build_image_path_map(base_dir)

    # Đọc danh sách file Validation
    with open(val_txt, 'r') as f:
        val_lines = f.readlines()

    val_image_ids = [line.strip().split()[0] for line in val_lines if
                     line.strip() and line.strip().split()[0] in img_map]
    print(f"-> Tổng số ảnh Validation hợp lệ: {len(val_image_ids)}")

    # Bộ tiền xử lý ảnh
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Khởi tạo biến đếm lỗi
    total_mae = 0.0
    total_mse = 0.0

    # Mở file để ghi báo cáo chi tiết
    report_path = os.path.join(base_dir, "Bao_Cao_Validation.txt")

    with open(report_path, "w", encoding="utf-8") as f_log:
        f_log.write("Image_ID\tGT_Count(Thực tế)\tPred_Count(AI Đoán)\tSai_số\n")

        # Chạy vòng lặp đánh giá
        for img_id in tqdm(val_image_ids, desc="Đang đánh giá tập Valid"):
            img_path = img_map[img_id]
            json_path = os.path.join(json_dir, f"{img_id}.json")

            # 1. Lấy Ground Truth (Đáp án thực tế) trực tiếp từ file JSON cho chuẩn xác 100%
            gt_count = 0
            if os.path.exists(json_path):
                with open(json_path, 'r') as jf:
                    data = json.load(jf)
                    gt_count = data.get("human_num", 0)

            # 2. Tiền xử lý ảnh giống hệt lúc Train
            try:
                image_org = Image.open(img_path).convert('RGB')
                image_org = ImageOps.exif_transpose(image_org)
            except:
                continue

            w, h = image_org.size
            max_size = 1024  # Giới hạn size để không tràn RAM như lúc Train
            if max(w, h) > max_size:
                ratio = max_size / float(max(w, h))
                w, h = int(w * ratio), int(h * ratio)

            new_w, new_h = (w // 16) * 16, (h // 16) * 16
            image_resized = image_org.resize((new_w, new_h), Image.BILINEAR)
            img_tensor = transform(image_resized).unsqueeze(0).to(device)

            # 3. AI Dự đoán
            with torch.no_grad():
                output_map = model(img_tensor)
                pred_count = torch.sum(output_map).item()

            # 4. Tính toán sai số
            error = abs(pred_count - gt_count)
            total_mae += error
            total_mse += (error ** 2)

            # Ghi vào file log
            f_log.write(f"{img_id}\t{gt_count}\t{pred_count:.2f}\t{error:.2f}\n")

    # ==========================================
    # TỔNG KẾT KẾT QUẢ ĐẦU RA
    # ==========================================
    final_mae = total_mae / len(val_image_ids)
    final_rmse = math.sqrt(total_mse / len(val_image_ids))

    print("\n" + "=" * 50)
    print("🏆 KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP VALIDATION 🏆")
    print("=" * 50)
    print(f"Tổng số ảnh đã test: {len(val_image_ids)} ảnh")
    print(f"📉 Chỉ số MAE (Sai số trung bình):   {final_mae:.2f} người/ảnh")
    print(f"📉 Chỉ số RMSE (Độ lệch chuẩn lỗi):  {final_rmse:.2f}")
    print("=" * 50)
    print(f"📄 Bảng báo cáo chi tiết từng ảnh đã được lưu tại: {report_path}")


if __name__ == "__main__":

    # Lấy đường dẫn tự động
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "best_crowd_model.pth")

    print(f"📁 Đang chạy đánh giá tại thư mục gốc: {BASE_DIR}")

    evaluate_model(BASE_DIR, MODEL_PATH)