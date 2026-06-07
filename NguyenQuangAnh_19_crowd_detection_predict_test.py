import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
import os
import glob
from tqdm import tqdm


# ==============================================================================
# 1. KHAI BÁO CẤU TRÚC MẠNG (EfficientNet-B0)
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
# 2. HÀM DỰ ĐOÁN HÀNG LOẠT (BATCH PREDICTION)
# ==============================================================================
def run_test_evaluation(test_image_dir, model_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Đang khởi động AI trên thiết bị: {device}")

    # 1. Khởi tạo mô hình và nạp trọng số
    model = EfficientNetCrowdCounter().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Đã nạp thành công bộ não AI (best_crowd_model.pth)!")
    else:
        print(f"❌ Không tìm thấy file trọng số tại: {model_path}")
        return
    model.eval()

    # 2. Tạo thư mục chứa ảnh kết quả
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. Quét toàn bộ file .jpg trong thư mục Test
    image_paths = glob.glob(os.path.join(test_image_dir, "*.jpg"))
    if len(image_paths) == 0:
        print(f"⚠️ Không tìm thấy ảnh .jpg nào trong thư mục {test_image_dir}")
        return

    print(f"-> Đã tìm thấy {len(image_paths)} ảnh Test. Bắt đầu phân tích...\n")

    # Mở file txt để ghi log kết quả
    txt_log_path = os.path.join(output_dir, "ket_qua_test.txt")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with open(txt_log_path, "w", encoding="utf-8") as f_log:
        f_log.write("Image_ID\tPredicted_Count\n")

        # 4. Vòng lặp xử lý từng ảnh
        for img_path in tqdm(image_paths, desc="Đang phân tích Test Set"):
            filename = os.path.basename(img_path)

            try:
                image_org = Image.open(img_path).convert('RGB')
                image_org = ImageOps.exif_transpose(image_org)
            except Exception:
                continue

            w, h = image_org.size
            max_size = 1280
            if max(w, h) > max_size:
                ratio = max_size / float(max(w, h))
                w, h = int(w * ratio), int(h * ratio)

            new_w, new_h = (w // 16) * 16, (h // 16) * 16
            image_resized = image_org.resize((new_w, new_h), Image.BILINEAR)
            img_tensor = transform(image_resized).unsqueeze(0).to(device)

            # Dự đoán
            with torch.no_grad():
                output_map = model(img_tensor)
                predicted_count = torch.sum(output_map).item()

            # Ghi kết quả đếm vào file text
            f_log.write(f"{filename}\t{int(round(predicted_count))}\n")

            # Xử lý Heatmap chống nhiễu
            density_np = output_map.squeeze().cpu().numpy()
            density_np = cv2.resize(density_np, (image_org.width, image_org.height))
            max_val = np.max(density_np)

            if max_val < 0.05:
                heatmap = cv2.applyColorMap(np.zeros_like(density_np, dtype=np.uint8), cv2.COLORMAP_JET)
            else:
                density_img = density_np / max_val
                density_img = np.clip(density_img * 255, 0, 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(density_img, cv2.COLORMAP_JET)

            # Vẽ Heatmap đè lên ảnh gốc
            image_org_cv = cv2.cvtColor(np.array(image_org), cv2.COLOR_RGB2BGR)
            blended = cv2.addWeighted(image_org_cv, 0.5, heatmap, 0.5, 0)

            # Gắn chữ số lượng lên góc ảnh cho ngầu
            cv2.putText(blended, f"Count: {int(round(predicted_count))}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            # Lưu ảnh kết quả
            out_img_path = os.path.join(output_dir, f"result_{filename}")
            cv2.imwrite(out_img_path, blended)

    print(f"\n🎉 HOÀN TẤT! Đã phân tích xong toàn bộ tập Test.")
    print(f"📁 Ảnh Heatmap được lưu tại: {output_dir}")
    print(f"📄 Bảng thống kê số lượng được lưu tại: {txt_log_path}")

if __name__ == "__main__":

    # Tự động lấy thư mục gốc của project
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # =====================================================================
    # KHAI BÁO ĐƯỜNG DẪN ĐỘNG (TỰ ĐỘNG CHẠY TRÊN MỌI MÁY)
    # =====================================================================

    TEST_IMAGES_DIR = os.path.join(BASE_DIR, "images_part5")

    # 2. File trọng số tốt nhất nằm ngay trong thư mục gốc
    MODEL_WEIGHTS = os.path.join(BASE_DIR, "best_crowd_model.pth")

    # 3. Nơi xuất kết quả ra (Code sẽ tự tạo folder này trong thư mục gốc)
    OUTPUT_RESULTS_DIR = os.path.join(BASE_DIR, "test_results")

    print(f"📁 Thư mục Test đang trỏ tới: {TEST_IMAGES_DIR}")

    run_test_evaluation(TEST_IMAGES_DIR, MODEL_WEIGHTS, OUTPUT_RESULTS_DIR)