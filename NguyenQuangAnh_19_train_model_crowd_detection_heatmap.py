import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageFile, ImageOps
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
# ==============================================================================
# HÀM HỖ TRỢ: QUÉT TÌM ĐƯỜNG DẪN ẢNH TRONG CÁC FOLDER PART
# ==============================================================================
def build_image_path_map(base_dir):
    """
    Quét qua các thư mục images_part1 -> images_part5
    Trả về một Dictionary map ID ảnh với đường dẫn tuyệt đối của nó.
    Ví dụ: {'0001': 'E:/Project/Crowd Detection/images_part1/0001.jpg'}
    """
    img_map = {}
    print("Đang quét cấu trúc thư mục ảnh...")
    for i in range(1, 6):
        part_folder = os.path.join(base_dir, f"images_part{i}")
        if os.path.exists(part_folder):
            for filename in os.listdir(part_folder):
                if filename.endswith(".jpg"):
                    img_id = filename.split('.')[0]
                    img_map[img_id] = os.path.join(part_folder, filename)
    print(f"-> Đã tìm thấy tổng cộng {len(img_map)} ảnh gốc.")
    return img_map


# ==============================================================================
# BƯỚC 1: TIỀN XỬ LÝ - SINH BẢN ĐỒ MẬT ĐỘ (DENSITY MAP GENERATOR)
# ==============================================================================
def precompute_density_maps(txt_list_path, img_map, json_dir, output_gt_dir, sigma=15):
    if not os.path.exists(output_gt_dir):
        os.makedirs(output_gt_dir)

    print(f"\n--- Bắt đầu kiểm tra/tạo Density Maps từ {os.path.basename(txt_list_path)} ---")

    with open(txt_list_path, 'r') as f:
        lines = f.readlines()

    # Thêm tqdm để vẽ thanh tiến trình
    for line in tqdm(lines, desc="Đang sinh Heatmap GT"):
        parts = line.strip().split()
        if len(parts) == 0 or not parts[0].isdigit():
            continue

        img_id = parts[0]
        image_path = img_map.get(img_id)
        if not image_path:
            continue

        json_path = os.path.join(json_dir, f"{img_id}.json")
        save_path = os.path.join(output_gt_dir, f"{img_id}.npy")

        # Nếu đã có file npy thì bỏ qua
        if os.path.exists(save_path):
            continue

        try:
            img = Image.open(image_path).convert('RGB')
            img = ImageOps.exif_transpose(img)
            w, h = img.size
        except Exception:
            continue

        density_map = np.zeros((h, w), dtype=np.float32)

        if os.path.exists(json_path):
            with open(json_path, 'r') as jf:
                data = json.load(jf)
            points = data.get('points', [])

            for point in points:
                x, y = int(point[0]), int(point[1])
                if 0 <= y < h and 0 <= x < w:
                    # 2. CỘNG DỒN: Đảm bảo không bị mất người nếu trùng pixel
                    density_map[y, x] += 1.0

            if len(points) > 0:
                # 3. CHỐNG BÓNG MA: Chặn viền ảnh không cho dội ngược mật độ
                density_map = gaussian_filter(density_map, sigma=sigma, mode='constant', cval=0.0)

        np.save(save_path, density_map)
    print("--- Tiền xử lý Density Maps hoàn tất! ---")


# ==============================================================================
# BƯỚC 2: DATASET & DATALOADER CHO CROWD COUNTING
# ==============================================================================
class NWPUCrowdDataset(Dataset):
    def __init__(self, txt_list_path, img_map, gt_density_dir, crop_size=512, is_train=True):
        self.img_map = img_map
        self.gt_density_dir = gt_density_dir
        self.crop_size = crop_size
        self.is_train = is_train
        self.data_list = []

        with open(txt_list_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 1 and parts[0].isdigit():
                    img_id = parts[0]
                    # Chỉ đưa vào danh sách train nếu ảnh thực sự tồn tại trong các thư mục part
                    if img_id in self.img_map:
                        self.data_list.append(img_id)

        self.img_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),  # 20% xác suất biến ảnh thành trắng đen
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_id = self.data_list[idx]

        img_path = self.img_map[img_id]
        gt_path = os.path.join(self.gt_density_dir, f"{img_id}.npy")

        # 1. Đọc numpy array trước
        density_np = np.load(gt_path).copy()

        # Lấy kích thước CHUẨN từ mảng numpy (h, w)
        h_gt, w_gt = density_np.shape

        # 2. Đọc ảnh bằng PIL
        image = Image.open(img_path).convert('RGB')
        image = ImageOps.exif_transpose(image)

        # ĐỒNG BỘ KÍCH THƯỚC: Nếu PIL đọc sai lệch so với Numpy, ép nó về cùng kích thước
        if image.size != (w_gt, h_gt):
            image = image.resize((w_gt, h_gt), Image.BILINEAR)

        # Gán lại w, h để code bên dưới chạy đúng
        w, h = w_gt, h_gt

        # 3. Chuyển sang Tensor
        density_map = torch.tensor(density_np, dtype=torch.float32).unsqueeze(0)

        if self.is_train:
            if w > self.crop_size and h > self.crop_size:
                x1 = random.randint(0, w - self.crop_size)
                y1 = random.randint(0, h - self.crop_size)
                image = image.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

                # Cắt tensor
                density_map = density_map[:, y1:y1 + self.crop_size, x1:x1 + self.crop_size]
                if random.random() > 0.5:
                    # Lật ngang ảnh gốc
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    # Lật ngang bản đồ mật độ (Trục số 2 là trục W - chiều rộng)
                    density_map = torch.flip(density_map, dims=[2])
            else:
                image = image.resize((self.crop_size, self.crop_size), Image.BILINEAR)
                density_map = F.interpolate(density_map.unsqueeze(0), size=(self.crop_size, self.crop_size),
                                            mode='bilinear', align_corners=False).squeeze(0)
        else:
            max_size = 1024
            if max(w, h) > max_size:
                ratio = max_size / float(max(w, h))
                w = int(w * ratio)
                h = int(h * ratio)

            new_w = (w // 16) * 16
            new_h = (h // 16) * 16

            image = image.resize((new_w, new_h), Image.BILINEAR)

            original_sum = torch.sum(density_map).item()
            density_map = F.interpolate(density_map.unsqueeze(0), size=(new_h, new_w), mode='bilinear',
                                        align_corners=False).squeeze(0)

            current_sum = torch.sum(density_map).item()
            if current_sum > 0:
                density_map = density_map * (original_sum / current_sum)

        image_tensor = self.img_transform(image)

        # Ép clone() và contiguous() cho cả 2 output
        # Điều kiện tiên quyết để collate (gộp batch) qua multiprocessing an toàn
        return image_tensor.clone().contiguous(), density_map.clone().contiguous()


# =============================================================================
# # BƯỚC 3: MÔ HÌNH NHẸ (EfficientNet-B0 + REGRESSION HEAD)
# ==============================================================================
class EfficientNetCrowdCounter(nn.Module):
    def __init__(self):
        super(EfficientNetCrowdCounter, self).__init__()
        # Load pre-trained EfficientNet-B0 (dùng cú pháp weights mới để không bị cảnh báo)
        efficientnet = models.efficientnet_b0(weights='DEFAULT')
        self.features = efficientnet.features

        # Đầu ra của EfficientNet-B0 có 1280 kênh
        self.regressor = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.features(x)
        # Phóng to bản đồ đặc trưng lên 8 lần
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        return self.regressor(x)


# ==============================================================================
# BƯỚC 4: VÒNG LẶP HUẤN LUYỆN
# ==============================================================================
def main_pipeline():
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    JSON_DIR = os.path.join(BASE_DIR, "jsons")
    GT_DENSITY_DIR = os.path.join(BASE_DIR, "generated_density")  # Thư mục này sẽ tự động được tạo

    TRAIN_TXT = os.path.join(BASE_DIR, "train.txt")
    VAL_TXT = os.path.join(BASE_DIR, "val.txt")

    # Cấu hình thuật toán
    BATCH_SIZE = 4
    EPOCHS = 30
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy trên thiết bị: {DEVICE}")

    # 1. Quét tìm toàn bộ ảnh trong các thư mục part
    img_path_map = build_image_path_map(BASE_DIR)

    # 2. Sinh Ground Truth
    precompute_density_maps(TRAIN_TXT, img_path_map, JSON_DIR, GT_DENSITY_DIR)
    precompute_density_maps(VAL_TXT, img_path_map, JSON_DIR, GT_DENSITY_DIR)

    # 3. Nạp dữ liệu
    train_dataset = NWPUCrowdDataset(TRAIN_TXT, img_path_map, GT_DENSITY_DIR, crop_size=512, is_train=True)
    val_dataset = NWPUCrowdDataset(VAL_TXT, img_path_map, GT_DENSITY_DIR, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 4. Khởi tạo mô hình (TÍCH HỢP FINE-TUNING)
    model = EfficientNetCrowdCounter().to(DEVICE)

    # --- CƠ CHẾ KẾ THỪA VÀ KHÓA LỚP (FREEZE) ---
    WEIGHTS_PATH = os.path.join(BASE_DIR, "best_crowd_model.pth")
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print("♻️ Đã nạp thành công bộ não cũ (Epoch 30). Bắt đầu Fine-tuning!")

        # Khóa (Freeze) toàn bộ các lớp của EfficientNet (không cho học lại phần này)
        for param in model.features.parameters():
            param.requires_grad = False
        print("🔒 Đã khóa lớp trích xuất nền tảng. Chỉ huấn luyện Regressor.")
    else:
        print("⚠️ Không tìm thấy file trọng số cũ, mô hình sẽ học lại từ đầu.")
    # ------------------------------------------------
    criterion = nn.L1Loss()

    # GIẢM LEARNING RATE VÀ CHỈ CẬP NHẬT CÁC LỚP CHƯA BỊ KHÓA
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    # 5. Training Loop
    best_mae = float('inf')

    epochs_no_improve = 0
    patience = 10
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        print(f"\n================ EPOCH {epoch + 1}/{EPOCHS} ================")

        # Bọc train_loader bằng tqdm để tạo thanh tiến trình
        train_loop = tqdm(train_loader, desc="Training", leave=False)

        for images, gt_maps in train_loop:
            images = images.to(DEVICE)
            gt_maps = gt_maps.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)

            gt_downsampled = F.interpolate(gt_maps, size=(outputs.shape[2], outputs.shape[3]), mode='bilinear',
                                           align_corners=False)
            scale_factor = (gt_maps.shape[2] * gt_maps.shape[3]) / (outputs.shape[2] * outputs.shape[3])
            gt_downsampled = gt_downsampled * scale_factor

            loss = criterion(outputs, gt_downsampled)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Cập nhật loss liên tục lên thanh tiến trình
            train_loop.set_postfix(loss=loss.item())

        print(f"-> Training Loss (Epoch {epoch + 1}): {epoch_loss / len(train_loader):.6f}")

        # --- Đánh giá Validation ---
        model.eval()
        total_mae = 0.0
        torch.cuda.empty_cache()

        # Thêm thanh tiến trình cho phần Validation
        val_loop = tqdm(val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for images, gt_maps in val_loop:
                images = images.to(DEVICE)
                outputs = model(images)

                pred_count = torch.sum(outputs).item()
                gt_count = torch.sum(gt_maps).item()

                total_mae += abs(pred_count - gt_count)

        mae = total_mae / len(val_loader)
        print(f"=== Validation MAE: {mae:.2f} ===")

        scheduler.step(mae)
        print(f"--> Tốc độ học (Learning Rate) hiện tại: {optimizer.param_groups[0]['lr']}")

        if mae < best_mae:
            best_mae = mae
            epochs_no_improve = 0  # Reset lại bộ đếm nếu có kỷ lục mới
            torch.save(model.state_dict(), os.path.join(BASE_DIR, "best_crowd_model.pth"))
            print("--> Đã lưu mô hình tốt nhất!")
        else:
            epochs_no_improve += 1
            print(f"--> Không cải thiện: {epochs_no_improve}/{patience} epoch.")

            if epochs_no_improve >= patience:
                print("\n⛔ AI đã có dấu hiệu học vẹt. Kích hoạt Early Stopping để dừng huấn luyện sớm!")
                break  # Phá vỡ vòng lặp, kết thúc chương trình


if __name__ == "__main__":
    main_pipeline()