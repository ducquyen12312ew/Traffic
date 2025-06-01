# quick_train_vn.py - Train nhanh model biển báo VN
import os
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import glob

def quick_setup_and_train():
    """Setup và train nhanh model VN"""
    print("🚀 TRAIN NHANH MODEL BIỂN BÁO VIỆT NAM")
    print("="*50)
    
    # Kiểm tra dataset
    dataset_root = "vietnamese_traffic_signs"
    if not os.path.exists(dataset_root):
        print("❌ Không tìm thấy dataset 'vietnamese_traffic_signs'")
        print("💡 Chạy trước: python setup_kaggle.py")
        return False
    
    # Tìm thư mục images và labels
    images_dir = None
    labels_dir = None
    
    for root, dirs, files in os.walk(dataset_root):
        if 'images' in dirs:
            images_dir = os.path.join(root, 'images')
        if 'labels' in dirs:
            labels_dir = os.path.join(root, 'labels')
    
    if not images_dir or not labels_dir:
        print("❌ Không tìm thấy thư mục images hoặc labels!")
        return False
    
    print(f"✅ Tìm thấy dataset: {images_dir}, {labels_dir}")
    
    # Đếm files
    image_files = glob.glob(os.path.join(images_dir, '*.jpg'))
    label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    
    print(f"📸 Images: {len(image_files)}")
    print(f"📝 Labels: {len(label_files)}")
    
    # Setup training data nhanh
    train_dir = "vn_training_quick"
    
    # Xóa thư mục cũ nếu có
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    
    # Tạo cấu trúc
    dirs = [
        os.path.join(train_dir, 'images', 'train'),
        os.path.join(train_dir, 'images', 'val'),
        os.path.join(train_dir, 'labels', 'train'),
        os.path.join(train_dir, 'labels', 'val')
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Tìm cặp ảnh-annotation hợp lệ
    matched_pairs = []
    for img_file in image_files:
        img_name = Path(img_file).stem
        label_file = os.path.join(labels_dir, f"{img_name}.txt")
        if os.path.exists(label_file):
            matched_pairs.append((img_file, label_file))
    
    print(f"✅ Cặp hợp lệ: {len(matched_pairs)}")
    
    # Chia dataset: 80% train, 20% val (đơn giản hóa)
    train_pairs, val_pairs = train_test_split(matched_pairs, test_size=0.2, random_state=42)
    
    print(f"📊 Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    # Copy files (chỉ lấy subset để train nhanh)
    max_train = 1000  # Giới hạn để train nhanh
    max_val = 200
    
    train_pairs = train_pairs[:max_train]
    val_pairs = val_pairs[:max_val]
    
    print(f"🚀 Sử dụng subset để train nhanh: {len(train_pairs)} train, {len(val_pairs)} val")
    
    # Copy train files
    for img_file, label_file in train_pairs:
        img_name = os.path.basename(img_file)
        label_name = os.path.basename(label_file)
        
        shutil.copy2(img_file, os.path.join(train_dir, 'images', 'train', img_name))
        shutil.copy2(label_file, os.path.join(train_dir, 'labels', 'train', label_name))
    
    # Copy val files
    for img_file, label_file in val_pairs:
        img_name = os.path.basename(img_file)
        label_name = os.path.basename(label_file)
        
        shutil.copy2(img_file, os.path.join(train_dir, 'images', 'val', img_name))
        shutil.copy2(label_file, os.path.join(train_dir, 'labels', 'val', label_name))
    
    print("✅ Đã copy files xong!")
    
    # Tạo YAML config
    classes = [
        'Đường người đi bộ cắt ngang', 'Đường giao nhau (ngã ba bên phải)',
        'Cấm đi ngược chiều', 'Phải đi vòng sang bên phải',
        'Giao nhau với đường đồng cấp', 'Giao nhau với đường không ưu tiên',
        'Chỗ ngoặt nguy hiểm vòng bên trái', 'Cấm rẽ trái',
        'Bến xe buýt', 'Nơi giao nhau chạy theo vòng xuyến',
        'Cấm dừng và đỗ xe', 'Chỗ quay xe',
        'Biển gộp làn đường theo phương tiện', 'Đi chậm',
        'Cấm xe tải', 'Đường bị thu hẹp về phía phải',
        'Giới hạn chiều cao', 'Cấm quay đầu',
        'Cấm ô tô khách và ô tô tải', 'Cấm rẽ phải và quay đầu',
        'Cấm ô tô', 'Đường bị thu hẹp về phía trái',
        'Gồ giảm tốc phía trước', 'Cấm xe hai và ba bánh',
        'Kiểm tra', 'Chỉ dành cho xe máy',
        'Chướng ngoại vật phía trước', 'Trẻ em',
        'Xe tải và xe công', 'Cấm mô tô và xe máy',
        'Chỉ dành cho xe tải', 'Đường có camera giám sát',
        'Cấm rẽ phải', 'Nhiều chỗ ngoặt nguy hiểm liên tiếp',
        'Cấm xe sơ-mi rơ-moóc', 'Cấm rẽ trái và phải',
        'Cấm đi thẳng và rẽ phải', 'Đường giao nhau (ngã ba bên trái)',
        'Giới hạn tốc độ (50km/h)', 'Giới hạn tốc độ (60km/h)',
        'Giới hạn tốc độ (80km/h)', 'Giới hạn tốc độ (40km/h)',
        'Các xe chỉ được rẽ trái', 'Chiều cao tĩnh không thực tế',
        'Nguy hiểm khác', 'Đường một chiều',
        'Cấm đỗ xe', 'Cấm ô tô quay đầu xe',
        'Giao nhau với đường sắt có rào chắn', 'Cấm rẽ trái và quay đầu xe',
        'Chỗ ngoặt nguy hiểm vòng bên phải', 'Chú ý chướng ngại vật – vòng tránh sang bên phải'
    ]
    
    yaml_config = {
        'path': os.path.abspath(train_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes
    }
    
    yaml_path = os.path.join(train_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"📄 Đã tạo config: {yaml_path}")
    
    # Bắt đầu training với config tối ưu cho tốc độ
    print("🤖 Bắt đầu training...")
    print("⚡ Config tối ưu tốc độ: YOLOv8n, 30 epochs, batch=8, img=416")
    
    try:
        # Load model nhỏ nhất để train nhanh
        model = YOLO('yolov8n.pt')
        
        # Train với config tối ưu tốc độ
        results = model.train(
            data=yaml_path,
            epochs=30,          # Ít epochs
            imgsz=416,          # Kích thước nhỏ
            batch=8,            # Batch nhỏ
            name='vn_quick',
            project='runs/detect',
            save=True,
            patience=10,        # Early stopping
            verbose=True,
            device='cpu'        # Force CPU nếu GPU có vấn đề
        )
        
        # Lưu model vào thư mục models
        best_model_path = model.trainer.best
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        target_path = os.path.join(models_dir, 'vn_traffic_signs.pt')
        
        shutil.copy2(best_model_path, target_path)
        
        print(f"✅ TRAINING HOÀN THÀNH!")
        print(f"📁 Model đã lưu tại: {target_path}")
        print(f"🎉 Giờ có thể sử dụng chức năng phát hiện biển báo VN!")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi training: {e}")
        print("💡 Thử giảm batch size hoặc sử dụng CPU")
        return False

if __name__ == "__main__":
    quick_setup_and_train()