# fix_class_mapping.py - Sửa mapping classes từ dataset thực tế
import os
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO
import glob

def read_classes_from_dataset():
    """Đọc classes từ file dataset gốc"""
    print("🔍 ĐỌC CLASSES TỪ DATASET GỐC")
    print("="*50)
    
    # Tìm file classes trong dataset
    dataset_root = "vietnamese_traffic_signs"
    classes_files = []
    
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if 'class' in file.lower() and file.endswith('.txt'):
                classes_files.append(os.path.join(root, file))
    
    if not classes_files:
        print("❌ Không tìm thấy file classes!")
        return None
    
    print(f"📋 Tìm thấy {len(classes_files)} file classes:")
    for cf in classes_files:
        print(f"   - {os.path.relpath(cf)}")
    
    # Đọc file classes đầu tiên
    classes_file = classes_files[0]
    print(f"\n📖 Đọc từ: {classes_file}")
    
    try:
        with open(classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"✅ Đọc được {len(classes)} classes:")
        for i, cls in enumerate(classes[:10]):
            print(f"   {i}: {cls}")
        if len(classes) > 10:
            print(f"   ... và {len(classes) - 10} classes khác")
        
        return classes
        
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return None

def create_mapping_without_accents(original_classes):
    """Tạo mapping từ classes có dấu sang không dấu"""
    print("\n🔤 TẠO MAPPING KHÔNG DẤU")
    print("="*50)
    
    # Bảng chuyển đổi dấu tiếng Việt
    accent_map = {
        'á': 'a', 'à': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
        'ă': 'a', 'ắ': 'a', 'ằ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
        'â': 'a', 'ấ': 'a', 'ầ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
        'é': 'e', 'è': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
        'ê': 'e', 'ế': 'e', 'ề': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
        'í': 'i', 'ì': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
        'ó': 'o', 'ò': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
        'ô': 'o', 'ố': 'o', 'ồ': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
        'ơ': 'o', 'ớ': 'o', 'ờ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
        'ú': 'u', 'ù': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
        'ư': 'u', 'ứ': 'u', 'ừ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
        'ý': 'y', 'ỳ': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
        'đ': 'd',
        'Á': 'A', 'À': 'A', 'Ả': 'A', 'Ã': 'A', 'Ạ': 'A',
        'Ă': 'A', 'Ắ': 'A', 'Ằ': 'A', 'Ẳ': 'A', 'Ẵ': 'A', 'Ặ': 'A',
        'Â': 'A', 'Ấ': 'A', 'Ầ': 'A', 'Ẩ': 'A', 'Ẫ': 'A', 'Ậ': 'A',
        'É': 'E', 'È': 'E', 'Ẻ': 'E', 'Ẽ': 'E', 'Ẹ': 'E',
        'Ê': 'E', 'Ế': 'E', 'Ề': 'E', 'Ể': 'E', 'Ễ': 'E', 'Ệ': 'E',
        'Í': 'I', 'Ì': 'I', 'Ỉ': 'I', 'Ĩ': 'I', 'Ị': 'I',
        'Ó': 'O', 'Ò': 'O', 'Ỏ': 'O', 'Õ': 'O', 'Ọ': 'O',
        'Ô': 'O', 'Ố': 'O', 'Ồ': 'O', 'Ổ': 'O', 'Ỗ': 'O', 'Ộ': 'O',
        'Ơ': 'O', 'Ớ': 'O', 'Ờ': 'O', 'Ở': 'O', 'Ỡ': 'O', 'Ợ': 'O',
        'Ú': 'U', 'Ù': 'U', 'Ủ': 'U', 'Ũ': 'U', 'Ụ': 'U',
        'Ư': 'U', 'Ứ': 'U', 'Ừ': 'U', 'Ử': 'U', 'Ữ': 'U', 'Ự': 'U',
        'Ý': 'Y', 'Ỳ': 'Y', 'Ỷ': 'Y', 'Ỹ': 'Y', 'Ỵ': 'Y',
        'Đ': 'D'
    }
    
    def remove_accents(text):
        """Bỏ dấu tiếng Việt"""
        for accented, unaccented in accent_map.items():
            text = text.replace(accented, unaccented)
        return text
    
    def clean_class_name(name):
        """Làm sạch tên class"""
        # Bỏ dấu
        name = remove_accents(name)
        # Thay thế ký tự đặc biệt
        name = name.replace(' ', '_')
        name = name.replace('-', '_')
        name = name.replace('/', '_')
        name = name.replace('(', '')
        name = name.replace(')', '')
        name = name.replace('[', '')
        name = name.replace(']', '')
        # Chuyển thành chữ thường
        name = name.lower()
        # Xóa ký tự đặc biệt
        import re
        name = re.sub(r'[^a-z0-9_]', '', name)
        # Xóa underscore liên tiếp
        name = re.sub(r'_+', '_', name)
        name = name.strip('_')
        return name
    
    # Tạo mapping
    original_to_clean = {}
    clean_classes = []
    
    for i, original in enumerate(original_classes):
        clean = clean_class_name(original)
        original_to_clean[i] = {
            'original': original,
            'clean': clean,
            'index': i
        }
        clean_classes.append(clean)
    
    print("📋 MAPPING CLASSES:")
    for i, mapping in enumerate(list(original_to_clean.values())[:10]):
        print(f"   {i}: '{mapping['original']}' -> '{mapping['clean']}'")
    if len(original_to_clean) > 10:
        print(f"   ... và {len(original_to_clean) - 10} mappings khác")
    
    return original_to_clean, clean_classes

def update_main_py_classes(clean_classes):
    """Cập nhật classes trong main.py"""
    print("\n🔧 CẬP NHẬT MAIN.PY")
    print("="*50)
    
    # Đọc file main.py hiện tại
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Tạo dict classes mới
    new_classes_dict = "{\n"
    for i, cls in enumerate(clean_classes):
        new_classes_dict += f"            {i}: '{cls}'"
        if i < len(clean_classes) - 1:
            new_classes_dict += ","
        new_classes_dict += "\n"
    new_classes_dict += "        }"
    
    # Tìm và thay thế phần self.vn_classes
    import re
    pattern = r'self\.vn_classes\s*=\s*\{[^}]+\}'
    replacement = f'self.vn_classes = {new_classes_dict}'
    
    updated_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Backup file cũ
    shutil.copy2('main.py', 'main.py.backup')
    print("📦 Đã backup main.py -> main.py.backup")
    
    # Ghi file mới
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("✅ Đã cập nhật main.py với classes chính xác!")

def retrain_with_correct_classes(clean_classes):
    """Retrain model với classes chính xác"""
    print("\n🤖 RETRAIN VỚI CLASSES CHÍNH XÁC")
    print("="*50)
    
    # Kiểm tra training data
    train_dir = "vn_training_quick"
    if not os.path.exists(train_dir):
        print("❌ Không tìm thấy training data!")
        print("💡 Chạy trước: python quick_train_vn.py")
        return False
    
    # Tạo YAML với classes chính xác
    yaml_config = {
        'path': os.path.abspath(train_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(clean_classes),
        'names': clean_classes  # Classes đã được làm sạch
    }
    
    yaml_path = os.path.join(train_dir, 'data_corrected.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"📄 Đã tạo YAML chính xác: {yaml_path}")
    
    try:
        # Load model cũ hoặc tạo mới
        old_model_path = os.path.join("models", "vn_traffic_signs.pt")
        if os.path.exists(old_model_path):
            print("🔄 Sử dụng model cũ làm base")
            model = YOLO(old_model_path)
        else:
            print("🆕 Tạo model mới")
            model = YOLO('yolov8n.pt')
        
        # Train với config tối ưu
        print("⚡ Training với 25 epochs, batch=6, img=384...")
        results = model.train(
            data=yaml_path,
            epochs=25,
            imgsz=384,
            batch=6,
            name='vn_corrected',
            project='runs/detect',
            save=True,
            patience=8,
            verbose=True
        )
        
        # Lưu model mới
        best_model_path = model.trainer.best
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        target_path = os.path.join(models_dir, 'vn_traffic_signs.pt')
        
        # Backup model cũ
        if os.path.exists(target_path):
            backup_path = os.path.join(models_dir, 'vn_traffic_signs_old.pt')
            shutil.copy2(target_path, backup_path)
            print(f"📦 Backup model cũ: {backup_path}")
        
        shutil.copy2(best_model_path, target_path)
        
        print(f"✅ MODEL ĐÃ ĐƯỢC SỬA!")
        print(f"📁 Model mới: {target_path}")
        print(f"🎉 Giờ classes sẽ hiển thị CHÍNH XÁC!")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi retrain: {e}")
        return False

def create_class_reference(original_to_clean):
    """Tạo file tham chiếu classes"""
    print("\n📋 TẠO FILE THAM CHIẾU")
    print("="*30)
    
    ref_content = "# THAM CHIẾU CLASSES BIỂN BÁO VIỆT NAM\n"
    ref_content += "# Mapping từ classes gốc (có dấu) sang classes model (không dấu)\n\n"
    
    for i, mapping in original_to_clean.items():
        ref_content += f"Class {i}:\n"
        ref_content += f"  Tên gốc: {mapping['original']}\n"
        ref_content += f"  Tên model: {mapping['clean']}\n\n"
    
    with open('classes_reference.txt', 'w', encoding='utf-8') as f:
        f.write(ref_content)
    
    print("✅ Đã tạo classes_reference.txt")

def main():
    """Main function"""
    print("🔧 SỬA MAPPING CLASSES BIỂN BÁO VN")
    print("="*60)
    
    # Bước 1: Đọc classes từ dataset gốc
    original_classes = read_classes_from_dataset()
    if not original_classes:
        print("❌ Không thể đọc classes từ dataset!")
        return
    
    # Bước 2: Tạo mapping không dấu
    original_to_clean, clean_classes = create_mapping_without_accents(original_classes)
    
    # Bước 3: Cập nhật main.py
    update_main_py_classes(clean_classes)
    
    # Bước 4: Tạo file tham chiếu
    create_class_reference(original_to_clean)
    
    # Bước 5: Retrain model
    print(f"\n🤖 RETRAIN MODEL?")
    choice = input("Có muốn retrain model với classes chính xác? (y/n): ").strip().lower()
    
    if choice in ['y', 'yes']:
        success = retrain_with_correct_classes(clean_classes)
        if success:
            print(f"\n🎉 HOÀN THÀNH!")
            print("✅ Classes đã được sửa chính xác")
            print("✅ Model đã được retrain")
            print("✅ main.py đã được cập nhật")
            print(f"\n📋 Kiểm tra file: classes_reference.txt")
            print(f"🚀 Chạy lại: python main.py")
        else:
            print(f"\n⚠️ Sửa classes thành công nhưng retrain thất bại")
            print("💡 Có thể chạy lại retrain sau")
    else:
        print(f"\n✅ ĐÃ SỬA CLASSES!")
        print("📋 main.py đã được cập nhật với classes chính xác")
        print("💡 Cần retrain model để có kết quả tốt nhất")

if __name__ == "__main__":
    main()