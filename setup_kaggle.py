# setup_kaggle.py - Setup Kaggle API từng bước
import os
import json
from pathlib import Path

def check_kaggle_setup():
    """Kiểm tra Kaggle đã setup chưa"""
    print("🔍 KIỂM TRA KAGGLE SETUP")
    print("="*40)
    
    # Kiểm tra file kaggle.json
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_file = kaggle_dir / 'kaggle.json'
    
    print(f"📁 Thư mục Kaggle: {kaggle_dir}")
    print(f"📄 File config: {kaggle_file}")
    
    if kaggle_file.exists():
        print("✅ File kaggle.json đã tồn tại")
        
        # Kiểm tra nội dung
        try:
            with open(kaggle_file, 'r') as f:
                config = json.load(f)
            
            if 'username' in config and 'key' in config:
                print(f"✅ Username: {config['username']}")
                print("✅ API Key: ***có***")
                return True
            else:
                print("❌ File kaggle.json thiếu username hoặc key")
                return False
                
        except Exception as e:
            print(f"❌ Lỗi đọc file kaggle.json: {e}")
            return False
    else:
        print("❌ File kaggle.json không tồn tại")
        return False

def create_kaggle_config():
    """Tạo file kaggle.json"""
    print("\n🔧 TẠO FILE KAGGLE CONFIG")
    print("="*40)
    
    print("📋 Để lấy API credentials:")
    print("1. Vào https://www.kaggle.com/account")
    print("2. Scroll xuống phần 'API'")
    print("3. Click 'Create New API Token'")
    print("4. File kaggle.json sẽ được tải về")
    
    print(f"\n📝 Nhập thông tin API:")
    username = input("Kaggle Username: ").strip()
    key = input("Kaggle Key: ").strip()
    
    if not username or not key:
        print("❌ Vui lòng nhập đầy đủ thông tin!")
        return False
    
    # Tạo thư mục .kaggle
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    # Tạo file config
    config = {
        "username": username,
        "key": key
    }
    
    kaggle_file = kaggle_dir / 'kaggle.json'
    
    try:
        with open(kaggle_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Set permissions (chỉ user được đọc)
        os.chmod(kaggle_file, 0o600)
        
        print(f"✅ Đã tạo file: {kaggle_file}")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi tạo file: {e}")
        return False

def test_kaggle_connection():
    """Test kết nối Kaggle"""
    print("\n🧪 TEST KẾT NỐI KAGGLE")
    print("="*40)
    
    try:
        import kaggle
        
        # Test bằng cách list competitions
        print("📡 Đang test kết nối...")
        competitions = kaggle.api.competitions_list()
        
        print("✅ Kết nối thành công!")
        print(f"📊 Tìm thấy {len(competitions)} competitions")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi kết nối: {e}")
        return False

def download_vn_dataset():
    """Tải dataset biển báo VN"""
    print("\n📥 TẢI DATASET BIỂN BÁO VIỆT NAM")
    print("="*40)
    
    try:
        import kaggle
        
        # Tạo thư mục đích
        dataset_dir = "vietnamese_traffic_signs"
        if os.path.exists(dataset_dir):
            import shutil
            shutil.rmtree(dataset_dir)
        
        print("📥 Đang tải dataset...")
        kaggle.api.dataset_download_files(
            'maitam/vietnamese-traffic-signs',
            path=dataset_dir,
            unzip=True
        )
        
        print(f"✅ Đã tải thành công!")
        print(f"📁 Dataset tại: {os.path.abspath(dataset_dir)}")
        
        # Kiểm tra nội dung
        if os.path.exists(dataset_dir):
            files = os.listdir(dataset_dir)
            print(f"📋 Có {len(files)} files/folders:")
            for i, file in enumerate(files[:10]):  # Hiển thị 10 file đầu
                print(f"   - {file}")
            if len(files) > 10:
                print(f"   ... và {len(files) - 10} files khác")
        
        return dataset_dir
        
    except Exception as e:
        print(f"❌ Lỗi tải dataset: {e}")
        return None

def find_annotation_files(dataset_dir):
    """Tìm file annotation trong dataset"""
    print(f"\n🔍 TÌM FILE ANNOTATION TRONG {dataset_dir}")
    print("="*50)
    
    if not os.path.exists(dataset_dir):
        print("❌ Thư mục không tồn tại!")
        return None
    
    # Tìm tất cả file .txt
    txt_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    
    print(f"📝 Tìm thấy {len(txt_files)} file .txt")
    
    # Tìm file classes
    class_files = []
    annotation_files = []
    
    for txt_file in txt_files:
        filename = os.path.basename(txt_file).lower()
        if 'class' in filename:
            class_files.append(txt_file)
        else:
            annotation_files.append(txt_file)
    
    print(f"📋 File classes: {len(class_files)}")
    for cf in class_files:
        print(f"   - {os.path.relpath(cf)}")
    
    print(f"📝 File annotations: {len(annotation_files)}")
    if annotation_files:
        print(f"   Ví dụ: {os.path.relpath(annotation_files[0])}")
    
    # Tìm ảnh
    image_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    print(f"📸 Tìm thấy {len(image_files)} ảnh")
    
    # Đề xuất thư mục chính
    if annotation_files and image_files:
        # Tìm thư mục chung của annotations và images
        ann_dir = os.path.dirname(annotation_files[0])
        img_dir = os.path.dirname(image_files[0])
        
        if ann_dir == img_dir:
            main_dir = ann_dir
        else:
            main_dir = dataset_dir
        
        print(f"\n✅ ĐỀ XUẤT SỬ DỤNG THỦ MỤC: {os.path.abspath(main_dir)}")
        return main_dir
    
    return dataset_dir

def main():
    """Main function"""
    print("🚀 SETUP KAGGLE VÀ TẢI DATASET")
    print("="*50)
    
    while True:
        print(f"\n📋 MENU:")
        print("1. Kiểm tra Kaggle setup")
        print("2. Tạo file kaggle.json")
        print("3. Test kết nối Kaggle")
        print("4. Tải dataset biển báo VN")
        print("5. Chạy full setup")
        print("0. Thoát")
        
        choice = input("\n👉 Chọn (0-5): ").strip()
        
        if choice == '1':
            check_kaggle_setup()
            
        elif choice == '2':
            create_kaggle_config()
            
        elif choice == '3':
            test_kaggle_connection()
            
        elif choice == '4':
            dataset_dir = download_vn_dataset()
            if dataset_dir:
                main_dir = find_annotation_files(dataset_dir)
                print(f"\n🎯 BƯỚC TIẾP THEO:")
                print("1. Chạy: python check_dataset.py")
                print(f"2. Nhập đường dẫn: {main_dir}")
                
        elif choice == '5':
            # Full setup
            print("🚀 CHẠY FULL SETUP...")
            
            if not check_kaggle_setup():
                print("\n🔧 Cần setup Kaggle API...")
                if not create_kaggle_config():
                    print("❌ Setup thất bại!")
                    continue
            
            if test_kaggle_connection():
                dataset_dir = download_vn_dataset()
                if dataset_dir:
                    main_dir = find_annotation_files(dataset_dir)
                    print(f"\n🎉 SETUP HOÀN TẤT!")
                    print(f"📁 Dataset ready tại: {main_dir}")
                    print(f"\n🚀 BƯỚC TIẾP THEO:")
                    print("1. Chạy: python check_dataset.py")
                    print(f"2. Nhập đường dẫn: {main_dir}")
                    break
            
        elif choice == '0':
            print("👋 Tạm biệt!")
            break
        
        else:
            print("❌ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()