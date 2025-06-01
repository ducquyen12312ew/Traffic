# explore_dataset.py - Khám phá cấu trúc dataset thực tế
import os
import glob
from pathlib import Path

def explore_directory_structure(root_path, max_depth=3):
    """Khám phá cấu trúc thư mục"""
    print(f"📁 KHÁM PHÁ CẤU TRÚC: {root_path}")
    print("="*60)
    
    if not os.path.exists(root_path):
        print(f"❌ Thư mục không tồn tại: {root_path}")
        return
    
    def explore_recursive(path, current_depth=0, prefix=""):
        if current_depth > max_depth:
            return
        
        try:
            items = sorted(os.listdir(path))
        except PermissionError:
            print(f"{prefix}❌ Không có quyền truy cập")
            return
        
        # Phân loại files và folders
        folders = [item for item in items if os.path.isdir(os.path.join(path, item))]
        files = [item for item in items if os.path.isfile(os.path.join(path, item))]
        
        # Hiển thị folders
        for i, folder in enumerate(folders):
            is_last_folder = (i == len(folders) - 1) and len(files) == 0
            folder_prefix = "└── " if is_last_folder else "├── "
            print(f"{prefix}{folder_prefix}📁 {folder}/")
            
            # Recursive cho subfolder
            if current_depth < max_depth:
                next_prefix = prefix + ("    " if is_last_folder else "│   ")
                folder_path = os.path.join(path, folder)
                explore_recursive(folder_path, current_depth + 1, next_prefix)
        
        # Hiển thị files (chỉ một số đầu)
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        txt_files = [f for f in files if f.lower().endswith('.txt')]
        other_files = [f for f in files if f not in image_files and f not in txt_files]
        
        if image_files:
            print(f"{prefix}├── 📸 Images ({len(image_files)} files):")
            for i, file in enumerate(image_files[:3]):
                symbol = "└──" if i == len(image_files[:3]) - 1 and not txt_files and not other_files else "├──"
                print(f"{prefix}│   {symbol} {file}")
            if len(image_files) > 3:
                print(f"{prefix}│   └── ... và {len(image_files) - 3} files khác")
        
        if txt_files:
            print(f"{prefix}├── 📝 Text files ({len(txt_files)} files):")
            for i, file in enumerate(txt_files[:3]):
                symbol = "└──" if i == len(txt_files[:3]) - 1 and not other_files else "├──"
                print(f"{prefix}│   {symbol} {file}")
            if len(txt_files) > 3:
                print(f"{prefix}│   └── ... và {len(txt_files) - 3} files khác")
        
        if other_files:
            print(f"{prefix}└── 📄 Other files ({len(other_files)} files):")
            for i, file in enumerate(other_files[:3]):
                symbol = "└──" if i == len(other_files[:3]) - 1 else "├──"
                print(f"{prefix}    {symbol} {file}")
            if len(other_files) > 3:
                print(f"{prefix}    └── ... và {len(other_files) - 3} files khác")
    
    explore_recursive(root_path)

def find_all_images_and_annotations(root_path):
    """Tìm tất cả ảnh và annotation trong toàn bộ cây thư mục"""
    print(f"\n🔍 TÌM TẤT CẢ ẢNH VÀ ANNOTATION TRONG: {root_path}")
    print("="*60)
    
    all_images = []
    all_annotations = []
    all_class_files = []
    
    # Tìm tất cả files trong toàn bộ cây thư mục
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Phân loại files
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                all_images.append(file_path)
            elif file.lower().endswith('.txt'):
                if 'class' in file.lower():
                    all_class_files.append(file_path)
                else:
                    all_annotations.append(file_path)
    
    print(f"📸 Tổng số ảnh: {len(all_images)}")
    print(f"📝 Tổng số annotation: {len(all_annotations)}")
    print(f"📋 Tổng số file classes: {len(all_class_files)}")
    
    # Hiển thị distribution theo thư mục
    if all_images:
        print(f"\n📊 PHÂN BỐ ẢNH THEO THỦ MỤC:")
        image_dirs = {}
        for img in all_images:
            dir_name = os.path.dirname(img)
            relative_dir = os.path.relpath(dir_name, root_path)
            image_dirs[relative_dir] = image_dirs.get(relative_dir, 0) + 1
        
        for dir_name, count in sorted(image_dirs.items()):
            print(f"   {dir_name}: {count} ảnh")
    
    if all_annotations:
        print(f"\n📊 PHÂN BỐ ANNOTATION THEO THỦ MỤC:")
        ann_dirs = {}
        for ann in all_annotations:
            dir_name = os.path.dirname(ann)
            relative_dir = os.path.relpath(dir_name, root_path)
            ann_dirs[relative_dir] = ann_dirs.get(relative_dir, 0) + 1
        
        for dir_name, count in sorted(ann_dirs.items()):
            print(f"   {dir_name}: {count} annotations")
    
    if all_class_files:
        print(f"\n📋 FILE CLASSES:")
        for class_file in all_class_files:
            relative_path = os.path.relpath(class_file, root_path)
            print(f"   {relative_path}")
    
    return all_images, all_annotations, all_class_files

def suggest_main_directories(all_images, all_annotations, root_path):
    """Đề xuất thư mục chính chứa data"""
    print(f"\n💡 ĐỀ XUẤT THỦ MỤC CHO TRAINING:")
    print("="*40)
    
    if not all_images:
        print("❌ Không tìm thấy ảnh nào!")
        return []
    
    # Tìm thư mục có nhiều ảnh nhất
    image_dirs = {}
    for img in all_images:
        dir_name = os.path.dirname(img)
        image_dirs[dir_name] = image_dirs.get(dir_name, 0) + 1
    
    # Sắp xếp theo số lượng
    sorted_dirs = sorted(image_dirs.items(), key=lambda x: x[1], reverse=True)
    
    suggestions = []
    
    for i, (dir_path, img_count) in enumerate(sorted_dirs[:3]):
        relative_dir = os.path.relpath(dir_path, root_path)
        
        # Kiểm tra có annotation không
        ann_count = 0
        for ann in all_annotations:
            if os.path.dirname(ann) == dir_path:
                ann_count += 1
        
        # Kiểm tra tỷ lệ ảnh/annotation
        ratio = ann_count / img_count if img_count > 0 else 0
        
        print(f"🎯 TÙY CHỌN {i+1}: {relative_dir}")
        print(f"   📸 Ảnh: {img_count}")
        print(f"   📝 Annotations: {ann_count}")
        print(f"   📊 Tỷ lệ: {ratio:.2%}")
        
        if ratio > 0.8:  # >= 80% có annotation
            print(f"   ✅ KHUYẾN NGHỊ (tỷ lệ annotation cao)")
            suggestions.append(dir_path)
        elif ratio > 0.5:  # >= 50% có annotation
            print(f"   🟡 CÓ THỂ DÙNG (tỷ lệ annotation trung bình)")
            suggestions.append(dir_path)
        else:
            print(f"   ❌ KHÔNG KHUYẾN NGHỊ (ít annotation)")
        
        print()
    
    return suggestions

def main():
    """Main function"""
    print("🔍 KHÁM PHÁ CẤU TRÚC DATASET")
    print("="*50)
    
    # Nhập đường dẫn
    dataset_path = input("📁 Nhập đường dẫn thư mục dataset: ").strip()
    
    if not dataset_path:
        dataset_path = "vietnamese_traffic_signs"  # Default
        print(f"🔄 Sử dụng default: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Thư mục không tồn tại: {dataset_path}")
        return
    
    # Bước 1: Khám phá cấu trúc
    explore_directory_structure(dataset_path)
    
    # Bước 2: Tìm tất cả files
    all_images, all_annotations, all_class_files = find_all_images_and_annotations(dataset_path)
    
    # Bước 3: Đề xuất thư mục
    suggestions = suggest_main_directories(all_images, all_annotations, dataset_path)
    
    # Kết luận
    print(f"📋 TÓM TẮT:")
    print(f"   📸 Tổng ảnh: {len(all_images)}")
    print(f"   📝 Tổng annotation: {len(all_annotations)}")
    print(f"   📋 File classes: {len(all_class_files)}")
    
    if suggestions:
        print(f"\n🎯 BƯỚC TIẾP THEO:")
        print(f"1. Chạy lại: python check_dataset.py")
        print(f"2. Nhập đường dẫn: {suggestions[0]}")
        print(f"   (Thư mục được khuyến nghị nhất)")
    else:
        print(f"\n⚠️ DATASET CẦN ĐIỀU CHỈNH:")
        print("- Kiểm tra format annotation")
        print("- Đảm bảo có đủ cặp ảnh-annotation")

if __name__ == "__main__":
    main()