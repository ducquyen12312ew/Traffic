# check_yolo_dataset.py - Kiểm tra dataset với cấu trúc images/ và labels/ riêng biệt
import os
import cv2
import glob
import random
from pathlib import Path

def check_yolo_dataset_structure(dataset_root):
    """Kiểm tra cấu trúc dataset YOLO"""
    print("🔍 KIỂM TRA DATASET YOLO")
    print("="*50)
    
    if not os.path.exists(dataset_root):
        print(f"❌ Thư mục không tồn tại: {dataset_root}")
        return False
    
    # Tìm thư mục images và labels
    images_dir = None
    labels_dir = None
    classes_files = []
    
    # Tìm trong thư mục gốc và thư mục con
    for root, dirs, files in os.walk(dataset_root):
        # Tìm thư mục images
        if 'images' in dirs:
            images_dir = os.path.join(root, 'images')
        
        # Tìm thư mục labels
        if 'labels' in dirs:
            labels_dir = os.path.join(root, 'labels')
        
        # Tìm file classes
        for file in files:
            if 'class' in file.lower() and file.endswith('.txt'):
                classes_files.append(os.path.join(root, file))
    
    print(f"📁 Thư mục images: {images_dir}")
    print(f"📁 Thư mục labels: {labels_dir}")
    print(f"📋 File classes: {len(classes_files)}")
    
    if not images_dir or not labels_dir:
        print("❌ Không tìm thấy thư mục images hoặc labels!")
        return False
    
    # Kiểm tra ảnh
    image_files = []
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    
    print(f"📸 Tìm thấy {len(image_files)} ảnh")
    
    # Kiểm tra annotations
    annotation_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    print(f"📝 Tìm thấy {len(annotation_files)} annotations")
    
    if len(image_files) == 0:
        print("❌ Không có ảnh nào!")
        return False
    
    if len(annotation_files) == 0:
        print("❌ Không có annotation nào!")
        return False
    
    # Kiểm tra cặp ảnh-annotation
    matched_pairs = []
    missing_annotations = []
    missing_images = []
    
    for img_file in image_files:
        img_name = Path(img_file).stem  # Tên file không có extension
        ann_file = os.path.join(labels_dir, f"{img_name}.txt")
        
        if os.path.exists(ann_file):
            matched_pairs.append((img_file, ann_file))
        else:
            missing_annotations.append(img_name)
    
    for ann_file in annotation_files:
        ann_name = Path(ann_file).stem
        found = False
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
            img_file = os.path.join(images_dir, f"{ann_name}{ext}")
            if os.path.exists(img_file):
                found = True
                break
        if not found:
            missing_images.append(ann_name)
    
    print(f"✅ Cặp hợp lệ: {len(matched_pairs)}")
    print(f"⚠️ Thiếu annotation: {len(missing_annotations)}")
    print(f"⚠️ Thiếu ảnh: {len(missing_images)}")
    
    if missing_annotations:
        print(f"📝 Một số ảnh thiếu annotation:")
        for i, name in enumerate(missing_annotations[:5]):
            print(f"   - {name}.txt")
        if len(missing_annotations) > 5:
            print(f"   ... và {len(missing_annotations) - 5} files khác")
    
    if missing_images:
        print(f"📸 Một số annotation thiếu ảnh:")
        for i, name in enumerate(missing_images[:5]):
            print(f"   - {name}.jpg")
        if len(missing_images) > 5:
            print(f"   ... và {len(missing_images) - 5} files khác")
    
    # Đọc file classes
    if classes_files:
        print(f"\n📋 FILE CLASSES:")
        for class_file in classes_files:
            filename = os.path.basename(class_file)
            print(f"📄 {filename}:")
            try:
                with open(class_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                print(f"   Có {len(lines)} classes")
                print(f"   Ví dụ:")
                for i, line in enumerate(lines[:3]):
                    print(f"   {i}: {line.strip()}")
                if len(lines) > 3:
                    print(f"   ... và {len(lines) - 3} classes khác")
                print()
            except Exception as e:
                print(f"   ❌ Lỗi đọc file: {e}")
    
    return len(matched_pairs) > 0, matched_pairs, images_dir, labels_dir, classes_files

def check_annotation_format(matched_pairs, sample_size=5):
    """Kiểm tra format annotation YOLO"""
    print(f"🔍 KIỂM TRA FORMAT ANNOTATION")
    print("="*50)
    
    if not matched_pairs:
        print("❌ Không có cặp ảnh-annotation để kiểm tra!")
        return False
    
    # Lấy mẫu ngẫu nhiên
    sample_pairs = random.sample(matched_pairs, min(sample_size, len(matched_pairs)))
    
    valid_count = 0
    total_objects = 0
    class_ids = set()
    
    for img_file, ann_file in sample_pairs:
        print(f"\n📄 Kiểm tra: {os.path.basename(ann_file)}")
        
        try:
            # Đọc ảnh để lấy kích thước
            img = cv2.imread(img_file)
            if img is None:
                print(f"   ❌ Không thể đọc ảnh: {os.path.basename(img_file)}")
                continue
                
            h, w = img.shape[:2]
            print(f"   📏 Ảnh: {w}x{h}")
            
            # Đọc annotation
            with open(ann_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                print("   ⚠️ File annotation rỗng")
                continue
            
            print(f"   📊 Có {len(lines)} objects")
            
            valid_lines = 0
            for i, line in enumerate(lines):
                parts = line.strip().split()
                
                if len(parts) >= 5:
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Kiểm tra range hợp lệ (0-1 cho YOLO format)
                        if (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                            0 <= width <= 1 and 0 <= height <= 1):
                            
                            # Convert to pixel coordinates để kiểm tra
                            x1 = int((x_center - width/2) * w)
                            y1 = int((y_center - height/2) * h)
                            x2 = int((x_center + width/2) * w)
                            y2 = int((y_center + height/2) * h)
                            
                            print(f"   ✅ Object {i+1}: class={class_id}, bbox=[{x1},{y1},{x2},{y2}]")
                            valid_lines += 1
                            class_ids.add(class_id)
                        else:
                            print(f"   ❌ Object {i+1}: Tọa độ ngoài range [0,1]")
                    except ValueError:
                        print(f"   ❌ Object {i+1}: Lỗi format số")
                else:
                    print(f"   ❌ Object {i+1}: Thiếu thông tin (cần ≥5 giá trị)")
            
            if valid_lines > 0:
                valid_count += 1
                total_objects += valid_lines
                print(f"   ✅ File hợp lệ ({valid_lines}/{len(lines)} objects OK)")
            
        except Exception as e:
            print(f"   ❌ Lỗi xử lý: {e}")
    
    print(f"\n📊 KẾT QUẢ KIỂM TRA FORMAT:")
    print(f"   ✅ Files hợp lệ: {valid_count}/{len(sample_pairs)}")
    print(f"   📦 Tổng objects: {total_objects}")
    print(f"   🏷️ Classes xuất hiện: {sorted(class_ids)}")
    print(f"   📊 Số classes: {len(class_ids)}")
    
    return valid_count > 0

def visualize_samples(matched_pairs, num_samples=3):
    """Hiển thị mẫu ảnh với annotation"""
    print(f"\n👁️ HIỂN THỊ MẪU ẢNH VỚI ANNOTATION")
    print("="*50)
    
    if not matched_pairs:
        print("❌ Không có cặp ảnh-annotation để hiển thị!")
        return False
    
    # Lấy mẫu ngẫu nhiên
    sample_pairs = random.sample(matched_pairs, min(num_samples, len(matched_pairs)))
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for img_file, ann_file in sample_pairs:
        print(f"\n📷 Hiển thị: {os.path.basename(img_file)}")
        
        # Đọc ảnh
        img = cv2.imread(img_file)
        if img is None:
            print(f"   ❌ Không thể đọc ảnh")
            continue
        
        h, w = img.shape[:2]
        print(f"   📏 Kích thước: {w}x{h}")
        
        # Đọc annotation
        try:
            with open(ann_file, 'r') as f:
                lines = f.readlines()
            
            print(f"   📝 Có {len(lines)} objects")
            
            # Vẽ bounding boxes
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert YOLO format to pixel coordinates
                        x1 = int((x_center - width/2) * w)
                        y1 = int((y_center - height/2) * h)
                        x2 = int((x_center + width/2) * w)
                        y2 = int((y_center + height/2) * h)
                        
                        # Vẽ box
                        color = colors[i % len(colors)]
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        
                        # Vẽ label
                        cv2.putText(img, f'C{class_id}', (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        print(f"     Object {i+1}: Class {class_id} at [{x1},{y1},{x2},{y2}]")
                        
                    except ValueError as e:
                        print(f"     ❌ Lỗi object {i+1}: {e}")
            
            # Resize nếu ảnh quá lớn
            if max(h, w) > 1000:
                scale = 1000 / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h))
            
            cv2.imshow(f'Sample: {os.path.basename(img_file)}', img)
            print("   👁️ Nhấn: SPACE=tiếp tục, ESC=dừng, Q=thoát")
            
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == 32:  # SPACE
                    break
                elif key == 27:  # ESC
                    cv2.destroyAllWindows()
                    return True
                elif key == ord('q') or key == ord('Q'):
                    cv2.destroyAllWindows() 
                    return True
                elif key != 255:  # Phím bất kỳ khác
                    break
            
            cv2.destroyAllWindows()  # Đóng cửa sổ trước khi chuyển ảnh tiếp theo
            
        except Exception as e:
            print(f"   ❌ Lỗi xử lý annotation: {e}")
    
    # Đóng tất cả cửa sổ cuối cùng
    cv2.destroyAllWindows()
    return True

def main():
    """Main function"""
    print("🔍 KIỂM TRA YOLO DATASET")
    print("="*60)
    
    # Nhập đường dẫn
    dataset_path = input("📁 Nhập đường dẫn dataset root (Enter = vietnamese_traffic_signs): ").strip()
    
    if not dataset_path:
        dataset_path = "vietnamese_traffic_signs"
    
    # Bước 1: Kiểm tra cấu trúc
    print("\n🔍 BƯỚC 1: KIỂM TRA CẤU TRÚC DATASET")
    result = check_yolo_dataset_structure(dataset_path)
    
    if not result or not result[0]:
        print("❌ Dataset có vấn đề cấu trúc!")
        return
    
    is_valid, matched_pairs, images_dir, labels_dir, classes_files = result
    
    # Bước 2: Kiểm tra format annotation
    print("\n🔍 BƯỚC 2: KIỂM TRA FORMAT ANNOTATION")
    format_ok = check_annotation_format(matched_pairs)
    
    if not format_ok:
        print("❌ Dataset có vấn đề format annotation!")
        return
    
    # Bước 3: Hiển thị mẫu
    print("\n🔍 BƯỚC 3: HIỂN THỊ MẪU ẢNH")
    show_samples = input("Có muốn hiển thị mẫu ảnh không? (y/n): ").strip().lower()
    
    if show_samples in ['y', 'yes']:
        visualize_samples(matched_pairs)
    
    # Kết luận
    print(f"\n✅ DATASET SẴN SÀNG TRAINING!")
    print("="*50)
    print("📋 Tóm tắt:")
    print(f"   ✅ {len(matched_pairs)} cặp ảnh-annotation hợp lệ")
    print(f"   ✅ Format annotation đúng YOLO")
    print(f"   ✅ Có {len(classes_files)} file classes")
    print(f"   📁 Images: {images_dir}")
    print(f"   📁 Labels: {labels_dir}")
    
    print(f"\n🚀 BƯỚC TIẾP THEO:")
    print("1. Tiến hành training với dataset này")
    print("2. Cấu trúc đã chuẩn YOLO format")
    
    # Ghi thông tin để sử dụng cho training
    info_file = "dataset_info.txt"
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"DATASET_ROOT={dataset_path}\n")
        f.write(f"IMAGES_DIR={images_dir}\n")
        f.write(f"LABELS_DIR={labels_dir}\n")
        f.write(f"CLASSES_FILES={';'.join(classes_files)}\n")
        f.write(f"TOTAL_PAIRS={len(matched_pairs)}\n")
    
    print(f"💾 Đã lưu thông tin dataset vào: {info_file}")

if __name__ == "__main__":
    main()