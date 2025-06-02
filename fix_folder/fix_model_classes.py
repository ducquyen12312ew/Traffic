# fix_model_classes.py - Fix model với classes không dấu
import os
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO

def fix_classes_and_retrain():
    """Fix classes không dấu và retrain nhanh"""
    print("🔧 FIX MODEL VỚI CLASSES KHÔNG DẤU")
    print("="*50)
    
    # Classes không dấu
    classes_no_accent = [
        'duong_nguoi_di_bo_cat_ngang', 'duong_giao_nhau_nga_ba_ben_phai',
        'cam_di_nguoc_chieu', 'phai_di_vong_sang_ben_phai',
        'giao_nhau_voi_duong_dong_cap', 'giao_nhau_voi_duong_khong_uu_tien',
        'cho_ngoat_nguy_hiem_vong_ben_trai', 'cam_re_trai',
        'ben_xe_buyt', 'noi_giao_nhau_chay_theo_vong_xuyen',
        'cam_dung_va_do_xe', 'cho_quay_xe',
        'bien_ghop_lan_duong_theo_phuong_tien', 'di_cham',
        'cam_xe_tai', 'duong_bi_thu_hep_ve_phia_phai',
        'gioi_han_chieu_cao', 'cam_quay_dau',
        'cam_oto_khach_va_oto_tai', 'cam_re_phai_va_quay_dau',
        'cam_oto', 'duong_bi_thu_hep_ve_phia_trai',
        'go_giam_toc_phia_truoc', 'cam_xe_hai_va_ba_banh',
        'kiem_tra', 'chi_danh_cho_xe_may',
        'chuong_ngoai_vat_phia_truoc', 'tre_em',
        'xe_tai_va_xe_cong', 'cam_moto_va_xe_may',
        'chi_danh_cho_xe_tai', 'duong_co_camera_giam_sat',
        'cam_re_phai', 'nhieu_cho_ngoat_nguy_hiem_lien_tiep',
        'cam_xe_somi_romooc', 'cam_re_trai_va_phai',
        'cam_di_thang_va_re_phai', 'duong_giao_nhau_nga_ba_ben_trai',
        'gioi_han_toc_do_50kmh', 'gioi_han_toc_do_60kmh',
        'gioi_han_toc_do_80kmh', 'gioi_han_toc_do_40kmh',
        'cac_xe_chi_duoc_re_trai', 'chieu_cao_tinh_khong_thuc_te',
        'nguy_hiem_khac', 'duong_mot_chieu',
        'cam_do_xe', 'cam_oto_quay_dau_xe',
        'giao_nhau_voi_duong_sat_co_rao_chan', 'cam_re_trai_va_quay_dau_xe',
        'cho_ngoat_nguy_hiem_vong_ben_phai', 'chu_y_chuong_ngai_vat_vong_tranh_sang_ben_phai'
    ]
    
    # Kiểm tra training data cũ
    train_dir = "vn_training_quick"
    if not os.path.exists(train_dir):
        print("❌ Không tìm thấy training data cũ!")
        print("💡 Chạy lại: python quick_train_vn.py")
        return False
    
    # Tạo YAML mới với classes không dấu
    yaml_config = {
        'path': os.path.abspath(train_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes_no_accent),
        'names': classes_no_accent  # Classes mới không dấu
    }
    
    yaml_path = os.path.join(train_dir, 'data_fixed.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"📄 Đã tạo YAML mới: {yaml_path}")
    
    # Retrain nhanh với classes mới
    print("🤖 Retrain với classes không dấu...")
    print("⚡ Config: 20 epochs, batch=4, img=320 (siêu nhanh)")
    
    try:
        # Load model cũ hoặc tạo mới
        old_model_path = os.path.join("models", "vn_traffic_signs.pt")
        if os.path.exists(old_model_path):
            print("🔄 Sử dụng model cũ làm base")
            model = YOLO(old_model_path)
        else:
            print("🆕 Tạo model mới")
            model = YOLO('yolov8n.pt')
        
        # Train với config siêu nhanh
        results = model.train(
            data=yaml_path,
            epochs=20,          # Rất ít epochs
            imgsz=320,          # Kích thước rất nhỏ
            batch=4,            # Batch rất nhỏ
            name='vn_fixed',
            project='runs/detect',
            save=True,
            patience=5,         # Early stopping nhanh
            verbose=True
        )
        
        # Lưu model mới
        best_model_path = model.trainer.best
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        target_path = os.path.join(models_dir, 'vn_traffic_signs.pt')
        
        # Backup model cũ
        if os.path.exists(target_path):
            backup_path = os.path.join(models_dir, 'vn_traffic_signs_backup.pt')
            shutil.copy2(target_path, backup_path)
            print(f"📦 Backup model cũ: {backup_path}")
        
        shutil.copy2(best_model_path, target_path)
        
        print(f"✅ MODEL ĐÃ ĐƯỢC FIX!")
        print(f"📁 Model mới: {target_path}")
        print(f"🎉 Giờ sẽ hiển thị classes không dấu!")
        
        # Test model mới
        test_model(target_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi retrain: {e}")
        return False

def test_model(model_path):
    """Test model mới"""
    print(f"\n🧪 TEST MODEL MỚI")
    print("="*30)
    
    try:
        model = YOLO(model_path)
        print("✅ Model load thành công!")
        
        # Hiển thị classes
        if hasattr(model, 'names'):
            print(f"📊 Có {len(model.names)} classes:")
            for i, name in enumerate(list(model.names.values())[:5]):
                print(f"   {i}: {name}")
            if len(model.names) > 5:
                print(f"   ... và {len(model.names) - 5} classes khác")
        
        print("\n🎯 Model sẵn sàng sử dụng!")
        
    except Exception as e:
        print(f"❌ Lỗi test model: {e}")

if __name__ == "__main__":
    fix_classes_and_retrain()