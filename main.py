import cv2
import os
from ultralytics import YOLO
from config import *
from utils import draw_boxes, get_files_in_folder

class TrafficDetector:
    def __init__(self):
        """Khởi tạo detector"""
        print("🚀 Đang khởi tạo Traffic Detector...")
        
        # Load model YOLO
        if not os.path.exists(MODEL_PATH):
            print(f"📥 Đang tải model {MODEL_NAME}...")
        
        self.model = YOLO(MODEL_PATH)
        print("✅ Model đã sẵn sàng!")
    
    def detect_image(self, image_path, save_result=True):
        """Phát hiện object trong ảnh"""
        print(f"🔍 Đang xử lý ảnh: {os.path.basename(image_path)}")
        
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Không thể đọc ảnh: {image_path}")
            return None
        
        print(f"📏 Kích thước ảnh: {image.shape[1]}x{image.shape[0]}")
        
        # Chạy detection với nhiều cấu hình
        print("🔍 Đang chạy detection...")
        
        # Thử với ngưỡng thấp trước
        results_low = self.model(image, conf=0.1, iou=IOU_THRESHOLD)
        results_normal = self.model(image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
        
        # Đếm số object phát hiện được
        total_objects_low = sum(len(r.boxes) if r.boxes is not None else 0 for r in results_low)
        total_objects_normal = sum(len(r.boxes) if r.boxes is not None else 0 for r in results_normal)
        
        print(f"📊 Phát hiện với conf=0.1: {total_objects_low} objects")
        print(f"📊 Phát hiện với conf={CONFIDENCE_THRESHOLD}: {total_objects_normal} objects")
        
        # Sử dụng kết quả tốt hơn
        results = results_low if total_objects_low > total_objects_normal else results_normal
        
        # In chi tiết tất cả object được phát hiện
        print("📋 Chi tiết các object:")
        for result in results:
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    class_name = TRAFFIC_CLASSES.get(class_id, f'class_{class_id}')
                    print(f"   {i+1}. {class_name} - {confidence:.3f} - [{x1},{y1},{x2},{y2}]")
        
        # Phân tích đèn giao thông
        from utils import analyze_traffic_light_stats
        traffic_lights = analyze_traffic_light_stats(image, results)
        
        if traffic_lights:
            print(f"🚦 Phát hiện {len(traffic_lights)} đèn giao thông:")
            for i, light in enumerate(traffic_lights):
                print(f"   - Đèn {i+1}: {light['color'].upper()} (độ tin cậy: {light['confidence']:.2f})")
        else:
            print("🚦 Không phát hiện được đèn giao thông")
            print("💡 Thử các cách khắc phục:")
            print("   - Giảm CONFIDENCE_THRESHOLD trong config.py")
            print("   - Sử dụng model lớn hơn (yolov8m.pt, yolov8l.pt)")
            print("   - Kiểm tra ảnh có đủ rõ nét không")
        
        # Vẽ kết quả
        result_image = draw_boxes(image.copy(), results)
        
        # Lưu kết quả
        if save_result:
            output_path = os.path.join(OUTPUT_DIR, 'images', f'detected_{os.path.basename(image_path)}')
            cv2.imwrite(output_path, result_image)
            print(f"💾 Đã lưu kết quả: {output_path}")
        
        return result_image
    
    def detect_video(self, video_path, save_result=True):
        """Phát hiện object trong video"""
        print(f"🎥 Đang xử lý video: {os.path.basename(video_path)}")
        
        # Mở video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Không thể mở video: {video_path}")
            return
        
        # Lấy thông tin video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Thiết lập video writer
        if save_result:
            output_path = os.path.join(OUTPUT_DIR, 'videos', f'detected_{os.path.basename(video_path)}')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"⏳ Xử lý frame {frame_count}/{total_frames}", end='\r')
            
            # Chạy detection
            results = self.model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
            
            # Vẽ kết quả
            result_frame = draw_boxes(frame, results)
            
            # Lưu frame
            if save_result:
                out.write(result_frame)
            
            # Hiển thị (tùy chọn)
            cv2.imshow('Traffic Detection', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Dọn dẹp
        cap.release()
        if save_result:
            out.release()
            print(f"\n💾 Đã lưu video kết quả: {output_path}")
        
        cv2.destroyAllWindows()
    
    def detect_webcam(self):
        """Phát hiện object từ webcam"""
        print("🎥 Đang mở webcam...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Không thể mở webcam!")
            return
        
        print("✅ Webcam đã sẵn sàng! Nhấn 'q' để thoát.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Chạy detection
            results = self.model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
            
            # Vẽ kết quả
            result_frame = draw_boxes(frame, results)
            
            # Hiển thị
            cv2.imshow('Traffic Detection - Webcam', result_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Hàm main"""
    detector = TrafficDetector()
    
    while True:
        print("\n" + "="*50)
        print("🚦 TRAFFIC OBJECT DETECTION")
        print("="*50)
        print("1. Phát hiện từ ảnh")
        print("2. Phát hiện từ video")
        print("3. Phát hiện từ webcam")
        print("4. Xử lý tất cả ảnh trong thư mục")
        print("5. Xử lý tất cả video trong thư mục")
        print("6. Demo phát hiện màu đèn giao thông")
        print("7. Debug mode - Kiểm tra tại sao không phát hiện được")
        print("9. Test màu đèn chi tiết (Debug color detection)")
        print("0. Thoát")
        print("-"*50)
        
        choice = input("👉 Chọn chức năng (0-8): ").strip()
        
        if choice == '1':
            image_path = input("📁 Nhập đường dẫn ảnh: ").strip()
            if os.path.exists(image_path):
                detector.detect_image(image_path)
            else:
                print("❌ File không tồn tại!")
        
        elif choice == '2':
            video_path = input("📁 Nhập đường dẫn video: ").strip()
            if os.path.exists(video_path):
                detector.detect_video(video_path)
            else:
                print("❌ File không tồn tại!")
        
        elif choice == '3':
            detector.detect_webcam()
        
        elif choice == '4':
            image_files = get_files_in_folder(
                os.path.join(INPUT_DIR, 'images'), 
                ['.jpg', '.jpeg', '.png', '.bmp']
            )
            if image_files:
                print(f"🔍 Tìm thấy {len(image_files)} ảnh")
                for img_path in image_files:
                    detector.detect_image(img_path)
            else:
                print("❌ Không tìm thấy ảnh nào trong thư mục input/images/")
        
        elif choice == '5':
            video_files = get_files_in_folder(
                os.path.join(INPUT_DIR, 'videos'), 
                ['.mp4', '.avi', '.mov', '.mkv']
            )
            if video_files:
                print(f"🔍 Tìm thấy {len(video_files)} video")
                for vid_path in video_files:
                    detector.detect_video(vid_path)
            else:
                print("❌ Không tìm thấy video nào trong thư mục input/videos/")
        
        elif choice == '7':
            # Debug mode
            image_path = input("📁 Nhập đường dẫn ảnh để debug: ").strip()
            if os.path.exists(image_path):
                print("🔍 Chế độ debug - Kiểm tra chi tiết...")
                
                # Đọc ảnh
                image = cv2.imread(image_path)
                if image is None:
                    print("❌ Không thể đọc ảnh!")
                    continue
                
                print(f"📏 Kích thước ảnh: {image.shape}")
                
                # Thử với nhiều ngưỡng confidence khác nhau
                confidence_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
                
                for conf in confidence_levels:
                    print(f"\n🎯 Thử với confidence = {conf}")
                    results = detector.model(image, conf=conf, iou=0.4)
                    
                    total_objects = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
                    traffic_lights = 0
                    
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                class_id = int(box.cls[0])
                                confidence = float(box.conf[0])
                                if class_id == 9:  # traffic light
                                    traffic_lights += 1
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    print(f"   🚦 Đèn giao thông: conf={confidence:.3f}, box=[{x1},{y1},{x2},{y2}]")
                    
                    print(f"   📊 Tổng: {total_objects} objects, {traffic_lights} đèn giao thông")
                    
                    if traffic_lights > 0:
                        print(f"✅ Tìm thấy đèn với confidence {conf}!")
                        break
                
                # Thử với model lớn hơn nếu có
                print(f"\n🤖 Model hiện tại: {detector.model.model_name}")
                
            else:
                print("❌ File không tồn tại!")
        
        elif choice == '9':
            # Test màu đèn chi tiết
            image_path = input("📁 Nhập đường dẫn ảnh để test màu đèn: ").strip()
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is None:
                    print("❌ Không thể đọc ảnh!")
                    continue
                
                print("🔍 Đang phân tích màu đèn giao thông...")
                
                # Chạy detection để tìm đèn
                results = detector.model(image, conf=0.2, iou=0.4)
                
                light_count = 0
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            if class_id == 9:  # traffic light
                                light_count += 1
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                confidence = float(box.conf[0])
                                
                                print(f"\n🚦 Đèn số {light_count}:")
                                print(f"   Vị trí: [{x1}, {y1}, {x2}, {y2}]")
                                print(f"   Confidence: {confidence:.3f}")
                                
                                # Test với debug chi tiết
                                from utils import detect_traffic_light_color, detect_traffic_light_color_advanced, debug_traffic_light_color
                                
                                print("🔬 Debug chi tiết màu sắc:")
                                color1, bgr1 = debug_traffic_light_color(image, x1, y1, x2, y2)
                                color2, bgr2 = detect_traffic_light_color_advanced(image, x1, y1, x2, y2)
                                
                                print(f"   Phương pháp HSV: {color1.upper()}")
                                print(f"   Phương pháp vùng: {color2.upper()}")
                                
                                # Hiển thị vùng đèn để kiểm tra
                                roi = image[y1:y2, x1:x2]
                                if roi.size > 0:
                                    roi_resized = cv2.resize(roi, (100, 200))  # Phóng to để xem rõ
                                    cv2.imshow(f'Traffic Light {light_count} - ROI', roi_resized)
                
                if light_count == 0:
                    print("❌ Không tìm thấy đèn giao thông trong ảnh!")
                else:
                    print(f"\n✅ Phân tích {light_count} đèn hoàn tất!")
                    print("👁️  Nhấn phím bất kỳ để đóng cửa sổ...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                print("❌ File không tồn tại!")
        
        elif choice == '8':
            # Thử model lớn hơn
            print("🚀 Đang tải model yolov8m (lớn hơn, chính xác hơn)...")
            try:
                from ultralytics import YOLO
                detector_m = YOLO('yolov8m.pt')
                
                image_path = input("📁 Nhập đường dẫn ảnh để test với model lớn: ").strip()
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    
                    print("🔍 Đang phân tích với YOLOv8m...")
                    results = detector_m(image, conf=0.2, iou=0.4)
                    
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                class_id = int(box.cls[0])
                                confidence = float(box.conf[0])
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                
                                if class_id == 9:
                                    print(f"🚦 Model lớn phát hiện đèn: conf={confidence:.3f}")
                                
                                class_name = TRAFFIC_CLASSES.get(class_id, f'class_{class_id}')
                                print(f"📋 {class_name}: {confidence:.3f}")
                    
                    # Vẽ kết quả
                    result_image = draw_boxes(image.copy(), results)
                    cv2.imshow('YOLOv8m Results', result_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
                else:
                    print("❌ File không tồn tại!")
                    
            except Exception as e:
                print(f"❌ Lỗi khi tải model lớn: {e}")
        
        elif choice == '6':
            # Demo phát hiện màu đèn giao thông
            print("🚦 Demo phát hiện màu đèn giao thông")
            print("Tìm kiếm ảnh có đèn giao thông trong thư mục input/images/...")
            
            image_files = get_files_in_folder(
                os.path.join(INPUT_DIR, 'images'), 
                ['.jpg', '.jpeg', '.png', '.bmp']
            )
            
            if image_files:
                for img_path in image_files:
                    print(f"\n🔍 Kiểm tra: {os.path.basename(img_path)}")
                    result_img = detector.detect_image(img_path)
                    
                    if result_img is not None:
                        # Hiển thị ảnh kết quả
                        cv2.imshow('Traffic Light Detection', result_img)
                        print("👁️  Nhấn phím bất kỳ để tiếp tục, 'q' để thoát")
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord('q'):
                            break
                cv2.destroyAllWindows()
            else:
                print("❌ Không tìm thấy ảnh nào!")
                print("💡 Hãy đặt ảnh có đèn giao thông vào thư mục input/images/")
        
        elif choice == '0':
            print("👋 Tạm biệt!")
            break
        
        else:
            print("❌ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()