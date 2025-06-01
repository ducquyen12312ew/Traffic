import cv2
import numpy as np
import os
from config import TRAFFIC_CLASSES

def get_class_name(class_id):
    """Lấy tên class từ ID"""
    return TRAFFIC_CLASSES.get(class_id, f'class_{class_id}')

def detect_traffic_light_color(image, x1, y1, x2, y2):
    """Phát hiện màu đèn giao thông - Fix đặc biệt cho GREEN vs YELLOW"""
    # Cắt vùng đèn giao thông
    traffic_light_roi = image[y1:y2, x1:x2]
    
    if traffic_light_roi.size == 0:
        return "unknown", (128, 128, 128)
    
    # Phân tích BGR trực tiếp (chính xác nhất cho green vs yellow)
    b, g, r = cv2.split(traffic_light_roi)
    
    # Tính giá trị trung bình và max
    avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
    max_b, max_g, max_r = np.max(b), np.max(g), np.max(r)
    
    # ** LOGIC CHÍNH XÁC CHO GREEN VS YELLOW **
    
    # Điều kiện cho màu XANH LÁ (GREEN):
    # 1. G channel phải cao hơn R và B rõ rệt
    # 2. Tỷ lệ G/(R+B) phải lớn  
    green_condition1 = avg_g > avg_r * 1.15 and avg_g > avg_b * 1.3
    green_condition2 = max_g > max_r * 1.1 and max_g > max_b * 1.2
    green_ratio = avg_g / (avg_r + avg_b + 1)  # +1 tránh chia 0
    
    # Điều kiện cho màu VÀNG (YELLOW):
    # 1. R và G gần bằng nhau
    # 2. B thấp hơn đáng kể
    green_condition1 = abs(avg_r - avg_g) < 25 and avg_b < min(avg_r, avg_g) * 0.7
    green_condition2 = max_r > max_b * 1.5 and max_g > max_b * 1.5
    
    # Điều kiện cho màu ĐỎ (RED):
    # 1. R channel cao nhất
    red_condition1 = avg_r > avg_g * 1.2 and avg_r > avg_b * 1.3
    red_condition2 = max_r > max_g * 1.1 and max_r > max_b * 1.2
    
    # Debug info (uncomment để debug)
    # print(f"🔍 BGR: R={avg_r:.1f}, G={avg_g:.1f}, B={avg_b:.1f}")
    # print(f"🔍 Max: R={max_r:.1f}, G={max_g:.1f}, B={max_b:.1f}")
    # print(f"🔍 Green ratio: {green_ratio:.2f}")
    # print(f"🔍 Green conditions: {green_condition1}, {green_condition2}")
    
    # Kiểm tra độ sáng tối thiểu (đèn phải đủ sáng)
    min_brightness = 30
    if max(avg_r, avg_g, avg_b) < min_brightness:
        return "off", (128, 128, 128)
    
    # ** QUY ẾT CUỐI CÙNG - ƯU TIÊN GREEN **
    
    if (green_condition1 and green_condition2) or green_ratio > 0.65:
        return "green", (0, 255, 0)
    elif red_condition1 and red_condition2:
        return "red", (0, 0, 255)  
    elif green_condition1 and green_condition2:
        return "green", (0, 255, 255)
    else:
        # Fallback: so sánh trực tiếp
        if avg_g > avg_r and avg_g > avg_b:
            return "green", (0, 255, 0)
        elif avg_r > avg_g and avg_r > avg_b:
            return "red", (0, 0, 255)
        elif abs(avg_r - avg_g) < 20 and avg_b < min(avg_r, avg_g):
            return "green", (0, 255, 255)
        else:
            return "unknown", (128, 128, 128)

def detect_traffic_light_color_advanced(image, x1, y1, x2, y2):
    """Phát hiện màu đèn giao thông nâng cao - phân tích từng vùng"""
    # Cắt vùng đèn giao thông
    traffic_light_roi = image[y1:y2, x1:x2]
    
    if traffic_light_roi.size == 0:
        return "unknown", (128, 128, 128)
    
    h, w = traffic_light_roi.shape[:2]
    
    # Mở rộng vùng phân tích để tránh nhầm lẫn
    margin = max(1, min(h, w) // 10)  # Thêm margin
    
    # Chia đèn thành 3 phần với margin
    regions = [
        traffic_light_roi[margin:h//3-margin, margin:w-margin] if h//3-margin > margin else traffic_light_roi[0:h//3, :],     # Phần trên (đỏ)
        traffic_light_roi[h//3+margin:2*h//3-margin, margin:w-margin] if 2*h//3-margin > h//3+margin else traffic_light_roi[h//3:2*h//3, :],  # Phần giữa (vàng)  
        traffic_light_roi[2*h//3+margin:h-margin, margin:w-margin] if h-margin > 2*h//3+margin else traffic_light_roi[2*h//3:h, :]       # Phần dưới (xanh)
    ]
    
    colors = ["red", "green", "green"]
    color_bgr = [(0, 0, 255), (0, 255, 0), (0, 255, 0)]
    
    max_brightness = 0
    detected_color = "off"
    detected_bgr = (128, 128, 128)
    
    # Thêm phân tích HSV để chính xác hơn
    hsv_roi = cv2.cvtColor(traffic_light_roi, cv2.COLOR_BGR2HSV)
    
    for i, region in enumerate(regions):
        if region.size == 0:
            continue
            
        # Chuyển sang grayscale để đo độ sáng
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Phân tích HSV của vùng này
        region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Tìm điểm sáng nhất
        max_val = np.max(gray)
        mean_val = np.mean(gray)
        bright_pixels = np.sum(gray > 180)  # Đếm pixel rất sáng
        
        # Kiểm tra màu sắc trong HSV
        h_channel = region_hsv[:, :, 0]
        s_channel = region_hsv[:, :, 1]
        v_channel = region_hsv[:, :, 2]
        
        # Xác định màu dựa trên HSV
        color_score = 0
        if i == 0:  # Vùng đỏ
            red_mask1 = (h_channel <= 10) & (s_channel > 100) & (v_channel > 100)
            red_mask2 = (h_channel >= 170) & (s_channel > 100) & (v_channel > 100)
            color_score = np.sum(red_mask1) + np.sum(red_mask2)
        elif i == 1:  # Vùng vàng
            green_mask = (h_channel >= 15) & (h_channel <= 35) & (s_channel > 100) & (v_channel > 100)
            color_score = np.sum(green_mask)
        elif i == 2:  # Vùng xanh
            green_mask = (h_channel >= 40) & (h_channel <= 80) & (s_channel > 100) & (v_channel > 100)
            color_score = np.sum(green_mask)
        
        # Tổng hợp điểm số
        total_score = max_val * 0.4 + mean_val * 0.3 + bright_pixels * 0.2 + color_score * 0.1
        
        # Kiểm tra độ sáng và màu sắc
        if (max_val > 120 and mean_val > 60 and bright_pixels > 3) or color_score > 20:
            if total_score > max_brightness:
                max_brightness = total_score
                detected_color = colors[i]
                detected_bgr = color_bgr[i]
    
    return detected_color, detected_bgr

def detect_arrow_direction(image, x1, y1, x2, y2):
    """Phát hiện hướng mũi tên trong đèn giao thông"""
    try:
        # Cắt vùng đèn
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return "none"
        
        # Chuyển sang grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Tìm contours
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Tính diện tích và tỷ lệ khung hình
            area = cv2.contourArea(contour)
            if area > 50:  # Đủ lớn để là mũi tên
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Phân tích hình dạng để xác định hướng
                if aspect_ratio > 1.2:  # Rộng hơn cao -> có thể là mũi tên ngang
                    # Kiểm tra vị trí khối lượng để xác định hướng
                    moments = cv2.moments(contour)
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"])
                        if cx > w * 0.6:  # Khối lượng lệch phải
                            return "right"
                        elif cx < w * 0.4:  # Khối lượng lệch trái
                            return "left"
        
        return "none"
    except:
        return "none"

def draw_boxes(image, results):
    """Vẽ bounding box lên ảnh với phát hiện màu đèn giao thông"""
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Lấy tọa độ
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Hiển thị TẤT CẢ object được phát hiện (không chỉ traffic classes)
                class_name = TRAFFIC_CLASSES.get(class_id, f'object_{class_id}')
                box_color = (0, 255, 0)  # Màu xanh lá mặc định
                
                # Nếu là đèn giao thông, phát hiện màu và hướng
                if class_id == 9:  # traffic light
                    try:
                        # Thử cả 2 phương pháp phát hiện màu
                        light_color1, color_bgr1 = detect_traffic_light_color(image, x1, y1, x2, y2)
                        light_color2, color_bgr2 = detect_traffic_light_color_advanced(image, x1, y1, x2, y2)
                        
                        # Ưu tiên kết quả của phương pháp nâng cao
                        if light_color2 != "off" and light_color2 != "unknown":
                            light_color, color_bgr = light_color2, color_bgr2
                        else:
                            light_color, color_bgr = light_color1, color_bgr1
                        
                        # Phát hiện hướng mũi tên
                        arrow_direction = detect_arrow_direction(image, x1, y1, x2, y2)
                        
                        if arrow_direction != "none":
                            class_name = f"Traffic Light ({light_color.upper()}) - {arrow_direction.upper()} ARROW"
                        else:
                            class_name = f"Traffic Light ({light_color.upper()})"
                            
                        box_color = color_bgr
                        
                    except Exception as e:
                        light_color = "unknown"
                        class_name = "Traffic Light (UNKNOWN)"
                
                # Vẽ box cho object có confidence > ngưỡng tối thiểu
                if confidence > 0.1:  # Hiển thị gần như tất cả
                    # Vẽ bounding box
                    thickness = 3 if class_id == 9 else 2  # Đèn giao thông dày hơn
                    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)
                    
                    # Vẽ label với nền
                    label = f'{class_name}: {confidence:.2f}'
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Vẽ nền cho text
                    cv2.rectangle(image, (x1, y1-label_size[1]-10), 
                                (x1+label_size[0]+5, y1), box_color, -1)
                    
                    # Vẽ text
                    text_color = (0, 0, 0) if class_id == 9 and box_color == (0, 255, 255) else (255, 255, 255)
                    cv2.putText(image, label, (x1+2, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    return image

def get_files_in_folder(folder_path, extensions):
    """Lấy danh sách file theo extension"""
    files = []
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in extensions):
                files.append(os.path.join(folder_path, file))
    return files

def analyze_traffic_light_stats(image, results):
    """Phân tích thống kê đèn giao thông trong ảnh"""
    traffic_lights = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id == 9:  # traffic light
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Phát hiện màu
                    light_color, _ = detect_traffic_light_color_advanced(image, x1, y1, x2, y2)
                    
                    traffic_lights.append({
                        'position': (x1, y1, x2, y2),
                        'color': light_color,
                        'confidence': confidence
                    })
    
    return traffic_lights