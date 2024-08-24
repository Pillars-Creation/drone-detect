import cv2
import sys
from object_detection import ObjectDetection
import time



if __name__ == '__main__':
    # 创建对象检测实例
    od = ObjectDetection()
    cap = cv2.VideoCapture("../wrj.mp4")

    # 初始化变量
    tracker = None
    drone_detected = False
    last_check_time = 0  # 上次检测时间
    saved_bbox = None  # 保存的边界框

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = time.time()


        if not drone_detected:
            # 检测无人机
            (class_ids, scores, boxes) = od.detect(frame)

            for i, class_id in enumerate(class_ids):
                if class_id == 4:  # 假设类别ID 4 表示无人机
                    drone_detected = True
                    bbox = tuple(boxes[i])

                    tracker = cv2.TrackerMIL_create()

                    # 初始化追踪器
                    ok = tracker.init(frame, bbox)
                    if ok:
                        print("无人机检测到并开始跟踪！")
                    else:
                        print("无法初始化跟踪器")
                    break

        else:
            # 使用追踪器跟踪无人机
            success, bbox = tracker.update(frame)
            saved_bbox = bbox  # 保存边界框
            # 每秒检查一次跟踪的对象是否仍然是无人机
            if current_time - last_check_time >= 1000:
                last_check_time = current_time

                # 从 saved_bbox 中获取坐标
                (saved_x, saved_y, saved_w, saved_h) = [int(v) for v in saved_bbox]

                # 裁剪帧到 saved_bbox 区域
                cropped_frame = frame[saved_y:saved_y + saved_h, saved_x:saved_x + saved_w]

                # 在裁剪后的区域内进行检测
                (class_ids, _, boxes) = od.detect(cropped_frame)

                # 检查检测到的对象是否包含无人机
                drone_detected_in_bbox = any(class_id == 4 for class_id in class_ids)

                if not drone_detected_in_bbox:
                    print("跟踪区域不再包含无人机，重新检测。")
                    drone_detected = False
                    saved_bbox = None
                    tracker = None
                    success = False

            if success:
                # 跟踪成功，绘制边界框
                (x, y, w, h) = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            else:
                # 跟踪失败
                cv2.putText(frame, "Lost", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow("Drone Detection and Tracking", frame)


    cap.release()
    cv2.destroyAllWindows()