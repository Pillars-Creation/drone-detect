import cv2
import sys
from object_detection import ObjectDetection

global isTracking
global bbox
global ok
global img2

od = ObjectDetection()

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

def on_mouse(event, x, y, flags, param):
    global img2, point1, point2, g_rect, isTracking
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击,则在原图打点
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 5)
        cv2.imshow('Tracking', img2)

    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳，画框
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), thickness=2)
        cv2.imshow('Tracking', img2)

    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放，显示
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), thickness=2)
        cv2.imshow('Tracking', img2)
        if point1 != point2:
            min_x = min(point1[0], point2[0])
            min_y = min(point1[1], point2[1])
            width = abs(point1[0] - point2[0])
            height = abs(point1[1] - point2[1])

            # 定义一个初始边界框
            bbox = (min_x, min_y, width, height)
            # 用第一帧和包围框初始化跟踪器
            tracker.init(frame, bbox)
            isTracking=True


if __name__ == '__main__':

    # 建立追踪器
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[1]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

    # 读取视频
    video = cv2.VideoCapture("../wrj.mp4")

    # 如果视频没有打开，退出。
    if not video.isOpened():
        print
        "Could not open video"
        sys.exit()

    isTracking=False

    cv2.namedWindow('Tracking')

    while True:
        # cv2.setMouseCallback('Tracking', on_mouse)
        # 读取一个新的帧
        ok, frame = video.read()
        if not ok:
            break

        img2=frame

        #取一帧判断是否有无人机
        (class_ids, scores, boxes) = od.detect(frame)

        #如果有无人机 选定无人机

        # 启动计时器
        timer = cv2.getTickCount()

        if isTracking:
            # 更新跟踪器
            ok, bbox = tracker.update(frame)

            # 计算帧率(FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

            # 绘制包围框
            if ok:
                # 跟踪成功
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            else:
                # 跟踪失败
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # 在帧上显示跟踪器类型名字
            cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

            # 在帧上显示帧率FPS
            cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # 显示结果
        cv2.imshow("Tracking", frame)

        # 按ESC键退出
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

video.release()
cv2.destroyAllWindows()