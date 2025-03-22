#!/usr/bin/env python
#-- coding:utf-8 --
####################################################################
# 프로그램이름 : line_tracking.py
# 코드작성팀명 : 슈티어링
####################################################################

import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class LineTracker:
    def __init__(self):
        # line_tracking 노드 초기화
        rospy.init_node('line_tracking_node', anonymous=True)
        self.bridge = CvBridge()

        # image_raw 노드 구독자
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)

        # traffic_light.py에서 퍼블리시된 속도와 각도 구독
        self.speed_sub = rospy.Subscriber("/traffic_light/speed", Float32, self.speed_callback)
        self.angle_sub = rospy.Subscriber("/traffic_light/angle", Float32, self.angle_callback)

        # 속도와 각도 퍼블리셔 초기화
        self.speed_pub = rospy.Publisher("/line_tracking/speed", Float32, queue_size=10)
        self.angle_pub = rospy.Publisher('/line_tracking/angle', Float32, queue_size=1)

        # 속도 및 각도 초기화
        self.speed = 0.0
        self.angle = 0.0

    def speed_callback(self, msg):
        self.speed = msg.data

    def angle_callback(self, msg):
        self.angle = msg.data

    def image_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")  # ROS 이미지 메시지를 OpenCV 이미지로 변환
        height, width, _ = cv_image.shape
        
        roi_vertices = [(0, height), (width / 2, height / 2), (width, height)]
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)                                           # 그레이스케일로 변환
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
        inpainted_image = cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)
        blur = cv2.GaussianBlur(inpainted_image, (5, 5), 0)                                                    # 가우시안 블러 적용
        edges = cv2.Canny(blur, 50, 150)                                                            # Canny 엣지 검출
        cropped_edges = self.region_of_interest(edges, np.array([roi_vertices], np.int32))          # ROI 설정
        lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=20) # 허프 변환으로 선 검출
        
        # 속도와 각도 초기화
        speed = self.speed
        angle = self.angle

        if lines is not None:
            left_line, right_line = self.average_slope_intercept(lines, width, height)

            if left_line is not None and right_line is not None:
                left_slope, left_intercept = left_line
                right_slope, right_intercept = right_line

                midpoint = width / 2
                lane_center = (left_intercept + right_intercept) / 2
                deviation = midpoint - lane_center

                # 비례 제어를 통한 조향 각도 조절
                k_p = 0.005  # 조향 각도를 더 부드럽게 만들기 위해 값을 더 작게 조정
                angle = k_p * deviation

                # 검출된 선을 이미지에 그리기
                cv_image = self.draw_lines(cv_image, left_line, right_line, height)

        # 속도와 각도 퍼블리시
        rospy.loginfo(f"Detected lines, publishing speed: {speed}, angle: {angle}")
        self.speed_pub.publish(speed)
        self.angle_pub.publish(angle)

        # 결과 이미지 보여주기
        cv2.imshow('Lane Tracking', cv_image)
        cv2.waitKey(3)

    def region_of_interest(self, img, vertices):
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, 255)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def average_slope_intercept(self, lines, width, height):
        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 0.0001)
            intercept = y1 - slope * x1

            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))

        left_line = np.mean(left_lines, axis=0) if len(left_lines) > 0 else None
        right_line = np.mean(right_lines, axis=0) if len(right_lines) > 0 else None

        return left_line, right_line

    def draw_lines(self, img, left_line, right_line, height):
        if left_line is not None:
            left_slope, left_intercept = left_line
            cv2.line(img, (int(left_intercept), height), (int(left_intercept + left_slope * (height / 2)), height // 2), (0, 255, 0), 2)
        if right_line is not None:
            right_slope, right_intercept = right_line
            cv2.line(img, (int(right_intercept), height), (int(right_intercept + right_slope * (height / 2)), height // 2), (0, 255, 0), 2)
        return img

if __name__ == '__main__':
    try:
        LineTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
