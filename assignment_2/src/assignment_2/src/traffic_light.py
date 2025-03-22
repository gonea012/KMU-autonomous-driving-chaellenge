#!/usr/bin/env python
#-- coding:utf-8 --
####################################################################
# 프로그램이름 : traffic_light.py
# 코드작성팀명 : 슈티어링
####################################################################

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32

class TrafficDetect:
    def __init__(self):
        # traffic_light 노드 초기화 
        rospy.init_node('traffic_light_node', anonymous=True)
        self.bridge = CvBridge()

        # image_raw 노드 구독자
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)

        # 속도와 각도 퍼블리셔 초기화
        self.speed_pub = rospy.Publisher('/traffic_light/speed', Float32, queue_size=10)
        self.angle_pub = rospy.Publisher('/traffic_light/angle', Float32, queue_size=1)

    # 이미지 처리 메소드
    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            print(e)
            return

        self.detect_traffic(cv_image)

    # 신호등 처리 메소드
    def detect_traffic(self, data):
        font = cv2.FONT_HERSHEY_SIMPLEX              # 사용할 폰트 설정
        copy_img = data.copy()                       # 원본 이미지 복사본 생성
        hsv = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)  # BGR 이미지를 HSV로 변환

        # 색상 범위 설정
        lower_red1 = np.array([0, 100, 100])    # 빨간색 범위 1 하한
        upper_red1 = np.array([10, 255, 255])   # 빨간색 범위 1 상한
        lower_red2 = np.array([160, 100, 100])  # 빨간색 범위 2 하한
        upper_red2 = np.array([180, 255, 255])  # 빨간색 범위 2 상한

        lower_green = np.array([40, 70, 70])    # 초록색 범위 하한
        upper_green = np.array([90, 255, 255])  # 초록색 범위 상한

        lower_yellow = np.array([15, 100, 100]) # 노란색 범위 하한
        upper_yellow = np.array([35, 255, 255]) # 노란색 범위 상한

        # 색상에 따른 마스크 생성
        maskr1 = cv2.inRange(hsv, lower_red1, upper_red1)     # 빨간색 마스크 1
        maskr2 = cv2.inRange(hsv, lower_red2, upper_red2)     # 빨간색 마스크 2
        maskg = cv2.inRange(hsv, lower_green, upper_green)    # 초록색 마스크
        masky = cv2.inRange(hsv, lower_yellow, upper_yellow)  # 노란색 마스크
        maskr = cv2.add(maskr1, maskr2)                       # 두 빨간색 마스크 결합

        size = data.shape  # 이미지 크기

        # 호프 서클 변환을 이용한 원형 감지
        r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80, param1=50, param2=10, minRadius=0, maxRadius=30)
        g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 60, param1=50, param2=10, minRadius=0, maxRadius=30)
        y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 30, param1=50, param2=5, minRadius=30, maxRadius=30)

        # 감지된 각 신호등에 대한 처리
        r = 5             # 검사할 반경 설정
        bound = 5.0 / 10  # 이미지의 상한선까지만 검사

        # 빨간 신호등 처리
        if r_circles is not None:
            # 감지된 원의 좌표 반올림 및 정수화
            r_circles = np.uint16(np.around(r_circles)) 

            # 각 원에 대해 
            for i in r_circles[0, :]:  
                if i[0] > size[1] or i[1] > size[0] or i[1] > size[0] * bound:
                    continue    # 이미지 범위를 벗어나면 무시

                # 픽셀 평균 값 계산을 위한 변수
                h, s = 0.0, 0.0  

                # 주변 픽셀에 대해
                for m in range(-r, r):  
                    for n in range(-r, r):
                        if (i[1] + m) >= size[0] or (i[0] + n) >= size[1]:
                            continue    # 이미지 범위를 벗어나면 무시

                        h += maskr[i[1] + m, i[0] + n]  # 해당 픽셀 값 합산
                        s += 1                          # 픽셀 수 카운트

                # 평균 픽셀 값이 임계값을 넘으면
                if h / s > 50:  
                    cv2.circle(copy_img, (i[0], i[1]), i[2] + 10, (0, 255, 0), 2)                     # 원 그리기
                    cv2.circle(maskr, (i[0], i[1]), i[2] + 30, (255, 255, 255), 2)                    # 마스크에 원 그리기
                    cv2.putText(copy_img, 'RED', (i[0], i[1]), font, 1, (255, 0, 0), 2, cv2.LINE_AA)  # 텍스트 쓰기
            
                    # angle과 speed를 퍼블리시 
                    self.angle_pub.publish(0.0)
                    self.speed_pub.publish(0.0)
                    # 디버깅용
                    rospy.loginfo("red")   

        # 초록 신호등 처리
        if g_circles is not None:
            # 감지된 원의 좌표 반올림 및 정수화
            g_circles = np.uint16(np.around(g_circles))

            # 각 원에 대해
            for i in g_circles[0, :]:
                if i[0] > size[1] or i[1] > size[0] or i[1] > size[0] * bound:
                    continue    # 이미지 범위를 벗어나면 무시

                # 픽셀 평균 값 계산을 위한 변수
                h, s = 0.0, 0.0

                # 주변 픽셀에 대해
                for m in range(-r, r):
                    for n in range(-r, r):
                        if (i[1] + m) >= size[0] or (i[0] + n) >= size[1]:
                            continue    # 이미지 범위를 벗어나면 무시

                        h += maskg[i[1] + m, i[0] + n]  # 해당 픽셀 값 합산
                        s += 1                          # 픽셀 수 카운트

                # 평균 픽셀 값이 임계값을 넘으면
                if h / s > 100:
                    cv2.circle(copy_img, (i[0], i[1]), i[2] + 10, (0, 255, 0), 2)                       # 원 그리기 
                    cv2.circle(maskg, (i[0], i[1]), i[2] + 30, (255, 255, 255), 2)                      # 마스크에 원 그리기
                    cv2.putText(copy_img, 'GREEN', (i[0], i[1]), font, 1, (255, 0, 0), 2, cv2.LINE_AA)  # 텍스트 쓰기

                    # angle과 speed를 퍼블리시 
                    self.angle_pub.publish(0.0)
                    self.speed_pub.publish(15.0)
                    # 디버깅용
                    rospy.loginfo("green") 
                    return 

        # 노란 신호등 처리
        if y_circles is not None:
            # 감지된 원의 좌표 반올림 및 정수화
            y_circles = np.uint16(np.around(y_circles))

            # 각 원에 대해
            for i in y_circles[0, :]:
                if i[0] > size[1] or i[1] > size[0] or i[1] > size[0] * bound:
                    continue    # 이미지 범위를 벗어나면 무시

                # 픽셀 평균 값 계산을 위한 변수
                h, s = 0.0, 0.0

                # 주변 픽셀에 대해
                for m in range(-r, r):
                    for n in range(-r, r):
                        if (i[1] + m) >= size[0] or (i[0] + n) >= size[1]:
                            continue    # 이미지 범위를 벗어나면 무시

                        h += masky[i[1] + m, i[0] + n]  # 해당 픽셀 값 합산
                        s += 1                          # 픽셀 수 카운트
                if h / s > 50:
                    cv2.circle(copy_img, (i[0], i[1]), i[2] + 10, (0, 255, 0), 2)                       # 원 그리기
                    cv2.circle(masky, (i[0], i[1]), i[2] + 30, (255, 255, 255), 2)                      # 마스크에 원 그리기
                    cv2.putText(copy_img, 'YELLOW', (i[0], i[1]), font, 1, (255, 0, 0), 2, cv2.LINE_AA) # 텍스트 쓰기

                    # angle과 speed를 퍼블리시 
                    self.angle_pub.publish(0.0)
                    self.speed_pub.publish(0.0)
                    # 디버깅용
                    rospy.loginfo("yellow")  

        # 디버깅용
        # cv2.imshow('detected results', copy_img)
        # cv2.waitKey(3)

if __name__ == '__main__':
    try:
        TrafficDetect()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
