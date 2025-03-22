#!/usr/bin/env python
#-- coding:utf-8 --
####################################################################
# 프로그램이름 : control.py
# 코드작성팀명 : 슈티어링
####################################################################

import rospy
from std_msgs.msg import Float32
from xycar_msgs.msg import xycar_motor

class CarController:
    def __init__(self):
        # control_node 노드 초기화
        rospy.init_node('control_node', anonymous=True)

        # line_tracking.py에서 퍼블리시된 속도와 각도 구독
        self.speed_pub = rospy.Subscriber("/line_tracking/speed", Float32, self.speed_callback)
        self.angle_pub = rospy.Subscriber("/line_tracking/angle", Float32, self.angle_callback)

        # 차량 제어를 위한 퍼블리셔 초기화
        self.motor_pub = rospy.Publisher('/xycar_motor', xycar_motor, queue_size=10)
        
        # speed와 angle 초기화
        self.speed = 0.0
        self.angle = 0.0

    def speed_callback(self, msg):
        self.speed = msg.data

    def angle_callback(self, msg):
        self.angle = msg.data

    def control_car(self):
        # 10 Hz 주기로 반복
        rate = rospy.Rate(10) 
        
        while not rospy.is_shutdown():
            # 구독한 속도와 각도 값을 퍼블리시
            motor_msg = xycar_motor()
            motor_msg.speed = self.speed
            motor_msg.angle = self.angle
            rospy.loginfo(f"Publishing speed: {self.speed}, angle: {self.angle}")   # 디버깅용
            self.motor_pub.publish(motor_msg)
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = CarController()
        controller.control_car()
    except rospy.ROSInterruptException:
        pass
