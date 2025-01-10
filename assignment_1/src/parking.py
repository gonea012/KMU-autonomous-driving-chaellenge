#!/usr/bin/env python
#-- coding:utf-8 --
####################################################################
# 프로그램이름 : parking.py
# 코드작성팀명 : 슈티어링
####################################################################

#=============================================
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부
#=============================================
import pygame
import numpy as np
import math
import rospy
from xycar_msgs.msg import xycar_motor

# motion_planning 소스코드를 import 해서 tracking 에서 경로를 추적한다.
# 파일 구조 : parking.py / motion_planning.py / path_planning.py
import motion_planning as pl 


#=============================================
# 모터 토픽을 발행할 것임을 선언
#============================================= 
motor_pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
xycar_msg = xycar_motor()


#=============================================
# 프로그램에서 사용할 상수 선언부
#=============================================
AR = (1142, 62) # AR 태그의 위치
P_ENTRY = (1036, 162) # 주차라인 진입 시점의 좌표
P_END = (1129, 69) # 주차라인 끝의 좌표

#=============================================
# 모터 토픽을 발행하는 함수
# 입력으로 받은 angle과 speed 값을
# 모터 토픽에 옮겨 담은 후에 토픽을 발행함.
#=============================================
def drive(angle, speed):
    xycar_msg.angle = int(angle)
    xycar_msg.speed = int(speed)
    motor_pub.publish(xycar_msg)



# --------------------------------
# planning 함수를 위한 기본적인 세팅
# planning 함수를 이용해서 생성한 경로를 tracking 함수에서 사용해야 하기 때문에 아래 변수들을 전역변수로 선언
# 변수들은 2차원 배열로 저장 - 각각의 배열은 각각의 경로를 의미
# rx, rt : 모든 경로의 x좌표, 주행 방향 - motion planning에서 waypoint를 탐색할 때 사용하는 변수
# path_x, path_y : 경로의 x, y 좌표
# rdirect : 경로의 주행 방향
rx, ry, ryaw, rdirect, path_x, path_y = [], [], [], [], [], []
# planning 함수가 실행될 때 1로 초기화 - tracking 함수 시작 시 1회 실행
flag_start = 1


# --------------------------------
# planning : 경로 생성 함수 (path_planning.py 사용)
# sx, sy : 차량의 시작위치, syaw : 시작각도
# max_acceleration : 최대가속도, dt : 단위시간
# 자이트론 카페 문의를 통해 휠 베이스(WB)와 차량의 너비(W)에 대한 답변 받음
# 휠 베이스(WB)를 path planning 알고리즘에 적용 -> 회전 반경 계산
# 경로 생성의 경우 reeds shepp 알고리즘을 사용 (path_planning 참고)
# path_planning은 motion_planning에서 사용됨
# --------------------------------
def planning(sx, sy, syaw, max_acceleration, dt):
    # 동일한 변수를 tracking 함수에서도 사용하기 위해 전역 변수로 생성
    global rx, ry, ryaw, rdirect, path_x, path_y
    # tracking 함수를 시작할 때 1회만 실행되는 flag
    # planning 함수 실행 시 1로 초기화
    global flag_start
    print("Planning Path")
    flag_start = 1

    # AR = (1142, 62) # AR 태그의 위치
    # P_ENTRY = (1036, 162) # 주차라인 진입 시점의 좌표
    # P_END = (1129, 69) # 주차라인 끝의 좌표

    # 경로 좌표를 states에 저장
    # 초기 각도에 90도 더해줌 : 차량의 초기 각도와 path_planning에서 사용하는 각도가 90도 틀어져 있음
    # 실행 순서 요약
    # 1. 주차 라인에 진입하기 100 픽셀 전에 경로 생성
    # 2. AR 태그 60 픽셀 전에 경로 생성
    states = [(sx, sy, syaw+90), (P_ENTRY[0]-100, P_ENTRY[1]+100, -45), (AR[0]-56, AR[1]+56, -35)]

    # 경로 생성 : path planning 알고리즘 사용 (reeds_shepp)
    # -------path planning 실행 원리 요약--------
    # 1. 차량의 초기 위치 및 주차 위치를 입력
    # 2. 차량의 초기 위치 및 주차 위치 기반으로 초기 각도 계산
    # 3. 계산된 초기 각도, 휠 베이스, 최대 조향각을 이용하여 회전 반경 계산
    # 4. 차량의 초기 위치, 목표 위치, 회전 반경 바탕으로 경로 생성
    # --------------------------------
    rx, ry, ryaw, rdirect, path_x, path_y = pl.generate_path_sec(states)

    # 경로 반환 (for 시각화)
    return path_x, path_y


# --------------------------------
# tracking 함수를 위한 기본적인 세팅

# 차량의 주행 했던 경로를 저장하는 리스트 - 디버깅시 사용
x_rec, y_rec = [], []

# path planning 알고리즘에서 생성된 리스트는 2차원 리스트로 생성되어 있기 때문에 - 방향이 바뀔때 마다 다른 경로를 생성하기 때문에 여러개의 경로가 저장 되기 때문
# 1차원 리스트로 변환하여 사용해야 한다.
# 1차원 리스트로 변환하기 위해 사용되는 리스트 - cx(경로의 x좌표), cy(경로의 y좌표), cyaw(경로의 yaw), cdirect(경로의 전 후진 방향)
cx, cy, cyaw, cdirect = [], [], [], []

# n번째 경로를 저장하기 위한 cnt 변수 
cnt = 0

# 차량의 현재 정보를 저장하는 클래스 - x, y, yaw, v(속도), direct(전 후진 방향)
node = pl.Node(x=0.0, y=0.0, yaw=0.0, v=0.0, direct=0.0)
# 차량의 현재 정보를 저장하는 클래스를 저장하는 리스트
node_list = pl.Node_list()

# 차량이 제어되는 시간을 저장하는 변수
t = 0.0

# 경로 추적을 위해 PATH 클래스의 인스턴스를 생성한다.
# 이 클래스는 경로의 좌표와 목표 인덱스를 계산하는 기능을 제공한다.
ref_trajectory = pl.RefPath(cx, cy)
# 목표 인덱스를 저장하는 변수
target_ind, _ = 0, 0

# n번째 경로를 시작할 때 1회만 실행 되로록 하기 위한 flag
flag_for = 1

# ------------------------------------------------
# tracking : 경로 추적 함수 (motion_planning.py 사용)
# Tracking 버튼을 누르면 tracking 함수가 계속 실행됨.
# screen : pygame의 screen
# velocity : 차량의 현재 속도
# max_acceleration : 차량의 최대 가속도
# dt : 단위 시간
#------------------------------------------------
def tracking(screen, x, y, yaw, velocity, max_acceleration, dt):
    # planning 함수에서 생성한 경로를 tracking 함수에서 사용해야 하기 때문에 전역변수로 선언한다.
    global rx, ry, ryaw, rdirect, path_x, path_y

    # tracking 함수를 시작하기 위한 flag (1회)
    global flag_start

    # 차량의 경로를 저정하는 변수 
    global x_rec, y_rec
    
    # 차량이 제어되는 시간을 저장하는 변수
    global t
    
    # RefPath 클래스를 이용해 경로 추적을 위해 사용되는 인스턴스
    global ref_trajectory
  
    # 목표 인덱스를 저장하는 변수
    global target_ind, _
    
    # 차량의 현재 상태를 저장하고 이를 다시 리스트에 추가하는 두 변수
    global node, node_list

    # 앞서 planning 에서 생성한 경로는 2차원 리스트 형태이므로, 
    # tracking 을 위해서 이를 1차원 리스트로 변환해줘야 하는데 이때 사용되는 변수들
    global cx, cy, cyaw, cdirect

    # 몇번째 경로인지 카운트 하는 변수 (0부터 4까지)
    global cnt
   
    # 특정 경로를 한번 추적할 때 경로가 한번만 실행되도록 하는 flag
    global flag_for

    # motion_planning / pure_pusuit 알고리즘에서 atna2() 함수를 통해 각도가 라디안 단위로 변환됐다.
    # 따라서, tracking 에서도 yaw 값을 라디안 단위로 변환해준다.
    # 그리고, puresuit 알고리즘에서의 좌표계와 차량의 실제 좌표계의 회전 방향을 맞추기 위해 -1 을 yaw 값에 곱해준다. 
    yaw = np.deg2rad(-yaw)

    # motion_planning 에서 선언한 상수 클래스를 config 로 받아온다. 
    # 이때, 제어 주기와 휠 베이스 그리고 차량 너비를 각 변수들에 저장한다. 
    config = pl.Config
    config.dt = dt
    config.WB = 84
    config.W = 64

    # tracking 함수 실행
    if flag_start:
        flag_start = 0 # 1회만 실행되도록 설정한다. 
        x_rec, y_rec = [], [] # 앞서 선언한 x_rec, y_rec 로 경로 저장하는 리스트를 받아온다. 
        cnt = 0 # 0번째 경로에서 시작하도록 한다. (0부터 4까지)
        flag_for = 1 # 그 경로가 한번만 실행되도록 한다. 
        print("Start Tracking.") # tracking 함수를 실행할 때 터미널에서 확인할 수 있도록 한다. 
        
    # planning 함수에서 계획한 경로들을 저장하기 위해서 2차원 리스트를 사용했었다. 
    # 따라서, 2차원 리스트의 원소 개수가 경로 개수가 되므로 이를 turn_direct 에 저장한다. 
    turn_direct = len(list(zip(rx, ry, ryaw, rdirect)))

    # 실행하려고 하는 n번째 경로가 planning 에서 생성된 경로의 개수 보다 많으면 안되므로,
    # 실행하려고 하는 n번째 경로가 생성된 경로 개수 범위 안에 있을 때 실행한다. 
    # 동시에, 그 n 번째 경로가 한번만 실행될 수 있도록 and 조건을 함께 만족해야 한다. 
    if cnt < turn_direct and flag_for: # 아래 if 문은 tracking 하기 전에 기본적인 세팅을 위한 코드이다. 

        flag_for = 0 # 한번만 실행되도록 설정한다. 
        t = 0.0 # 차량의 제어 시간을 초기화 한다. 
        cx, cy, cyaw, cdirect = rx[cnt], ry[cnt], ryaw[cnt], rdirect[cnt] # 향후 정지 거리 등을 계산하기 위해서 받아온 2차원 리스트를 모두 1차원 리스트로 바꿔준다. 

        # motion_planning 에서 선언한 차량의 상태를 나타내는 클래스들을 초기화 한다. 
        node = pl.Node(x=x, y=y, yaw=-yaw, v=velocity, direct=cdirect)
        nodes = pl.Node_list()

        # 경로 추적을 위해 motion_planning 에서 선언한 RefPath 클래스를 통해 경로 간 거리나 인덱스 위치 등을 계산하는 인스턴스 ref_trajectory 를 생성한다.  
        ref_trajectory = pl.RefPath(cx, cy)

        # motion_planning 의 find_target_index 함수를 통해 목표 인덱스를 찾고, 이를 저장한다. 
        target_ind, _ = ref_trajectory.find_index_target(node) 

        # 몇번째 경로가 실행되고 있는 지 터미널 창에서 확인할 수 있도록 한다.  
        print(cnt, " 번째 경로") 


    if cnt < turn_direct: # 아래 if 문은 본격적인 추적을 위한 코드이다.
        if cdirect[0] > 0: # 받아온 방향 경로 리스트의 첫번째 원소가 1이면,
            # 50.0 의 속도로 전진한다. 
            target_speed = 50.0 
            config.Ld = 100.0
            config.dist_stop = 10
            config.dc = -8.4

        else: # 만약 받아온 방향 경로 리스트의 첫번째 원소가 0보다 작거나 같으면
            # 50.0 의 속도로 후진한다.
            target_speed = 50.0 
            config.Ld = 100
            config.dist_stop = 20
            config.dc = 80.68

        # 각각 현재 x,y 위치와 목표 인덱스의 x,y 위치 차이를 좌표계상의 각도로 계산하고,
        # hypot() 함수를 통해 이를 거리로 변환하여 목표 인덱스까지 남은 거리를 dist 에 저장한다. 
        xt = node.x + config.dc * math.cos(node.yaw)
        yt = node.y + config.dc * math.sin(node.yaw)
        dist = math.hypot(xt - cx[-1], yt - cy[-1]) 

        # 위에서 계산한 남은 거리보다 정지 거리가 커지면 차량을 정지시켜야 하므로,
        if dist < config.dist_stop: # 아래 if 문은 정지를 위한 코드이다.
            cnt += 1 # 해당 경로를 정지 시킨 뒤 다음 경로를 실행하기 위해서 경로를 카운트 하는 cnt 변수의 값을 1만큼 올린다.  
            flag_for = 1 # 한번만 실행되기 위해서 설정한다. 
            drive(0, 0) # 발행한 모터토픽을 통해 차량을 정지시킨다. 

            if cnt == turn_direct: # 더 이상 실행할 경로가 없음을 출력한다. 
                print("마지막 경로 도착")
                drive(0, 0)
                return
       
        # motion_planning 에서 선언한 Node 클래스를 통해 차량의 현재 정보를 받아와서 저장한다. 
        node = pl.Node(x=x, y=y, yaw=yaw, v=velocity, direct=cdirect[0])
        # motion_planning / pure_pursuit 함수를 통해 계산한 조향각과 목표 인덱스를 저장한다. 
        delta, target_ind = pl.pure_pursuit(node, ref_trajectory, target_ind)
        delta = np.rad2deg(delta) # 이때, 받아온 조향각의 단위가 라디안이므로 rad2deg() 함수를 이용해 다시 "도" 단위로 변환한다.

        # 차량이 정해진 경로에서 지나치게 벗어나지 않게 하기 위해 조향각의 범위를 제한한다. 
        if delta > 20:
            delta = 20
        elif delta < -20:
            delta = -20

        # 차량 제어 시간을 업데이트 한다.
        t += config.dt

        # 앞서 계산하고 설정해둔 조향각과 목표 속도(방향 포함)를 이용해 차량을 제어한다. 
        drive(delta, target_speed*cdirect[0])

        # 차량의 상태를 nodes 리스트에 추가하여 업데이트 한다.
        nodes.add_state(t, node)
        # 각각 x,y 경로를 저장한다. 
        x_rec.append(node.x)
        y_rec.append(node.y)
