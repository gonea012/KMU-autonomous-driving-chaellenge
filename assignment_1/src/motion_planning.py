#!/usr/bin/env python
#-- coding:utf-8 --
####################################################################
# 프로그램이름 : parking.py
# 코드작성팀명 : 슈티어링
####################################################################

"""
Pure Pursuit
author: huiming zhou
"""
# 경로 추종을 위한 알고리즘 중 Pure Pursuit 을 채택 (사용하지 않는 함수 및 변수 일부 삭제)
# Pure Pursuit 알고리즘은 차량이 현재 위치에서 목표점을 조향각을 계산하는 알고리즘으로,
# parking.py 에 pl로 import 되어 실행된다. 

# 움직임 계획을 위한 라이브러리들
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

# 경로 생성 함수들을 불러오기 위해 경로 계획을 위한 알고리즘(path_planning)을 import 한다.
import path_planning as rs

# 제어에 필요한 값들을 저장하는 상수 클래스 C
class Config:
    # PID 제어를 위한 매개변수
    Kp = 0.3  # 비례 이득값

    # 시스템 매개변수
    Ld = 2.6  # 차량의 목표점을 설정하기 위한 거리로, 이 반경 안의 경로에서 목표점을 찾아서 그 점으로 조향한다. 
    kf = 0.1  # Ld의 거리를 조절하기 위한 이득값으로, 차량의 속도를 고려해 목표점을 조절한다. 
    dt = 0.1  # 시간 단위, 이 단위를 주기로 차량의 상태를 업데이트 한다. 
    dist_stop = 0.7  # 정지 기준, 목표 지점에서 정지하기 위해 이 거리 안에 들어왔을 때 주행을 멈춘다. 
    dc = 0.0 

    # 차량 매개변수 (시뮬레이터 내에서 배율을 맞추기 위해 변환계수 *25 적용)
    RF = 3.3 * 25  # 차량 전장 
    RB = 0.8 * 25 # 후방 오버행
    W = 2.4 * 25 # 차량의 너비
    WD = 0.7 * W  # 좌우 바퀴 간 거리
    WB = 84  # 휠베이스, 전륜 후륜의 중심간 수평거리
    TR = 0.44 * 25 # 타이어 반지름
    TW = 0.7 * 25 # 타이어 너비
    MAX_STEER = 0.30 # 최대 조향각 
    MAX_ACCELERATION = 5.0 # 최대 가속도

# 차량의 상태를 저장하는 클래스
class Node:
    def __init__(self, x, y, yaw, v, direct):
        self.x = x # 차량의 현재 x 좌표값
        self.y = y # 차량의 현재 y 좌표값
        self.yaw = yaw # 차량의 현재 각도값
        self.v = v # 차량의 현재 속도값
        self.direct = direct # 차량의 현재 방향값

# Node 에서 받아온 차량의 상태들을 리스트 형태로 저장하는 클래스
class Node_list:
    def __init__(self):
        self.x_list = []  # 차량의 x 좌표들을 저장하는 리스트
        self.y_list = []  # 차량의 y 좌표들을 저장하는 리스트
        self.yaw_list = []  # 차량의 각도들을 저장하는 리스트
        self.v_list = []  # 차량의 속도들을 저장하는 리스트
        self.time_list = []  # 차량의 시간들을 저장하는 리스트
        self.direct_list = []  # 차량의 방향들을 저장하는 리스트

    # 현재 차량의 상태를 업데이트 하기 위한 함수로, x/y/yaw/v/t/direct 값을 리스트에 저장한다.
    def add_state(self, time, state):
        self.x_list.append(state.x)
        self.y_list.append(state.y)
        self.yaw_list.append(state.yaw)
        self.v_list.append(state.v)
        self.time_list.append(time)
        self.direct_list.append(state.direct)

# 즉, Node는 차량의 "값"을 가져오고 Node_list는 그 값들을 리스트에 저장하는 역할을 한다.

# path_planning(rs)를 통해서 경로를 생성하고 그 경로들을 저장하는 클래스
class RefPath:
    def __init__(self, x_coords, y_coords):
        # x 경로 저장
        self.x_coords = x_coords
        # y 경로 저장
        self.y_coords = y_coords
        # 경로의 끝 지점 인덱스 계산 : (경로 길이 - 1) 
        self.index_end = len(self.x_coords) - 1
        # 목표 인덱스의 초기값
        self.index_target = None

    # 목표 인덱스를 찾기 위한 함수
    # 앞서 클래스 C에서 설명한 바와 같이, Ld 반경 내 경로의 인덱스를 목표점으로 잡고 경로를 추종한다. 
    def find_index_target(self, state):
        # Nodes 클래스의 add_state 함수에서 받아온 차량의 현재 상태 : state
        # 이 state를 이용해 목표 인덱스를 찾는 내부 함수로, 목표 인덱스를 리턴한다. 

        if self.index_target is None:
            self.calc_nearest_index(state) #calc_nearest_index 라는 내부함수를 이용해 현재 위치에서 가장 가까운 인덱스를 계산한다.

        # 목표 인덱스를 추적하기 위해 필요한 거리값 구하는 방법 (추적 거리)
        Lf = Config.kf * state.v + Config.Ld # (Ld 거리의 이득값 * 현재 차량 속도 + Ld 거리)  를 계산해서 Lf 에 저장

        # 목표 인덱스와 그 인덱스까지의 거리를 찾기 위한 반복문
        for ind in range(self.index_target, self.index_end + 1):
            # 내부 함수 calc_distance 를 이용해 현재 state와 목표 인덱스 간의 거리를 계산
            # 이때, 계산한 거리가 추적 거리 Lf 보다 크다면 해당 인덱스는 더 이상 목표 index가 아니다.
            if self.calc_distance(state, ind) > Lf:
                self.index_target = ind # 따라서 해당 인덱스를 index_target 로 반환한다. 
                return ind, Lf # 최종적으로는 해당 인덱스와 추적 거리 Lf를 반환한다. 

        # 만약 계산한 거리가 Lf 보다 작으면 해당 인덱스의 끝을 목표 인덱스로 설정해야 한다.
        self.index_target = self.index_end # 따라서, self.index_target를 self.index_end 로 설정하고,

        return self.index_end, Lf # 인덱스의 끝 지점과 추적 거리를 반환한다. 

    def calc_nearest_index(self, state):
        # Nodes 클래스의 add_state 함수에서 받아온 차량의 현재 상태 : state
        # 이 state를 이용해 현재 위치에서 가장 가까운 인덱스를 찾는 내부 함수로, 최근접 인덱스를 리턴한다. 

        # 현재 x 위치와 x 경로 지점 간의 좌표 차이를 계산해서 dx에 저장
        # 현재 y 위치와 y 경로 지점 간의 좌표 차이를 계산해서 dy에 저장
        dx = [state.x - x for x in self.x_coords] 
        dy = [state.y - y for y in self.y_coords]

        # numpy의 함수 np.hypot() : 두 배열 dx, dy 의 요소별 유클리드 거리를 계산한다. 
        # numpy의 함수 np.argmin() : 계산된 유클리드 거리 배열에서 최소값 인덱스를 반환한다. 
        nearest_index = np.argmin(np.hypot(dx, dy))
        self.index_target = nearest_index

    # 현재 위치와 목표 인덱스 간의 거리를 계산하는 내부 함수
    def calc_distance(self, state, index):
        # hypot() 함수를 이용해서 유클리드 거리를 계산
        # 즉, 현재 위치와 목표 인덱스 간의 직선 거리를 반환한다. 
        return math.hypot(state.x - self.x_coords[index], state.y - self.y_coords[index])


def pure_pursuit(state, ref_path, index_old):
    # pure pursuit 알고리즘 - 다음 변수들을 가지고 조향각을 계산한다.
    # state : 현재 정보
    # ref_path : 참조 경로 (x,y,yaw,curvature) 
    # 최종적으로 최적의 조향각을 반환한다.

    # find_index_target 내부 함수를 이용해 현재 차량의 위치를 기준으로 목표 인덱스와 추적 거리 Lf를 계산한다.
    ind, Lf = ref_path.find_index_target(state)  
    # 경로 추종을 하면서 전진하려면 항상 더 큰 인덱스 값을 취해야 하기 때문에 
    # ind 와 index_old 중에서 최대값을 목표 인덱스로 잡는다.   
    ind = max(ind, index_old) 

    # 참조 경로의 x 좌표에서 목표 인덱스를 가져온다. (target_x에 저장)
    target_x = ref_path.x_coords[ind]
    # 참조 경로의 y 좌표에서 목표 인덱스를 가져온다. (target_y에 저장)
    target_y = ref_path.y_coords[ind]

    # 조향각 구하는 과정
    # 1. target_y와 현재 y 위치 간의 각도, target_x와 현재 x 위치 간의 각도를 계산한다. 
    # 2. 계산한 각도에서 현재 yaw (진행 방향)을 뺀다. 
    #    즉, 현재 차량의 진행 방향을 기준으로 목표 지점과의 어느 정도 각도차가 있는 지 상대적으로 계산해서 alpha에 저장한다.
    # 3. (상수 2 * 휠베이스 * alpha에 sin 취한 값) 을 추적거리로 나눈다. 
    # 4. 위 계산을 통해 차량의 최적 조향각을 얻게 되고, 이를 delta에 저장한다. 
    #    이때, atan2() 함수는 arctan 삼각함수로 라디안 단위의 각도를 반환한다. 
    alpha = math.atan2(target_y - state.y, target_x - state.x) - state.yaw
    delta = math.atan2(2.0 * Config.WB * math.sin(alpha), Lf) # alpha 각도의 사인값을 이용해 차량의 회전 반경을 고려한다.

    # 최적의 조향각과 목표 인덱스를 반환한다.
    return delta, ind

# 경로를 생성하는 내부함수
def generate_path_sec(s):
    # 경로를 몇 개의 섹션으로 나눈다. 각 섹션에서 방향은 동일하다.
    # s : 목표 위치와 yaw 값
    # 최종적으로, 섹션을 반환한다. 

    # 최대 조향각 MAX_STEER의 tan 값을 계산하고 이를 휠베이스 값으로 나눠서 최대 곡률을 얻는다.
    # 그리고 그 값을 max_c에 저장
    max_curve = math.tan(Config.MAX_STEER) / Config.WB  

    # 경로 생성에 사용될 x,y,yaw,direct 값을 각각 담을 리스트를 선언한다.
    path_x, path_y, yaw, direct = [], [], [], []
    x_rec, y_rec, yaw_rec, direct_rec = [], [], [], []

    # 경로의 방향을 위한 flag를 설정한다. 
    direct_flag = 1.0

    # 경로의 끝 지점에 도달할 때까지 반복하는 for문을 선언
    for i in range(len(s) - 1):
        # x,y,yaw 값은 순서대로 리스트의 첫번째, 두번째, 세번째에 저장되고,
        # 경로의 시작점이 되는 x,y,yaw 값을 각각 s_x,s_y,s_yaw 에 저장한다.
        sx, sy, syaw = s[i][0], s[i][1], np.deg2rad(s[i][2])
        # 경로의 끝점이 되는 x,y,yaw 값을 각각 g_x,g_y,g_yaw 에 저장한다.
        ex, ey, eyaw = s[i + 1][0], s[i + 1][1], np.deg2rad(s[i + 1][2])

        # path_planning(rs)의 create_opt_path 함수에 경로의 시작점과 끝점, 최대 곡률을 넣고 최적 경로를 계산한다.
        # 계산된 최적 경로를 path_i 에 저장한다. 
        opt_path = rs.create_opt_path(sx, sy, syaw,
                                      ex, ey, eyaw, max_curve)

        ox = opt_path.final_x # 최적 경로의 x을 ox에 저장한다.
        oy = opt_path.final_y # 최적 경로의 y를 oy에 저장한다.
        oyaw = opt_path.final_yaw # 최적 경로의 yaw을 oyaw에 저장한다.
        odirect = opt_path.directions # 최적 경로의 direct를 odirect에 저장한다.

        # 최적 경로의 x 좌표 길이 만큼 반복하는 for문 선언 
        for j in range(len(ox)):
            # 앞서 설정한 섹션의 방향(flag)과 iderect의 섹션 방향이 같으면,
            if odirect[j] == direct_flag:
                # 현재 섹션의 x,y,yaw,direct 값을 각각 x_rec, y_rec, yaw_rec, direct_rec에 저장한다. (리스트 추가)
                x_rec.append(ox[j])
                y_rec.append(oy[j])
                yaw_rec.append(oyaw[j])
                direct_rec.append(odirect[j])
            else: 
                # 만약 이전 섹션이 없거나 섹션의 방향이 서로 다르면
                if len(x_rec) == 0 or direct_rec[0] != direct_flag:
                    direct_flag = odirect[j] # 새로운 섹션을 시작한다. 
                    continue
                
                # 진행되고 있는 섹션의 x,y,yaw,direct 값을 각각 path_x, path_y, yaw, direct에 저장한다. (리스트 추가)
                path_x.append(x_rec)
                path_y.append(y_rec)
                yaw.append(yaw_rec)
                direct.append(direct_rec)
                # 새로운 섹션으로 나아가기 위해서 이전 섹션의 마지막 점[-1] 을 첫번째 점으로 초기화한다.
                x_rec, y_rec, yaw_rec, direct_rec = [x_rec[-1]], [y_rec[-1]], [yaw_rec[-1]], [-direct_rec[-1]]

    # 모든 섹션의 x,y,yaw 값을 각각 path_x, path_y, yaw, direct에 저장하고,
    path_x.append(x_rec)
    path_y.append(y_rec)
    yaw.append(yaw_rec)
    direct.append(direct_rec)

    # 경로의 모든 x,y 값을 담을 리스트를 선언하고
    x_all, y_all = [], []

    # 모든 섹션의 x,y 값을 하나로 합쳐서 각각 x_all, y_all에 저장한다.
    for ox, oy in zip(path_x, path_y):
         x_all += ox
         y_all += oy
 
    # 최종적으로 경로의 모든 x,y 값까지 반환한다.
    return path_x, path_y, yaw, direct, x_all, y_all
 
# control.py 는 움직임 계획을 위한 알고리즘이고,  
# 구동은 parking.py를 통해 이루어지므로 Pure Pursuit 알고리즘의 main()은 삭제
