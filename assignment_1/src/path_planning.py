#!/usr/bin/env python
#-- coding:utf-8 --
####################################################################
# 프로그램이름 : parking.py
# 코드작성팀명 : 슈티어링
####################################################################

"""
Reeds Sheepp
author: huiming zhou
"""
# 경로 추종을 위한 알고리즘 중 Reeds Shepp 을 채택
# Reeds Shepp Curve 알고리즘은 자동 주차에서 경로 계획을 할 때 사용되는 알고리즘으로,
# motion_planning.py 에 rs로 import 되어 실행된다. 

# 필요한 라이브러리 설치
import time
import math
import numpy as np


# 파라미터 초기화
# 주행 경로 간격 - 단위 m
PATH_STEP_SIZE = 0.2
# 최대 주행 거리 설정 - 단위 m
PATH_MAX_LENGTH = 1000.0
# 파이 값 설정
PI = math.pi


# 경로 요소 클래스
class PATH:
    def __init__(self, path_lengths, path_types, total_path_length, final_x, final_y, final_yaw, directions):
        # 각 부분 경로의 길이 - float 형태
        # +: 전진, -: 후진
        self.path_lengths = path_lengths
        # 각 부분 경로의 유형 - 문자열 형태
        self.path_types = path_types
        # 전체 경로 길이 - float 형태
        self.total_path_length = total_path_length
        # 최종 x 위치 - 단위 m
        self.final_x = final_x
        # 최종 y 위치 - 단위 m
        self.final_y = final_y
        # 최종 yaw 각도 - 단위 rad
        self.final_yaw = final_yaw
        # 전진: 1, 후진: -1
        self.directions = directions

# create_opt_path : 시작점과 끝점을 이용하여 최적의 경로 생성
def create_opt_path(start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw, max_steering, path_step_size=PATH_STEP_SIZE):
    # 시작점과 끝점을 이용하여 가능한 모든 경로 생성
    all_paths = create_all_paths(start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw, max_steering, path_step_size=path_step_size)
    
    # 생성된 경로 중 가장 짧은 경로 선택 -> 시간이 최소로 걸리는 경로를 선택하기 위함
    # min_length_path 초기화 (모든 경로 중 0 번째 경로로 초기화 함)
    min_length_path = all_paths[0].total_path_length
    # min_index 초기화
    min_index = 0
    # 모든 경로 길이 비교 -> 가장 짧은 경로의 인덱스 찾기
    for i in range(len(all_paths)):
        # 만약 모든 경로 중 현재 경로의 길이가 현재 가장 짧은 경로의 길이보다 짧다면,
        if all_paths[i].total_path_length <= min_length_path:
            # min_length_path를 현재 경로의 길이로 저장
            # min_index를 현재 경로의 index로 저장
            min_length_path, min_index = all_paths[i].total_path_length, i

    # 가장 짧은 경로의 인덱스를 이용하여 모든 경로 중 최적의 경로 반환
    return all_paths[min_index]


# create_all_paths : 전역 경로(global_path) 각각에 대해 지역 경로를 생성하고, 그 결과를 전역 좌표계로 변환하여 최종 경로 데이터를 업데이트
def create_all_paths(start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw, max_steering, path_step_size=PATH_STEP_SIZE):

    # 시작점과 끝점을 각각 리스트로 저장
    start_point = [start_x, start_y, start_yaw]
    goal_point = [goal_x, goal_y, goal_yaw]

    # 시작점, 끝점, 최대 회전 반경을 이용하여 전역 경로 생성
    global_path = generate_path(start_point, goal_point, max_steering)

    # generate_local_course 함수를 사용하여 각 경로(path)에 대한 지역 경로 생성
    for path in global_path:
        local_x, local_y, local_yaw, local_directions = \
            generate_local_course(path.total_path_length, path.path_lengths,
                                  path.path_types, max_steering, path_step_size * max_steering)

        # 전역 좌표계로 변환
        # x 좌표 변환: 시작점의 yaw 각도(start_point[2])를 사용하여 각 지역 x, y 좌표를 전역 좌표계로 변환.이를 위해 회전 행렬을 적용하고, 시작점의 x 좌표(start_point[0])를 더함.
        # y 좌표 변환: x좌표와 동일하게 회전 행렬을 적용하고, 시작점의 y 좌표(start_point[1])를 더함
        # yaw 각도 변환: 지역 경로에서 계산된 각 방향각에 시작점의 yaw 각도를 더하고, 결과 각도를 적절한 범위(-π ~ π)로 조정(pi_2_pi 함수 사용).
        path.final_x = [math.cos(-start_point[2]) * s_x + math.sin(-start_point[2]) * s_y + start_point[0] for (s_x, s_y) in zip(local_x, local_y)]
        path.final_y = [-math.sin(-start_point[2]) * s_x + math.cos(-start_point[2]) * s_y + start_point[1] for (s_x, s_y) in zip(local_x, local_y)]
        path.final_yaw = [pi_2_pi(s_yaw + start_point[2]) for s_yaw in local_yaw]
        # 전진, 후진 방향을 저장한다.
        path.directions = local_directions
        # 경로의 길이를 저장한다.
        path.path_lengths = [l / max_steering for l in path.path_lengths]
        path.total_path_length = path.total_path_length / max_steering
    # 생성한 전역 경로 반환
    return global_path


# create_local_path : 모든 경로, 길이, 유형을 이용하여 지역 경로 생성
def create_local_path(paths, path_lengths, path_types):
    # 경로 요소 클래스 생성
    path = PATH([], [], 0.0, [], [], [], [])
    # 경로 요소 클래스에 경로, 길이, 유형 저장
    path.path_types = path_types
    path.path_lengths = path_lengths

    # 생성된 경로가 이미 존재하는 경로인지 확인
    for path_exist in paths:
        if path_exist.path_types == path.path_types:
            # 경로가 이미 존재하는 경로라면 경로를 반환
            if sum([x - y for x, y in zip(path_exist.path_lengths, path.path_lengths)]) <= 0.01:
                return paths
    # 경로가 존재하지 않는다면 경로 추가
    path.total_path_length = sum([abs(i) for i in path_lengths])
    # 만약 경로의 길이가 최대 길이를 넘어간다면 경로 반환
    if path.total_path_length >= PATH_MAX_LENGTH:
        return paths
    
    # 길이가 0.01 이상이라면 경로 추가
    assert path.total_path_length >= 0.01
    paths.append(path)
    # 경로 반환
    return paths


def LeftStraightLeft(x, y, start2goal_dir):
    # LeftStraightLeft 유형의 Reeds-Shepp 경로 생성
    # LeftStraightLeft 유형은 시작점에서 시작 방향으로 왼쪽으로 회전하고, 직진한 후 다시 왼쪽으로 회전하여 끝점에 도달하는 경로
    # 시작점 (x, y)와 시작점에서 끝점까지의 방향인 start2goal_dir을 사용
    # 경로 생성에 필요한 중간 변수 계산
    # 중간 변수 u, t를 계산
    # t가 0 이상인 경우에만 경로 생성
    u, t = get_polar_coord(x - math.sin(start2goal_dir), y - 1.0 + math.cos(start2goal_dir))

    # 중간 변수 v를 계산
    # v가 0 이상인 경우에만 경로를 생성한다.
    # 경로 생성이 불가능한 경우 False 반환
    # 가능한 경우 True와 경로 생성에 필요한 변수들을 반환
    if t >= 0.0:
        v = regulate_theta(start2goal_dir - t)
        if v >= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0

def LeftStraightRight(x, y, start2goal_dir):
    # LeftStraightRight 유형의 Reeds-Shepp 경로 생성
    # LeftStraightRight 유형은 시작점에서 시작 방향으로 왼쪽으로 회전하고, 직진한 후 다시 오른쪽으로 회전하여 끝점에 도달하는 경로
    # 시작점 (x, y)와 시작점에서 끝점까지의 방향인 start2goal_dir을 사용
    # 경로 생성에 필요한 중간 변수 계산
    u1, t1 = get_polar_coord(x + math.sin(start2goal_dir), y - 1.0 - math.cos(start2goal_dir))
    u1 = u1 ** 2
    # 중간 변수 u1, t1를 계산
    # u1이 4 이상인 경우에만 경로를 생성
    # 중간 변수 u를 계산하고
    # u가 0 이상인 경우에만 경로를 생성
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        # 중간 변수 theta를 계산
        theta = math.atan2(2.0, u)
        # 중간 변수 t를 계산
        # t가 0 이상인 경우에만 경로 생성
        t = regulate_theta(t1 + theta)
        # 중간 변수 v를 계산
        # v가 0 이상인 경우에만 경로 생성
        v = regulate_theta(t - start2goal_dir)
        
        # 경로 생성이 불가능한 경우 False 반환
        # 가능한 경우 True와 경로 생성에 필요한 변수 반환
        if t >= 0.0 and v >= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0

def LeftRightLeft(x, y, start2goal_dir):
    # LeftRightLeft 유형의 Reeds-Shepp 경로 생성
    # LeftRightLeft 유형은 시작점에서 시작 방향으로 왼쪽으로 회전하고, 오른쪽으로 회전한 후 다시 왼쪽으로 회전하여 끝점에 도달하는 경로
    # 시작점 (x, y)와 시작점에서 끝점까지의 방향인 start2goal_dir을 사용
    # 경로 생성에 필요한 중간 변수들을 계산
    u1, t1 = get_polar_coord(x - math.sin(start2goal_dir), y - 1.0 + math.cos(start2goal_dir))
    # 중간 변수 u1, t1를 계산
    # u1이 4 이하인 경우에만 경로 생성
    if u1 <= 4.0:
        u = -2.0 * math.asin(0.25 * u1)
        t = regulate_theta(t1 + 0.5 * u + PI)
        v = regulate_theta(start2goal_dir - t + u)

        if t >= 0.0 and u <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0



def StraightCurveStraight(x, y, start2goal_dir, paths):
    # StraightCurveStraight 유형의 Reeds-Shepp 경로 생성
    # StraightCurveStraight 유형은 시작점에서 시작 방향으로 직진하고, 회전한 후 다시 직진하여 끝점에 도달하는 경로
    # 시작점 (x, y)와 시작점에서 끝점까지의 방향인 start2goal_dir을 사용
    # StraightLeftStraight 함수를 호출하여 경로 생성 가능 여부 확인하고 경로 리스트 업데이트

    # x, y, start2goal_dir를 사용하여 StraightLeftStraight 경로 계산
    flag, t, u, v = StraightLeftStraight(x, y, start2goal_dir)
    # 계산된 경로가 유효하면 paths 리스트에 추가
    if flag:
        paths = create_local_path(paths, [t, u, v], ["S", "WB", "S"])

    # y를 반전시킨 채로 동일한 계산 수행하여 또 다른 가능한 경로를 검토
    flag, t, u, v = StraightLeftStraight(x, -y, -start2goal_dir)
    # 계산된 경로가 유효하면 paths 리스트에 추가
    if flag:
        paths = create_local_path(paths, [t, u, v], ["S", "R", "S"])

    # 수정된 paths 리스트 반환
    return paths



def StraightLeftStraight(x, y, start2goal_dir):
    # StraightLeftStraight 유형의 Reeds-Shepp 경로 생성
    # 시작점에서 시작 방향으로 직진, 왼쪽으로 회전, 다시 직진하여 끝점에 도달하는 경로
    # 시작점 (x, y)와 시작점에서 끝점까지의 방향인 start2goal_dir을 사용

    # start2goal_dir을 적절한 범위(-π ~ π)로 조정
    start2goal_dir = regulate_theta(start2goal_dir)
    
    # y가 양수이고, start2goal_dir이 0과 0.99π 사이일 때 경로 계산 실행
    if y > 0.0 and 0.0 < start2goal_dir < PI * 0.99:
        xd = -y / math.tan(start2goal_dir) + x  # x 방향으로의 변화 계산
        t = xd - math.tan(start2goal_dir / 2.0)  # 첫 직진 구간의 길이 계산
        u = start2goal_dir  # 회전 각도
        v = math.sqrt((x - xd) ** 2 + y ** 2) - math.tan(start2goal_dir / 2.0)  # 두 번째 직진 구간의 길이 계산
        return True, t, u, v  # 계산된 경로의 세그먼트 길이와 각도 반환

    # y가 음수이고, start2goal_dir이 0과 0.99π 사이일 때 경로 계산 실행
    elif y < 0.0 and 0.0 < start2goal_dir < PI * 0.99:
        xd = -y / math.tan(start2goal_dir) + x  # x 방향으로의 변화 계산
        t = xd - math.tan(start2goal_dir / 2.0)  # 첫 직진 구간의 길이 계산
        u = start2goal_dir  # 회전 각도
        v = -math.sqrt((x - xd) ** 2 + y ** 2) - math.tan(start2goal_dir / 2.0)  # 두 번째 직진 구간의 길이 계산, 음수 표시로 후진 의미
        return True, t, u, v  # 계산된 경로의 세그먼트 길이와 각도 반환

    # 계산에 실패하거나 조건을 만족하지 못하는 경우
    return False, 0.0, 0.0, 0.0  # 경로 생성 실패 반환




def CurveStraightCurve(x, y, start2goal_dir, paths):
    # CurveStraightCurve 유형의 Reeds-Shepp 경로 생성
    # CurveStraightCurve 유형은 시작점에서 회전하고, 직진한 후 다시 회전하여 끝점에 도달하는 경로
    # 시작점 (x, y)와 시작점에서 끝점까지의 방향인 start2goal_dir, 경로 리스트 paths 사용
    # LeftStraightLeft, LeftStraightRight 함수를 호출하여 경로 생성 가능 여부를 확인하고 경로 리스트 업데이트
    # ('S' for Straight, 'WB' for Wide Bend, 'R' for Right)
    
    # x, y, start2goal_dir를 사용하여 LeftStraightLeft 경로 계산
    flag, t, u, v = LeftStraightLeft(x, y, start2goal_dir)
    # 계산된 경로가 유효하면 paths 리스트에 추가
    if flag:
        paths = create_local_path(paths, [t, u, v], ["WB", "S", "WB"])

    # x를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftStraightLeft(-x, y, -start2goal_dir)
    # 계산된 경로가 유효하면 paths 리스트에 추가
    if flag:
        paths = create_local_path(paths, [-t, -u, -v], ["WB", "S", "WB"])

    # y를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftStraightLeft(x, -y, -start2goal_dir)
    # 계산된 경로가 유효하면 paths 리스트에 추가
    if flag:
        paths = create_local_path(paths, [t, u, v], ["R", "S", "R"])

    # x와 y를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftStraightLeft(-x, -y, start2goal_dir)
    # 계산된 경로가 유효하면 paths 리스트에 추가
    if flag:
        paths = create_local_path(paths, [-t, -u, -v], ["R", "S", "R"])

    # x, y, start2goal_dir를 사용하여 LeftStraightRight 경로 계산
    flag, t, u, v = LeftStraightRight(x, y, start2goal_dir)
    # 계산된 경로가 유효하면 paths 리스트에 추가
    if flag:
        paths = create_local_path(paths, [t, u, v], ["WB", "S", "R"])

    # x를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftStraightRight(-x, y, -start2goal_dir)
    # 계산된 경로가 유효하면 paths 리스트에 추가
    if flag:
        paths = create_local_path(paths, [-t, -u, -v], ["WB", "S", "R"])

    # y를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftStraightRight(x, -y, -start2goal_dir)
    # 계산된 경로가 유효하면 paths 리스트에 추가
    if flag:
        paths = create_local_path(paths, [t, u, v], ["R", "S", "WB"])

    # x와 y를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftStraightRight(-x, -y, start2goal_dir)
    # 계산된 경로가 유효하면 paths 리스트에 추가
    if flag:
        paths = create_local_path(paths, [-t, -u, -v], ["R", "S", "WB"])

    # 모든 가능한 경로를 계산하고 추가한 후 최종 경로 리스트 반환
    return paths



def CurveCurveCurve(x, y, start2goal_dir, paths):
    # CurveCurveCurve 유형의 Reeds-Shepp 경로 생성
    # CurveCurveCurve 유형은 시작점에서 회전하고, 회전한 후 다시 회전하여 끝점에 도달하는 경로
    # 시작점 (x, y)와 시작점에서 끝점까지의 방향인 start2goal_dir, 경로 리스트 paths 사용
    # LeftRightLeft 함수를 호출하여 경로 생성 가능 여부를 확인하고 경로 리스트 업데이트
    # ('S' for Straight, 'WB' for Wide Bend, 'R' for Right)
    
    # x, y, start2goal_dir를 사용하여 LeftRightLeft 경로 계산
    flag, t, u, v = LeftRightLeft(x, y, start2goal_dir)
    # 계산된 경로가 유효하면 paths 리스트에 추가
    if flag:
        paths = create_local_path(paths, [t, u, v], ["WB", "R", "WB"])

    # x를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightLeft(-x, y, -start2goal_dir)
    if flag:
        paths = create_local_path(paths, [-t, -u, -v], ["WB", "R", "WB"])

    # y를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightLeft(x, -y, -start2goal_dir)
    if flag:
        paths = create_local_path(paths, [t, u, v], ["R", "WB", "R"])

    # x와 y를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightLeft(-x, -y, start2goal_dir)
    if flag:
        paths = create_local_path(paths, [-t, -u, -v], ["R", "WB", "R"])

    # 시작점을 회전 변환하여 새로운 좌표계로 계산
    xb = x * math.cos(start2goal_dir) + y * math.sin(start2goal_dir)
    yb = x * math.sin(start2goal_dir) - y * math.cos(start2goal_dir)

    # 변환된 좌표에서 LeftRightLeft 경로 계산
    flag, t, u, v = LeftRightLeft(xb, yb, start2goal_dir)
    if flag:
        paths = create_local_path(paths, [v, u, t], ["WB", "R", "WB"])

    # 변환된 x 좌표를 반전시키고 계산
    flag, t, u, v = LeftRightLeft(-xb, yb, -start2goal_dir)
    if flag:
        paths = create_local_path(paths, [-v, -u, -t], ["WB", "R", "WB"])

    # 변환된 y 좌표를 반전시키고 계산
    flag, t, u, v = LeftRightLeft(xb, -yb, -start2goal_dir)
    if flag:
        paths = create_local_path(paths, [v, u, t], ["R", "WB", "R"])

    # 변환된 x와 y 좌표를 모두 반전시키고 계산
    flag, t, u, v = LeftRightLeft(-xb, -yb, start2goal_dir)
    if flag:
        paths = create_local_path(paths, [-v, -u, -t], ["R", "WB", "R"])

    # 모든 가능한 경로를 계산하고 추가한 후 최종 경로 리스트 반환
    return paths



def calc_tauOmega(u, v, xi, eta, start2goal_dir):
    # tau와 omega 계산: 두 각도 u, v와 좌표 xi, eta를 이용하여 회전 각도 tau와 omega를 계산
    # u, v: 각각 시작점과 끝점의 각도
    # xi, eta: 각각 시작점과 끝점의 x, y 좌표
    # start2goal_dir: 시작점에서 끝점을 향하는 방향

    # u와 v의 차이를 적절한 범위(-π ~ π)로 조정
    delta = regulate_theta(u - v)
    
    # 회전 경로의 계산을 위한 중간 계수 A와 B 계산
    A = math.sin(u) - math.sin(delta)  # sin(u)와 sin(delta)의 차를 계산하여 A 생성
    B = math.cos(u) - math.cos(delta) - 1.0  # cos(u)와 cos(delta)의 차를 1.0 감소시켜 B 생성

    # 타우(tau) 계산을 위한 atan2 함수 사용
    t1 = math.atan2(eta * A - xi * B, xi * A + eta * B)  # eta와 xi를 이용하여 atan2 함수로 각도 t1 계산

    # t1을 이용하여 최종 타우 각도 결정을 위한 추가 계산
    t2 = 2.0 * (math.cos(delta) - math.cos(v) - math.cos(u)) + 3.0  # t2 계산으로 조건 검사 준비

    # t2의 값에 따라 tau 각도를 조정
    if t2 < 0:
        tau = regulate_theta(t1 + PI)  # t2가 음수인 경우, PI를 더해준 후 조정
    else:
        tau = regulate_theta(t1)  # t2가 음수가 아닌 경우, t1을 그대로 조정

    # 최종 omega 값 계산: 시작 각도 u에서 tau를 더하고, v를 빼고, start2goal_dir을 고려하여 omega 계산
    omega = regulate_theta(tau - u + v - start2goal_dir)

    # 계산된 tau와 omega 반환
    return tau, omega



def LeftRightLeftRight_Negative(x, y, start2goal_dir):
    # LeftRightLeftRight_Negative 유형의 Reeds-Shepp 경로 생성 (backward, 후진)
    # 시작점 (x, y)와 시작점에서 끝점까지의 방향인 start2goal_dir을 사용하여 경로 생성
    # 경로 생성에 필요한 중간 변수들을 계산하고, 경로 생성 가능 여부를 확인

    # xi와 eta를 계산하여 시작점에서 회전의 중심점까지의 오프셋 계산
    xi = x + math.sin(start2goal_dir)  # x 위치에 시작 방향 각도의 사인 값을 더해 xi를 계산
    eta = y - 1.0 - math.cos(start2goal_dir)  # y 위치에서 1을 빼고 시작 방향 각도의 코사인 값을 빼서 eta 계산

    # rho를 계산하여 시작점에서 끝점까지의 근사 반경을 정의
    rho = 0.25 * (2.0 + math.sqrt(xi * xi + eta * eta))  # rho는 xi와 eta를 이용하여 계산된 유클리디안 거리에 기반한 값

    # rho 값이 1.0 이하인 경우에만 경로 계산 실행
    if rho <= 1.0:
        u = math.acos(rho)  # acos 함수를 사용하여 rho에 대한 각도 u 계산
        t, v = calc_tauOmega(u, -u, xi, eta, start2goal_dir)  # calc_tauOmega 함수를 호출하여 t와 v 계산
        # t와 v의 값이 특정 조건을 만족하는 경우 (t가 0 이상, v가 0 이하)에만 경로 생성을 승인
        if t >= 0.0 and v <= 0.0:
            return True, t, u, v  # 계산된 값 t, u, v를 반환하고 경로 생성 승인

    # 위 조건을 만족하지 않을 경우 경로 생성 실패
    return False, 0.0, 0.0, 0.0  # 경로 생성 실패를 나타내는 값 반환



def LeftRightLeftRight_Positive(x, y, start2goal_dir):
    # LeftRightLeftRight_Positive 유형의 Reeds-Shepp 경로 생성 (forward, 직진)
    # 시작점 (x, y)와 시작점에서 끝점까지의 방향인 start2goal_dir을 사용하여 경로 생성
    # 경로 생성에 필요한 중간 변수들을 계산하고, 경로 생성 가능 여부를 확인

    # xi와 eta 계산: 시작점에서 회전의 중심점까지의 오프셋 계산
    xi = x + math.sin(start2goal_dir)  # x 위치에 시작 방향 각도의 사인 값을 더해 xi 계산
    eta = y - 1.0 - math.cos(start2goal_dir)  # y 위치에서 1을 빼고 시작 방향 각도의 코사인 값을 빼서 eta 계산

    # rho 계산: xi와 eta를 사용하여 시작점과 끝점 사이의 조정된 거리 계산
    rho = (20.0 - xi * xi - eta * eta) / 16.0  # rho 값은 수정된 식을 사용하여 계산

    # rho의 값이 0.0과 1.0 사이에 있는 경우에만 경로 계산 수행
    if 0.0 <= rho <= 1.0:
        u = -math.acos(rho)  # acos 함수를 사용하여 rho에 대한 각도 u를 계산, 부호 반전으로 forward 방향 설정
        # u 값이 -π/2 이상인 경우에만 계속 진행
        if u >= -0.5 * PI:
            t, v = calc_tauOmega(u, u, xi, eta, start2goal_dir)  # calc_tauOmega 함수를 호출하여 t와 v 계산
            # t와 v의 값이 모두 0 이상인 경우에만 경로 생성을 승인
            if t >= 0.0 and v >= 0.0:
                return True, t, u, v  # 계산된 값 t, u, v를 반환하고 경로 생성 승인

    # 위 조건을 만족하지 않을 경우 경로 생성 실패
    return False, 0.0, 0.0, 0.0  # 경로 생성 실패를 나타내는 값 반환



def CurveCurveCurveCurve(x, y, start2goal_dir, paths):
    # CurveCurveCurveCurve 유형의 Reeds-Shepp 경로 생성
    # 시작점 (x, y)와 시작점에서 끝점까지의 방향인 start2goal_dir을 사용하여 경로 생성
    # 각 유형별로 중간 변수들을 계산하여 경로 생성 가능 여부를 확인
    # ('S' for Straight, 'WB' for Wide Bend, 'R' for Right)
    
    # x, y, start2goal_dir를 사용하여 LeftRightLeftRight_Negative 경로 계산
    flag, t, u, v = LeftRightLeftRight_Negative(x, y, start2goal_dir)
    # 계산된 경로가 유효하면 paths 리스트에 추가
    if flag:
        paths = create_local_path(paths, [t, u, -u, v], ["WB", "R", "WB", "R"])

    # x를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightLeftRight_Negative(-x, y, -start2goal_dir)
    if flag:
        paths = create_local_path(paths, [-t, -u, u, -v], ["WB", "R", "WB", "R"])

    # y를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightLeftRight_Negative(x, -y, -start2goal_dir)
    if flag:
        paths = create_local_path(paths, [t, u, -u, v], ["R", "WB", "R", "WB"])

    # x와 y를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightLeftRight_Negative(-x, -y, start2goal_dir)
    if flag:
        paths = create_local_path(paths, [-t, -u, u, -v], ["R", "WB", "R", "WB"])

    # x, y, start2goal_dir를 사용하여 LeftRightLeftRight_Positive 경로 계산
    flag, t, u, v = LeftRightLeftRight_Positive(x, y, start2goal_dir)
    if flag:
        paths = create_local_path(paths, [t, u, u, v], ["WB", "R", "WB", "R"])

    # x를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightLeftRight_Positive(-x, y, -start2goal_dir)
    if flag:
        paths = create_local_path(paths, [-t, -u, -u, -v], ["WB", "R", "WB", "R"])

    # y를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightLeftRight_Positive(x, -y, -start2goal_dir)
    if flag:
        paths = create_local_path(paths, [t, u, u, v], ["R", "WB", "R", "WB"])

    # x와 y를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightLeftRight_Positive(-x, -y, start2goal_dir)
    if flag:
        paths = create_local_path(paths, [-t, -u, -u, -v], ["R", "WB", "R", "WB"])

    # 모든 가능한 경로를 계산하고 추가한 후 최종 경로 리스트 반환
    return paths



def LeftRightStraightRight(x, y, start2goal_dir):
    # LeftRightStraightRight 유형의 Reeds-Shepp 경로 생성
    # 시작점 (x, y)와 시작점에서 끝점까지의 방향인 start2goal_dir을 사용
    # 중간 변수들을 계산하고, 경로 생성 가능 여부를 확인

    # 시작점에서 시작 방향을 고려하여 변경된 x, y 좌표 계산
    xi = x + math.sin(start2goal_dir)  # 변경된 x 좌표
    eta = y - 1.0 - math.cos(start2goal_dir)  # 변경된 y 좌표

    # 변경된 좌표를 극좌표로 변환
    rho, theta = get_polar_coord(-eta, xi)  # xi, -eta를 사용하여 극좌표 rho, theta 계산

    # rho가 2.0 이상인 경우 경로 계산을 시도
    if rho >= 2.0:
        t = theta  # 각도 theta를 t로 설정
        u = 2.0 - rho  # rho로부터 u 계산
        v = regulate_theta(t + 0.5 * PI - start2goal_dir)  # 전체 경로 방향 조정을 위한 v 계산
        # 모든 계산된 각도가 조건을 만족하는 경우
        if t >= 0.0 and u <= 0.0 and v <= 0.0:
            return True, t, u, v  # 조건을 만족하면 경로 생성 승인 및 값 반환

    # 위 조건을 만족하지 않을 경우 경로 생성 실패
    return False, 0.0, 0.0, 0.0



def LeftRightStraightLeft(x, y, start2goal_dir):
    # LeftRightStraightLeft 유형의 Reeds-Shepp 경로 생성
    # 시작점 (x, y)와 시작점에서 끝점까지의 방향인 start2goal_dir 사용
    # 중간 변수들을 계산하고, 경로 생성 가능 여부를 확인

    # 시작점에서 시작 방향을 고려하여 변경된 x, y 좌표 계산
    xi = x - math.sin(start2goal_dir)  # 변경된 x 좌표
    eta = y - 1.0 + math.cos(start2goal_dir)  # 변경된 y 좌표

    # 변경된 좌표를 극좌표로 변환
    rho, theta = get_polar_coord(xi, eta)  # xi, eta를 사용하여 극좌표 rho, theta 계산

    # rho가 2.0 이상인 경우 경로 계산을 시도
    if rho >= 2.0:
        r = math.sqrt(rho * rho - 4.0)  # rho로부터 r 계산
        u = 2.0 - r  # r을 사용하여 u 계산
        t = regulate_theta(theta + math.atan2(r, -2.0))  # 전체 각도 조정을 위한 t 계산
        v = regulate_theta(start2goal_dir - 0.5 * PI - t)  # 전체 경로 방향 조정을 위한 v 계산
        # 모든 계산된 각도가 조건을 만족하는 경우
        if t >= 0.0 and u <= 0.0 and v <= 0.0:
            return True, t, u, v  # 조건을 만족하면 경로 생성 승인 및 값 반환

    # 위 조건을 만족하지 않을 경우 경로 생성 실패
    return False, 0.0, 0.0, 0.0



def CurveCurveStraightCurve(x, y, start2goal_dir, paths):
    # CurveCurveStraightCurve 유형의 Reeds-Shepp 경로 생성
    # 시작점 (x, y)와 시작점에서 끝점까지의 방향인 start2goal_dir 사용
    # 중간 변수들을 계산하고, 경로 생성 가능 여부를 확인
    # ('S' for Straight, 'WB' for Wide Bend, 'R' for Right)

    # x, y, start2goal_dir를 사용하여 LeftRightStraightLeft 경로 계산
    flag, t, u, v = LeftRightStraightLeft(x, y, start2goal_dir)
    # 계산된 경로가 유효하면 paths 리스트에 추가
    if flag:
        paths = create_local_path(paths, [t, -0.5 * PI, u, v], ["WB", "R", "S", "WB"])

    # x를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightStraightLeft(-x, y, -start2goal_dir)
    if flag:
        paths = create_local_path(paths, [-t, 0.5 * PI, -u, -v], ["WB", "R", "S", "WB"])

    # y를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightStraightLeft(x, -y, -start2goal_dir)
    if flag:
        paths = create_local_path(paths, [t, -0.5 * PI, u, v], ["R", "WB", "S", "R"])

    # x와 y를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightStraightLeft(-x, -y, start2goal_dir)
    if flag:
        paths = create_local_path(paths, [-t, 0.5 * PI, -u, -v], ["R", "WB", "S", "R"])

    # x, y, start2goal_dir를 사용하여 LeftRightStraightRight 경로 계산
    flag, t, u, v = LeftRightStraightRight(x, y, start2goal_dir)
    if flag:
        paths = create_local_path(paths, [t, -0.5 * PI, u, v], ["WB", "R", "S", "R"])

    # x를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightStraightRight(-x, y, -start2goal_dir)
    if flag:
        paths = create_local_path(paths, [-t, 0.5 * PI, -u, -v], ["WB", "R", "S", "R"])

    # y를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightStraightRight(x, -y, -start2goal_dir)
    if flag:
        paths = create_local_path(paths, [t, -0.5 * PI, u, v], ["R", "WB", "S", "WB"])

    # x와 y를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightStraightRight(-x, -y, start2goal_dir)
    if flag:
        paths = create_local_path(paths, [-t, 0.5 * PI, -u, -v], ["R", "WB", "S", "WB"])

    # 후진 계산을 위한 좌표 변환
    xb = x * math.cos(start2goal_dir) + y * math.sin(start2goal_dir)
    yb = x * math.sin(start2goal_dir) - y * math.cos(start2goal_dir)

    # 변환된 좌표를 사용하여 LeftRightStraightLeft 경로 계산
    flag, t, u, v = LeftRightStraightLeft(xb, yb, start2goal_dir)
    if flag:
        paths = create_local_path(paths, [v, u, -0.5 * PI, t], ["WB", "S", "R", "WB"])

    # xb를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightStraightLeft(-xb, yb, -start2goal_dir)
    if flag:
        paths = create_local_path(paths, [-v, -u, 0.5 * PI, -t], ["WB", "S", "R", "WB"])

    # yb를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightStraightLeft(xb, -yb, -start2goal_dir)
    if flag:
        paths = create_local_path(paths, [v, u, -0.5 * PI, t], ["R", "S", "WB", "R"])

    # xb와 yb를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightStraightLeft(-xb, -yb, start2goal_dir)
    if flag:
        paths = create_local_path(paths, [-v, -u, 0.5 * PI, -t], ["R", "S", "WB", "R"])

    # xb, yb를 사용하여 LeftRightStraightRight 경로 계산
    flag, t, u, v = LeftRightStraightRight(xb, yb, start2goal_dir)
    if flag:
        paths = create_local_path(paths, [v, u, -0.5 * PI, t], ["R", "S", "R", "WB"])

    # xb를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightStraightRight(-xb, yb, -start2goal_dir)
    if flag:
        paths = create_local_path(paths, [-v, -u, 0.5 * PI, -t], ["R", "S", "R", "WB"])

    # yb를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightStraightRight(xb, -yb, -start2goal_dir)
    if flag:
        paths = create_local_path(paths, [v, u, -0.5 * PI, t], ["WB", "S", "WB", "R"])

    # xb와 yb를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightStraightRight(-xb, -yb, start2goal_dir)
    if flag:
        paths = create_local_path(paths, [-v, -u, 0.5 * PI, -t], ["WB", "S", "WB", "R"])

    # 모든 가능한 경로를 계산하고 추가한 후 최종 경로 리스트 반환
    return paths



def LeftRightStraightLeftRight(x, y, start2goal_dir):
    # LeftRightStraightLeftRight 유형의 Reeds-Shepp 경로 생성
    # 시작점 (x, y)와 시작점에서 끝점까지의 방향인 start2goal_dir 사용
    # 중간 변수들을 계산하고, 경로 생성 가능 여부 확인
    # 문헌의 오류를 지적하며 정정한 공식을 사용 (8.11 공식)

    # 시작점에서 변형된 좌표 xi, eta 계산
    xi = x + math.sin(start2goal_dir)  # x 좌표에 시작점에서 끝점까지의 방향의 사인 값을 더함
    eta = y - 1.0 - math.cos(start2goal_dir)  # y 좌표에서 1을 뺀 후 시작점에서 끝점까지의 방향의 코사인 값을 뺌

    # 변형된 좌표를 극좌표로 변환하여 rho와 theta 계산
    rho, theta = get_polar_coord(xi, eta)  # xi와 eta를 사용하여 극좌표로 변환

    # rho 값이 2.0 이상인 경우에만 경로 계산을 수행
    if rho >= 2.0:
        u = 4.0 - math.sqrt(rho * rho - 4.0)  # rho를 사용하여 u 계산, 여기서 u는 경로에서 회전의 정도를 결정

        # u 값이 0.0 이하인 경우에만 계산을 계속 진행
        if u <= 0.0:
            # t 계산, 특정 각도의 탄젠트를 사용하여 계산
            t = regulate_theta(math.atan2((4.0 - u) * xi - 2.0 * eta, -2.0 * xi + (u - 4.0) * eta))
            # v 계산, t에서 시작점에서 끝점까지의 방향을 뺀 후 적절한 범위로 조정
            v = regulate_theta(t - start2goal_dir)

            # t와 v가 모두 0.0 이상인 경우 경로 생성 승인
            if t >= 0.0 and v >= 0.0:
                return True, t, u, v  # 계산된 t, u, v 값을 반환하며 경로 생성 승인

    # 위의 조건을 만족하지 않는 경우, 경로 생성 실패
    return False, 0.0, 0.0, 0.0  # 경로 생성 실패를 나타내는 값 반환



def CurveCurveStraightCurveCurve(x, y, start2goal_dir, paths):
    # CurveCurveStraightCurveCurve 유형의 Reeds-Shepp 경로 생성
    # 시작점 (x, y)와 시작점에서 끝점까지의 방향인 start2goal_dir 사용
    # 중간 변수들을 계산하고, 경로 생성 가능 여부를 확인
    # ('S' for Straight, 'WB' for Wide Bend, 'R' for Right)

    # x, y, start2goal_dir를 사용하여 LeftRightStraightLeftRight 경로 계산
    flag, t, u, v = LeftRightStraightLeftRight(x, y, start2goal_dir)
    # 계산된 경로가 유효하면 paths 리스트에 추가
    if flag:
        paths = create_local_path(paths, [t, -0.5 * PI, u, -0.5 * PI, v], ["WB", "R", "S", "WB", "R"])

    # x를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightStraightLeftRight(-x, y, -start2goal_dir)
    if flag:
        paths = create_local_path(paths, [-t, 0.5 * PI, -u, 0.5 * PI, -v], ["WB", "R", "S", "WB", "R"])

    # y를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightStraightLeftRight(x, -y, -start2goal_dir)
    if flag:
        paths = create_local_path(paths, [t, -0.5 * PI, u, -0.5 * PI, v], ["R", "WB", "S", "R", "WB"])

    # x와 y를 반전시켜 동일한 계산 수행
    flag, t, u, v = LeftRightStraightLeftRight(-x, -y, start2goal_dir)
    if flag:
        paths = create_local_path(paths, [-t, 0.5 * PI, -u, 0.5 * PI, -v], ["R", "WB", "S", "R", "WB"])

    # 모든 가능한 경로를 계산하고 추가한 후 최종 경로 리스트 반환
    return paths


def generate_local_course(total_path_length, path_lengths, mode, max_steering, path_step_size):
    # 지역 경로를 생성하는 함수, 주어진 경로의 총 길이, 각 부분 경로의 길이, 경로 모드, 최대 조향각 및 스텝 크기를 기반으로 함

    # 경로 점의 총 개수를 계산
    point_num = int(total_path_length / path_step_size) + len(path_lengths) + 3

    # 경로의 x, y, yaw 좌표와 방향을 저장할 리스트를 초기화
    path_x = [0.0 for _ in range(point_num)]
    path_y = [0.0 for _ in range(point_num)]
    path_yaw = [0.0 for _ in range(point_num)]
    path_directions = [0 for _ in range(point_num)]
    idx = 1

    # 첫 번째 경로 세그먼트의 방향을 설정
    if path_lengths[0] > 0.0:
        path_directions[0] = 1  # 전진
    else:
        path_directions[0] = -1  # 후진

    # 첫 번째 경로 세그먼트의 스텝 크기를 설정
    if path_lengths[0] > 0.0:
        d = path_step_size  # 전진 스텝 크기
    else:
        d = -path_step_size  # 후진 스텝 크기

    ll = 0.0  # 현재 세그먼트의 남은 길이를 초기화

    # 각 경로 세그먼트에 대해 반복
    for m, l, i in zip(mode, path_lengths, range(len(mode))):
        # 각 경로 세그먼트의 스텝 크기를 결정
        if l > 0.0:
            d = path_step_size  # 전진
        else:
            d = -path_step_size  # 후진

        prev_x, prev_y, prev_yaw = path_x[idx], path_y[idx], path_yaw[idx]  # 이전 점의 좌표 및 방향

        idx -= 1  # 인덱스를 감소시켜 세그먼트의 시작점으로 설정
        if i >= 1 and (path_lengths[i - 1] * path_lengths[i]) > 0:
            pd = -d - ll  # 경로의 방향이 전환되지 않았을 경우, 기존 방향으로 스텝 계산
        else:
            pd = d - ll  # 경로의 방향이 전환되었을 경우, 새로운 방향으로 스텝 계산

        # 스텝별로 경로의 점 생성
        while abs(pd) <= abs(l):
            idx += 1
            path_x, path_y, path_yaw, path_directions = \
                interpolate(idx, pd, m, max_steering, prev_x, prev_y, prev_yaw, path_x, path_y, path_yaw, path_directions)
            pd += d

        # 세그먼트의 남은 길이를 계산
        ll = l - pd - d

        idx += 1
        path_x, path_y, path_yaw, path_directions = \
            interpolate(idx, l, m, max_steering, prev_x, prev_y, prev_yaw, path_x, path_y, path_yaw, path_directions)

    # 사용되지 않은 데이터 제거
    while len(path_x) >= 1 and path_x[-1] == 0.0:
        path_x.pop()
        path_y.pop()
        path_yaw.pop()
        path_directions.pop()

    # 경로의 x, y, yaw 좌표와 방향 리스트 반환
    return path_x, path_y, path_yaw, path_directions



def interpolate(idx, l, m, max_steering, prev_x, prev_y, prev_yaw, path_x, path_y, path_yaw, path_directions):
    # 선형 보간 함수를 정의하여 주어진 모드에 따라 경로 점을 생성
    # idx: 현재 인덱스
    # l: 보간할 거리
    # m: 모드 ('S' for Straight, 'WB' for Wide Bend, 'R' for Right)
    # max_steering: 최대 조향각
    # prev_x, prev_y, prev_yaw: 이전 경로 점의 x, y, yaw 값
    # path_x, path_y, path_yaw: 각 경로의 x, y, yaw 값을 저장할 리스트
    # path_directions: 각 경로 점의 방향을 저장할 리스트

    if m == "S":
        # 모드가 'S'(직선)인 경우 직선 경로 계산
        path_x[idx] = prev_x + l / max_steering * math.cos(prev_yaw)  # 새로운 x 좌표 계산
        path_y[idx] = prev_y + l / max_steering * math.sin(prev_yaw)  # 새로운 y 좌표 계산
        path_yaw[idx] = prev_yaw  # yaw 값은 변경 없음

    else:
        # 회전이 있는 경우, 회전 반경을 고려한 보간 계산
        ldx = math.sin(l) / max_steering  # x 방향의 변화량 계산
        if m == "WB":
            ldy = (1.0 - math.cos(l)) / max_steering  # 'WB'(Wide Bend)의 y 방향 변화량
        elif m == "R":
            ldy = (1.0 - math.cos(l)) / (-max_steering)  # 'R'(Right)의 y 방향 변화량

        # 글로벌 좌표계에서의 변화량 계산
        gdx = math.cos(-prev_yaw) * ldx + math.sin(-prev_yaw) * ldy  # x 좌표 변화량
        gdy = -math.sin(-prev_yaw) * ldx + math.cos(-prev_yaw) * ldy  # y 좌표 변화량
        path_x[idx] = prev_x + gdx  # 새로운 x 좌표
        path_y[idx] = prev_y + gdy  # 새로운 y 좌표

    # yaw 값 업데이트
    if m == "WB":
        path_yaw[idx] = prev_yaw + l  # 'WB' 시 yaw 값 증가
    elif m == "R":
        path_yaw[idx] = prev_yaw - l  # 'R' 시 yaw 값 감소

    # 경로 방향 설정
    if l > 0.0:
        path_directions[idx] = 1  # 전진
    else:
        path_directions[idx] = -1  # 후진

    # 변경된 경로의 x, y, yaw 값과 방향을 반환
    return path_x, path_y, path_yaw, path_directions


def generate_path(start_point, goal_point, max_steering):
    # 시작점부터 끝점까지의 경로를 생성하는 함수
    # start_point: 시작 위치 및 방향 정보를 포함하는 튜플 (x, y, yaw)
    # goal_point: 목표 위치 및 방향 정보를 포함하는 튜플 (x, y, yaw)
    # max_steering: 조향 최대 각도

    # 시작점과 끝점 사이의 x, y 좌표 차이 계산
    diff_x = goal_point[0] - start_point[0]
    diff_y = goal_point[1] - start_point[1]
    diff_yaw = goal_point[2] - start_point[2]

    # 시작점의 yaw 각도를 이용하여 회전 변환 계수 계산
    c = math.cos(start_point[2])
    s = math.sin(start_point[2])

    # 로컬 좌표계로 변환된 x, y 좌표 계산
    x = (c * diff_x + s * diff_y) * max_steering
    y = (-s * diff_x + c * diff_y) * max_steering

    # 경로 리스트 초기화
    paths = []

    # 다양한 경로 생성 함수 호출
    paths = StraightCurveStraight(x, y, diff_yaw, paths)
    paths = CurveStraightCurve(x, y, diff_yaw, paths)
    paths = CurveCurveCurve(x, y, diff_yaw, paths)
    paths = CurveCurveCurveCurve(x, y, diff_yaw, paths)
    paths = CurveCurveStraightCurve(x, y, diff_yaw, paths)
    paths = CurveCurveStraightCurveCurve(x, y, diff_yaw, paths)

    # 생성된 경로들 반환
    return paths



# utils
# 각도를 -π에서 π 범위로 정규화하는 함수
def pi_2_pi(theta):
    while theta > PI:
        theta -= 2.0 * PI  # θ가 π를 초과하면 2π를 빼준다

    while theta < -PI:
        theta += 2.0 * PI  # θ가 -π보다 작으면 2π를 더한다

    return theta  # 정규화된 각도 반환

# 주어진 (x, y) 좌표에 대해 극좌표 (r, θ)를 계산하고 반환하는 함수
def get_polar_coord(x, y):
    r = math.hypot(x, y)  # (x, y)의 유클리드 거리 계산
    theta = math.atan2(y, x)  # atan2 함수를 이용해 각도 계산

    return r, theta  # 극좌표 (r, θ) 반환


# 두 점 사이의 거리, 각도 계산
def get_polar_coord(x, y):
    """
    (x, y)점의 극좌표 (r, theta)를 반환한다.
    """
    r = math.hypot(x, y)  # (x, y)의 유클리디안 거리(반지름 r) 계산
    theta = math.atan2(y, x)  # (x, y)점에 대한 각도 theta 계산

    return r, theta  # 계산된 반지름 r과 각도 theta 반환


# 각도 세타를 -pi ~ pi 범위로 정규화
def regulate_theta(theta):
    """
    세타(theta)를 -pi <= theta < pi 범위로 정규화한다.
    """
    start2goal_dir = theta % (2.0 * PI)  # theta를 2π의 나머지 연산으로 일단 0 ~ 2π 범위로 조정

    if start2goal_dir < -PI:
        start2goal_dir += 2.0 * PI  # -π보다 작으면 2π를 더해 양의 범위로 조정
    if start2goal_dir > PI:
        start2goal_dir -= 2.0 * PI  # π보다 크면 2π를 빼 범위를 -π ~ π 사이로 조정

    return start2goal_dir  # 정규화된 각도 반환


def get_label(path):
    # 경로 레이블을 생성하는 함수
    label = ""  # 레이블을 저장할 빈 문자열 초기화

    # 경로 유형과 길이를 순회하면서 레이블을 구성
    for m, l in zip(path.path_types, path.path_lengths):
        label = label + m  # 경로 유형을 레이블에 추가
        if l > 0.0:
            label = label + "+"  # 길이가 양수일 경우 '+' 추가
        else:
            label = label + "-"  # 길이가 음수일 경우 '-' 추가

    return label  # 생성된 레이블 반환


def calc_curvature(x, y, yaw, directions):
    # 주어진 x, y 좌표 및 yaw, 이동 방향을 기반으로 곡률과 거리 계산
    c, ds = [], []  # 곡률과 거리를 저장할 리스트 초기화

    # 곡률 계산을 위한 루프
    for i in range(1, len(x) - 1):
        dxn = x[i] - x[i - 1]  # 이전 x 좌표와 현재 x 좌표의 차이
        dxp = x[i + 1] - x[i]  # 현재 x 좌표와 다음 x 좌표의 차이
        dyn = y[i] - y[i - 1]  # 이전 y 좌표와 현재 y 좌표의 차이
        dyp = y[i + 1] - y[i]  # 현재 y 좌표와 다음 y 좌표의 차이
        dn = math.hypot(dxn, dyn)  # 이전 점과 현재 점 사이의 거리
        dp = math.hypot(dxp, dyp)  # 현재 점과 다음 점 사이의 거리
        dx = 1.0 / (dn + dp) * (dp / dn * dxn + dn / dp * dxp)  # 가중 평균으로 x 방향의 미분 계산
        ddx = 2.0 / (dn + dp) * (dxp / dp - dxn / dn)  # x 방향의 이차 미분 계산
        dy = 1.0 / (dn + dp) * (dp / dn * dyn + dn / dp * dyp)  # 가중 평균으로 y 방향의 미분 계산
        ddy = 2.0 / (dn + dp) * (dyp / dp - dyn / dn)  # y 방향의 이차 미분 계산
        curvature = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2)  # 곡률 계산 공식 적용

        d = (dn + dp) / 2.0  # 현재 세그먼트의 평균 거리

        # 곡률이 NaN이면 0으로 설정
        if np.isnan(curvature):
            curvature = 0.0
        # 이동 방향이 음수면 곡률을 음수로 설정
        if directions[i] <= 0.0:
            curvature = -curvature
        # 이전 곡률과 현재 곡률을 리스트에 추가
        ds.append(d)
        c.append(curvature)
    # 마지막 곡률을 이전 곡률과 동일하게 설정
    ds.append(ds[-1])
    c.append(c[-1])

    return c, ds  # 계산된 곡률과 거리 리스트 반환


def check_path(start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw, max_steering):
    # 주어진 시작점과 목표점 사이의 경로가 유효한지 확인하는 함수
    # 주어진 시작점과 목표점에 대한 모든 경로를 생성
    paths = create_all_paths(start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw, max_steering)
    # 경로 리스트가 비어 있으면 검증 실패
    assert len(paths) >= 1
    
    # 생성된 모든 경로에 대해 유효성 검사
    for path in paths:
        # 경로의 시작점이 주어진 시작점과 거의 일치하는지 검사
        assert abs(path.final_x[0] - start_x) <= 0.01
        assert abs(path.final_y[0] - start_y) <= 0.01
        assert abs(path.final_yaw[0] - start_yaw) <= 0.01
        # 경로의 종료점이 주어진 목표점과 거의 일치하는지 검사
        assert abs(path.final_x[-1] - goal_x) <= 0.01
        assert abs(path.final_y[-1] - goal_y) <= 0.01
        assert abs(path.final_yaw[-1] - goal_yaw) <= 0.01

        # 경로의 각 세그먼트의 길이를 계산하여 검증
        d = [math.hypot(dx, dy)
             for dx, dy in zip(np.diff(path.final_x[0:len(path.final_x) - 1]),
                               np.diff(path.final_y[0:len(path.final_y) - 1]))]
        # 각 세그먼트의 길이가 주어진 스텝 크기와 거의 일치하는지 검사
        for i in range(len(d)):
            assert abs(d[i] - PATH_STEP_SIZE) <= 0.001


def main():
    # 메인 실행 함수
    start_x = 3.0  # 시작점 x 좌표 [m]
    start_y = 10.0  # 시작점 y 좌표 [m]
    start_yaw = np.deg2rad(40.0)  # 시작점 방향 [rad]
    end_x = 0.0  # 목표점 x 좌표 [m]
    end_y = 1.0  # 목표점 y 좌표 [m]
    end_yaw = np.deg2rad(0.0)  # 목표점 방향 [rad]
    max_curvature = 0.1  # 최대 곡률

    t0 = time.time()  # 실행 시간 측정 시작

    # 1000번의 경로 생성 시도
    for i in range(1000):
        _ = create_opt_path(start_x, start_y, start_yaw, end_x, end_y, end_yaw, max_curvature)

    t1 = time.time()  # 실행 시간 측정 종료
    print(t1 - t0)  # 실행 시간 출력

if __name__ == '__main__':
    main()
