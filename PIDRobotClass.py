import random
import numpy as np
import matplotlib.pyplot as plt
import math

k = 0.25  # 前视增益
Lfc = 0.25  # [m] 前视距离
Kp = 2.5  # 速度比例增益
dt = 0.05  # [s] 时间步长
WB = 1  # [m] 车辆轴距,老师所给(即wheellengh)

# Robot类由老师给的Matlab代码改写而来
class Robot(object):
    def __init__(self, length=20.0):
        """
        创建机器人对象并初始化位置和方向为0
        """
        self.x = 0.0 # x position
        self.y = 0.0 # y position
        self.orientation = 0.0 # heading（朝向）,弧度制
        self.length = length # wheel length(轮轴长度)
        self.steering_noise = 0.0 # steering noise(转向噪声)
        self.distance_noise = 0.0 # distance noise(距离噪声)
        self.steering_drift = 0.0 # steering drift(转向漂移)
        self.v = 0.0  # 初始化速度

    def set(self, x, y, orientation):
        """
        设置机器人的坐标
        """
        self.x = x
        self.y = y
        self.orientation = orientation % (2.0 * np.pi)

    def set_noise(self, steering_noise, distance_noise):
        """
        设置噪声参数
        """
        self.steering_noise = steering_noise
        self.distance_noise = distance_noise

    def set_steering_drift(self, drift):
        """
        设置系统性转向漂移参数
        """
        self.steering_drift = drift

    def move(self, steering, distance, tolerance=0.001, max_steering_angle=np.pi / 4.0):
        """
        控制机器人进行运动，根据给定的转向和距离
        steering = 前轮转向角度, 受max_steering_angle控制
        distance = 总行驶距离, 必须为非负数
        """
        if steering > max_steering_angle:
            steering = max_steering_angle
        if steering < -max_steering_angle:
            steering = -max_steering_angle
        if distance < 0.0:
            distance = 0.0

        # 应用噪声
        steering2 = random.gauss(steering, self.steering_noise)
        distance2 = random.gauss(distance, self.distance_noise)

        # 应用转向漂移
        steering2 += self.steering_drift

        # 执行运动
        turn = np.tan(steering2) * distance2 / self.length

        if abs(turn) < tolerance:
            # 近似直线运动
            self.x += distance2 * np.cos(self.orientation)
            self.y += distance2 * np.sin(self.orientation)
            self.orientation = (self.orientation + turn) % (2.0 * np.pi)
        else:
            # 近似自行车模型的运动
            radius = distance2 / turn
            cx = self.x - (np.sin(self.orientation) * radius)
            cy = self.y + (np.cos(self.orientation) * radius)
            self.orientation = (self.orientation + turn) % (2.0 * np.pi)
            self.x = cx + (np.sin(self.orientation) * radius)
            self.y = cy - (np.cos(self.orientation) * radius)

    def __repr__(self):
        return '[x=%.5f y=%.5f orient=%.5f v=%.5f]' % (self.x, self.y, self.orientation, self.v)

# 存储目标轨迹的坐标点(cx,cy)，并提供搜索最近目标点的方法
class TargetCourse:
    def __init__(self, cx, cy):
        """
        初始化目标轨迹对象
        """
        self.cx = cx # 轨迹点的x坐标数组
        self.cy = cy # 轨迹点的y坐标数组
        self.old_nearest_point_index = None # 上一个最近的轨迹点的索引

    def search_target_index(self, robot):
        """
        搜索机器人当前位置最近的目标点索引，并计算前视距离Lf
        """
        # 初始化最近目标点的索引，为后续搜索提供起点
        if self.old_nearest_point_index is None: # 第一次调用该函数，之前未保存过最近目标点的索引
            dx = [robot.x - icx for icx in self.cx]
            dy = [robot.y - icy for icy in self.cy]
            d = np.hypot(dx, dy) # 勾股定理
            ind = np.argmin(d) # 找到距离最小的点的索引，即最近目标点
            self.old_nearest_point_index = ind # 保存以便下次调用时使用
        else:
            ind = self.old_nearest_point_index
            distance_this_index = self.calc_distance(robot, self.cx[ind], self.cy[ind]) # 计算当前位置到目标点的距离
            while True:
                if (ind + 1) >= len(self.cx): # 到达轨迹最后一个点，跳出循环
                    break
                distance_next_index = self.calc_distance(robot, self.cx[ind + 1], self.cy[ind + 1]) # 计算当前位置到下一个目标点的距离
                if distance_this_index < distance_next_index:
                    break
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        Lf = k * robot.v + Lfc # 计算跟随误差
        while Lf > self.calc_distance(robot, self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break
            ind += 1

        return ind, Lf

    def calc_distance(self, robot, point_x, point_y):
        """
        计算机器人当前位置到指定点的距离
        """
        dx = robot.x - point_x
        dy = robot.y - point_y
        return np.hypot(dx, dy)


def pure_pursuit_steer_control(robot, trajectory, pind):
    """
    控制函数：纯追踪算法控制机器人转向，根据机器人当前位置和目标轨迹计算出需要的转向角度
    """
    ind, Lf = trajectory.search_target_index(robot)
    # 检查当前目标索引pind是否大于等于计算得到的目标点索引ind，如果是，则将目标索引设为pind，以确保机器人一直朝着当前目标点行驶
    if pind >= ind:
        ind = pind
    # 获取目标点的坐标 (tx, ty)
    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
    else:
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1

    alpha = math.atan2(ty - robot.y, tx - robot.x) - robot.orientation # 计算航向角alpha，即机器人当前位置指向目标点的角度差
    delta = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0) # 计算转向角度delta，根据给定公式、航向角alpha、车辆特性参数WB和跟随误差Lf计算得到

    return delta, ind


def proportional_control(target, current):
    """
    实现速度的比例控制，根据目标速度和当前速度计算加速度，使用PID的P(比例项)
    """
    a = Kp * (target - current)
    return a


def reference_trajectory(x):
    """
    参考轨迹：定义了一条sin形状的参考轨迹，与老师给的参考轨迹相同
    """
    omega = 0.30
    alt = 10
    return alt * np.sin(omega * x)

### 计算上升时间和调节时间

def find_rise_time(error_list, max_overshoot):
    """
    遍历误差列表，找到第一次达到最大超调的时间点
    """
    for i, error in enumerate(error_list):
        if error >= max_overshoot:
            return i * dt
    return None

def find_settling_time(error_list, steady_state_error, rise_time):
    """
    遍历误差列表，找到误差在最大超调值上下steady_state_error范围内的时间点，这通常表示系统已经进入稳态
    """
    settling_time = 0
    # print(f"steady_state_error:{steady_state_error}")
    for i, error in enumerate(error_list):
        # print(f"i:{i},error:{error}")
        if i * dt < rise_time:
            continue
        else:
            if -steady_state_error < error < steady_state_error:
                if settling_time == 0:
                    settling_time = i * dt
                else:
                    return settling_time
    return settling_time


def main():
    """
    主函数，执行路径跟踪仿真
    """
    cx = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    cy = reference_trajectory(cx)
    target_speed = 10.0 / 3.6  # [m/s] 机器人的目标速度，可调整
    T = 200.0  # 最大仿真时间，可调整

    robot = Robot(length=WB)
    robot.set(-2 * np.pi, reference_trajectory(-2 * np.pi), 0.0) # 设定起点
    target_course = TargetCourse(cx, cy)
    target_ind, _ = target_course.search_target_index(robot) # 返回最近点索引及对应的跟随误差

    time = 0.0
    x, y, orientation, v, t = [], [], [], [], []

    error_list = [] # 存储机器人路径坐标与参考轨迹坐标之差

    while T >= time:
        ai = proportional_control(target_speed, robot.v) # 加速度
        di, target_ind = pure_pursuit_steer_control(robot, target_course, target_ind) # 转向角度 目标点索引
        robot.move(di, robot.v * dt)  # 使用速度乘以时间步长计算距离
        robot.v += ai * dt  # 更新速度

        x.append(robot.x)
        y.append(robot.y)
        orientation.append(robot.orientation)
        v.append(robot.v)
        t.append(time)
        time += dt

        # 计算机器人当前位置与目标位置之间的差
        tmp_y = reference_trajectory(robot.x)
        error_list.append(robot.y - tmp_y)  # 将误差存储到列表中
        # error_list.append(abs(robot.y - tmp_y)) # 将误差存储到列表中

        # 检查是否接近轨迹末端
        if target_ind >= len(cx) - 1:
            break

        # 跟踪轨迹并动态显示
        plt.clf()
        plt.plot(cx, cy, "-r", label="reference")
        plt.plot(x, y, "ob", label="tracking")
        plt.xlim(-8, 8)
        plt.ylim(-11, 11)
        plt.title("Pure Pursuit Path Tracking Simulation")
        plt.pause(0.01)

        # 调试信息
        print(f"Time: {time:.2f}, Position: ({robot.x:.2f}, {robot.y:.2f}), Speed: {robot.v:.2f}, Steering: {di:.2f}")

    # 绘图
    plt.figure()
    plt.plot(cx, cy, "-r", label="reference")
    plt.scatter(x, y, c='b', label="tracking", s=30, alpha=0.6)  # 使用scatter绘制追踪的点
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

    # 绘制y-t和cy-t的图
    # tmp_cx = np.linspace(-2 * np.pi, 2 * np.pi, len(t))
    # tmp_cy = reference_trajectory(tmp_cx)
    # plt.figure()
    # plt.plot(t, y, "-b", label="y")
    # # print(f"len(t):{len(t)},len(cy):{len(cy)}")
    # plt.plot(t, tmp_cy, "-r", label="cy")
    # plt.xlabel("Time[s]")
    # plt.ylabel("Y")
    # plt.grid(True)
    # plt.title("Robot Position Y and Target Path Y")
    # plt.legend()
    # plt.show()

    # 绘制路径坐标与参考轨迹坐标之差的图
    plt.figure()
    plt.plot(t, error_list, "-b")
    plt.xlabel("Time[s]")
    plt.ylabel("Error")
    plt.grid(True)
    plt.title("Error between Robot Position and Target Path")
    plt.show()

    max_overshoot = max(error_list) # 计算最大超调量
    rise_time = find_rise_time(error_list, max_overshoot) # 计算上升时间
    steady_state_error = abs(error_list[-1])
    settling_time = find_settling_time(error_list, steady_state_error, rise_time) # 计算调节时间

    # 输出分析结果
    print(f"最大超调: {max_overshoot:.2f}")
    print(f"上升时间: {rise_time:.2f} s")
    print(f"调节时间: {settling_time:.2f} s")
    print(f"稳态误差: {steady_state_error:.2f}")

if __name__ == '__main__':
    print("纯追踪路径跟踪仿真开始")
    main()
