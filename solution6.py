
# from code import interact
# from functools import partial
# from operator import mod
# from time import time
import numpy as np
import matplotlib.pyplot as plt
from cmath import inf, pi, sin
from scipy.integrate import RK45
from scipy.signal import tf2ss

# 模块库


class block(object):
    def __init__(self):
        pass

    def output(self):
        pass

    def acceptInput(self):
        pass


class sink(block):
    def __init__(self, n):
        self.memory = np.zeros([0, n+1])
        if n == 1:
            self.singleInput = True

    def acceptInput(self, input):
        self.input = input

    def output(self, time):
        if self.singleInput:
            self.memory = np.append(
                self.memory, [[time]+[self.input]], axis=0)
        else:
            self.memory = np.append(self.memorty, [[time]+self.input], axis=0)


class scope(sink):
    def plot(self, type='-x', label='ModelOut'):
        time = self.memory[:, 0]
        value = self.memory[:, 1]
        plt.plot(time, value, type, label=label)
        plt.grid()
        plt.legend()
        plt.xlabel('Time/s')
        plt.ylabel("ModelOutput")
        plt.show()


# 积分模块
class integral(block):
    def __init__(self, initialState):
        self.X = initialState

    def output(self):
        return self.X

    def acceptInput(self, input):
        self.input = input

    def setState(self, X):
        self.X = X

    def ode(self, X):
        dotX = self.input
        return dotX

# 离散积分模块


class discreteIntegral(block):
    def __init__(self, initialState):
        self.state = initialState
        self.lastTime = 0

    def output(self):
        return self.state

    def acceptInput(self, input):
        self.input = input

    def setState(self, time):
        h = time - self.lastTime
        self.state += self.input * h
        self.lastTime = time

# 离散传函


class discreteTransferfunction(block):
    """
    """

    def __init__(self, num, den):
        #
        if len(num) == len(den):
            self.feedThrough = True
        elif len(num) < len(den):
            self.feedThrough = False
        else:
            assert('no supported for transerfunction')

        self.nState = len(den)-1
        self.X = np.zeros(0)
        self.dX = np.zeros(self.nState)
        # 传函转换状态空间
        self.A, self.B, self.C, self.D = tf2ss(num, den)

    def output(self):
        if self.feedThrough:
            # return np.dot(self.C, self.X)+np.dot(self.D, self.input)
            out = np.dot(self.C, self.dX)+self.D.T[0]*self.input
        else:
            # return np.dot(self.C, self.X)
            out = np.dot(self.C, self.dX)
        return out[0]

    def acceptInput(self, input):
        self.input = input

    def setState(self):
        self.dX = np.dot(self.A, self.dX)+self.B.T[0]*self.input


# 连续传函
class transferfunction(block):
    """
    """

    def __init__(self, num, den):
        #
        if len(num) == len(den):
            self.feedThrough = True
        elif len(num) < len(den):
            self.feedThrough = False
        else:
            assert('no supported for transerfunction')

        self.nState = len(den)-1
        self.X = np.zeros(self.nState)
        # 传函转换状态空间
        self.A, self.B, self.C, self.D = tf2ss(num, den)

    def output(self):
        if self.feedThrough:
            out = np.dot(self.C, self.X)+np.dot(self.D, self.input)
            # return np.dot(self.C, self.X)+self.D.T[0]*self.input

        else:
            # return np.dot(self.C, self.X)
            out = np.dot(self.C, self.X)
        return out[0]

    def acceptInput(self, input):
        self.input = np.array(input)

    def setState(self, X):
        self.X = X

    def ode(self, X):
        # 微分方程转化为ODE方程组,用状态空间表示
        dotX = np.dot(self.A, X)+self.B.T[0]*self.input
        # dotX = np.dot(self.A, X)+np.dot(self.B, self.input)
        return dotX


class gain(block):
    def __init__(self, gain):

        self.gain = gain

    def acceptInput(self, input):
        self.input = input

    def output(self):
        return self.input * self.gain


class sum(block):
    '''
        gain is a numpy vector with elements:1 or -1
        lenth should be the same with input vector.
    '''

    def __init__(self, gain):
        self.gain = np.array(gain)

    def acceptInput(self, input):
        self.input = np.array(input, dtype=object)

    def output(self):
        return np.dot(self.input, self.gain)


class source(block):
    def __init__(self):
        pass

    def output(self, time):
        pass

# 斜坡信号源模块


class ramp(source):
    def __init__(self, slope):
        self.slope = slope

    def output(self, time):
        return self.slope * time

# 阶跃信号源模块


class step(source):
    def __init__(self, stepTime):
        self.stepTime = stepTime
        self.clock = np.array([stepTime, stepTime+0.00001])

    def output(self, time):
        if time <= self.stepTime:
            return 0
        else:
            return 1

# 正弦信号源模块


class sinWave(source):
    def __init__(self, T=2*pi):
        self.omega = 2 * pi / T

    def output(self, time):
        return sin(self.omega * time)

# 常数输出


class constant(source):
    def __init__(self, const):
        self.const = const

    def output(self):
        return self.const


# 单位延迟
class unitdelay(block):
    def __init__(self, initial):
        self.input = initial

    def output(self, time):
        return self.state

    def acceptInput(self, input):
        self.input = input

    def setState(self):
        self.state = self.input

# PID


class pid(object):
    def __init__(self, kp, ki, kd, N, i0=0) -> None:
        self.kp = gain(kp)
        self.ki = gain(ki)
        self.kd = gain(kd)
        self.integral = integral(i0)
        self.derivative = transferfunction([N, 0], [1, N])
        self.sum = sum([1, 1, 1])
        # 模块状态向量
        self.X = np.hstack((self.integral.X, self.derivative.X))
        self.discrete = False
        self.feedThrough = False

    def setState(self, X):
        # 更新状态到各个模块
        self.integral .setState(X[0])
        self.derivative.setState(X[1])

    def ode(self, X):

        # 状态微分向量为所有模型的状态微分向量
        return np.hstack((self.integral.ode(X[0]), self.derivative.ode(X[1])[0][0]))

    def output(self, time):
        self.kp.acceptInput(self.input)
        self.ki.acceptInput(self.input)
        self.integral.acceptInput(self.ki.output())
        self.kd.acceptInput(self.input)
        self.derivative.acceptInput(self.kd.output())
        self.sum.acceptInput(
            [self.kp.output(), self.integral.output(), self.derivative.output()])
        return self.sum.output()

    def acceptInput(self, input):
        self.input = input


class model1(object):
    def __init__(self):
        # 本模型由斜坡，增益和积分三个模块组成
        self.integral1 = discreteIntegral(0)
        self.gain1 = gain(2)
        self.ramp1 = ramp(1)
        self.X = np.ones(0)
        self.discrete = True

    def doStep(self, time, *X):
        # 调用各模块输出函数，并传递给下一个模块,输出函数的调用顺序和模型执行顺序一致
        self.modeloutput = self.integral1.output()
        # gain1的 输入为ramp1的输出
        self.gain1.acceptInput(self.ramp1.output(time))
        # integral1的输入为gain1的 输出
        self.integral1.acceptInput(self.gain1.output())
        # 模型更新状态，本模型只有一个状态
        self.integral1.setState(time)


class model2(object):
    def __init__(self):
        # 本模型由斜坡，离散传函组成
        self.transfer = discreteTransferfunction([1, 2], [1, 0.5, 0.2])
        self.step = sinWave()
        self.X = np.ones(0)
        self.discrete = True

    def doStep(self, time, *X):
        # 调用各模块输出函数，并传递给下一个模块,输出函数的调用顺序和模型执行顺序一致
        self.modeloutput = self.transfer.output()
        self.transfer.acceptInput(self.step.output(time))
        # 模型更新状态，本模型只有一个状态
        self.transfer.setState()


class model5(object):
    def __init__(self):
        # 本模型由阶跃，单位延迟，一阶系统和二阶系统4个模块组成
        self.step = step(1)
        self.unitdelay = unitdelay(0)
        self.oneOrder = transferfunction([2], [1, 2])
        self.twoOrder = transferfunction([1], [0.2, 0.2, 1])
        self.discrete = False
        # 模型状态向量为所有模块的状态向量
        self.X = np.hstack((self.oneOrder.X, self.twoOrder.X))

    def ode(self, X):
        # 状态微分向量为所有模型的状态微分向量
        return np.hstack((self.oneOrder.ode(X[0]), self.twoOrder.ode(X[1:3])))

    def doStep(self, time, X):
        # 更新状态到各个模块
        self.unitdelay.setState()
        self.oneOrder.setState(X[0])
        self.twoOrder.setState(X[1:3])
        # 调用各模块输出函数，并传递给下一个模块,输出函数的调用顺序和模型执行顺序一致
        self.modeloutput = self.twoOrder.output()
        self.twoOrder.acceptInput(self.oneOrder.output())
        self.oneOrder.acceptInput(self.unitdelay.output(time))
        self.unitdelay.acceptInput(self.step.output(time))
        # 返回模型的状态微分向量，供求解器调用
        return self.ode(X)


class model6(object):
    def __init__(self):
        # 本模型实现二阶系统的PI闭环控制
        self.step = step(1)
        self.kp = gain(2.5)
        self.ki = gain(2)
        self.integrator = integral(0)
        self.feedback = sum([1, -1])
        self.sum = sum([1, 1])
        self.twoOrder = transferfunction([1], [0.2, 0.5, 1])
        self.scope = scope(1)
        self.discrete = False
        # 模型状态向量为所有模块的状态向量
        self.X = np.hstack((self.integrator.X, self.twoOrder.X))
        self.clock = self.step.clock

    def ode(self, X):
        # 状态微分向量为所有模型的状态微分向量
        return np.hstack((self.integrator.ode(X[0]), self.twoOrder.ode(X[1:3])))

    def output(self, time):
        # 调用各模块输出函数，并传递给下一个模块,输出函数的调用顺序和模型执行顺序一致
        self.scope.acceptInput(self.twoOrder.output())
        self.feedback.acceptInput(
            [self.step.output(time), self.twoOrder.output()])
        self.kp.acceptInput(self.feedback.output())
        self.ki.acceptInput(self.feedback.output())
        self.integrator.acceptInput(self.ki.output())
        self.sum.acceptInput([self.kp.output(), self.integrator.output()])
        self.twoOrder.acceptInput(self.sum.output())

    def setState(self, X):
        # 更新状态到各个模块
        self.integrator .setState(X[0])
        self.twoOrder.setState(X[1:3])

    def doStep(self, time, X):
        self.setState(X)

        self.output(time)
        # 返回模型的状态微分向量，供求解器调用
        return self.ode(X)

    def recorde(self, time):
        self.scope.output(time)

    def scopeshow(self):
        self.scope.plot()


class model61(object):
    def __init__(self):
        # 本模型实现二阶系统的PI闭环控制
        self.step = step(1)
        self.kp = gain(2.5)
        self.feedback = sum([1, -1])
        self.twoOrder = transferfunction([1], [0.2, 0.5, 1])
        self.discrete = False
        self.scope = scope(1)
        # 模型状态向量为所有模块的状态向量
        self.X = self.twoOrder.X
        self.clock = self.step.clock
        # self.clock = np.zeros(0)

    def ode(self, X):
        # 状态微分向量为所有模型的状态微分向量
        return self.twoOrder.ode(X)

    def output(self, time):
        # 调用各模块输出函数，并传递给下一个模块,输出函数的调用顺序和模型执行顺序一致
        self.scope.acceptInput(self.twoOrder.output())
        self.feedback.acceptInput(
            [self.step.output(time), self.twoOrder.output()])
        self.kp.acceptInput(self.feedback.output())
        self.twoOrder.acceptInput(self.kp.output())

    def setState(self, X):
        # 更新状态到各个模块
        self.twoOrder.setState(X)

    def doStep(self, time, X):
        self.setState(X)
        self.output(time)
        # 返回模型的状态微分向量，供求解器调用
        return self.ode(X)

    def recorde(self, time):
        self.scope.output(time)

    def scopeshow(self):
        self.scope.plot()


class model62(object):
    def __init__(self):
        # 本模型实现二阶系统的PID闭环控制
        self.step = step(1)
        self.pid = pid(2.5, 2, 0.12, 10)
        self.feedback = sum([1, -1])
        self.twoOrder = transferfunction([1], [0.2, 0.5, 1])
        self.scope = scope(1)
        self.discrete = False
        # 模型状态向量为所有模块的状态向量
        self.X = np.hstack((self.pid.X, self.twoOrder.X))
        self.clock = self.step.clock

    def ode(self, X):
        # 状态微分向量为所有模型的状态微分向量
        return np.hstack((self.pid.ode(X[0:2]), self.twoOrder.ode(X[2:4])))

    def output(self, time):
        # 调用各模块输出函数，并传递给下一个模块,输出函数的调用顺序和模型执行顺序一致
        self.scope.acceptInput(self.twoOrder.output())
        self.feedback.acceptInput(
            [self.step.output(time), self.twoOrder.output()])
        self.pid.acceptInput(self.feedback.output())
        self.twoOrder.acceptInput(self.pid.output(time))

    def setState(self, X):
        # 更新状态到各个模块
        self.pid .setState(X[0:2])
        self.twoOrder.setState(X[2:4])

    def doStep(self, time, X):
        self.setState(X)
        self.output(time)
        # 返回模型的状态微分向量，供求解器调用
        return self.ode(X)

    def recorde(self, time):
        self.scope.output(time)

    def scopeshow(self):
        self.scope.plot()


class model7(object):
    def __init__(self):
        # 本模型实现二阶系统的PI闭环控制
        self.step = step(1)
        self.derivative = transferfunction([5, 0], [1, 5])
        self.scope = scope(1)
        # 模型状态向量为所有模块的状态向量
        self.X = self.derivative.X
        self.clock = self.step.clock
        self.discrete = False

    def ode(self, X):
        # 状态微分向量为所有模型的状态微分向量
        return self.derivative.ode(X)

    def output(self, time):
        # 调用各模块输出函数，并传递给下一个模块,输出函数的调用顺序和模型执行顺序一致
        self.derivative.acceptInput(self.step.output(time))
        self.scope.acceptInput(self.derivative.output())

    def doStep(self, time, X):
        # 更新状态到各个模块
        self.derivative.setState(X)
        self.output(time)
        # 返回模型的状态微分向量，供求解器调用
        return self.ode(X)

    def recorde(self, time):
        self.scope.output(time)

    def scopeshow(self):
        self.scope.plot()


def solve(model, Time, solver, Type):

    # 计算第0步的输出
    model.output(0)
    model.recorde(0)
    if Type == "Fixed":
        # 计算步长
        step = Time[1]-Time[0]
        # 外推迭代
        for time in Time:
            if model.discrete:
                # 纯离散系统
                model.doStep(time)
                model.recorde(time)

            else:
                # 包含连续状态的系统
                # 调用求解器，求解模型状态
                integrator = solver(
                    model.doStep, time, model.X, time+step)
                integrator.step()
                model.recorde(integrator.t)
                model.X = integrator.y
    else:
        # 离散事件发生的时间添加到时间向量中
        Time = np.hstack([Time, model.clock])
        Time.sort()
        for i, time in enumerate(Time):
            try:
                integrator = solver(model.doStep, t0=time,
                                    y0=model.X, t_bound=Time[i+1], max_step=10, rtol=1e-5, atol=1e-06, vectorized=False)
            except Exception as e:
                print(str(e))
                break
            while integrator.status == "running":
                integrator.step()
                model.X = integrator.y
                model.recorde(integrator.t)
                if integrator.status == "finished":
                    break


class fixedSolver():
    def __init__(self, ode, time, X, timeNext, type='RK45'):
        self.ode = ode
        self.t = time
        self.y = X
        self.h = timeNext-time
        self.timeNext = timeNext
        self.integrator = type

    def step(self):
        pass


class RK4(fixedSolver):
    def step(self):
        t = self.t
        y = self.y
        h = self.h
        ode = self.ode
        k1 = ode(t, y)
        k2 = ode(t+h/2, y+h/2*k1)
        k3 = ode(t+h/2, y+h/2*k2)
        k4 = ode(t+h, y+h*k3)
        self.y = y + h/6*(k1+2*k2+2*k3+k4)
        self.t = self.timeNext


class BackeEular(fixedSolver):
    def step(self):
        '''
        改进欧拉法
        '''
        t = self.t
        y = self.y
        h = self.h
        ode = self.ode
        k1 = ode(t, y)
        k2 = ode(t+h, y+h*k1)
        self.y = y+h/2*(k1+k2)


if __name__ == "__main__":
    # 选择连续系统的数值求解器，Eular，BackEular，RK4
    # Model = model2()
    # Time = np.arange(0, 5.1, 0.2)
    # solve(Model, Time, RK4, "Fixed")
    # plt.plot(Model.Time, Model.DataLog, '-*', label='RK4_model1')

    # Model = model5()

    # Model.solve(0.1, 10, RK4, 'Fixed')
    # plt.plot(Model.Time, Model.DataLog, '-', label='Runge-Kutta4 Model5')
    # Model = model5()
    # Model.solve(10, 10, scipy_integrator, 'Variable')
    # plt.plot(Model.Time, Model.DataLog, '-*', label='Runge-Kutta45 Model5')

    # Model = model6()
    # Model = model6()
    # Time = np.arange(0, 8, 0.1)
    # solve(Model, Time, RK4, "Fixed")
    # Model.scopeshow()

    # Model = model6()
    # Model = model61()
    Model = model62()
    scipy_integrator = RK45
    Time = np.array([0, 8])
    solve(Model, Time, scipy_integrator, "Variable")
    Model.scopeshow()
