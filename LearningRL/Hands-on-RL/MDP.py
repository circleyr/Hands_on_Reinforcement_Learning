import numpy as np
import random


# 给定一条序列,计算从某个索引（起始状态）开始到序列最后（终止状态）得到的回报
def ComputeSequenceReward(Start_idx, Sequence, RewardVector, gamma=0.5):
    TotalReward = 0.0
    for i in reversed(range(Start_idx, len(Sequence))):
        TotalReward = gamma * TotalReward + RewardVector[Sequence[i] - 1]
    return TotalReward

# Exploit Bellman equation to compute value of all states
def ComputeValue(RewardVector, Statesize, TransitionMatrix, gamma=0.5):
    RewardVector = np.array(RewardVector).reshape(-1, 1)
    try:
        Value = np.dot(np.linalg.inv(np.eye(Statesize, Statesize) - gamma * TransitionMatrix),
                   RewardVector)
    except:
        print("-------------状态转移矩阵为奇异矩阵，存在求解误差-------------")
        TransitionMatrix[Statesize - 1][Statesize - 1] += 1e-7
        I = np.eye(Statesize, Statesize)
        Value = np.dot(np.linalg.inv(I - gamma * TransitionMatrix),
                   RewardVector)
    return Value

def Set_MDPParameterAndPolicy():
    # 状态集合
    S = ["C1", "C2", "Pass", "FB", "Sleep"]
    # 动作集合
    A = ["Facebook", "Study", "Sleep", "Pub", "Quit"]
    # 状态转移函数
    P = {
        "C1-Study-C2": 1.0,
        "C1-Facebook-FB": 1.0,
        "FB-Facebook-FB": 1.0,
        "FB-Quit-C1": 1.0,
        "C2-Study-Pass": 1.0,
        "C2-Sleep-Sleep": 1.0,
        "Pass-Study-Sleep": 1.0,
        "Pass-Pub-C1": 0.2,
        "Pass-Pub-C2": 0.4,
        "Pass-Pub-Pass": 0.4,
    }
    # 奖励函数
    R = {
        "C1-Study": -2,
        "C1-Facebook": -1,
        "FB-Facebook": -1,
        "FB-Quit": 0,
        "C2-Study": -2,
        "C2-Sleep": 0,
        "Pass-Study": 10,
        "Pass-Pub": 1,
    }
    # 折扣因子
    gamma = 0.5
    MDP = (S, A, P, R, gamma)

    # 策略1,随机策略
    Pi_1 = {
        "C1-Study": 0.5,
        "C1-Facebook": 0.5,
        "FB-Facebook": 0.5,
        "FB-Quit": 0.5,
        "C2-Study": 0.5,
        "C2-Sleep": 0.5,
        "Pass-Study": 0.5,
        "Pass-Pub": 0.5,
    }
    # 策略2
    Pi_2 = {
        "C1-Study": 0.7,
        "C1-Facebook": 0.3,
        "FB-Facebook": 0.3,
        "FB-Quit": 0.7,
        "C2-Study": 0.5,
        "C2-Sleep": 0.5,
        "Pass-Study": 0.2,
        "Pass-Pub": 0.8,
    }
    return MDP, Pi_1

# 把输入的两个字符串通过“-”连接,便于使用上述定义的P、R变量
def join(str1, str2):
    return str1 + '-' + str2

def MonteCarloSampling(MDP, Policy, MAXTimeStep, SamplingNum):
    ''' 采样函数,策略Pi,限制最长时间步MaxTimeStep,总共采样序列数SamplingNum '''
    S, A, P, R, gamma = MDP
    StateNum = len(S)
    Sequences = []
    for _ in range(SamplingNum):
        Sequence = []
        TimeStep = 0
        # 随机选择一个除Sleep以外的状态s作为起点
        s = S[np.random.randint(StateNum - 1)]
        # 当前状态为终止状态或者时间步太长时,一次采样结束
        while s != "Sleep" and TimeStep <= MAXTimeStep:
            TimeStep += 1
            rand, temp = np.random.rand(), 0
            # 在状态s下根据策略选择动作
            for a_ in A:
                temp += Policy.get(join(s, a_), 0.0)
                if temp >= rand:
                    a = a_
                    r = R.get(join(s, a_), 0.0)
                    break
            rand, temp = np.random.rand(), 0
            # 根据状态转移概率得到下一个状态s_next
            for s_ in S:
                temp += P.get(join(join(s, a), s_), 0.0)
                if temp >= rand:
                    s_next = s_
                    break
            # 把（s,a,r,s_next）元组放入序列中
            Sequence.append((s, a, r, s_next))
            # s_next变成当前状态,开始接下来的循环
            s = s_next
        Sequences.append(Sequence)
    return Sequences

# 对所有采样序列计算所有状态的价值
def MonteCarloComputeValue(Sequences, MDP):
    gamma = MDP[4]
    V = {"C1": 0, "C2": 0, "Pass": 0, "FB": 0, "Sleep": 0}
    N = {"C1": 0, "C2": 0, "Pass": 0, "FB": 0, "Sleep": 0}
    for Sequence in Sequences:
        G = 0
        # 一个序列从后往前计算
        for i in reversed(range(len(Sequence))):
            s, r = Sequence[i][0], Sequence[i][2]
            G = r + gamma * G
            N[s] = N[s] + 1
            V[s] = V[s] + (G - V[s]) / N[s]
    return V

def ComputeOccupancy(s, a, Sequences, MAXTimeStep, MDP):
    ''' 计算状态动作对（s,a）出现的频率,以此来估算策略的占用度量 '''
    gamma = MDP[4]
    rho = 0
    total_times = np.zeros(MAXTimeStep)  # 记录每个时间步t各被经历过几次
    occur_times = np.zeros(MAXTimeStep)  # 记录(s_t,a_t)=(s,a)的次数
    for Sequence in Sequences:
        for i in range(len(Sequence)):
            try:
                s_, a_ = Sequence[i][0], Sequence[i][1]
                total_times[i] += 1
                if s_ == s and a_ == a:
                    occur_times[i] += 1
            except IndexError:
                continue
    for i in reversed(range(MAXTimeStep)):
        if total_times[i]:
            # 用频率来估算策略的占用度量
            rho = gamma ** i * occur_times[i] / total_times[i]
    return (1 - gamma) * rho

def SampleTEXT():
    MDP, Policy = Set_MDPParameterAndPolicy()
    Sequences = MonteCarloSampling(MDP, Policy, MAXTimeStep=8, SamplingNum=5)
    for Sequence in Sequences:
        print(Sequence)

def MonteCarloTEXT():
    MDP, Policy = Set_MDPParameterAndPolicy()
    Sequences = MonteCarloSampling(MDP, Policy, MAXTimeStep=8, SamplingNum=5000)
    V = MonteCarloComputeValue(Sequences, MDP)
    print("使用蒙特卡洛方法计算MDP的状态价值为\n", V)

def test01():
    # Define the transition Matrix
    # C1 C2 C3 Pass Pub FB Sleep
    P = [
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2],
        [0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ]
    P = np.array(P)
    RewardVector = [-2, -2, -2, 10, 1, -1, 0]
    chain = [1, 6, 6, 1, 2, 7]
    start_index = 0
    print("根据本序列计算得到回报为：%s。"% ComputeSequenceReward(start_index, chain, RewardVector, gamma=0.5))
    print("MRP中每个状态价值分别为\n", ComputeValue(RewardVector, 7, P))

# MDP2MRP
def test02():
    # Define the transition Matrix
    # C1 C2  Pass FB Sleep
    P_TransformMDP2MRP = [
        [0.0, 0.5, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.5],
        [0.1, 0.2, 0.2, 0.0, 0.5],
        [0.5, 0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0]
    ]
    P_TransformMDP2MRP = np.array(P_TransformMDP2MRP)
    R_TransformMDP2MRP = [-1.5, -1, 5.5, -0.5, 0.0]
    print("MDP中每个状态价值分别为\n", ComputeValue(R_TransformMDP2MRP, 5, P_TransformMDP2MRP, gamma=0.5))

# MonteCarlo
def test03():
    # SampleTEXT(
    MonteCarloTEXT()
    test02()

# Occupancy
def test04():
    # 策略2
    Policy_2 = {
        "C1-Study": 0.6,
        "C1-Facebook": 0.4,
        "FB-Facebook": 0.3,
        "FB-Quit": 0.7,
        "C2-Study": 0.5,
        "C2-Sleep": 0.5,
        "Pass-Study": 0.1,
        "Pass-Pub": 0.9,
    }
    MAXTimeStep = 8
    MDP, Policy_1 = Set_MDPParameterAndPolicy()
    Sequences1 = MonteCarloSampling(MDP, Policy_1, MAXTimeStep, SamplingNum=1000)
    Sequences2 = MonteCarloSampling(MDP, Policy_2, MAXTimeStep, SamplingNum=1000)
    rho1 = ComputeOccupancy("Pass", "Pub", Sequences1, MAXTimeStep, MDP)
    rho2 = ComputeOccupancy("Pass", "Pub", Sequences2, MAXTimeStep, MDP)
    print(rho1, rho2)

if __name__ == "__main__":
    test03()
    test04()
