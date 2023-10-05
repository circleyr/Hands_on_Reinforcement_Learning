import copy


class CliffWalkingEnv:
    """ 悬崖漫步环境"""
    def __init__(self, ncol=12, nrow=4):
        # 定义网格世界的列
        self.ncol = ncol
        # 定义网格世界的行
        self.nrow = nrow
        # 转移矩阵P[state][action] = [(prob, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.CreateP()

    def CreateP(self):
        # 初始化
        P = [[() for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    # 最后一行除了第一列和最后一列，都是悬崖
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [1, i * self.ncol + j, 0, True]
                        continue
                    # 其他位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = (1, next_state, reward, done)
        return P

class PolicyIteration:
    """ 策略迭代算法 """
    def __init__(self, env, theta, gamma):
        self.env = env
        # 策略评估收敛阈值
        self.theta = theta
        # 折扣因子
        self.gamma = gamma
        # 初始化随机策略
        self.policy = [[0.25, 0.25, 0.25, 0.25] for i in range(self.env.nrow * self.env.ncol)]
        # 初始化价值函数
        self.v = [0] * self.env.ncol * self.env.nrow

    def PolicyEvaluation(self):
        count = 0
        Delta_v = 100
        while Delta_v >= self.theta:
            Delta_v = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                # 用动作价值函数来替换公式后面的一部分,计算状态s下的所有Q(s,a)价值
                Qsa_list = []
                for a in range(4):
                    Qsa = 0
                    prob, next_state, reward, done = self.env.P[s][a]
                    Qsa = prob * (reward + self.gamma * self.v[next_state] * (1 - done))
                    Qsa_list.append(Qsa * self.policy[s][a])
                new_v[s] = sum(Qsa_list)
                Delta_v = max(Delta_v, abs(self.v[s] - new_v[s]))
            self.v = new_v
            count += 1
        print("策略评估进行%d轮后完成" % count)

    def PolicyImprovement(self):
        for s in range(self.env.nrow * self.env.ncol):
            Qsa_list = []
            for a in range(4):
                Qsa = 0
                prob, next_state, reward, done = self.env.P[s][a]
                Qsa = prob * (reward + self.gamma * self.v[next_state] * (1 - done))
                Qsa_list.append(Qsa)
            MaxQ = max(Qsa_list)
            # 计算有几个动作得到了最大的Q值
            CountQ = Qsa_list.count(MaxQ)
            # 让这些动作均分概率
            self.policy[s] = [1 / CountQ if Q == MaxQ else 0.0 for Q in Qsa_list]
        print("策略提升完成")
        return self.policy

    def policy_iteration(self):  # 策略迭代
        while 1:
            self.PolicyEvaluation()
            # 将列表进行深拷贝,方便接下来进行比较
            old_policy = copy.deepcopy(self.policy)
            new_policy = self.PolicyImprovement()
            if old_policy == new_policy: break


class ValueIteration():
    def __init__(self, env, gamma, theta):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        # 用于存储价值迭代后的策略
        self.policy = [None for i in range(self.env.nrow * self.env.ncol)]
        self.v = [0] * self.env.ncol * self.env.nrow

    def Valueiteration(self):
        count = 0
        Delta_v = 100
        while Delta_v >= self.theta:
            Delta_v = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                # 用动作价值函数来替换公式后面的一部分,计算状态s下的所有Q(s,a)价值
                Qsa_list = []
                for a in range(4):
                    Qsa = 0
                    prob, next_state, reward, done = self.env.P[s][a]
                    Qsa = prob * (reward + self.gamma * self.v[next_state] * (1 - done))
                    Qsa_list.append(Qsa)
                new_v[s] = max(Qsa_list)
                Delta_v = max(Delta_v, abs(self.v[s] - new_v[s]))
            self.v = new_v
            count += 1
        print("价值迭代进行%d轮后完成" % count)
        self.PolicyOutput()

    def PolicyOutput(self):
        for s in range(self.env.nrow * self.env.ncol):
            Qsa_list = []
            for a in range(4):
                Qsa = 0
                prob, next_state, reward, done = self.env.P[s][a]
                Qsa = prob * (reward + self.gamma * self.v[next_state] * (1 - done))
                Qsa_list.append(Qsa)
            MaxQ = max(Qsa_list)
            # 计算有几个动作得到了最大的Q值
            CountQ = Qsa_list.count(MaxQ)
            # 让这些动作均分概率
            self.policy[s] = [1 / CountQ if Q == MaxQ else 0.0 for Q in Qsa_list]



def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # 目标状态
                print('EEEE', end=' ')
            else:
                a = agent.policy[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()

def test01():
    env = CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001
    gamma = 0.9
    agent = PolicyIteration(env, theta, gamma)
    agent.policy_iteration()
    print_agent(agent, action_meaning, list(range(37, 47)), [47])

def test02():
    env = CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001
    gamma = 0.9
    agent = ValueIteration(env, gamma, theta)
    agent.Valueiteration()
    print_agent(agent, action_meaning, list(range(37, 47)), [47])

if __name__ == "__main__":
    test02()




