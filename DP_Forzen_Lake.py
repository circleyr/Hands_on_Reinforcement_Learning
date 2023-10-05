import gym
import copy
import time


def PrintEnvINFO(env):
    holes = set()
    ends = set()
    for s in env.P:
        for a in env.P[s]:
            for s_ in env.P[s][a]:
                if s_[2] == 1.0:  # 获得奖励为1,代表是目标
                    ends.add(s_[1])
                if s_[3] == True:
                    holes.add(s_[1])
    holes = holes - ends
    print("冰洞的索引:", holes)
    print("目标的索引:", ends)
    for a in env.P[14]:  # 查看目标左边一格的状态转移信息
        print(env.P[14][a])

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
        Delta_v = 1.1 * self.theta
        while Delta_v >= self.theta:
            Delta_v = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                # 用动作价值函数来替换公式后面的一部分,计算状态s下的所有Q(s,a)价值
                Qsa_list = []
                for a in range(4):
                    Qsa = 0
                    # 每次行走都有一定的概率滑行到附近的其它状态，因此env.P[s][a]是一个含多个元组的列表
                    for res in self.env.P[s][a]:
                        prob, next_state, reward, done = res
                        Qsa += prob * (reward + self.gamma * self.v[next_state] * (1 - done))
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
                # 每次行走都有一定的概率滑行到附近的其它状态，因此env.P[s][a]是一个含多个元组的列表
                for res in self.env.P[s][a]:
                    prob, next_state, reward, done = res
                    Qsa += prob * (reward + self.gamma * self.v[next_state] * (1 - done))
                Qsa_list.append(Qsa)
            MaxQ = max(Qsa_list)
            # 计算有几个动作得到了最大的Q值
            CountQ = Qsa_list.count(MaxQ)
            # 让这些动作均分概率
            self.policy[s] = [1 / CountQ if Q == MaxQ else 0.0 for Q in Qsa_list]
        print("策略提升完成")
        return self.policy

    def policy_iteration(self):  # 策略迭代
        self.env.reset()
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
        Delta_v = 1.1 * self.theta
        while Delta_v >= self.theta:
            Delta_v = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                # 用动作价值函数来替换公式后面的一部分,计算状态s下的所有Q(s,a)价值
                Qsa_list = []
                for a in range(4):
                    Qsa = 0
                    # 每次行走都有一定的概率滑行到附近的其它状态，因此env.P[s][a]是一个含多个元组的列表
                    for res in self.env.P[s][a]:
                        prob, next_state, reward, done = res
                        Qsa += prob * (reward + self.gamma * self.v[next_state] * (1 - done))
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
                # 每次行走都有一定的概率滑行到附近的其它状态，因此env.P[s][a]是一个含多个元组的列表
                for res in self.env.P[s][a]:
                    prob, next_state, reward, done = res
                    Qsa += prob * (reward + self.gamma * self.v[next_state] * (1 - done))
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

# 以后再改，step.action不知道返回了啥
def PlotGame(env, agent):
    for _ in range(1000):
        observation = env.reset()[0]
        env.render()
        time.sleep(0.1)
        actions = agent.policy[observation]
        for k in range(4):
            if actions[k] > 0:
                action = k
                break
            else:
                continue
        observation, reward, done, info, _  = env.step(action)



def test01():
    env = gym.make("FrozenLake-v1", render_mode="human")  # 创建环境
    env = env.unwrapped  # 解封装才能访问状态转移矩阵P
    # 这个动作意义是Gym库针对冰湖环境事先规定好的
    action_meaning = ['<', 'v', '>', '^']
    theta = 1e-5
    gamma = 0.9
    agent = PolicyIteration(env, theta, gamma)
    agent.policy_iteration()
    print_agent(agent, action_meaning, [5, 7, 11, 12], [15])
    # PlotGame(env, agent)
    env.close()

def test02():
    action_meaning = ['<', 'v', '>', '^']
    theta = 1e-5
    gamma = 0.9
    agent = ValueIteration(env, gamma, theta)
    agent.Valueiteration()
    print_agent(agent, action_meaning, [5, 7, 11, 12], [15])


if __name__ == "__main__":
    test01()