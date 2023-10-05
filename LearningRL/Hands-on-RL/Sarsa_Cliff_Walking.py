import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # tqdm是显示循环进度条的库

"""粗糙的版本，以后会重构 """

class CliffWalkingEnv():
    def __init__(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols
        # 记录当前智能体位置的横坐标
        self.x = 0
        # 记录当前智能体位置的纵坐标
        self.y = self.nrows - 1

    # 外部调用这个函数来改变当前位置
    def Step(self, action):
        # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncols - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrows - 1, max(0, self.y + change[action][1]))
        next_state = self.x + self.y * self.ncols
        reward = -1
        done = False
        # 下一个位置在悬崖或者终点
        if self.y == self.nrows - 1 and self.x > 0:
            done = True
            if self.x != self.ncols - 1:  # 下一个位置在悬崖
                reward = -100
        return next_state, reward, done

    # 回归初始状态,坐标轴原点在左上角
    def Reset(self):
        self.x = 0
        self.y = self.nrows - 1
        return self.x + self.y * self.ncols

class Sarsa():
    def __init__(self, env, gamma, alpha, epsilon, numOfEpisodes, numOfActions=4):
        self.env = env
        # 折扣因子
        self.gamma = gamma
        # epsilon-贪婪策略中的参数
        self.epsilon = epsilon
        # 学习率
        self.alpha = alpha
        # 动作个数
        self.numOfActions = numOfActions
        # 初始化Q(s, a)表
        self.Q_table = np.zeros([self.env.nrows * self.env.ncols, numOfActions])
        self.numOfEpisodes = numOfEpisodes

    # Choose A from S using policy derived from Q (e.g., epsilon-greedy)
    def ChooseAction(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.numOfActions)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def SarsaRun(self):
        # 记录每一条序列的回报
        returnList = []
        # 显示10个进度条
        for i in range(10):
            # tqdm的进度条功能
            with tqdm(total=int(self.numOfEpisodes / 10), desc='Iteration %d' % i) as pbar:
                # 每个进度条的序列数
                for episode in range(int(self.numOfEpisodes / 10)):
                    # initialize state
                    state = self.env.Reset()
                    # Choose A from S using policy derived from Q (e.g., epsilon-greedy)
                    action = self.ChooseAction(state)
                    done = False
                    episodeReward = 0
                    # Loop for each step of episode:
                    while not done:
                        # Take action A, observe R, S'
                        stateprime, reward, done = self.env.Step(action)
                        # Choose A' from S' using policy derived from Q (e.g., epsilon-greedy)
                        actionprime = self.ChooseAction(stateprime)
                        episodeReward += reward
                        # Update
                        TD_error = reward + self.gamma * self.Q_table[stateprime, actionprime] \
                                   - self.Q_table[state, action]
                        self.Q_table[state, action] += self.alpha * TD_error
                        state = stateprime
                        action = actionprime
                    returnList.append(episodeReward)
                    if (episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                        pbar.set_postfix({
                            'episode':
                                '%d' % (self.numOfEpisodes / 10 * i + episode + 1),
                            'return':
                                '%.3f' % np.mean(returnList[-10:])
                        })
                    pbar.update(1)
        return returnList

    # 用于打印策略
    def best_action(self, state):
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.numOfActions)]
        for i in range(self.numOfActions):  # 若两个动作的价值一样,都会记录下来
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

class NStepSarsa():
    def __init__(self, env, N, gamma, alpha, epsilon, numOfEpisodes, numOfActions=4):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.numOfActions = numOfActions
        self.Q_table = np.zeros([self.env.nrows * self.env.ncols, numOfActions])
        self.numOfEpisodes = numOfEpisodes
        # 采用n步Sarsa算法
        self.N = N
        # 保存之前的状态
        self.stateList = []
        # 保存之前的动作
        self.actionList = []
        # 保存之前的奖励
        self.rewardList = []

    # Choose A from S using policy derived from Q (e.g., epsilon-greedy)
    def ChooseAction(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.numOfActions)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def NStepSarsaRun(self):
        # 记录每一条序列的回报
        returnList = []
        # 显示10个进度条
        for i in range(10):
            # tqdm的进度条功能
            with tqdm(total=int(self.numOfEpisodes / 10), desc='Iteration %d' % i) as pbar:
                # 每个进度条的序列数
                for episode in range(int(self.numOfEpisodes / 10)):
                    # initialize state
                    state = self.env.Reset()
                    # Choose A from S using policy derived from Q (e.g., epsilon-greedy)
                    action = self.ChooseAction(state)
                    done = False
                    episodeReward = 0
                    # Loop for each step of episode:
                    while not done:
                        # Take action A, observe R, S'
                        stateprime, reward, done = self.env.Step(action)
                        # Choose A' from S' using policy derived from Q (e.g., epsilon-greedy)
                        actionprime = self.ChooseAction(stateprime)
                        episodeReward += reward
                        self.stateList.append(state)
                        self.actionList.append(action)
                        self.rewardList.append(reward)
                        # Update
                        # 若保存的数据可以进行n步更新
                        if len(self.stateList) == self.N:
                            # 得到Q(s_{t+n}, a_{t+n})
                            Q_n = self.Q_table[stateprime, actionprime]
                            for i in reversed(range(self.N)):
                                # 不断向前计算每一步的回报
                                Q_n = Q_n * self.gamma + self.rewardList[i]
                                # 如果到达终止状态,最后几步虽然长度不够n步,也将其进行更新
                                if done and i > 0:
                                    s = self.stateList[i]
                                    a = self.actionList[i]
                                    self.Q_table[s, a] += self.alpha * (Q_n - self.Q_table[s, a])
                                # 将需要更新的状态动作从列表中删除,下次不必更新
                            s = self.stateList.pop(0)
                            a = self.actionList.pop(0)
                            self.rewardList.pop(0)
                            # n步Sarsa的主要更新步骤
                            self.Q_table[s, a] += self.alpha * (Q_n - self.Q_table[s, a])
                        if done:  # 如果到达终止状态,即将开始下一条序列,则将列表全清空
                            self.stateList = []
                            self.actionList = []
                            self.rewardList = []
                        state = stateprime
                        action = actionprime
                    returnList.append(episodeReward)
                    if (episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                        pbar.set_postfix({
                            'episode':
                                '%d' % (self.numOfEpisodes / 10 * i + episode + 1),
                            'return':
                                '%.3f' % np.mean(returnList[-10:])
                        })
                    pbar.update(1)
        return returnList

    # 用于打印策略
    def best_action(self, state):
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.numOfActions)]
        for i in range(self.numOfActions):  # 若两个动作的价值一样,都会记录下来
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a


class ForwardViewSarsaLambda():
    def __init__(self, env, N, gamma, alpha, epsilon, numOfEpisodes, numOfActions=4, lambda_=0.5):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.numOfActions = numOfActions
        self.Q_table = np.zeros([self.env.nrows * self.env.ncols, numOfActions])
        self.numOfEpisodes = numOfEpisodes
        # 采用n步Sarsa算法
        self.N = N
        # 保存之前的状态
        self.stateList = []
        # 保存之前的动作
        self.actionList = []
        # 保存之前的奖励
        self.rewardList = []
        self.lambda_ = lambda_

    # Choose A from S using policy derived from Q (e.g., epsilon-greedy)
    def ChooseAction(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.numOfActions)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def NStepSarsaRun(self):
        # 记录每一条序列的回报
        returnList = []
        # 显示10个进度条
        for i in range(10):
            # tqdm的进度条功能
            with tqdm(total=int(self.numOfEpisodes / 10), desc='Iteration %d' % i) as pbar:
                # 每个进度条的序列数
                for episode in range(int(self.numOfEpisodes / 10)):
                    # initialize state
                    state = self.env.Reset()
                    # Choose A from S using policy derived from Q (e.g., epsilon-greedy)
                    action = self.ChooseAction(state)
                    done = False
                    episodeReward = 0
                    # Loop for each step of episode:
                    while not done:
                        # Take action A, observe R, S'
                        stateprime, reward, done = self.env.Step(action)
                        # Choose A' from S' using policy derived from Q (e.g., epsilon-greedy)
                        actionprime = self.ChooseAction(stateprime)
                        episodeReward += reward
                        self.stateList.append(state)
                        self.actionList.append(action)
                        self.rewardList.append(reward)
                        # Update
                        # 若保存的数据可以进行n步更新
                        Q_nList = []
                        Q_lambda = 0
                        if len(self.stateList) == self.N:
                            # 得到Q(s_{t+n}, a_{t+n})
                            Q_n = self.Q_table[stateprime, actionprime]
                            for num in range(1, self.N + 1):
                                for i in reversed(range(num)):
                                    Q_n = Q_n * self.gamma + self.rewardList[i]
                                    if done and i > 0 and num == self.N:
                                        s = self.stateList[i]
                                        a = self.actionList[i]
                                        self.Q_table[s, a] += self.alpha * (Q_n - self.Q_table[s, a])
                                    Q_n = (1 - self.lambda_) * Q_n * (self.lambda_) ** (num - 1)
                                Q_nList.append(Q_n)
                                # 继续下一轮Q_n的计算
                                Q_n = self.Q_table[stateprime, actionprime]
                            for i in range(len(Q_nList)):
                                Q_lambda += Q_nList[i]
                            # 将需要更新的状态动作从列表中删除,下次不必更新
                            s = self.stateList.pop(0)
                            a = self.actionList.pop(0)
                            self.rewardList.pop(0)
                            # n步Sarsa的主要更新步骤
                            self.Q_table[s, a] += self.alpha * (Q_lambda - self.Q_table[s, a])
                        if done:  # 如果到达终止状态,即将开始下一条序列,则将列表全清空
                            self.stateList = []
                            self.actionList = []
                            self.rewardList = []
                        state = stateprime
                        action = actionprime
                    returnList.append(episodeReward)
                    if (episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                        pbar.set_postfix({
                            'episode':
                                '%d' % (self.numOfEpisodes / 10 * i + episode + 1),
                            'return':
                                '%.3f' % np.mean(returnList[-10:])
                        })
                    pbar.update(1)
        return returnList

    # 用于打印策略
    def best_action(self, state):
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.numOfActions)]
        for i in range(self.numOfActions):  # 若两个动作的价值一样,都会记录下来
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

class BackViewSarsaLambda():
    def __init__(self, env, gamma, alpha, epsilon, numOfEpisodes, numOfActions=4, lambda_=0.85):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.numOfActions = numOfActions
        self.Q_table = np.zeros([self.env.nrows * self.env.ncols, numOfActions])
        self.numOfEpisodes = numOfEpisodes
        self.lambda_ = lambda_

    # Choose A from S using policy derived from Q (e.g., epsilon-greedy)
    def ChooseAction(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.numOfActions)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    # 用于打印策略
    def best_action(self, state):
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.numOfActions)]
        for i in range(self.numOfActions):  # 若两个动作的价值一样,都会记录下来
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def BackViewSarsaRun(self):
        # 记录每一条序列的回报
        returnList = []
        # 显示10个进度条
        for i in range(10):
            # tqdm的进度条功能
            with tqdm(total=int(self.numOfEpisodes / 10), desc='Iteration %d' % i) as pbar:
                # 每个进度条的序列数
                for episode in range(int(self.numOfEpisodes / 10)):
                    # initialize state
                    state = self.env.Reset()
                    # Choose A from S using policy derived from Q (e.g., epsilon-greedy)
                    action = self.ChooseAction(state)
                    done = False
                    episodeReward = 0
                    # E(s, a) = 0
                    E_table = np.zeros([self.env.ncols * self.env.nrows, self.numOfActions])
                    # Loop for each step of episode:
                    while not done:
                        # Take action A, observe R, S'
                        stateprime, reward, done = self.env.Step(action)
                        # Choose A' from S' using policy derived from Q (e.g., epsilon-greedy)
                        actionprime = self.ChooseAction(stateprime)
                        episodeReward += reward
                        # Update
                        TD_error = reward + self.gamma * self.Q_table[stateprime, actionprime] \
                                   - self.Q_table[state, action]
                        E_table[state, action] += 1
                        self.Q_table[state, action] += self.alpha * TD_error * E_table[state, action]
                        state = stateprime
                        action = actionprime
                        E_table = E_table * self.gamma * self.lambda_
                    returnList.append(episodeReward)
                    if (episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                        pbar.set_postfix({
                            'episode':
                                '%d' % (self.numOfEpisodes / 10 * i + episode + 1),
                            'return':
                                '%.3f' % np.mean(returnList[-10:])
                        })
                    pbar.update(1)
        return returnList

def PlotReturn(returnList):
    episodes_list = list(range(len(returnList)))
    plt.plot(episodes_list, returnList)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    # plt.title('Sarsa on {}'.format('Cliff Walking'))
    plt.title('5-step Sarsa on {}'.format('Cliff Walking'))
    plt.show()

def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrows):
        for j in range(env.ncols):
            if (i * env.ncols + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncols + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * env.ncols + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()

# Sarsa
def test01():
    env = CliffWalkingEnv(4, 12)
    np.random.seed(0)
    agent = Sarsa(env, gamma=0.9, alpha=0.1, epsilon=0.1, numOfEpisodes=500)
    returnList = agent.SarsaRun()
    PlotReturn(returnList)
    action_meaning = ['^', 'v', '<', '>']
    print('Sarsa算法最终收敛得到的策略为：')
    print_agent(agent, env, action_meaning, list(range(37, 47)), [47])

# NStepSarsa
def test02():
    env = CliffWalkingEnv(4, 12)
    np.random.seed(0)
    agent = NStepSarsa(env, N=5, gamma=0.9, alpha=0.1, epsilon=0.1, numOfEpisodes=500)
    returnList = agent.NStepSarsaRun()
    PlotReturn(returnList)
    action_meaning = ['^', 'v', '<', '>']
    print('NStepSarsa算法最终收敛得到的策略为：')
    print_agent(agent, env, action_meaning, list(range(37, 47)), [47])

# ForwardViewSarsaLambda
def test03():
    env = CliffWalkingEnv(4, 12)
    np.random.seed(0)
    agent = ForwardViewSarsaLambda(env, N=5, gamma=0.9, alpha=0.1, epsilon=0.1, numOfEpisodes=500)
    returnList = agent.NStepSarsaRun()
    PlotReturn(returnList)
    action_meaning = ['^', 'v', '<', '>']
    print('ForwardView Sarsa算法最终收敛得到的策略为：')
    print_agent(agent, env, action_meaning, list(range(37, 47)), [47])

# BackViewSarsaLambda
def test04():
    env = CliffWalkingEnv(4, 12)
    np.random.seed(0)
    agent = BackViewSarsaLambda(env, gamma=0.9, alpha=0.1, epsilon=0.1, numOfEpisodes=500)
    returnList = agent.BackViewSarsaRun()
    PlotReturn(returnList)
    action_meaning = ['^', 'v', '<', '>']
    print('BackView Sarsa算法最终收敛得到的策略为：')
    print_agent(agent, env, action_meaning, list(range(37, 47)), [47])

def test05():
    returnLists = []
    env = CliffWalkingEnv(4, 12)

    np.random.seed(0)
    agent = Sarsa(env, gamma=0.9, alpha=0.1, epsilon=0.1, numOfEpisodes=500)
    returnLists.append(agent.SarsaRun())

    np.random.seed(0)
    agent = NStepSarsa(env, N=5, gamma=0.9, alpha=0.1, epsilon=0.1, numOfEpisodes=500)
    returnLists.append(agent.NStepSarsaRun())

    np.random.seed(0)
    agent = ForwardViewSarsaLambda(env, N=5, gamma=0.9, alpha=0.1, epsilon=0.1, numOfEpisodes=500)
    returnLists.append(agent.NStepSarsaRun())

    np.random.seed(0)
    agent = BackViewSarsaLambda(env, gamma=0.9, alpha=0.1, epsilon=0.1, numOfEpisodes=500)
    returnLists.append(agent.BackViewSarsaRun())

    for i in range(4):
        plt.plot(list(range(500)), returnLists[i])
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Sarsa on {}'.format('Cliff Walking'))
    plt.show()



if __name__ == "__main__":
    test03()
