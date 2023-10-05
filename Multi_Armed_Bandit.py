import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    """伯努利多臂老虎机,输入K表示拉杆个数"""
    def __init__(self, K):
        # 随机生成K个0～1的数,作为拉动每根拉杆的获奖概率
        self.probs = np.random.uniform(size=K)
        # 获奖概率最大的拉杆
        self.best_idx = np.argmax(self.probs);
        # 最大的获奖概率
        self.best_prob = self.probs[self.best_idx]
        self.K = K

    def step(self, Kth):
        # 当玩家选择了k号拉杆后,根据拉动该老虎机的k号拉杆获得奖励的概率返回1（获奖）或0（未获奖）
        if np.random.rand() < self.probs[Kth]:
            return 1
        else:
            return 0

class ProblemSolver:
    """多臂老虎机算法基本框架 """
    def __init__(self, bandit):
        self.bandit = bandit
        # 每根拉杆的尝试次数
        self.counts = np.zeros(self.bandit.K)
        # 当前步的累积懊悔
        self.regret = 0.0
        # 维护一个列表,记录每一步的动作
        self.actions = []
        # 维护一个列表,记录每一步的累积懊悔
        self.regrets = []

    def UpdateRegret(self, Kth):
        # 计算累积懊悔并保存, Kth为本次动作选择的拉杆的编号
        self.regret += self.bandit.best_prob - self.bandit.probs[Kth]
        self.regrets.append(self.regret)

    def RunOnce(self):
        # 返回当前动作选择哪一根拉杆, 由每个具体的策略实现,需要继承后重写
        raise NotImplementedError

    def RunLoop(self, NumofSteps):
        # 运行一定次数, num_steps为总运行次数
        for _ in range(NumofSteps):
            Kth = self.RunOnce()
            self.counts[Kth] += 1
            self.UpdateRegret(Kth)
            self.actions.append(Kth)

class EpsilonGreedy(ProblemSolver):
    """ epsilon贪婪算法,继承ProblemSolver类"""
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        # 初始化拉动所有拉杆的期望奖励估值
        self.EstimateReward = np.array([init_prob] * self.bandit.K)

    def RunOnce(self):
        if np.random.rand() < self.epsilon:
            # 随机选择一根拉杆
            Kth = np.random.randint(0, self.bandit.K)
        else:
            # 选择期望奖励估值最大的拉杆
            Kth = np.argmax(self.EstimateReward)
            # 得到本次动作的奖励
            Reward = self.bandit.step(Kth)
            # 更新期望奖励估值
            self.EstimateReward[Kth] += 1.0 / (self.counts[Kth] + 1) * (Reward - self.EstimateReward[Kth])
        return Kth

class DecayingEpsilonGreedy(ProblemSolver):
    """ epsilon值随时间衰减的epsilon-贪婪算法,继承Solver类 """
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        # 初始化拉动所有拉杆的期望奖励估值
        self.EstimateReward = np.array([init_prob] * self.bandit.K)
        self.TimeCount = 0

    def RunOnce(self):
        self.TimeCount += 1
        if np.random.rand() < 1.0 / self.TimeCount:
            # 随机选择一根拉杆
            Kth = np.random.randint(0, self.bandit.K)
        else:
            # 选择期望奖励估值最大的拉杆
            Kth = np.argmax(self.EstimateReward)
            # 得到本次动作的奖励
            Reward = self.bandit.step(Kth)
            # 更新期望奖励估值
            self.EstimateReward[Kth] += 1.0 / (self.counts[Kth] + 1) * (Reward - self.EstimateReward[Kth])
        return Kth

class DecayingEpsilonGreedy2(ProblemSolver):
    """ epsilon值随时间衰减的epsilon-贪婪算法（另一种衰减策略）,继承Solver类 """
    def __init__(self, bandit, init_prob=1.0, coef=1.0):
        super(DecayingEpsilonGreedy2, self).__init__(bandit)
        # 初始化拉动所有拉杆的期望奖励估值
        self.EstimateReward = np.array([init_prob] * self.bandit.K)
        self.TimeCount = 0
        self.coef = coef

    def RunOnce(self):
        self.TimeCount += 1
        if self.regret > 0:
            d = self.regret
        else:
            d = 1
        coef_epsilon = min(1, (self.coef * self.bandit.K) / (d * d * self.TimeCount))
        if np.random.rand() < coef_epsilon:
            # 随机选择一根拉杆
            Kth = np.random.randint(0, self.bandit.K)
        else:
            # 选择期望奖励估值最大的拉杆
            Kth = np.argmax(self.EstimateReward)
            # 得到本次动作的奖励
            Reward = self.bandit.step(Kth)
            # 更新期望奖励估值
            self.EstimateReward[Kth] += 1.0 / (self.counts[Kth] + 1) * (Reward - self.EstimateReward[Kth])
        return Kth

class UCB(ProblemSolver):
    """ UCB算法,继承Solver类 """
    def __init__(self, bandit, init_prob=1.0, coef=1.0):
        super(UCB, self).__init__(bandit)
        # 初始化拉动所有拉杆的期望奖励估值
        self.EstimateReward = np.array([init_prob] * self.bandit.K)
        self.TimeCount = 0
        self.coef = coef

    def RunOnce(self):
        self.TimeCount += 1
        # 计算上置信界
        ucb = self.EstimateReward + self.coef * np.sqrt(
            np.log(self.TimeCount) / 2 / (self.bandit.K + 1)
        )
        # 选出上置信界最大的拉杆
        Kth = np.argmax(ucb)
        # 得到本次动作的奖励
        Reward = self.bandit.step(Kth)
        # 更新期望奖励估值
        self.EstimateReward[Kth] += 1.0 / (self.counts[Kth] + 1) * (Reward - self.EstimateReward[Kth])
        return Kth

class ThompsonSampling(ProblemSolver):
    """ 汤普森采样算法,继承Solver类 """
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        # 列表,表示每根拉杆奖励为1的次数
        self.SuccessCounter = np.zeros(self.bandit.K)
        # 列表,表示每根拉杆奖励为0的次数
        self.FailureCounter = np.zeros(self.bandit.K)

    def RunOnce(self):
        # 按照Beta分布采样一组奖励样本
        Samples = np.random.beta(self.SuccessCounter + 1, self.FailureCounter + 1)
        # 选出采样奖励最大的拉杆
        Kth = np.argmax(Samples)
        # 得到本次动作的奖励
        Reward = self.bandit.step(Kth)
        if Reward == 1:
            self.SuccessCounter[Kth] += 1
        else:
            self.FailureCounter[Kth] += 1
        return Kth

def PlotResults(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    plt.style.use('seaborn-v0_8-paper')
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

# test01
def test01():
    # 设定随机种子,使实验具有可重复性
    np.random.seed(1)
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    print("随机生成了一个%d臂伯努利老虎机" % K)
    print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" %
          (bandit_10_arm.best_idx, bandit_10_arm.best_prob))

# test02-epsilon
def test02():
    np.random.seed(0)
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    EpsilonGreedySolver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
    EpsilonGreedySolver.RunLoop(5000)
    print('epsilon-贪婪算法的累积懊悔为：', EpsilonGreedySolver.regret)
    PlotResults([EpsilonGreedySolver], ["EpsilonGreedy"])

# test03-multi-epsilon
def test03():
    np.random.seed(0)
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    epsilon_lists = [1e-4, 0.01, 0.1, 0.25, 0.5]
    EpsilonGreedySolvers = [EpsilonGreedy(bandit_10_arm, e) for e in epsilon_lists]
    EpsilonGreedySolversNames = ["epsilon={}".format(e) for e in epsilon_lists]
    for Solver in  EpsilonGreedySolvers:
        Solver.RunLoop(5000)
        print('epsilon-贪婪算法的累积懊悔为：', Solver.regret)
    PlotResults(EpsilonGreedySolvers, EpsilonGreedySolversNames)

# test04-decay-epsilon
def test04():
    np.random.seed(0)
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    DecayingEpsilonGreedySolver = DecayingEpsilonGreedy(bandit_10_arm)
    DecayingEpsilonGreedySolver.RunLoop(5000)
    print('epsilon值衰减的贪婪算法的累积懊悔为：', DecayingEpsilonGreedySolver.regret)
    # print('epsilon值衰减的贪婪算法的每步懊悔为：', DecayingEpsilonGreedySolver.regrets)
    PlotResults([DecayingEpsilonGreedySolver], ["EpsilonGreedy"])

# test05-decay2-epsilon
def test05():
    np.random.seed(0)
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    DecayingEpsilonGreedySolver2 = DecayingEpsilonGreedy2(bandit_10_arm, coef=1.0)
    DecayingEpsilonGreedySolver2.RunLoop(5000)
    print('epsilon值衰减的贪婪算法的累积懊悔为：', DecayingEpsilonGreedySolver2.regret)
    PlotResults([DecayingEpsilonGreedySolver2], ["EpsilonGreedy"])

# test06-UCB
def test06():
    np.random.seed(0)
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    UCBSolver = UCB(bandit_10_arm, coef=2.0)
    UCBSolver.RunLoop(5000)
    print('epsilon值衰减的贪婪算法的累积懊悔为：', UCBSolver.regret)
    PlotResults([UCBSolver], ["UCB"])

# test07-ThompsonSampling
def test07():
    np.random.seed(0)
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    ThompsonSamplingSolver = ThompsonSampling(bandit_10_arm)
    ThompsonSamplingSolver.RunLoop(5000)
    print('epsilon值衰减的贪婪算法的累积懊悔为：', ThompsonSamplingSolver.regret)
    PlotResults([ThompsonSamplingSolver], ["ThompsonSampling"])

if __name__ == '__main__':
    test07()
