import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, m, upperLimit):
        self.m = m; # true mean
        self.mean = upperLimit;
        self.N = 1;

    def pull(self):
        return np.random.randn() + self.m # return value centered around true mean

    def update(self, x):
        self.N += 1
        self.mean = (1-1.0/ self.N) * self.mean + 1.0/self.N*x

def run_experiment(m1, m2, m3, N, upperLimit=10):
    bandits = [Bandit(m1, upperLimit), Bandit(m2, upperLimit), Bandit(m3, upperLimit)]

    data = np.empty(N)

    for i in xrange(N):
        # optimistic init vals
        j = np.argmax([b.mean for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)
        data[i] = x

    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

      # plot moving average ctr
    plt.plot(cumulative_average)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log')
    plt.show()

    for b in bandits:
        print(b.mean)

    return cumulative_average

if __name__ == '__main__':
  # c_1 = run_experiment_eps(1.0, 2.0, 3.0, 0.1, 100000)
  oiv = run_experiment(1.0, 2.0, 3.0, 100000)

  # log scale plot
  plt.plot(c_1, label='eps = 0.1')
  plt.plot(oiv, label='optimistic')
  plt.legend()
  plt.xscale('log')
  plt.show()


  # linear plot
  # plt.plot(c_1, label='eps = 0.1')
  plt.plot(oiv, label='optimistic')
  plt.legend()
  plt.show()
