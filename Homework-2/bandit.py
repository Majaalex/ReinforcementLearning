# What we need
# Q, N, epsilon, Number of arms, Pull lever and Update_Q.
# 1 simulation = 1000 pulls, average over 2000 simualtions and epsilon = 0.1, 0.01 and 0


import numpy as np
from matplotlib import pyplot as plt


class Bandit(object):
    def __init__(self, numArms, trueRewards, epsilon): # takes as parameters
        self.Q = [0 for i in range(numArms)]           #initial values 
        self.N = [0 for i in range(numArms)]           #initial values
        self.numArms = numArms
        self.epsilon = epsilon           
        self.trueRewards = trueRewards                 # save the true rewards and las action
        self.lastAction = None

    def pull(self):
        rand = np.random.random()                            # uniform distributed number
        if rand <= self.epsilon: 
            whichArm = np.random.choice(self.numArms)        # compare a random number with epsilon , if smaller we take random action 
        elif rand > self.epsilon:
            a = np.array([approx for approx in self.Q])      # otherwise we take the argmax of the estimates of the wewards
            whichArm = np.random.choice(np.where(a == a.max())[0]) # we create a list with the max values, if we get more than one then we choose randomly
        self.lastAction = whichArm                                 # we keep the value of the arm 

        return np.random.randn() + self.trueRewards[whichArm]      # we return a normal distribution reward centered in the true reward for that action

    def updateMean(self, sample):                                  # update the bandits estimate of the rewards
        whichArm = self.lastAction                                 
        self.N[whichArm] += 1                                      # update N
        self.Q[whichArm] = self.Q[whichArm] + 1.0/self.N[whichArm]*(sample - self.Q[whichArm]) # update Q

def simulate(numArms, epsilon, numPulls):
    rewardHistory = np.zeros(numPulls)                              # rewards history to calculate the average
    for j in range(2000):                                           # run 2000 simulations new reward and a new bandit
        rewards = [np.random.randn() for _ in range(numArms)]
        bandit = Bandit(numArms, rewards, epsilon)
        for i in range(numPulls):                                   # evaluate 1000 times and keep track of the weward 
            reward = bandit.pull()
            bandit.updateMean(reward)

            rewardHistory[i] += reward
    average = rewardHistory / 2000                                  # once we finish we compute the average

    return average


# main function

if __name__ == '__main__':
    numActions = 5
    run1 = simulate(numActions,  epsilon=0.1, numPulls=1000)
    run2 = simulate(numActions,  epsilon=0.01, numPulls=1000)
    run3 = simulate(numActions,  epsilon=0.0, numPulls=1000)
    plt.plot(run1, 'b--', run2, 'r--', run3, 'g--')
    plt.legend(['epsilon=0.1', 'epsilon=0.01', 'epsilon=0, Pure greedy'])
    plt.show()
