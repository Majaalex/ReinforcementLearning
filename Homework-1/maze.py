# final version for the environment, aka Maze
import numpy as np

actionSpace = {'U': (-1,0), 'D': (1,0), 'L': (0,-1), 'R': (0,1)}  # action space

class Maze(object):
    def __init__(self, maze):  
        self.maze = maze
        self.robotPosition = (0,0)
        self.steps = 0                                          # initial steps to zero
        self.constructAllowedStates()                           # construct the allowed states

    def printMaze(self):
        #%matplotlib inline
        plt.figure(figsize=(10,10))
        plt.show(sns.heatmap(self.maze, cmap='Dark2', cbar=False, linewidths=.5))
        
    def isAllowedMove(self, state, action):
        y, x = state
        y += actionSpace[action][0]                             # extract the coordinates from the action space
        x += actionSpace[action][1]
        if y < 0 or x < 0 or y > self.maze.shape[1]-1 or x > self.maze.shape[0]-1:                    # check if the move is allowed, inside the maze
            return False

        if self.maze[y,x] == 0 or self.maze[y,x] == 2:          #check if the new state is zero (or the actual position of the robot, because not moving is valid)
            return True
        else:
            return False

    def constructAllowedStates(self):                           # construct a dictionary and loop over the maze
        allowedStates= {}
        for y, row in enumerate(self.maze):
            for x, col in enumerate(row):                
                if self.maze[(y,x)] != 1:                       # It goes space by space checking if the actions are allowed, if yes it appended to allowed states dictionary
                    allowedStates[(y,x)] = []
                    for action in actionSpace:
                        if self.isAllowedMove((y,x), action):
                            allowedStates[(y,x)].append(action)
        self.allowedStates = allowedStates

    def updateMaze(self, action):                              
        y,x = self.robotPosition
        self.maze[y,x] = 0                                       # Get the current position of the robot and set to 0
        y += actionSpace[action][0]                              # read the coordinates from the action space diccionary 
        x += actionSpace[action][1]               
        self.robotPosition = (y,x)                               # updates the position of the robot
        self.maze[y,x] = 2                                       # update the mze
        self.steps += 1                                          # adds a new step

    def isGameOver(self):                                        # Check if the position is in the exit.
        x, y = self.get_matrix().shape
        if self.robotPosition == (x - 1, y - 1):
            return True
        else:
            return False
    
    def getStateAndReward(self):                                 
        reward = self.giveReward()
        return self.robotPosition, reward

    def giveReward(self):                                        # Gives the reward of 0 if the robots is in the exit.
        x, y = self.get_matrix().shape
        if self.robotPosition == (x - 1, y - 1):
            return 0
        else:
            return -1
        
    def get_matrix(self):
        return self.maze
