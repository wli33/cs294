import tensorflow as tf
import numpy as np
import random
from collections import deque
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game

# preprocess raw image to 80*80 gray image
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(80,80,1))

def playFlappyBird():
    # Step 1: init BrainDQN
    brain = BrainDQN()
    # Step 2: init Flappy Bird Game
    flappyBird = game.GameState()
    # Step 3: play game
    # Step 3.1: obtain init state
    action0 = np.array([1,0])  # do nothing
    observation0, reward0, terminal = flappyBird.frame_step(action0)
    observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
    brain.setInitState(observation0)

    # Step 3.2: run the game
    while True:
        action = brain.getAction()
        nextObservation,reward,terminal = flappyBird.frame_step(action)
        nextObservation = preprocess(nextObservation)
        brain.setPerception(nextObservation,action,reward,terminal)

def main():
    playFlappyBird()

class BrainDQN:
    ACTION = 2
    FRAME_PER_ACTION = 1
    GAMMA = 0.99 # decay rate of past observations
    OBSERVE = 1000. # timesteps to observe before training
    EXPLORE = 150000. # frames over which to anneal epsilon
    FINAL_EPSILON = 0.0 # final value of epsilon
    INITIAL_EPSILON = 0.0 # starting value of epsilon
    REPLAY_MEMORY = 50000 # number of previous transitions to remember
    BATCH_SIZE = 32 # size of minibatch
    
    def __init__(self):
        # init replay memory
        self.replayMemory = deque()
        # init Q network
        self.createQNetwork()
        self.timeStep = 0
        self.epsilon = self.INITIAL_EPSILON
        
    def createQNetwork(self):
        
        # input layer
        self.stateInput = tf.placeholder("float",[None,80,80,4])
        
        # network weights
        with tf.variable_scope('Conv1'):
            W = tf.get_variable("W",shape = [8,8,4,32],
                                initializer = tf.truncated_normal_initializer(stddev=0.01))
            b = tf.get_variable("b",shape = [32],
                                initializer=tf.constant_initializer(0.01))
            conv = tf.nn.relu(self.conv2d(self.stateInput,W,4) + b_conv1)
            conv1 = self.max_pool_2x2(h_conv1)
            
        with tf.variable_scope('Conv2'):
            W = tf.get_variable("W",shape = [4,4,32,64],
                                initializer = tf.truncated_normal_initializer(stddev=0.01))
            b = tf.get_variable("b",shape = [64],
                                initializer=tf.constant_initializer(0.01))
            conv2 = tf.nn.relu(self.conv2d(conv1,W,2) + b)
            
         with tf.variable_scope('Conv3'):
            W = tf.get_variable("W",shape = [3,3,64,64],
                                initializer = tf.truncated_normal_initializer(stddev=0.01))
            b = tf.get_variable("b",shape = [64],
                                initializer=tf.constant_initializer(0.01))
            conv3 = tf.nn.relu(self.conv2d(conv2,W,1) + b)
            conv3_flat = tf.reshape(conv3,[-1,1600])

        with tf.variable_scope('fc1'):
            W = tf.get_variable("W",shape = [1600,512],
                                initializer = tf.truncated_normal_initializer(stddev=0.01))
            b = tf.get_variable("b",shape = [512],
                                initializer=tf.constant_initializer(0.01))
            fc1 = tf.nn.relu(tf.matmul(conv3_flat,W) + b)
            
        with tf.variable_scope('fc2'):
            W = tf.get_variable("W",shape = [512,self.ACTION],
                                initializer = tf.truncated_normal_initializer(stddev=0.01))
            b = tf.get_variable("b",shape = [self.ACTION],
                                initializer=tf.constant_initializer(0.01))
            
            self.QValue = tf.matmul(fc1,W) + b

        self.actionInput = tf.placeholder("float",[None,self.ACTION])
        self.yInput = tf.placeholder("float", [None])

        Q_action = tf.reduce_sum(tf.mul(self.QValue, self.actionInput), axis = 1)
        self.cost = tf.reduce_mean(tf.square(self.yInput-Q_action))
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)
        
        # saving and loading networks
        saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(self.session, checkpoint.model_checkpoint_path)
                print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
                print "Could not find old network weights"

    def trainQNetwork(self):
        # Step 1: obtain random minibatch from replay memory
        # minibatch:(state,action,reward,nextstate,terminal)
        
        minibatch = random.sample(self.replayMemory,self.BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

         # Step 2: calculate y 
        y_batch = []
        # QValue_batch:(n,2)
        QValue_batch = self.QValue.eval(feed_dict={self.stateInput:nextState_batch})

        for i in range(self.BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        self.trainStep.run(feed_dict={
            self.yInput : y_batch,
            self.actionInput : action_batch,
            self.stateInput : state_batch
            })

        # save network every 100000 iteration
        if self.timeStep % 10000 == 0:
            saver.save(self.session, 'saved_networks/' + 'network' + '-dqn',
                       global_step = self.timeStep)
            
            

    def setPerception(self,nextObservation,action,reward,terminal):
        newState = np.append(self.currentState[:,:,1:],nextObservation,axis = 2)
        self.replayMemory.append((self.currentState,action,reward,newState,terminal))

        if len(self.replayMemory)> REPLAY_MEMORY:
            self.replayMemory.popleft()
            
        if self.timeStep > OBSERVE:
            self.trainQNetwork()

        self.currentState = newState()
        self.timeStep += 1
            
    def getAction(self):

        action = np.zeros(self.ACTION)
        action_index = 0
        if self.timeStep % self.FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.ACTION)
                action[action_index] = 1
            else:
                QValue = self.QValue.eval(feed_dict= {self.stateInput:[self.currentState]})[0]
                action_index = np.argmax(QValue)
                action[action_index] = 1
        else:
            action[0] = 1 # do nothing

        # change episilon
        if self.epsilon > self.FINAL_EPSILON and self.timeStep > self.OBSERVE:
            self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON)/self.EXPLORE

        return action
    
    def setInitState(self,observation):
        self.currentState = np.stack((observation, observation, observation, observation), axis = 2)

    def conv2d(self,x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

    
