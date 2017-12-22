import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque


class DQN():
  # Hyper Parameters for DQN
    GAMMA = 0.9 # discount factor for target Q
    INITIAL_EPSILON = 0.5 # starting value of epsilon
    FINAL_EPSILON = 0.01 # final value of epsilon
    REPLAY_SIZE = 10000 # experience replay buffer size
    BATCH_SIZE = 32 # size of minibatch
  
  # DQN Agent
    def __init__(self, env):
        self.replay_buffer = deque()
        self.time_step = 0
        self.epsilon = INITIAL_EPSOLON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.create_Q_network()
        self.create_training_method()

        self.session= tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
      
    def create_Q_network(self):
      
        self.stateInput = tf.placeholder("float",[None,self.state_dim])
      
        with tf.variable_scope('fc1'):
            W = tf.get_variable("W",shape = [self.state_dim,20],
                                initializer = tf.truncated_normal_initializer(stddev=0.01))
            b = tf.get_variable("b",shape = [20],
                                initializer=tf.constant_initializer(0.01))
            fc1 = tf.nn.relu(tf.matmul(self.stateInput,W) + b)
            
        with tf.variable_scope('fc2'):
            W = tf.get_variable("W",shape = [20,self.action_dim],
                                initializer = tf.truncated_normal_initializer(stddev=0.01))
            b = tf.get_variable("b",shape = [self.action_dim],
                                initializer=tf.constant_initializer(0.01))
            self.Q_value = tf.matmul(fc1,W)+ b
      
    def create_training_method(self):
        self.action_input = tf.placeholder("float",[None,self.action.dim])
        self.y_input = tf.placeholder("float",[None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self,action_input),axis = 1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self,state,action,reward,next_state,done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state,one_hot_action,reward,next_state,done))

        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer)> BATCH_SIZE:
            self.train_Q_network()
            self.time_step += 1
          
    def train_Q_network(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i]+GAMMA*np.max(Q_value_batch[i])

        self.optimizer.run(feed_dict = {
            self.y_input:y_batch,
            self.action_input:action_batch,
            self.state_input:state_batch
            })
        
    def egreedy_action(self,state):
        Q_value = self.Q_value.eval(feed_dict = {self.state_input:[state]})[0]
        if random.random()<= self.epsilon:
            action = random.randrange(self.action_dim)
        else:
            action = np.argmax(Q_value)

        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000                       
        return action
                               
    def action(self,state):                       
        Q_value = self.Q_value.eval(feed_dict = {self.state_input:[state]})[0]                       
        return np.argmax(Q_value)
                               

# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 10000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode

def main():
    env = gym.make(ENV_NAME)
    agent = DQN(env)

    for episode in range(EPISODE):
        state = env.reset()

        for step in range(STEP):
            action =agent.egreedy_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            
            if done:break
        #test
        if episode %100 == 0:
            total_reward = 0

            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = agent.action(state)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            if ave_reward >= 200:
        break

if __name__ == '__main__':
  main()
           
