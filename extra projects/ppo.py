import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym

EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization

class PPO(object):
    def __init__(self):
        self,sess = tf.Session()
        self.tfs = tf.placeholder(tf,float32,[None,S_DIM],'state')

        #critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs,100, tf.nn.relu)
            self.v = tf.layers.dense(l1,1)
            self.tfdc_r = tf.placeholder(tf.float32,[None,1],'discounted_r')
            self.advantage = self.tfdc_r - self.v
