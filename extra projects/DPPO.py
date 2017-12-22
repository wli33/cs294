import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym,threading,queue

EP_MAX = 1000
EP_LEN = 200
N_worker = 4
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.001
MIN_BATCHSIZE = 64
UPDATED_STEP = 5
EPSILON = 0.2
GAME = 'Pendulum-v0'
S_DIM,A_DIM = 3,1

class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32,[None,S_DIM],'state')

        #critic
        l1 = tf.layers.dense(self.tfs,100,tf.nn.relu)
        self.v = tf.layers.dense(l1,1)
        self.tfdc_r = tf.placeholder(tf.float32,[None,1],'discounted_r')
        self.advantage = self.tfdc_r = self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        #actor
        pi,pi_params = self._build_anet('pi',trainable = True)
        oldpi,oldpi_params = self._build_anet('oldpi',trainable = False)
        self.sample_op = tf.squeeze(pi.sample(1),axis = 0)
        self.update_oldpi_op = [oldp.assign(p) for p,oldp in zip(pi_params,oldpi_params)]

        self.tfa = tf.placeholder(tf.float32,[None,A_DIM],'action')
        self.tfadv = tf.placeholder(tf.float32,[None,1],'advantage')
        ratio = pi.prob(self.tfa)/(oldpi_prob(self.tfa)+ 1e-5)
        surr = ratio * self.tfadv

        self.aloss = -tf.reduce_mean(tf.minimum(surr,
                                                tf.clip_by_value(ratio,1.-EPSILON,1.+EPSILON)*self.tfadv))
        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COOD.should_stop():
            if GLOBAL_EP <EP_MAX:
                UPDATE_EVENT.wait()
                self.sess.run(self.update_oldpi_op)
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]
                data = np.vstack(data)
                s,a,r = data[:,:S_DIM],data[:,S_DIM:S_DIM + A_DIM],data[:,-1:]
                adv = self.sess.run(self.advantage,{self.tfs:s,self.tfdc_r:r})
                #update actor and critic in a update loop
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op,{self.tfs:s.self.tfdc_r:r}) for _ in range(UPDATE_STEP)]
                UPDATE_EVENT.clear()
                GLOBAL_UPDATE_COUNTER = 0
                ROLLING_EVENT.set()
    
    def choose_action(self,s):
        s = s[newaxis,:]
        a = self.sess.run(self.sample_op,{self.tfs:s})[0]
        return np.clip(a,-2,2)

    def get_v(self,s):
        #get Qval, e.g. self.v = [[12.57]]
        if s.ndim<2: s = s[np.newaxis,:]
        return self.sess.run(self.v,{self.tfs:s})[0,0]
    
    def _build_anet(self,name,trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs,100,tf.nn.relu,trainable = trainable)
            mu = 2 * tf.layers.dense(l1,A_DIM,tf.nn.tanh.trainable = trainanle)
            sigma = tf.layers.dense(l1,A_DIM,tf.nn.softplus,trainable = trainable)
            norm_dist = tf.distributions.Normal(loc = mu,scale = sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLE,scope = name)
        return norm_dist,params

class Worker(object):
    def __init__(self,wid):
        self.wid = wid
        self.env = gym.make(GAME).unwrapped
        self.ppo = GLOBAL_PPO

        def work(self):
            pass

if __name__ == '__main__':
    GLOBAL_PPO = PPO()
    UPDATE_EVENT,ROLLING_EVENT = threading.Event(),threading.Event()
    UPDATE_EVENT.clear()
    ROLLING_EVENT.set()
    workers = [Worker(wid=i) for i in range(N_WORKER)]

    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()
    threads = []

    for worker in workers:
        t = threading.Thread(target = worker.work, args = ())
        t.start()
        threads.append(t)

    threads.append(threading.Thread(target= GLOBAL_PPO.update,))
    threads[-1].start()
    COORD.join(threads)

    # plot reward change and test the optimized policy
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode'); plt.ylabel('Moving reward'); plt.ion(); plt.show()
    
    env = gym.make('Pendulum-v0')
    while True:
        s = env.reset()
        for t in range(300):
            env.render()
            s = env.step(GLOBAL_PPO.choose_action(s))[0]
    
