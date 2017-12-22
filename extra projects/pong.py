import numpy as np
import gym
import pickle

def sigmoid(x):
    return 1.0/(1.0 +np.exp(-x))

def prepro(I):
    I = I[35:195] #crop
    I = I[::2,::2,0]
    I[I ==144] = 0
    I[I == 109] = 0
    I[I !=0] = 1
    return I.astype(np.float).ravel()

def discount_reward(r):
    disocunted_r = np,zeros.like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        if r[t] != 0: running_add = 0
        running_add  = rnning_add * gamma + r[t]
        discounted_r[t] = running_add
    return disocount_r

def policy_forward(x):
    #w1:(h,d) x:(D,) w2:(h,)
    h = np.dot(model['W1'],x) #(h,1)
    h[h<0] = 0 #ReLu
    z = np.dot(model['W2'], h) #(1)
    p = sigmoid(z)
    return p,h

def policy_backward(eph,epdlogp,epx):
    """ backward pass. (eph is array of intermediate hidden states) """
    #epdlogp:(n,1) eph:(n,h)
    dW2 = eph.T.dot(epdlogp).ravel() #from (h,1) to (h,)
    dh = np.outer(epdlogp,model['W2'])#(n,h)
    dh[eph<=0] = 0 #(n,h)
    dW1 = np.dot(dh.T,epx)

    return {'W1':dW1,'W2':dW2}

  
# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False

#model initialization
D = 80*80

if resume:
    model = pickle.load(open('save.p','rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H,D)/np.sqrt(D)
    model['W2'] = np.random.randn(H)/np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None
xs,hs,dlogps,drs = [],[],[],[]
reward_sum = 0
episode_number = 0
running_reward = None

while True:
    if render: env.render()

    # preprocess the observation, set input to network to be difference image
    cur_x = propro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    #forward
    aprob, h = plocy_foward(x)
    action = 2 if np.random.unform()< aprob else 3

    # record various intermediates (needed later for backprop)
    xs.append(observation)
    hs.append(h) #hidden state
    y = 1 if action == 2 else 0
    dlogps.append(y - aprob)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    drs.append(reward)
    reward_sum += reward

    if done: #an episode finished
        episode_numbet += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs,hs,dlogps,drs = [],[],[],[] # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp, epx)

        for k in model:
            grad_buffer[k] += grad[k]

        if episoid_number % batch_size == 0:
            for k,v in model.iteritems():
                g = grad_buffer[k]
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print ('resetting env. episode reward total was %f. running mean: %f') % (reward_sum, running_reward)

        if episode_number % 100 == 0:
            pickle.dump(model, open('save.p', 'wb'))

        reward_sum = 0
        observation = env.reset() # reset env
        prev_x = None
    
