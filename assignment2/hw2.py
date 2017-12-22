# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:58:51 2017

@author: zdj07
"""
import gym
from frozen_lake import FrozenLakeEnv
import numpy as np, numpy.random as nr

env = FrozenLakeEnv()
# Some basic imports and setup

np.set_printoptions(precision=3)
def begin_grading(): print("\x1b[43m")
def end_grading(): print("\x1b[0m")
#
## Seed RNGs so you get the same printouts as me
#env.seed(0); from gym.spaces import prng; prng.seed(10)
## Generate the episode
#env.reset()
#for t in range(100):
#    env.render()
#    a = env.action_space.sample()
#    ob, rew, done, _ = env.step(a)
#    if done:
#        break
#assert done
#env.render();
          
class MDP(object):
    def __init__(self, P, nS, nA, desc=None):
        self.P = P # state transition and reward probabilities, explained below
        self.nS = nS # number of states
        self.nA = nA # number of actions
        self.desc = desc # 2D array specifying what each grid cell means (used for plotting)
mdp = MDP( {s : {a : [tup[:3] for tup in tups] for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()}, env.nS, env.nA, env.desc)

def value_iteration(mdp, gamma, nIt):
    """
    Inputs:
        mdp: MDP
        gamma: discount factor
        nIt: number of iterations, corresponding to n above
    Outputs:
        (value_functions, policies)
        
    len(value_functions) == nIt+1 and len(policies) == n
    """
    print("Iteration | max|V-Vprev| | # chg actions | V[0]")
    print("----------+--------------+---------------+---------")
    Vs = [np.zeros(mdp.nS)] # list of value functions contains the initial value function V^{(0)}, which is zero
    pis = []
    for it in range(nIt):
        oldpi = pis[-1] if len(pis) > 0 else None # \pi^{(it)} = Greedy[V^{(it-1)}]. Just used for printout
        Vprev = Vs[-1] # V^{(it)}
        # YOUR CODE HERE
        # Your code should define the following two variables
        # pi: greedy policy for Vprev, 
        #     corresponding to the math above: \pi^{(it)} = Greedy[V^{(it)}]
        #     numpy array of ints
        # V: bellman backup on Vprev
        #     corresponding to the math above: V^{(it+1)} = T[V^{(it)}]
        #     numpy array of floats
        
        nA = mdp.nA
        nS = mdp.nS
        pi = np.zeros(nS)
        V = np.zeros(nS)
        
        for state in range(nS):
            expected_values = np.zeros(nA)
            
            for action in range(nA):
                expected_value = 0
                for s_prime in mdp.P[state][action]:
                    expected_value += s_prime[0]*(s_prime[2]+ gamma*Vprev[s_prime[1]])
                expected_values[action] = expected_value

            pi[state] = np.argmax(expected_values)
            V[state] = np.max(expected_values)
                
        max_diff = np.abs(V - Vprev).max()
        nChgActions="N/A" if oldpi is None else (pi != oldpi).sum()
        print("%4i      | %6.5f      | %4s          | %5.3f"%(it, max_diff, nChgActions, V[0]))
        Vs.append(V)
        pis.append(pi)
    return Vs, pis

GAMMA=0.95 # we'll be using this same value in subsequent problems
begin_grading()
Vs_VI, pis_VI = value_iteration(mdp, gamma=GAMMA, nIt=20)
end_grading()

import matplotlib.pyplot as plt
plt.figure()
plt.plot(Vs_VI)
plt.title("Values of different states")

chg_iter = 50
# YOUR CODE HERE
# Your code will need to define an MDP (mymdp)
# like the frozen lake MDP defined above
num_states = 3 # 0, 1, 2
num_actions = 2 # 0, 1

transitions = {0: {0: [(1.0, 1, 0.0)], 1: [(1.0, 0, 0.0993)]}, 
 1: {0: [(1.0, 2, 1.0)], 1: [(0.5, 2, 0.0), (0.5, 0, 0.2)]}, 
 2: {0: [(1.0, 2, 0.0)], 1: [(1.0, 2, 0.0)]}}
          
mymdp = MDP(transitions, num_states , num_actions, 
          "This is an MDP designed for Value Iteration to take a long time to converge")

begin_grading()
Vs, pis = value_iteration(mymdp, gamma=GAMMA, nIt=chg_iter+1)
end_grading()

def compute_vpi(pi, mdp, gamma):
    # YOUR CODE HERE
    nS = mdp.nS
    b = np.zeros(nS)
    a = np.eye(nS)

    for state in range(nS):
        action = pi[state]
        b_state = 0
        for s_prime in mdp.P[state][action]:
            b_state += s_prime[0]*s_prime[2]
            a_coeff = gamma*s_prime[0]
            a[state][s_prime[1]] -= a_coeff
        b[state] = b_state

    V = np.linalg.solve(a,b) 
    return V
begin_grading()
print(compute_vpi(np.ones(16), mdp, gamma=GAMMA))
end_grading()

def compute_qpi(vpi,mdp,gamma):
    # YOUR CODE HERE
    nS = mdp.nS
    nA = mdp.nA
    Qpi = np.zeros((nS,nA))

    for state in range(nS):
        for action in range(nA):
            for s_prime in mdp.P[state][action]:
                Qpi[state][action] += s_prime[0]*(s_prime[2]+ gamma*vpi[s_prime[1]])
              
    return Qpi

begin_grading()
Qpi = compute_qpi(np.arange(mdp.nS), mdp, gamma=0.95)
print("Qpi:\n", Qpi)
end_grading()

def policy_iteration(mdp, gamma, nIt):
    Vs = []
    pis = []
    pi_prev = np.zeros(mdp.nS,dtype='int')
    pis.append(pi_prev)
    print("Iteration | # chg actions | V[0]")
    print("----------+---------------+---------")
    for it in range(nIt):        
        vpi = compute_vpi(pi_prev, mdp, gamma)
        qpi = compute_qpi(vpi, mdp, gamma)
        pi = qpi.argmax(axis=1)
        print("%4i      | %6i        | %6.5f"%(it, (pi != pi_prev).sum(), vpi[0]))
        Vs.append(vpi)
        pis.append(pi)
        pi_prev = pi
    return Vs, pis
Vs_PI, pis_PI = policy_iteration(mdp, gamma=0.95, nIt=20)
plt.plot(Vs_PI);