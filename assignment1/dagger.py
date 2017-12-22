"""
DAgger implemented to clone an expert policy.
Example usage:
    python dagger.py experts/Humanoid-v1.pkl Humanoid-v1 Humanoid-v1_10_data.pkl \
        --render --num_rollouts 20
"""

import pickle
import numpy as np
import tensorflow as tf
import tf_util
import gym
import load_policy
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def load_data(filename):
    with open(filename,'rb') as f:
        data = pickle.load(f,encoding='iso-8859-1')
        obs_data = np.array(data['observations'])
        obs_data = np.expand_dims(obs_data, axis=2)
        act_data = np.array(data['actions'])
        act_data = np.squeeze(act_data)
    return obs_data,act_data

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('data_file', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument("--max_timesteps", type=int)
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    # Set Parameters
    task = args.envname
    task_data = args.data_file
    mean_rewards = []
    stds = []

    # Load in expert policy observation data
    X,Y = load_data(task_data)
    
    # Create a feedforward neural network
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(Y.shape[2], activation='linear'))
    model.compile(loss='msle', optimizer='adam', metrics=['accuracy'])
    model.save('models/' + task + '_dagger_model.h5')

    # Main DAGGER Loop
    for i in range(5):
        # 1) Train policy on D
        # Split data into train and test set
        
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)

        # Train model on dataset
        model = load_model('models/' + task + '_dagger_model.h5')
        model.fit(X_train, Y_train, batch_size=64, nb_epoch=30, verbose=1)

        score = model.evaluate(X_test, Y_test, verbose=1)
        model.save('models/' + task + '_dagger_model.h5')

        # 2) Run policy on simulation and 3) Expert labels on these observations
        with tf.Session():
            tf_util.initialize()
            env = gym.make(args.envname)
            max_steps = args.max_timesteps or env.spec.timestep_limit

            returns = []
            new_observations = []
            new_exp_actions = []

            model = load_model('models/' + task + '_dagger_model.h5')
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    # get Expert labels u on these observations
                    obs = np.array(obs)
                    exp_action = policy_fn(obs[None,:])

                    new_observations.append(obs[None,:])
                    new_exp_actions.append(exp_action)
                    
                    # Run policy to get dataset O
                    obs = obs.reshape(1, len(obs), 1)
                    action = (model.predict(obs, batch_size=64, verbose=0))
 
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))
            mean_rewards.append(np.mean(returns))
            stds.append(np.std(returns))

            new_observations = np.array(new_observations)
            new_exp_actions = np.array(new_exp_actions)

        # 4) Aggregate new data to old
        
        X = np.concatenate((X, new_observations))
        Y = np.concatenate((Y, new_exp_actions))

    print(mean_rewards)
    print(stds)

if __name__ == '__main__':
    main()
