#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
'''add make_video function to record lateral_distance and heading_error'''

import gym
import gym_carla
import carla
import numpy as np
import tensorflow as tf
import os
import datetime
import cv2

from gym import wrappers


# design the network model
class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output

class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = MyModel(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        
    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)

        print("rewards")
        print(rewards)
        print("value_next")
        print(value_next)
        
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        print("actula_values")
        print(actual_values)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            print("all predicted Q")
            print(self.predict(states))
            print("selected_action_values")
            print(selected_action_values)

            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
            print("loss")
            print(loss)
        
        
        
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        print("gradients")
        print(gradients)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss
        
    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            output = self.predict(np.atleast_2d(states))[0]
            return np.argmax(output)
    
    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())
            
def play_game(env, TrainNet, TargetNet, epsilon, copy_step):
    rewards = 0
    iter = 0
    done = False
    observations = env.reset()
    losses = list()
    while not done:
        # only use state as model input
        state = observations['state']
        action = TrainNet.get_action(state, epsilon)
        prev_observations = observations
        observations, reward, done, _ = env.step(action)
        rewards += reward
        if done:
            reward = -200
            env.reset()

        exp = {'s': prev_observations['state'], 'a': action, 'r': reward, 's2': observations['state'], 'done': done}
        TrainNet.add_experience(exp)
        loss = TrainNet.train(TargetNet)
        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())
        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
    return rewards, np.mean(losses)

def make_video(env, TrainNet):
    rewards = 0
    steps = 0
    done = False
    observation = env.reset()
    
    #frameSize = observation['camera'].shape[0:2]
    #out = cv2.VideoWriter('output_video_10000.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)
    
    log_dir = 'logs/dqn/test_output'
    summary_writer = tf.summary.create_file_writer(log_dir)

    while not done:
        action = TrainNet.get_action(observation['state'], 0)
        observation, reward, done, _ = env.step(action)
        steps += 1
        rewards += reward

        with summary_writer.as_default():
            tf.summary.scalar('lateral distance', observation['state'][0], step=steps)
            tf.summary.scalar('heading error', observation['state'][1], step=steps)
        
        #img = observation['camera']
        #img_dir = 'img/image_{}.png'.format(steps)
        #cv2.imwrite(img_dir,img)
        #out.write(img)
    
    #out.release()
    print("Test model ends, steps: {} total rewards:{} ".format(steps, rewards))
                

def main_test(port=2000):
    # parameters for the gym_carla environment
    params = {
        'number_of_vehicles': 0,
        'number_of_walkers': 0,
        'display_size': 256,  # screen size of bird-eye render
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.1,  # time interval between two frames
        'discrete': True,  # whether to use discrete control space
        'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
        # discrete value of steering angles
        'discrete_steer': [-0.2, 0.0, 0.2],
        'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
        # continuous steering angle range
        'continuous_steer_range': [-0.3, 0.3],
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'port': port,  # connection port
        'town': 'Town04',  # which town to simulate
        # mode of the task, [random, roundabout (only for Town03)]
        'task_mode': 'random',
        'max_time_episode': 1000,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'obs_range': 32,  # observation range (meter)
        'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 8,  # desired speed (m/s)
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
        'display_route': True,  # whether to render the desired route
        'pixor_size': 64,  # size of the pixor labels
        'pixor': False,  # whether to output PIXOR observation
    }

    # Set gym-carla environment
    env = gym.make('carla-v0', params=params).unwrapped
    
    gamma = 0.999
    copy_step = 25
    num_states = len(env.observation_space.sample()['state'])
    # lateral distance between ego car and target lane line center (m)
    # heading error between ego car and target lane line center (rad)
    # ego car speed (m/s)
    # whether there is a front vehicle in safety margin

    #num_actions = env.action_space.n
    num_actions = 9
    hidden_units = [50]
    max_experience = 100000
    min_experience = 10
    batch_size = 2
    lr = 1e-2
    
    # save training log
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/dqn/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    # create DQN model
    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experience, min_experience, batch_size, lr)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experience, min_experience, batch_size, lr)
    
    N = 10
    total_rewards = np.empty(N)
    epsilon = 0.99
    decay = 0.95
    min_epsilon = 0.1
    for n in range(N):
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward, losses = play_game(env, TrainNet, TargetNet, epsilon, copy_step)
        total_rewards[n] = total_reward
        avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
        with summary_writer.as_default():
            tf.summary.scalar('episode reward', total_reward, step=n)
            tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)
            tf.summary.scalar('average loss', losses, step=n)
        if n % 100 == 0:
            print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):", avg_rewards,
                  "episode loss: ", losses)
    print("End: avg reward for last 100 episodes:", avg_rewards)
    
    # make_video(env, TrainNet)
    
    env.close()

"""
    while True:
        action = [2.0, 0.0]
        obs, r, done, info = env.step(action)
        print(obs['birdeye'].shape) # 256,256,3

        if done:
            obs = env.reset()
"""

if __name__ == '__main__':
    main_test()
