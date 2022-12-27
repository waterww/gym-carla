import gym
import gym_carla
import carla
import numpy as np
import tensorflow as tf
import os
import datetime

from carla_train import test_model,DQN


def main(port=2000, file_dir='/home/miao/hpc_output_files/final_model_1221/20221220-201435'):
    # parameters for the gym_carla environment
    params = {
        'number_of_vehicles': 0,
        'number_of_walkers': 0,
        'display_size': 256,  # screen size of bird-eye render
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.1,  # time interval between two frames
        'discrete': True,  # whether to use discrete control space
        'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
        'discrete_steer': [-0.2, 0.0, 0.2], # discrete value of steering angles
        'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
        'continuous_steer_range': [-0.3, 0.3], # continuous steering angle range
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

    # parameters for DQN
    gamma = 0.999
    copy_step = 25
    num_states = len(env.observation_space.sample()['state'])
    #num_actions = env.action_space.n
    num_actions = 9
    hidden_units = [50]
    max_experience = 100000
    min_experience = 100
    batch_size = 64
    lr = 5e-4

    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experience, min_experience, batch_size, lr)

    # load the saved model
    model = tf.keras.models.load_model(file_dir)
    TrainNet.model = model

    # test the model and make a video
    test_model(env, TrainNet, -4)

    env.close()

if __name__ == "__main__":
    main()