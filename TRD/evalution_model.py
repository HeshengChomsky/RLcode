import argparse
import sys
import gym
import numpy as np
import os

import d4rl
import uuid
import json

import time

import utils.model_utils as model_utils






def train_reverse_bc(env, args):
    fake_env = model_utils.initialize_fake_env(args)
    state=np.random.random([1,11])
    action=np.random.random([1,3])
    next_obs, rew, _ = fake_env.step(state, action)
    print(next_obs.shape)
    print(rew.shape)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="halfcheetah-random-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=1234, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Robust")  # Prepends name to filename
    parser.add_argument("--eval_freq", default=5e4, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=5e5,
                        type=int)  # Max time steps to run environment or train for (this defines buffer size)
    parser.add_argument("--start_timesteps", default=25e3,
                        type=int)  # Time steps initial random policy is used before training behavioral
    parser.add_argument("--rand_action_p", default=0.3,
                        type=float)  # Probability of selecting random action during batch generation
    parser.add_argument("--gaussian_std", default=0.3,
                        type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
    parser.add_argument("--batch_size", default=100, type=int)  # Mini batch size for networks
    parser.add_argument("--train_behavioral", action="store_true")  # If true, train behavioral (DDPG)
    parser.add_argument("--generate_buffer", action="store_true")  # If true, generate buffer
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--train_reverse_bc", action="store_true")  # If true, train RBC
    parser.add_argument("--save_interval", default=50000, type=int)  # Save interval
    parser.add_argument("--load_stamp", default="_205000.0", type=str)  # load_stamp

    # parser.add_argument("--is_uniform_rollout", action="store_true")
    # parser.add_argument("--is_prioritized_reverse_bc", action="store_true")
    parser.add_argument("--is_forward_rollout", default=True)
    parser.add_argument("--state_dim",default=11)
    parser.add_argument("--action_dim",default=3)

    args = parser.parse_args()
    # d4rl.set_dataset_path('/datasets')

    args.task_name = args.env_name
    args.is_uniform_rollout = False
    args.weight_k = 0
    args.is_prioritized_reverse_bc = False
    args.entropy_weight = 0.5
    args.rollout_batch_size = 1000
    args.rollout_length = 5


    if args.task_name[:7] == 'maze2d-' or args.task_name[:8] == 'antmaze-' or \
            args.task_name[:7] == 'hopper-' or args.task_name[:12] == 'halfcheetah-' or \
            args.task_name[:4] == 'ant-' or args.task_name[:9] == 'walker2d-':
        args.forward_model_load_path = 'mopo_models' +  '_forward_{}/'.format(args.seed)
        args.reverse_model_load_path = 'mopo_models' + '_reverse_{}/'.format(args.seed)
    else:
        raise NotImplementedError

    if args.task_name[:6] == 'maze2d':
        args.domain = 'maze2d'
        args.test_model_length = 2
    elif args.task_name[:7] == 'antmaze':
        args.domain = 'antmaze'
        args.test_model_length = 15
    elif args.task_name[:11] == 'halfcheetah':
        args.domain = 'halfcheetah'
        args.test_model_length = 13
    elif args.task_name[:6] == 'hopper':
        args.domain = 'hopper'
        args.test_model_length = 5
    elif args.task_name[:8] == 'walker2d':
        args.domain = 'walker2d'
        args.test_model_length = 13
    elif args.task_name[:4] == 'ant-':
        args.domain = 'ant'
        args.test_model_length = 13
    else:
        raise NotImplementedError

    if args.task_name[:6] == 'maze2d' or args.task_name[:7] == 'antmaze':
        args.test_padding = 0
    elif args.task_name[:6] == 'hopper' or args.task_name[:11] == 'halfcheetah' or args.task_name[:3] == 'ant' or args.task_name[:8] == 'walker2d':
        args.test_padding = 1
    else:
        raise NotImplementedError

    args.save_path = args.task_name + '/' + str(time.time()) + '/'
    print('args.save_path: ', args.save_path)

    # print("---------------------------------------")
    # if args.train_behavioral:
    #     print(f"Setting: Training behavioral, Env: {args.env_name}, Seed: {args.seed}")
    # elif args.generate_buffer:
    #     print(f"Setting: Generating buffer, Env: {args.env_name}, Seed: {args.seed}")
    # else:
    #     print(f"Setting: Training reverse_bc, Env: {args.env_name}, Seed: {args.seed}")
    # print("---------------------------------------")

    results_dir = os.path.join(args.output_dir, 'reverse_bc', str(uuid.uuid4()))
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'params.json'), 'w') as params_file:
        json.dump({
            'env_name': args.env_name,
            'seed': args.seed,
        }, params_file)

    if args.train_behavioral and args.generate_buffer:
        print("Train_behavioral and generate_buffer cannot both be true.")
        exit()

    env=None

    train_reverse_bc(env, args)
