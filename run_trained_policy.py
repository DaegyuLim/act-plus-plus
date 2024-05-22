#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ##
# @brief    [record_episode] 
# @author   Daegyu Lim (dglim@robros.co.kr)   

import rospy
import os
import time
import threading
from std_msgs.msg import Float32MultiArray, Float32
from quest2ros.msg import OVR2ROSInputs, OVR2ROSHapticFeedback
from geometry_msgs.msg import PoseStamped, Twist 
import numpy as np
from scipy.spatial.transform import Rotation
import h5py
import pickle
import argparse
from robot.constants import *
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
import torch
from einops import rearrange

from robot.metaquest_teleop import drlControl
from robot.schunk_gripper_control import gripperControl
from robot.robot_utils import ImageRecorder
# for single robot 
try:
    ROBOT_ID     = rospy.get_param('/dsr/robot_id')
except:
    ROBOT_ID     = "dsr_l"
ROBOT_MODEL = "a0509"


def main(args):
    image_recorder = ImageRecorder(init_node=False)
    dsr = drlControl(ROBOT_ID, HZ, init_node=True, teleop=False)
    gripper = gripperControl(ROBOT_ID, HZ, init_node=False, teleop=False)

    task_config = TASK_CONFIGS[args['task_name']]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']
    
    temporal_ensemble = True
    record_data = False
    use_depth = True
    overwrite = False
    reletive_obs_mode = False
    reletive_action_mode = False
    
    dataset_name = 'real_robot_evaluation'
    print(args['task_name'] + ', ' + dataset_name + '\n' )

    # rate = rospy.Rate(HZ)
    rate = rospy.Rate(15)

    # load model parameters
    ckpt_dir = args['ckpt_dir']
    ckpt_path = os.path.join(ckpt_dir, 'policy_last.ckpt')
    config_path = os.path.join(ckpt_dir, 'config.pkl')
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
        print('config: \n', config)
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.deserialize(torch.load(ckpt_path))
    print(f'Loaded policy from: {ckpt_path} ({loading_status})')
    policy.cuda()

    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # prepare action list for temporal ensemble
    num_queries = policy_config['num_queries']
    num_robot_obs = policy_config['num_robot_observations']
    num_image_obs = policy_config['num_image_observations']
    image_obs_every = policy_config['image_observation_skip']

    image_obs_history = dict()
    depth_obs_history = dict()
    if reletive_obs_mode:
        robot_obs_history = np.zeros((num_robot_obs+1, 7), dtype=np.float32)
        relative_robot_obs_history = np.zeros((num_robot_obs, 7), dtype=np.float32)
    else:
        robot_obs_history = np.zeros((num_robot_obs, 7), dtype=np.float32)
    all_time_actions = np.zeros((max_timesteps, max_timesteps+num_queries, 7), dtype=np.float32)
    


    # print('---dataset stats--- \n', stats)
    # if policy_class == 'Diffusion':
    #     post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    # else:
    pre_process = lambda s: (s - stats['state_mean']) / stats['state_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # saving dataset
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
        exit()

    data_dict = {
        '/observations/xpos': [],
        '/observations/euler': [],
        '/observations/gripper_pos': [],
        '/actions/pose': [],
        '/actions/gripper_pos': []
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        data_dict[f'/observations/depth_images/{cam_name}'] = []


    images = dict()
    depth_images = dict()
    actual_dt_history = []

    # check cameara streaming
    for cam_name in camera_names:
        images = image_recorder.get_images()
        depth_images = image_recorder.get_depth_images()
        while images[cam_name] is None:
            print('waiting '+cam_name+' image streaming...')
            if rospy.is_shutdown():
                break
            time.sleep(1.0)
            images = image_recorder.get_images()
        while depth_images[cam_name] is None:
            print('waiting '+cam_name+' depth image streaming...')
            if rospy.is_shutdown():
                break
            time.sleep(1.0)
            images = image_recorder.get_images()

    

    print('Are you ready?')
    for cnt in range(3):
        print(3 - cnt, 'seconds before to start!!!')
        time.sleep(1.0)

    #ready gripper thread
    gripper_thread = threading.Thread(target=gripper.control_loop)
    gripper_thread.daemon = True
    gripper_thread.start()

    #initialize
    robot_state = np.concatenate([dsr.get_xpos(), dsr.get_euler(), gripper.get_state()], axis=-1)
    # robot_state = pre_process(robot_state)
    robot_obs_history[:] = robot_state

    image_sampling = range(0, (num_image_obs-1)*image_obs_every+1, image_obs_every)
    for cam_name in camera_names:
        first_image = rearrange(image_recorder.get_images()[cam_name], 'h w c -> c h w')
        image_obs_history[cam_name] = np.repeat(first_image[np.newaxis, :, :, :], (num_image_obs-1)*image_obs_every + 1, axis=0) #0xxx0xxx0
        first_depth = np.array([image_recorder.get_depth_images()[cam_name], image_recorder.get_depth_images()[cam_name], image_recorder.get_depth_images()[cam_name]])
        depth_obs_history[cam_name] = np.repeat(first_depth[np.newaxis, :, :, :], (num_image_obs-1)*image_obs_every + 1, axis=0) #0xxx0xxx0
                
    print('Start!')
    time0 = time.time()
    for t in tqdm(range(max_timesteps)):
        t0 = time.time() #
        dsr_state_xpos = dsr.get_xpos()
        dsr_state_euler = dsr.get_euler()
        gripper_state = gripper.get_state()
        dsr_rotation = Rotation.from_euler("ZYZ", dsr_state_euler, degrees=True)

        # if t%60 == 0:
        
        with torch.inference_mode():
            ### get input data
            ## ROBOT STATE INPUT
            robot_state = np.concatenate([dsr_state_xpos, dsr_state_euler, gripper_state], axis=-1)
            
            if num_robot_obs >1:
                robot_obs_history[1:] = robot_obs_history[:-1]
            robot_obs_history[0] = robot_state[:]

            print('robot_obs_history: \n', robot_obs_history)
            if reletive_obs_mode:
                reference_state_pos = robot_obs_history[0, 0:3]
                reference_state_euler = robot_obs_history[0, 3:6]
                reference_state_rotation = Rotation.from_euler("ZYZ", reference_state_euler, degrees=True)
                for t_obs in range(num_robot_obs):
                    # orientation
                    robot_rotation = Rotation.from_euler("ZYZ", robot_obs_history[t_obs+1, 3:6], degrees=True)
                    robot_rotm = reference_state_rotation.as_matrix().transpose() * robot_rotation.as_matrix()
                    rel_robot_rotation = Rotation.from_matrix(robot_rotm)
                    relative_robot_obs_history[t_obs, 3:6] = rel_robot_rotation.as_rotvec()
                    # position
                    relative_robot_obs_history[t_obs, 0:3] = (reference_state_rotation.as_matrix().transpose()).dot(robot_obs_history[t_obs+1, 0:3] - reference_state_pos)
                # pass gripper position
                relative_robot_obs_history[:, 6] = robot_obs_history[:-1, 6]
                # print('relative_robot_obs_history pre: \n', relative_robot_obs_history)
                relative_robot_obs_history = pre_process(relative_robot_obs_history)
                # print('relative_robot_obs_history post: \n', relative_robot_obs_history)
                robot_obs_history_flat = relative_robot_obs_history.reshape([-1])
            else:
                robot_obs_history_flat = pre_process(robot_obs_history[:])
                robot_obs_history_flat = robot_obs_history_flat.reshape([-1])
            
            t1 = time.time()
            ## CAMERA INPUT
            for cam_name in camera_names:
                current_image = rearrange(image_recorder.get_images()[cam_name], 'h w c -> c h w')
                
                if num_image_obs >1:
                    image_obs_history[cam_name][1:] =  image_obs_history[cam_name][:-1]
                    image_obs_history[cam_name][0] = current_image
                else:
                    image_obs_history[cam_name][0] = current_image

            if use_depth:
                current_depth = np.array([image_recorder.get_depth_images()[cam_name], image_recorder.get_depth_images()[cam_name], image_recorder.get_depth_images()[cam_name]])
                
                if num_image_obs >1:
                    depth_obs_history[cam_name][1:] =  depth_obs_history[cam_name][:-1]
                    depth_obs_history[cam_name][0] = current_depth
                else:
                    depth_obs_history[cam_name][0] = current_depth
                

            all_cam_images = []
            for cam_name in camera_names:
                all_cam_images.append(np.array(image_obs_history[cam_name])[image_sampling])

            if use_depth:
                for cam_name in camera_names:
                    all_cam_images.append(np.array(depth_obs_history[cam_name])[image_sampling])

            all_cam_images = np.stack(all_cam_images, axis=0)

            t2 = time.time()
            # move data into GPU
            robot_obs_history_flat = torch.from_numpy(robot_obs_history_flat).float().cuda().unsqueeze(0)
            cam_images = torch.from_numpy(all_cam_images / 255.0).float().cuda().unsqueeze(0)
            
            # policy inference
            all_actions = policy(robot_obs_history_flat, cam_images) # action dim: [1, chunk_size, action_dim]
            all_actions = all_actions.cpu().numpy()
            all_actions = np.squeeze(all_actions, axis =0)
            all_actions[:] = post_process(all_actions[:])

            t3 = time.time()
            print('all_actions: \n', all_actions[:30])
            if reletive_action_mode:
                dsr_state_rotm = dsr_rotation.as_matrix()
                rel_all_actions = all_actions
                
                rel_all_actions_rotation = Rotation.from_rotvec(rel_all_actions[:, 3:6])
                all_actions_rotm = dsr_state_rotm * rel_all_actions_rotation.as_matrix()[:]
                all_actions_rotation = Rotation.from_matrix(all_actions_rotm)
                # position
                for i in range(num_queries):
                    all_actions[i, 0:3] = dsr_state_xpos[:] + dsr_state_rotm.dot(rel_all_actions[i, 0:3])
                    # euler
                    all_actions[i, 3:6] = all_actions_rotation[i].as_euler("ZYZ", degrees=True)
            
            # print('all_actions post: \n', all_actions)
            # 
            if temporal_ensemble:
                all_time_actions[t, t:t+num_queries, :] = all_actions[:]
                action_bottom_idx = max(t-num_queries, 0)
                actions_for_curr_step = all_time_actions[ action_bottom_idx:(t+1), t, :] # (chunk_size, action_dim)
                # actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                # print('actions_populated: ', actions_populated)
                # actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.05
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = np.expand_dims(exp_weights, axis=1).transpose() # (1, chunk_size)

                # print(actions_for_curr_step.shape)
                # print(exp_weights.shape)
                action = np.sum(  exp_weights.dot(actions_for_curr_step), axis=0, keepdims=False)
            else:
                # print('all_actions shape: ', all_actions.size())
                action = all_actions[0][:]
                # print('action: ', action)
            
            # action = np.squeeze(action, axis=0)
            
            # print('all_action: ', all_action)
        

            print('action: ', action)
            print('robot_state: ', robot_state)
            # dsr.set_action(action[0][:6])
            # gripper.set_action(action[0][-1])
            # for i in range(10):
            #     t1 = time.time()
            #     dsr.step()
            #     t2 = time.time()
            
            t4 = time.time() #

            # 

            # print("t1 - t0: ", t1-t0)
            # print("t2 - t1: ", t2-t1)
            # print("t3 - t2: ", t3-t2)
            # print("t4 - t3: ", t4-t3)
            # print("t4 - t0: ", t4-t0)
            if rospy.is_shutdown():
                break
        
        ## command robot
        dsr.set_action(action[:6])
        gripper.set_action(action[-1])

        if record_data:
            dsr_action = dsr.get_action()
            gripper_action = gripper.get_action()
            # print('dsr_action: ', dsr_action)
            # print('gripper_action: ', gripper_action)

            data_dict['/observations/xpos'].append(dsr_state_xpos)
            data_dict['/observations/euler'].append(dsr_state_euler)
            data_dict['/observations/gripper_pos'].append(gripper_state)
            data_dict['/actions/pose'].append(dsr_action)
            data_dict['/actions/gripper_pos'].append(gripper_action)

            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append((image_recorder.get_images())[cam_name])
                data_dict[f'/observations/depth_images/{cam_name}'].append((image_recorder.get_depth_images())[cam_name])
        # t2 = time.time() #
        # actual_dt_history.append([t0, t1, t2])

        # print("tdsr - t0: ", tdsr-t0)
        # print("t1 - tdsr: ", t1-tdsr)
        # print("t2 - t1: ", t2-t1)
        # if t2-t1>0.1:
        #     print("DSR commad duration is too long (over 100ms)")
        
        rate.sleep()
        
    
    print(f'Avg Control Frequency [Hz]: {dsr.tick / (time.time() - time0)}')

    # stop dsr
    dsr.stop()
    # open gripper
    gripper.open()
    time.sleep(0.1)
    gripper.stop_control_loop = True
    # shutdown controllers
    # dsr.shutdown()
    # gripper.shutdown()

    if record_data:
        COMPRESS = True

        if COMPRESS:
            # JPEG compression
            t0 = time.time()
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50] # tried as low as 20, seems fine
            compressed_image_len = []
            compressed_depth_len = []
            for cam_name in camera_names:
                image_list = data_dict[f'/observations/images/{cam_name}']
                depth_list = data_dict[f'/observations/depth_images/{cam_name}']
                compressed_list = []
                compressed_depth_list = []
                compressed_image_len.append([])
                compressed_depth_len.append([])
                
                for image in image_list:
                    result, encoded_image = cv2.imencode('.jpg', image, encode_param) # 0.02 sec # cv2.imdecode(encoded_image, 1)
                    compressed_list.append(encoded_image)
                    compressed_image_len[-1].append(len(encoded_image))
                    if result == False:
                        print('Error during image compression')
                for depth in depth_list:
                    result, encoded_depth = cv2.imencode('.jpg', depth, encode_param) # 0.02 sec # cv2.imdecode(encoded_image, 1)
                    compressed_depth_list.append(encoded_depth)
                    compressed_depth_len[-1].append(len(encoded_depth))
                    if result == False:
                        print('Error during depth image compression')
                data_dict[f'/observations/images/{cam_name}'] = compressed_list
                data_dict[f'/observations/depth_images/{cam_name}'] = compressed_depth_list
            print(f'compression: {time.time() - t0:.2f}s')

            # pad so it has same length
            t0 = time.time()
            compressed_image_len = np.array(compressed_image_len)
            compressed_depth_len = np.array(compressed_depth_len)

            padded_size = compressed_image_len.max()
            for cam_name in camera_names:
                compressed_image_list = data_dict[f'/observations/images/{cam_name}']
                padded_compressed_image_list = []
                for compressed_image in compressed_image_list:
                    padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                    image_len = len(compressed_image)
                    padded_compressed_image[:image_len] = compressed_image
                    padded_compressed_image_list.append(padded_compressed_image)
                data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list

            padded_size2 = compressed_depth_len.max()
            for cam_name in camera_names:
                compressed_depth_list = data_dict[f'/observations/depth_images/{cam_name}']
                padded_compressed_depth_list = []
                for compressed_depth in compressed_depth_list:
                    padded_compressed_depth = np.zeros(padded_size2, dtype='uint8')
                    depth_len = len(compressed_depth)
                    padded_compressed_depth[:depth_len] = compressed_depth
                    padded_compressed_depth_list.append(padded_compressed_depth)
                data_dict[f'/observations/depth_images/{cam_name}'] = padded_compressed_depth_list
            print(f'padding: {time.time() - t0:.2f}s')

        # HDF5
        t0 = time.time()
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
            # root.attrs['sim'] = False
            root.attrs['compress'] = COMPRESS
            obs = root.create_group('observations')
            actions = root.create_group('actions')
            image = obs.create_group('images')
            depth = obs.create_group('depth_images')
            for cam_name in camera_names:
                if COMPRESS:
                    _ = image.create_dataset(cam_name, (max_timesteps, padded_size), dtype='uint8',
                                            chunks=(1, padded_size), )
                    _ = depth.create_dataset(cam_name, (max_timesteps, padded_size2), dtype='uint8',
                                            chunks=(1, padded_size2), )
                else:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                            chunks=(1, 480, 640, 3), )
                    _ = depth.create_dataset(cam_name, (max_timesteps, 480, 640), dtype='uint8',
                                            chunks=(1, 480, 640, 1), )
            _ = obs.create_dataset('xpos', (max_timesteps, 3))
            _ = obs.create_dataset('euler', (max_timesteps, 3))
            _ = obs.create_dataset('gripper_pos', (max_timesteps, 1))
            _ = actions.create_dataset('pose', (max_timesteps, 6))
            _ = actions.create_dataset('gripper_pos', (max_timesteps, 1))

            for name, array in data_dict.items():
                root[name][...] = array

            if COMPRESS:
                _ = root.create_dataset('compressed_image_len', (len(camera_names), max_timesteps))
                _ = root.create_dataset('compressed_depth_len', (len(camera_names), max_timesteps))
                root['/compressed_image_len'][...] = compressed_image_len
                root['/compressed_depth_len'][...] = compressed_depth_len

        print(f'Saving: {time.time() - t0:.1f} secs')

def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='Check Point Directory.', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', default='dsr_block_collect', required=False)
    main(vars(parser.parse_args())) # TODO
    # debug()