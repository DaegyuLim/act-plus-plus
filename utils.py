import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import cv2
from time import time
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation

import IPython
e = IPython.embed

def flatten_list(l):
    return [item for sublist in l for item in sublist]

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list, camera_names, norm_stats, episode_ids, episode_len, chunk_size, robot_obs_size, img_obs_size, img_obs_skip, policy_class, use_depth):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.robot_obs_size = robot_obs_size
        self.img_obs_size = img_obs_size
        self.img_obs_skip = img_obs_skip
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.policy_class = policy_class
        
        self.augment_images = False
        self.transformations = False
        self.img_debug = False

        self.use_depth = use_depth
    
        self.reletive_action_mode = False
        self.reletive_obs_mode = False
        self.__getitem__(0) # initialize self.is_sim and self.transformations
        self.is_sim = False
        

    # def __len__(self):
    #     return sum(self.episode_len)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        t0 = time()
        try:
            with h5py.File(dataset_path, 'r') as root:
                try: # some legacy data does not have this attribute
                    is_sim = root.attrs['sim']
                except:
                    is_sim = False

                compressed = root.attrs.get('compress', False)
                if '/base_action' in root:
                    base_action = root['/base_action'][()]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([ root['/actions/pose'][()], root['/actions/gripper_pos'][()], base_action ], axis=-1)
                else:  
                    action = np.concatenate([ root['/actions/pose'][()], root['/actions/gripper_pos'][()] ], axis=-1)
                    # dummy_base_action = np.zeros([action.shape[0], 2])
                    # action = np.concatenate([action, dummy_base_action], axis=-1)

                # print('action[6]: ', action[:, 6])
                original_action_shape = action.shape
                episode_len = original_action_shape[0]
                # get observation at start_ts only
                xpos = root['/observations/xpos'][()]
                euler = root['/observations/euler'][()]
                gripper_pos = root['/observations/gripper_pos'][()]
                t1 = time()
                
                robot_state = np.concatenate( [xpos, euler, gripper_pos], axis=-1)
                original_robot_shape = robot_state.shape
                t2 = time()
                # print('robot_state shape', original_robot_shape) # (2000, 7)
                image_dict = dict()
                img_sampling = np.clip( range(start_ts, start_ts - self.img_obs_skip*self.img_obs_size, -self.img_obs_skip), 0, self.max_episode_len)
 
                for cam_name in self.camera_names:
                    image_dict[cam_name] = np.array(root[f'/observations/images/{cam_name}'])[img_sampling]
                t3 = time()
                if compressed:
                    for cam_name in image_dict.keys():
                        decompressed_image_array = []
                        for i in range(len(image_dict[cam_name])):
                            decompressed_image = cv2.imdecode(image_dict[cam_name][i], 1)
                            # cv2.imshow('decoding', decompressed_image)
                            # cv2.waitKey()
                            decompressed_image_array.append(decompressed_image)
                        image_dict[cam_name] = np.array(decompressed_image_array)
                        # print('image_dict[cam_name].shape', image_dict[cam_name].shape)
                t4 = time()
                
                if self.use_depth:
                    depth_image_dict = dict()

                    for cam_name in self.camera_names:
                        depth_image_dict[cam_name] = np.array(root[f'/observations/depth_images/{cam_name}'])[img_sampling]

                    if compressed:
                        for cam_name in depth_image_dict.keys():
                            decompressed_depth_array = []
                            for i in range(len(depth_image_dict[cam_name])):
                                decompressed_depth = cv2.imdecode(depth_image_dict[cam_name][i], 1)
                                # cv2.imshow('decoding', decompressed_depth)
                                # cv2.waitKey()
                                decompressed_depth_array.append(decompressed_depth)
                            depth_image_dict[cam_name] = np.array(decompressed_depth_array)
                # print('depth_image_dict[cam_name].size: ', np.shape( np.array(depth_image_dict[cam_name])) )
               
                # get all actions after and including start_ts 
                action = action[start_ts:] 
                action_len = episode_len - start_ts 
                # get all states before and including start_ts 
                robot_state = robot_state[:start_ts+1] 
                robot_state_len = start_ts+1 
            
            
            # self.is_sim = is_sim
            padded_action = np.zeros((self.max_episode_len, original_action_shape[1]), dtype=np.float32)
            padded_action[:action_len] = action
            is_pad = np.zeros(self.max_episode_len, dtype=np.bool_)
            is_pad[action_len:] = 1

            padded_action = padded_action[:self.chunk_size]
            is_pad = is_pad[:self.chunk_size]

            padded_robot_state = np.zeros((self.max_episode_len, original_robot_shape[1]), dtype=np.float32)
            padded_robot_state[:] = robot_state[0]
            padded_robot_state[-robot_state_len:] = robot_state
            
            if self.reletive_obs_mode == False:
                padded_robot_state = padded_robot_state[-self.robot_obs_size:]
            else:
                padded_robot_state = padded_robot_state[-(self.robot_obs_size+1):]
            padded_robot_state = padded_robot_state[::-1] # padded_robot_state[0] is the current observation at start_ts
            
            t5 = time()
            # print('padded_robot_state: ', padded_robot_state[0, 0:6])
            # print('padded_action pre: ', padded_action[:10, 0:6])

            if self.reletive_action_mode:
                reference_state_pos = padded_robot_state[0, 0:3]
                reference_state_euler = padded_robot_state[0, 3:6]
                reference_state_rotation = Rotation.from_euler("ZYZ", reference_state_euler, degrees=True)

                for t in range(self.chunk_size):
                    # orientation
                    action_rotation = Rotation.from_euler("ZYZ", padded_action[t, 3:6], degrees=True)
                    action_rotm = reference_state_rotation.as_matrix().transpose() * action_rotation.as_matrix()
                    rel_action_rotation = Rotation.from_matrix(action_rotm)
                    padded_action[t, 3:6] = rel_action_rotation.as_rotvec()
                    # position
                    padded_action[t, 0:3] = (reference_state_rotation.as_matrix().transpose()).dot(padded_action[t, 0:3] - reference_state_pos)
                    # print('padded_action: ', t, ', ', padded_action[t, 0:3], ', ', padded_action[t, 3:6])
                    # print( (reference_state_rotation.as_matrix()))
                    # print( (reference_state_rotation.as_matrix().transpose()))

            if self.reletive_obs_mode:
                reference_state_pos = padded_robot_state[0, 0:3]
                reference_state_euler = padded_robot_state[0, 3:6]
                reference_state_rotation = Rotation.from_euler("ZYZ", reference_state_euler, degrees=True)
                for t in range(self.robot_obs_size):
                    # orientation
                    robot_rotation = Rotation.from_euler("ZYZ", padded_robot_state[t+1, 3:6], degrees=True)
                    robot_rotm = reference_state_rotation.as_matrix().transpose() * robot_rotation.as_matrix()
                    rel_robot_rotation = Rotation.from_matrix(robot_rotm)
                    padded_robot_state[t+1, 3:6] = rel_robot_rotation.as_rotvec()
                    # position
                    padded_robot_state[t+1, 0:3] = (reference_state_rotation.as_matrix().transpose()).dot(padded_robot_state[t+1, 0:3] - reference_state_pos)
                    padded_robot_state[:-1, 0:6] = padded_robot_state[1:, 0:6] # pop relative current state which is always be identity
                
                padded_robot_state = padded_robot_state[:self.robot_obs_size] 

            t6 = time()
            # new axis for different cameras
            all_cam_images = []
            for cam_name in self.camera_names:
                all_cam_images.append(image_dict[cam_name])
            
            if self.use_depth:
                for cam_name in self.camera_names:
                    all_cam_images.append(depth_image_dict[cam_name])

            all_cam_images = np.stack(all_cam_images, axis=0)

            # print('all_cam_images:', all_cam_images.shape)
            # all_cam_images = all_cam_images[:, :start_ts+1]
            # image_len = start_ts+1 

            # original_image_stack_shape = all_cam_images.shape ## (camera num, epi len, h, w, c)

            # padded_all_cam_images = np.zeros((original_image_stack_shape[0], self.max_episode_len, original_image_stack_shape[2], original_image_stack_shape[3], original_image_stack_shape[4]), dtype=np.int8)
            # for cam_id in range(original_image_stack_shape[0]):
            #     padded_all_cam_images[cam_id, :] = all_cam_images[cam_id, 0]
            #     padded_all_cam_images[cam_id, :] = all_cam_images[cam_id, 0]

            # padded_all_cam_images[:, -image_len:] = all_cam_images[:, :start_ts+1] 
            # padded_all_cam_images = padded_all_cam_images[:, -self.img_obs_size:]
            # padded_all_cam_images = padded_all_cam_images[:, ::-1]
            t7 = time()

            # construct torch data
            action_data = torch.from_numpy(np.array(padded_action)).float()
            robot_state_data = torch.from_numpy(np.array(padded_robot_state)).float()
            image_data = torch.from_numpy(np.array(all_cam_images))
            is_pad = torch.from_numpy(is_pad).bool()

            
            # channel last
            image_data = torch.einsum('k t h w c -> k t c h w', image_data)
            t8 = time()
            # augmentation
            if self.transformations:
                original_size = image_data.shape[3:]
                # single_camera_size = original_size.copy()
                # single_camera_size[1] *= 0.5
                # print(original_size, single_camera_size)
                ratio = 0.95
                self.transformations = [
                    transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                    transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1]/2 * ratio)]),
                    transforms.Resize(original_size[0]),
                    transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.04)
                ]

            image_data_for_show = torch.einsum('c h w -> h w c', image_data[0, 0, :]).numpy()
            # print(image_data_for_show.shape)
            
            if self.img_debug:
                cv2.imshow('original', image_data_for_show)
                cv2.waitKey(0)

            if self.augment_images:
                for k in range(image_data.shape[0]):
                    for t in range(image_data.shape[1]):
                        temp_left_image = image_data[k, t, :, :, :int(original_size[1]/2)].clone()
                        print('temp_left_image.shape: ', temp_left_image.shape)
                        temp_right_image = image_data[k, t, :, :, int(original_size[1]/2):].clone()
                        print('temp_right_image.shape: ', temp_right_image.shape)
                        for transform in self.transformations:
                            temp_left_image = transform(temp_left_image)
                            temp_right_image = transform(temp_right_image)

                            image_data_for_show = torch.einsum('c h w -> h w c', temp_left_image).numpy()
                            if self.img_debug:
                                cv2.imshow(f'transform', image_data_for_show)
                                cv2.waitKey(0)
                        image_data[k, t, :, :, :int(original_size[1]/2)] = temp_left_image.clone()
                        image_data[k, t, :, :, int(original_size[1]/2):] = temp_right_image.clone()
                        if self.img_debug:
                            image_data_for_show = torch.einsum('c h w -> h w c', image_data[k, t, :]).numpy()
                            cv2.imshow('finally transfomed', image_data_for_show)
                            cv2.waitKey(0)
            # normalize image and change dtype to float
            image_data = image_data / 255.0

            if self.policy_class == 'Diffusion':
                # normalize to [-1, 1]
                action_data = ((action_data - self.norm_stats["action_min"]) / (self.norm_stats["action_max"] - self.norm_stats["action_min"])) * 2 - 1
            else:
                # normalize to mean 0 std 1
                action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

            # print('self.norm_stats["action_mean"]: ', self.norm_stats["action_mean"])
            # print('self.norm_stats["action_std"]: ', self.norm_stats["action_std"])
            # print('self.norm_stats["state_mean"]: ', self.norm_stats["state_mean"])
            # print('self.norm_stats["state_std"]: ', self.norm_stats["state_std"])
            # print('robot_state_data.shape: ', robot_state_data.shape)
            # print('action_data.shape: ', action_data.shape)

            robot_state_data = (robot_state_data - self.norm_stats["state_mean"]) / self.norm_stats["state_std"]

        except:
            print(f'Error loading {dataset_path} in __getitem__')
            quit()
        t9 = time()
        # print("duration 1: ", t1-t0)
        # print("duration 2: ", t2-t1)
        # print("duration 3: ", t3-t2)
        # print("duration 4: ", t4-t3)
        # print("duration 5: ", t5-t4)
        # print("duration 6: ", t6-t5)
        # print("duration 7: ", t7-t6)
        # print("duration 8: ", t8-t7)
        # print("duration 9: ", t9-t8)
        # print(image_data.dtype, qpos_data.dtype, action_data.dtype, is_pad.dtype)
        return image_data, robot_state_data, action_data, is_pad


def get_norm_stats(dataset_path_list):
    all_state_data = []
    all_action_data = []
    all_episode_len = []

    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                xpos = root['/observations/xpos'][()]
                euler = root['/observations/euler'][()]
                gripper_pos = root['/observations/gripper_pos'][()]
                state = np.concatenate([xpos, euler, gripper_pos], axis=-1)
                # qpos = root['/observations/qpos'][()]
                # qvel = root['/observations/qvel'][()]
                if '/base_action' in root:
                    base_action = root['/base_action'][()]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([ root['/actions/pose'][()], root['/actions/gripper_pos'][()], base_action ], axis=-1)
                else:
                    action = np.concatenate([ root['/actions/pose'][()], root['/actions/gripper_pos'][()] ], axis=-1)
                    # dummy_base_action = np.zeros([action.shape[0], 2])
                    # action = np.concatenate([action, dummy_base_action], axis=-1)
        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            print(e)
            quit()
        all_state_data.append(torch.from_numpy(state))
        all_action_data.append(torch.from_numpy(action))
        all_episode_len.append(len(state))
    all_state_data = torch.cat(all_state_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    state_mean = all_state_data.mean(dim=[0]).float()
    state_std = all_state_data.std(dim=[0]).float()
    state_std = torch.clip(state_std, 1e-2, np.inf) # clipping

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 0.0001
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
             "action_min": action_min.numpy() - eps,"action_max": action_max.numpy() + eps,
             "state_mean": state_mean.numpy(), "state_std": state_std.numpy()}

    return stats, all_episode_len

def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files

def BatchSampler(batch_size, episode_len_l, sample_weights):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch

def load_data(dataset_dir_l, name_filter, camera_names, batch_size_train, batch_size_val, chunk_size, robot_obs_size, img_obs_size, img_obs_skip, skip_mirrored_data=False, load_pretrain=False, policy_class=None, stats_dir_l=None, sample_weights=None, train_ratio=0.95, use_depth=False):
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    num_episodes_0 = len(dataset_path_list_list[0])
    dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]
    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)

    # obtain train test split on dataset_dir_l[0]
    shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    train_episode_ids_0 = shuffled_episode_ids_0[:int(train_ratio * num_episodes_0)]
    val_episode_ids_0 = shuffled_episode_ids_0[int(train_ratio * num_episodes_0):]
    train_episode_ids_l = [train_episode_ids_0] + [np.arange(num_episodes) + num_episodes_cumsum[idx] for idx, num_episodes in enumerate(num_episodes_l[1:])]
    val_episode_ids_l = [val_episode_ids_0]
    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    print(f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')

    # obtain normalization stats for qpos and action
    # if load_pretrain:
    #     with open(os.path.join('/home/zfu/interbotix_ws/src/act/ckpts/pretrain_all', 'dataset_stats.pkl'), 'rb') as f:
    #         norm_stats = pickle.load(f)
    #     print('Loaded pretrain dataset stats')
    _, all_episode_len = get_norm_stats(dataset_path_list)
    train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l]
    val_episode_len_l = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_l]
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]
    norm_stats, _ = get_norm_stats(flatten_list([find_all_hdf5(stats_dir, skip_mirrored_data) for stats_dir in stats_dir_l]))
    print(f'Norm stats from: {stats_dir_l}')

    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l, sample_weights)
    batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)

    # print(f'train_episode_len: {train_episode_len}, val_episode_len: {val_episode_len}, train_episode_ids: {train_episode_ids}, val_episode_ids: {val_episode_ids}')

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, train_episode_ids, train_episode_len, chunk_size, robot_obs_size, img_obs_size, img_obs_skip, policy_class, use_depth)
    val_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, val_episode_ids, val_episode_len, chunk_size, robot_obs_size, img_obs_size, img_obs_skip, policy_class, use_depth)
    train_num_workers = (16 if os.getlogin() == 'robrosdg' else 8)
    val_num_workers = 4
    print(f'Augment images: {train_dataset.augment_images}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True, num_workers=train_num_workers, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=True, num_workers=val_num_workers, prefetch_factor=2)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def calibrate_linear_vel(base_action, c=None):
    if c is None:
        c = 0.0 # 0.19
    v = base_action[..., 0]
    w = base_action[..., 1]
    base_action = base_action.copy()
    base_action[..., 0] = v - c * w
    return base_action

def smooth_base_action(base_action):
    return np.stack([
        np.convolve(base_action[:, i], np.ones(5)/5, mode='same') for i in range(base_action.shape[1])
    ], axis=-1).astype(np.float32)

def preprocess_base_action(base_action):
    # base_action = calibrate_linear_vel(base_action)
    base_action = smooth_base_action(base_action)

    return base_action

def postprocess_base_action(base_action):
    linear_vel, angular_vel = base_action
    linear_vel *= 1.0
    angular_vel *= 1.0
    # angular_vel = 0
    # if np.abs(linear_vel) < 0.05:
    #     linear_vel = 0
    return np.array([linear_vel, angular_vel])

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
