import os
import numpy as np
import cv2
import h5py
import argparse

import matplotlib.pyplot as plt
from robot.constants import DT

import IPython
e = IPython.embed

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]

def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        # is_sim = root.attrs['sim']
        compressed = root.attrs.get('compress', False)
        xpos = root['/observations/xpos'][()]
        euler = root['/observations/euler'][()]
        gripper_pos = root['/observations/gripper_pos'][()]
        action_pose = root['/actions/pose'][()]
        action_gripper = root['/actions/gripper_pos'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    return xpos, euler, gripper_pos,  action_pose, action_gripper, image_dict, compressed

def main(args):
    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    if episode_idx is None:
        index_list = get_auto_index_list(dataset_dir)
        print(f'video index_list = {index_list}')
        for episode_idx in index_list:
            dataset_name = f'episode_{episode_idx}'
            xpos, euler, gripper_pos,  action_pose, action_gripper, image_dict, compressed = load_hdf5(dataset_dir, dataset_name)
            save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'), is_compressed=compressed)
    else:
        dataset_name = f'episode_{episode_idx}'

        xpos, euler, gripper_pos,  action_pose, action_gripper, image_dict, compressed = load_hdf5(dataset_dir, dataset_name)
        save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'), is_compressed=compressed)
        # visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))
        # visualize_timestamp(t_list, dataset_path) # TODO addn timestamp back

def get_auto_index_list(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        raise Exception(f"dataset directory ({dataset_dir}) is not exist!")
    index_list = []
    for i in range(max_idx+1):
        if os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}_video.mp4')):
                index_list.append(i)
    return index_list
    
    



def save_videos(video, dt, video_path=None, is_compressed = True):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        cam_names = sorted(cam_names)
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]] # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        cam_names = sorted(cam_names)
        all_cam_videos = []

        if is_compressed:
            for cam_name in cam_names:
                decompressed_image_array = []
                for i in range(len(video[cam_name])):
                    decompressed_image = cv2.imdecode(video[cam_name][i], 1)
                    # cv2.imshow('decoding', decompressed_image)
                    # cv2.waitKey()
                    decompressed_image_array.append(decompressed_image)
                video[cam_name] = np.array(decompressed_image_array)
                all_cam_videos.append(video[cam_name])
        else:
            for cam_name in cam_names:
                all_cam_videos.append(video[cam_name])
        
        all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')


def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list) # ts, dim
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()

def visualize_timestamp(t_list, dataset_path):
    plot_path = dataset_path.replace('.pkl', '_timestamp.png')
    h, w = 4, 10
    fig, axs = plt.subplots(2, 1, figsize=(w, h*2))
    # process t_list
    t_float = []
    for secs, nsecs in t_list:
        t_float.append(secs + nsecs * 10E-10)
    t_float = np.array(t_float)

    ax = axs[0]
    ax.plot(np.arange(len(t_float)), t_float)
    ax.set_title(f'Camera frame timestamps')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    ax = axs[1]
    ax.plot(np.arange(len(t_float)-1), t_float[:-1] - t_float[1:])
    ax.set_title(f'dt')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved timestamp plot to: {plot_path}')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    main(vars(parser.parse_args()))
