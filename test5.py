import os
import numpy as np
import cv2
import h5py
import argparse
import matplotlib.pyplot as plt
import pandas as pd
DT = 0.05


# 디스어셈블인지 소트인지 (1과 0으로 나타내는 파트)
file_path = 'motionchange.xlsx'
df = pd.read_excel(file_path)
df = df.drop(columns=['contime'])       #초 단위로 표현한 셀 삭제
df_cleaned = df.dropna()
df_filled = df.fillna(0)
contick_values = df_cleaned['contick'].values
# 1부터 1800까지의 numpy 배열 만들어
numbers = np.arange(1, 1801)
# contick 값보다 작으면 1, 크면 0 기준으로 조건 처리하고 1 또는 0으로 변환
disorsort = np.where(numbers <= contick_values[:, np.newaxis], 1, 0)
# csv로 확인 가능
'''
disorsort = pd.DataFrame(disorsort, columns=numbers)
result_df.insert(0, 'epnum', df['epnum'])
result_df.to_csv('disassemble.csv', index=False)
print("CSV 파일이 성공적으로 저장되었습니다.")
'''

# 성공인지 아닌지 (1 or 0)
file_path = 'success.xlsx'
df = pd.read_excel(file_path)
columns_to_remove = [f'suctime{i}' for i in range(1, 13)]       # 초 단위로 표현한 셀 삭제
df.drop(columns=columns_to_remove, inplace=True)
df.dropna(inplace=True)
# 1800 개 틱 배열 생성
max_ticks = 1800
def create_tick_array(row):
    tick_array = np.zeros(max_ticks, dtype=int)
    for i in range(1, 13):
        suctick_col = f'suctick{i}'
        if suctick_col in row and pd.notna(row[suctick_col]) and row[suctick_col] > 0:
            tick_index = int(row[suctick_col]) - 1  # 인덱스는 0부터 시작하므로 -1
            tick_array[tick_index] = 1
    return tick_array
tick_arrays = df.apply(create_tick_array, axis=1)
sucornot = np.array(tick_arrays.tolist())
# csv로 확인 가능
'''
sucornot = pd.DataFrame(tick_arrays.tolist())
tick_df.insert(0, 'epnum', df['epnum'].values)
output_file_path = os.path.join(os.getcwd(), 'tick_data.csv')
tick_df.to_csv(output_file_path, index=False)
print(f"CSV 파일이 {output_file_path}에 저장되었습니다.")
'''

# HDF5 핸들링 시작
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]

def load_hdf5(dataset_dir, dataset_name):
    # 지정된 경로로 HDF5 파일 경로 설정
    dataset_path = os.path.join(dataset_dir, 'disassemble_and_sort', dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()
    # Extract column based on ep_idx
    ep_idx = int(dataset_name.replace('episode_', '').replace('.hdf5', ''))
    disorsort_col = disorsort.iloc[:, ep_idx].values
    sucornot_col = sucornot.iloc[:, ep_idx].values
    # HDF5 파일 열기
    with h5py.File(dataset_path, 'a') as root:  # 'a'는 append 모드
        # 그룹 생성
        disassemble_group = root.create_group('/task/disassemble')
        sort_group = root.create_group('/task/sort')
        # 하위 그룹 생성
        disassemble_success_group = disassemble_group.create_group('success')
        disassemble_label_group = disassemble_group.create_group('label')
        sort_success_group = sort_group.create_group('success')
        sort_label_group = sort_group.create_group('label')
        # 최하위 그룹에 데이터 채우기
        disassemble_success_group.create_dataset(f'episode_{ep_idx}', data=(disorsort_col == 1) & (sucornot_col == 1))
        disassemble_label_group.create_dataset(f'episode_{ep_idx}', data=(disorsort_col == 1) & (sucornot_col == 0))
        sort_success_group.create_dataset(f'episode_{ep_idx}', data=(disorsort_col == 0) & (sucornot_col == 1))
        sort_label_group.create_dataset(f'episode_{ep_idx}', data=(disorsort_col == 0) & (sucornot_col == 0))
        #새로운 이름으로 저장
        new_filename = 'new_' + dataset_name + '.hdf5'
        os.rename(dataset_name + '.hdf5' , new_filename)
        print(f'Modified HDF5 file saved as: {new_filename}')

        plot_path = 'C:\\task\\dg\\IL_data'
        visualize_task_data(disorsort_col, sucornot_col, os.path.join(dataset_dir, f'{dataset_name}_task_data.png'))
    print(f'Data added to HDF5 file: {dataset_path}')

def visualize_task_data(hdf5_file_path, plot_path, episode_idx):
    with h5py.File(hdf5_file_path, 'r') as f:
        fig, axs = plt.subplots(4, 1, figsize=(10, 20))
        episode_str = f'episode_{episode_idx}'
        # Disassemble success
        disassemble_success = f[f'/task/disassemble/success/{episode_str}'][:]
        axs[0].plot(disassemble_success, label='Disassemble Success', marker='o', linestyle='-')
        axs[0].set_title('Disassemble Success')
        axs[0].set_xlabel('Tick')
        axs[0].set_ylabel('Label')
        axs[0].legend()
        axs[0].grid(True)
        # Disassemble label
        disassemble_label = f[f'/task/disassemble/label/{episode_str}'][:]
        axs[1].plot(disassemble_label, label='Disassemble Label', marker='o', linestyle='-')
        axs[1].set_title('Disassemble Label')
        axs[1].set_xlabel('Tick')
        axs[1].set_ylabel('Label')
        axs[1].legend()
        axs[1].grid(True)
        # Sort success
        sort_success = f[f'/task/sort/success/{episode_str}'][:]
        axs[2].plot(sort_success, label='Sort Success', marker='o', linestyle='-')
        axs[2].set_title('Sort Success')
        axs[2].set_xlabel('Tick')
        axs[2].set_ylabel('Label')
        axs[2].legend()
        axs[2].grid(True)
        # Sort label
        sort_label = f[f'/task/sort/label/{episode_str}'][:]
        axs[3].plot(sort_label, label='Sort Label', marker='o', linestyle='-')
        axs[3].set_title('Sort Label')
        axs[3].set_xlabel('Tick')
        axs[3].set_ylabel('Label')
        axs[3].legend()
        axs[3].grid(True)
        # 플랏
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f'Saved labels plot to: {plot_path}')
        plt.close()

def main(args):
    path2file = 'C:\\task\\dg\\IL_data\\disassemble_and_sort'
    dataset_dir = args.get('dataset_name', path2file) 
    episode_idx = args['episode_idx']
    ismirror = args['ismirror']
    if episode_idx is None:
        # 자동으로 인덱스 리스트 가져오기
        print(f'dataset_dir = {dataset_dir}')
        index_list = get_auto_index_list(dataset_dir)
        print(f'video index_list = {index_list}')
        for episode_idx in index_list:
            dataset_name = f'episode_{episode_idx}'
            print("test1")
            load_hdf5(dataset_dir, dataset_name)
    else:
        if ismirror:
            dataset_name = f'mirror_episode_{episode_idx}'
        else:
            dataset_name = f'episode_{episode_idx}'
        print("test2")
        load_hdf5(dataset_dir, dataset_name)

def get_auto_index_list(dataset_dir, dataset_name_prefix='', data_suffix='hdf5'):
    max_idx = 1000
    # dataset_dir가 None이면 현재 작업 디렉터리를 기본값으로 설정
    if dataset_dir is None:
        dataset_dir = 'C:\task\dg\IL_data\disassemble_and_sort'
    if not os.path.isdir(dataset_dir):
        raise Exception(f"dataset directory ({dataset_dir}) does not exist!")
    index_list = []
    for i in range(max_idx+1):
        if os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            # if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}_video.mp4')):
            index_list.append(i)
    return index_list

if __name__ == '__main__':
    print('test123')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset directory.', required=False)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    parser.add_argument('--ismirror', action='store_true')
    args = parser.parse_args()
    main(vars(parser.parse_args()))



