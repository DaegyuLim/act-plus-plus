import pathlib
import os

### Task parameters
DATA_DIR = os.path.expanduser('~/dg/IL_data')
TASK_CONFIGS = {
    'dsr_block_collect':{
        'dataset_dir': DATA_DIR + '/dsr_block_collect',
        'episode_len': 4000,
        'camera_names': ['lhand_camera', 'head_camera'],
        'robot_id_list': ['dsr_l'],
    },
    'dsr_block_sort':{
        'dataset_dir': DATA_DIR + '/dsr_block_sort',
        'episode_len': 4000,
        'train_ratio': 0.95,
        'camera_names': ['lhand_camera'],
        'robot_id_list': ['dsr_l'],
    },
    'dsr_block_disassemble_and_sort':{
        'dataset_dir': DATA_DIR + '/dsr_block_disassemble_and_sort',
        'episode_len': 1800,
        'train_ratio': 0.95,
        'camera_names': ['lhand_camera', 'rhand_camera'],
        'robot_id_list': ['dsr_l', 'dsr_r'],
    }
}

HZ = 20
DT = 1/HZ

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path