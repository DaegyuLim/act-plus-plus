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
        'episode_len': 40000,
        'train_ratio': 0.95,
        'camera_names': ['lhand_camera'],
        'robot_id_list': ['dsr_l'],
    },
    'dsr_block_disassemble_and_sort':{
<<<<<<< HEAD
        'dataset_dir': DATA_DIR + '/dsr_block_sort_only_head_camera/240716',
        'episode_len': 1200,
=======
        'dataset_dir': DATA_DIR + '/dsr_block_disassemble_and_sort',
        'episode_len': 1800,
>>>>>>> 3de3af3c5cd83882709045d8e0f997d987262e54
        'train_ratio': 0.99,
        'camera_names': ['lhand_camera', 'rhand_camera'],
        'robot_id_list': ['dsr_l', 'dsr_r'],
        'name_filter': lambda n: 'sort_only' in n,
    },
    'dsr_block_sort_demo_head_camera':{
        'dataset_dir': DATA_DIR + '/dsr_block_sort_demo_head_camera/240717',
        'episode_len': 3600,
        'train_ratio': 0.99,
        'camera_names': ['lhand_camera', 'rhand_camera', 'head_camera'],
        'robot_id_list': ['dsr_l', 'dsr_r']
    }
}

HZ = 20
DT = 1/HZ

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path