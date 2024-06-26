import pathlib
import os

### Task parameters
DATA_DIR = os.path.expanduser('~/dg/IL_data')
TASK_CONFIGS = {
    'dsr_single_block_sorting':{
        'dataset_dir': DATA_DIR + '/dsr_single_block_sorting',
        'episode_len': 4000,
        'camera_names': ['lhand_camera', 'head_camera']
    }
}

HZ = 30
DT = 1/HZ

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path