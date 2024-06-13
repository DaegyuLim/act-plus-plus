INSTALL
==============
##### Dynamixel SDK
``` bash
    $ pip install dynamixel_sdk
```
- Dynamixel SDK ISSUE: groupSyncRead.getData function returns unsigend int. Chnage MAKEDWORD code into "return int.from_bytes(self.data_dict[dxl_id][address - self.start_address:address - self.start_address+4], byteorder='little', signed=True)"
  
##### [Pinocchio](https://stack-of-tasks.github.io/pinocchio/download.html) for Kinematics
``` bash
    $ conda install pinocchio -c conda-forge 
    or
    $ pip install pin
```

---------------------------------------
EXCUTION
===============
### Python 
```
python scripts/dynamixel_master_arm.py
```
### ROS
```
rosrun dsr_master_arm dynamixel_master_arm.py
```

USB STATIC NAME SETTING
===============
link: https://jh-byun.github.io/study/ubuntu-USB-static-name/