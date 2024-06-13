#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ##
# @brief    [record_episode] 
# @author   Daegyu Lim (dglim@robros.co.kr)   

from __future__ import print_function

from dxl_master_arm import dsrMasterArm


if __name__ == "__main__":
    master_arms = dsrMasterArm(robot_id_list = ['dsr_l', 'dsr_r'], hz=50, init_node=True)

    master_arms.set_init_q('dsr_l', [180.0,180.0,180.0,180.0,180.0,0.0,180.0])
    master_arms.set_joint_axis('dsr_l', [1,1,-1,1,-1,1,1])

    master_arms.set_init_q('dsr_r', [270.0,180.0,180.0,180.0,180.0,-180.0,180.0])
    master_arms.set_joint_axis('dsr_r', [1,1,-1,1,-1,1,1])

    master_arms.run_gui()






