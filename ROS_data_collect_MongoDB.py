#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ##
# @brief    [record_episode] 
# @author   Daegyu Lim (dglim@robros.co.kr)   

import rospy
import os
import time
import threading
from std_msgs.msg import Float32MultiArray, Float32, Bool
from quest2ros.msg import OVR2ROSInputs, OVR2ROSHapticFeedback
from geometry_msgs.msg import Pose, PoseStamped, Twist, PoseArray
import numpy as np
from scipy.spatial.transform import Rotation
import h5py
import pickle
import argparse
from robot.constants import *
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import os
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
import torch
import torch.nn as nn
from einops import rearrange

from robot.metaquest_teleop import drlControl
from robot.schunk_gripper_control import gripperControl
from robot.robot_utils import ImageRecorder

from pymongo import MongoClient
from datetime import datetime
import base64
from PIL import Image
import io
import gridfs
from pymongo import MongoClient
from datetime import datetime


# for single robot 
try:
    ROBOT_ID     = rospy.get_param('/dsr/robot_id')
except:
    ROBOT_ID     = "dsr_l"
ROBOT_MODEL = "a0509"

class DataRecorder:
    def __init__(self, robot_ids, camera_names, init_node=False, is_debug=False):
        import rospy
        from collections import deque
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image
        self.is_debug = is_debug
        self.bridge = CvBridge()
        self.robot_ids = robot_ids #['dsr_l', 'dsr_r']
        self.camera_names = camera_names #['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
        self.total_data = {
        '/observations/xpos': [],
        '/observations/quat': [],
        '/observations/gripper_pos': [],
        '/actions/xpos': [],
        '/actions/quat': [],
        '/actions/gripper_pos': [],
    }
        # MONGODB initialize
        self.client = MongoClient("mongodb://localhost:27017/")  # 여기서 MongoDB 연결을 초기화
        self.db = self.client["test_database"]  # 데이터베이스 선택
        self.collection = self.db["test_collection"]  # 컬렉션 선택
        
    
        if init_node:
            rospy.init_node('data_recorder', anonymous=True)
        
        rospy.Subscriber('/data_collect_call', Bool, self.data_collection_callback_func)
        
        for robot_id in self.robot_ids:
            
            setattr(self, f'{robot_id}_state_pose', None)
            setattr(self, f'{robot_id}_state_xpos_np', np.zeros(3))
            setattr(self, f'{robot_id}_state_quat_np', np.zeros(4))
            setattr(self, f'{robot_id}_state_gripper_pos', None)
            setattr(self, f'{robot_id}_state_gripper_pos_np', np.zeros(1))
            
            setattr(self, f'{robot_id}_action_pose', None)
            setattr(self, f'{robot_id}_action_xpos_np', np.zeros(3))
            setattr(self, f'{robot_id}_action_quat_np', np.zeros(4))
            setattr(self, f'{robot_id}_action_gripper_pos', None)
            setattr(self, f'{robot_id}_action_gripper_pos_np', np.zeros(1))
            
            if robot_id == 'dsr_l':
                gripper_pos_state_callback_func = self.gripper_pos_state_cb_left
                gripper_pos_action_callback_func = self.gripper_pos_action_cb_left
                pose_state_callback_func = self.pose_state_cb_left
                pose_action_callback_func = self.pose_action_cb_left
            elif robot_id == 'dsr_r':
                gripper_pos_state_callback_func = self.gripper_pos_state_cb_right
                gripper_pos_action_callback_func = self.gripper_pos_action_cb_right
                pose_state_callback_func = self.pose_state_cb_right
                pose_action_callback_func = self.pose_action_cb_right
            else:
                raise NotImplementedError  

            rospy.Subscriber(f'/'+robot_id +'/state/gripper', Float32, gripper_pos_state_callback_func)
            rospy.Subscriber(f'/'+robot_id +'/state/pose', PoseStamped, pose_state_callback_func)
            rospy.Subscriber(f'/'+robot_id +'/action/gripper', Float32, gripper_pos_action_callback_func)
            rospy.Subscriber(f'/'+robot_id +'/action/pose', PoseStamped, pose_action_callback_func)
            
            
        for cam_name in self.camera_names:
            setattr(self, f'{cam_name}_image', None)
            setattr(self, f'{cam_name}_depth', None)
            setattr(self, f'{cam_name}_secs', None)
            setattr(self, f'{cam_name}_nsecs', None)
            if cam_name == 'lhand_camera':
                image_callback_func = self.image_cb_lhand_cam
            elif cam_name == 'rhand_camera':
                image_callback_func = self.image_cb_rhand_cam
            elif cam_name == 'head_camera':
                image_callback_func = self.image_cb_head_cam
            else:
                raise NotImplementedError
            rospy.Subscriber(f"/{cam_name}/image_raw", Image, image_callback_func)
            # rospy.Subscriber(f"/{cam_name}/aligned_depth_to_color/image_raw", Image, depth_callback_func)
            if self.is_debug:
                setattr(self, f'{cam_name}_timestamps', deque(maxlen=50))
        time.sleep(0.5)

    
    def data_collect_MongoDB(self):
        start_time = time.time()
        current_time = datetime.now() # TO-DO: ms 단위까지만 보기

        total_record_data = {
            "timestamp": current_time,  
            "robot_data": [],
            "camera_data": []
        }
        
        ### Data Collection Target Variables ###
        for robot_id in self.robot_ids:
            setattr(self, f'{robot_id}_state_xpos_np'[0], getattr(self, f'{robot_id}_state_pose').position.x )
            setattr(self, f'{robot_id}_state_xpos_np'[1], getattr(self, f'{robot_id}_state_pose').position.y )
            setattr(self, f'{robot_id}_state_xpos_np'[2], getattr(self, f'{robot_id}_state_pose').position.z )
            
            setattr(self, f'{robot_id}_state_quat_np'[0], getattr(self, f'{robot_id}_state_pose').orientation.x )
            setattr(self, f'{robot_id}_state_quat_np'[1], getattr(self, f'{robot_id}_state_pose').orientation.y )
            setattr(self, f'{robot_id}_state_quat_np'[2], getattr(self, f'{robot_id}_state_pose').orientation.z )
            setattr(self, f'{robot_id}_state_quat_np'[3], getattr(self, f'{robot_id}_state_pose').orientation.w )
            
            setattr(self, f'{robot_id}_state_gripper_pos_np'[0], getattr(self, f'{robot_id}_state_gripper_pos') )
    
            setattr(self, f'{robot_id}_action_xpos_np'[0], getattr(self, f'{robot_id}_action_pose').position.x )
            setattr(self, f'{robot_id}_action_xpos_np'[1], getattr(self, f'{robot_id}_action_pose').position.y )
            setattr(self, f'{robot_id}_action_xpos_np'[2], getattr(self, f'{robot_id}_action_pose').position.z )
            
            setattr(self, f'{robot_id}_action_quat_np'[0], getattr(self, f'{robot_id}_action_pose').orientation.x )
            setattr(self, f'{robot_id}_action_quat_np'[1], getattr(self, f'{robot_id}_action_pose').orientation.y )
            setattr(self, f'{robot_id}_action_quat_np'[2], getattr(self, f'{robot_id}_action_pose').orientation.z )
            setattr(self, f'{robot_id}_action_quat_np'[3], getattr(self, f'{robot_id}_action_pose').orientation.w )

            setattr(self, f'{robot_id}_action_gripper_pos_np'[0], getattr(self, f'{robot_id}_state_gripper_pos') )
        
            self.total_data[f'/observations/{robot_id}/xpos'] = getattr(self, f'{robot_id}_state_xpos_np').tolist() 
            self.total_data[f'/observations/{robot_id}/quat'] = getattr(self, f'{robot_id}_state_quat_np').tolist() 
            self.total_data[f'/observations/{robot_id}/gripper_pos'] = getattr(self, f'{robot_id}_state_gripper_pos_np').tolist() 
            self.total_data[f'/actions/{robot_id}/xpos'] = getattr(self, f'{robot_id}_action_xpos_np').tolist() 
            self.total_data[f'/actions/{robot_id}/quat'] = getattr(self, f'{robot_id}_action_quat_np').tolist()
            self.total_data[f'/actions/{robot_id}/gripper_pos'] = getattr(self, f'{robot_id}_action_gripper_pos_np').tolist()
            
            # 각 로봇의 데이터를 record_data로 준비
            
            record_data = {
                "robot_id": robot_id,
                "dsr_state_xpos": getattr(self, f'{robot_id}_state_xpos_np').tolist(),
                "dsr_state_quat": getattr(self, f'{robot_id}_state_quat_np').tolist(),
                "gripper_state": getattr(self, f'{robot_id}_state_gripper_pos_np').tolist(),
                "dsr_action_xpos": getattr(self, f'{robot_id}_action_xpos_np').tolist(),
                "dsr_action_quat": getattr(self, f'{robot_id}_action_quat_np').tolist(),
                "action_gripper_state": getattr(self, f'{robot_id}_action_gripper_pos_np').tolist(),
            }
            
            # record_data를 total_record_data 리스트에 추가
            total_record_data["robot_data"].append(record_data)            
        camera_data_list = []
        


        for cam_name in self.camera_names:
            # 이미지 데이터 가져오기
            image_data = getattr(self, f'{cam_name}_image')
            image_file_path = f"/home/robros-ai/dg/robros_imitation_learning/MongoDB/image/{cam_name}_{current_time}.jpg"

            image = Image.fromarray(image_data)
            image.save(image_file_path)            # 종료 시간 기록

            # camera_data 추가
            camera_data = {
                "camera_name": cam_name,
                "image_url": image_file_path,
            }
            total_record_data["camera_data"].append(camera_data)


        
        # 몽고디비 저장
        self.collection.insert_one(total_record_data)

        end_time = time.time()
        
        # 시간 차이 계산
        elapsed_time = end_time - start_time
        print(f"데이터 수집에 소요된 시간: {elapsed_time:.10f} 초")
        
        #####################################################
    def data_collection_callback_func(self, data):
        if data.data == True:
            self.data_collect_MongoDB()
            if self.is_debug:
                print(f'a data sample is collected {time.time()}')
            
    def pose_state_cb_left(self, data):
        robot_id = 'dsr_l'
        setattr(self, f'{robot_id}_state_pose', data.pose)
    
    def pose_state_cb_right(self, data):
        robot_id = 'dsr_r'
        setattr(self, f'{robot_id}_state_pose', data.pose)
    
    def pose_action_cb_left(self, data):
        robot_id = 'dsr_l'
        setattr(self, f'{robot_id}_action_pose', data.pose)
    
    def pose_action_cb_right(self, data):
        robot_id = 'dsr_r'
        setattr(self, f'{robot_id}_action_pose', data.pose)
    
    
    def gripper_pos_state_cb_left(self, data):
        robot_id = 'dsr_l'
        setattr(self, f'{robot_id}_state_gripper_pos', data.data)
    
    def gripper_pos_state_cb_right(self, data):
        robot_id = 'dsr_r'
        setattr(self, f'{robot_id}_state_gripper_pos', data.data)
    
    def gripper_pos_action_cb_left(self, data):
        robot_id = 'dsr_l'
        setattr(self, f'{robot_id}_action_gripper_pos', data.data)
    
    def gripper_pos_action_cb_right(self, data):
        robot_id = 'dsr_r'
        setattr(self, f'{robot_id}_action_gripper_pos', data.data)
        
        
    def image_cb(self, cam_name, data):
        setattr(self, f'{cam_name}_image', self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough'))
        setattr(self, f'{cam_name}_secs', data.header.stamp.secs)
        setattr(self, f'{cam_name}_nsecs', data.header.stamp.nsecs)
        if self.is_debug:
            getattr(self, f'{cam_name}_timestamps').append(data.header.stamp.secs + data.header.stamp.nsecs * 1e-9)

    def image_cb_lhand_cam(self, data):
        cam_name = 'lhand_camera'
        return self.image_cb(cam_name, data)

    def image_cb_rhand_cam(self, data):
        cam_name = 'rhand_camera'
        return self.image_cb(cam_name, data)
    
    def image_cb_head_cam(self, data):
        cam_name = 'head_camera'
        return self.image_cb(cam_name, data)
        
    def get_images(self):
        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = getattr(self, f'{cam_name}_image')
        return image_dict


if __name__ == "__main__":
    task_config = TASK_CONFIGS['dsr_block_sort_demo_head_camera']

    data_recorder = DataRecorder(robot_ids = task_config['robot_id_list'], camera_names = task_config['camera_names'], init_node=True, is_debug = True)
    rate = rospy.Rate(1000)
    while not rospy.is_shutdown():
        rate.sleep()
    print('---Data Collection Exits---')

