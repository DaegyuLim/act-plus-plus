#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ##
# @brief    [schunk_gripper_control] control schunk gripper using quest controller input
# @author   Daegyu Lim (dglim@robros.co.kr)   

import serial
import threading
import time
import rospy
import numpy as np
from crc import Calculator, Crc16
import struct
from quest2ros.msg import OVR2ROSInputs
from std_msgs.msg import Float32

try:
    ROBOT_ID     = rospy.get_param('/dsr/robot_id')
except:
    ROBOT_ID     = "dsr_l"

ROBOT_MODEL  = "a0509"

class gripperControl:
    def __init__(self, robot_id = 'dsr_l', hz = 30, init_node = True, teleop=True):
        self.hz = hz
        self.dt = 1/self.hz
        self.teleop = teleop

        if init_node:
            rospy.init_node('schunk_gripper_control_py')
        rospy.on_shutdown(self.shutdown)

        self.ovr2ros_right_hand_inputs_sub = rospy.Subscriber("/q2r_right_hand_inputs",OVR2ROSInputs,self.ovr2ros_right_hand_inputs_callback)
        
        self.gripper_state_pub = rospy.Publisher('/'+robot_id +'/state/gripper', Float32, tcp_nodelay= True, queue_size= 1)
        self.gripper_action_pub = rospy.Publisher('/'+robot_id +'/action/gripper', Float32, tcp_nodelay= True, queue_size= 1)

        self.gripper_current_position = Float32()
        self.gripper_desired_position = Float32()

        self.controller_inputs_raw = OVR2ROSInputs()
        self.controller_inputs = OVR2ROSInputs()

        self.gripper_desired_position_buffer = np.zeros(8)

        self.gripper_state = 0.0
        self.gripper_command = 0.0
        self.policy_gripper_command = 0.0

        self.ser = serial.Serial()
        
        
        self.ser.baudrate = 115200
        self.ser.port = '/dev/ttyUSB0'
        # ser.port = 'COM7'
        #ser.timeout =0
        self.ser.stopbits = serial.STOPBITS_ONE
        self.ser.bytesize = 8
        self.ser.parity = serial.PARITY_EVEN
        self.ser.timeout = 1
        self.ser.xonxoff = False
        self.ser.rtscts = False
        self.ser.dsrdtr = False

        self.calculator = Calculator(Crc16.MODBUS, optimized=True)

        self.init_bytes = b"\x0C\x10\x00\x47\x00\x08\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x3A\x7C"
        self.sync_bytes = b"\x0C\x10\x00\x47\x00\x08\x10\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xF9\x2F"
        self.request_state_bytes = b"\x0C\x03\x00\x3F\x00\x08\x75\x1D"
        self.loop_tick = 0

        self.bytes_buffer = b''
        self.buffer_size = 0

        self.first_control_tick = False

        self.data_read_trigger = False
        
        self.initialize_gripper()

        self.thread1 = threading.Thread(target=self.read_state)
        self.thread1.daemon = True 
        self.thread1.start()

        self.stop_control_loop = False

    def initialize_gripper(self):
        self.ser.open()
        if not self.ser.is_open:
            print("[ERROR] gripper serial socket is not opened") 
        rospy.sleep(0.5)
        self.ser.write(self.init_bytes)
        rospy.sleep(0.5)
        self.ser.write(self.sync_bytes)
        time.sleep(0.01)
        self.init_time = rospy.Time.now()

    def read_state(self):
        rate2 = rospy.Rate(self.hz)

        while not rospy.is_shutdown():
            # self.ser.reset_output_buffer()
            # self.ser.reset_input_buffer()
            # self.ser.write(self.request_state_bytes)
            # time.sleep(0.01)
            # self.gripper_state_bytes = self.ser.read_until(expected=b"\x00\x00\x00\x00", size=15)
            read_time_start = rospy.Time.now()
            # while not self.data_read_trigger:
            #     time.sleep(0.0001)

            find_gripper_state = False
            read_time_idle = rospy.Time.now()
            while not find_gripper_state:
                self.data_read_trigger = False

                try:
                    self.bytes_buffer += self.ser.read(1)
                except:
                    time.sleep(0.0001)
                    # print("serial read error")
                self.buffer_size = len(self.bytes_buffer)
                # print('buffer size: ', self.buffer_size)
                if self.buffer_size >= 22:
                    while self.buffer_size > 21:
                        self.bytes_buffer = self.bytes_buffer[1:]
                        self.buffer_size -= 1
                # print("bytes: ", self.bytes_buffer)

                if self.bytes_buffer[0:3] == b'\x0C\x03\x10':
                    # print("find bytes!: ", self.bytes_buffer)
                    check_crc = self.calculator.checksum(self.bytes_buffer[0:19]).to_bytes(2, 'little', signed=False) == self.bytes_buffer[19:21]
                    if check_crc:
                        # self.ser.reset_input_buffer()
                        find_gripper_state = True
                        self.gripper_state_bytes = self.bytes_buffer[7:11]
                        self.gripper_state = struct.unpack('i', self.gripper_state_bytes)[0]
                        self.gripper_state = float( np.clip(1.0-(self.gripper_state/83000), 0.0, 1.0) )
                        self.gripper_current_position.data = self.gripper_state
                        self.gripper_state_pub.publish(self.gripper_current_position)

                        
                        read_time_end = rospy.Time.now()
                        # print("read duration: [ms]", (read_time_end - read_time_start).nsecs * 1e-6)
                        # print("read idle duration: [ms]", (read_time_idle - read_time_start).nsecs * 1e-6)
                        # print("gripper_state_bytes: ", self.gripper_state_bytes)
                        # print("gripper state [mm]: ", float(self.gripper_state/1e3) )
                    # else:
                    #     print("CRC error")
                # elif (rospy.Time.now() - read_time_start).nsecs > 30*1e6:
                #     self.gripper_state_pub.publish(self.gripper_current_position)
                #     print("[WARNING] Gripper position is not recieved in the time (30ms)")
                #     break


            # rate2.sleep()

    def control(self):        
        # sync signal
        time_start = rospy.Time.now()
        self.ser.reset_output_buffer()
        self.ser.reset_input_buffer()

        self.ser.write(self.request_state_bytes)
        self.data_read_trigger = True
        time.sleep(0.008)
        #calculate command bytes
        # elapsed_time = (rospy.Time.now() - self.init_time).to_nsec()/1e9

        # radian = [2*np.pi*(rospy.Time.now() - self.init_time)/5.0]
        # gripper_command = (np.sin(2*np.pi*elapsed_time/10.0) +1)/2
        # gripper_command = self.controller_inputs.press_index
        
        self.gripper_desired_position.data = self.gripper_command
        self.gripper_action_pub.publish(self.gripper_desired_position)
        self.send_gripper_pos(self.gripper_command)

        # desired_gripper_pos_um = int( np.clip(83000 *(1-self.gripper_command), 0, 83000) )
        # gripper_force = 1 ## dim:  percentage(%)
        # # print("desired_gripper_pos_um: ", desired_gripper_pos_um)
        
        # gripper_position_command = desired_gripper_pos_um.to_bytes(4, 'little', signed=True)
        
        # head_bytes = b"\x0C\x10\x00\x47\x00\x08\x10\x01\x20\x00\x00"
        # velocity_bytes = b"\x38\xC1\x01\x00"
        # force_bytes = gripper_force.to_bytes(4, 'little', signed=True)
        # gripper_command_bytes = head_bytes + gripper_position_command + velocity_bytes + force_bytes
        
        # gripper_command_bytes += self.calculator.checksum(gripper_command_bytes).to_bytes(2, 'little', signed=False)
        
        # self.ser.write(gripper_command_bytes)
        time.sleep(0.008)

        self.ser.write(self.sync_bytes)
        time.sleep(0.008)
        ## read current gripper position
        
        self.ser.reset_input_buffer()
        self.data_read_trigger = True
        self.ser.write(self.request_state_bytes)


        # time.sleep(0.01)
        # self.ser.write(self.sync_bytes)
        # time.sleep(0.01)
        # self.ser.write(self.sync_bytes)
        # time.sleep(0.01)
        # # self.gripper_state_bytes = self.ser.read_until(expected=b"\x00\x00\x00\x00", size=15)
        # self.gripper_state_bytes = self.ser.read(21)
        
        # self.gripper_state = struct.unpack('i', self.gripper_state_bytes[7:11])[0]
        # # self.gripper_state = int.from_bytes(self.gripper_state_bytes[7:11], "little", signed=True)
        # self.gripper_state = self.gripper_state * 1e-3 ## in [mm]

        time_end = rospy.Time.now()

        consumed_time_us = float((time_end - time_start).nsecs)*1e-3 # [us]  
        if consumed_time_us > self.dt*1e6:
            print("Warning! gripper control elapsed_time [us]: ", consumed_time_us, "is over the control period")

        # if self.loop_tick % self.hz == 0:
        #     print("gripper_state: ", self.gripper_state)
        #     print("gripper_state_bytes: ", self.gripper_state_bytes)
        self.loop_tick += 1

    def step(self):
        if self.teleop:
            self.readTriggerInputs()
        else:
            self.readPolicyCommand()
        self.control()

    def control_loop(self):
        rate = rospy.Rate(self.hz)
        print("GRIPPER CONTROL START")
        while not rospy.is_shutdown():
            if self.stop_control_loop:
                return
            self.step()
            rate.sleep()

    def open(self):
        self.gripper_command = 0.0

    def close(self):
        self.gripper_command = 1.0

    def get_state(self):
        return np.array([self.gripper_state])
    
    def get_action(self):
        return np.array([self.policy_gripper_command])
    
    def set_action(self, gripper_action):
        self.policy_gripper_command = gripper_action

    def readTriggerInputs(self):
        self.controller_inputs = self.controller_inputs_raw
        self.gripper_desired_position_buffer[1:] = self.gripper_desired_position_buffer[0:-1]
        self.gripper_desired_position_buffer[0] = self.controller_inputs.press_index
        self.gripper_command = np.mean(self.gripper_desired_position_buffer)

    def readPolicyCommand(self):
        self.gripper_command = self.policy_gripper_command

    def send_gripper_pos(self, des_pos = 1):
        desired_gripper_pos_um = int( np.clip(83000 *(1-des_pos), 0, 83000) )
        gripper_force = 1 ## dim:  percentage(%)
        # print("desired_gripper_pos_um: ", desired_gripper_pos_um)
        
        gripper_position_command = desired_gripper_pos_um.to_bytes(4, 'little', signed=True)
        
        head_bytes = b"\x0C\x10\x00\x47\x00\x08\x10\x01\x20\x00\x00"
        velocity_bytes = b"\x38\xC1\x01\x00"
        force_bytes = gripper_force.to_bytes(4, 'little', signed=True)
        gripper_command_bytes = head_bytes + gripper_position_command + velocity_bytes + force_bytes
        
        gripper_command_bytes += self.calculator.checksum(gripper_command_bytes).to_bytes(2, 'little', signed=False)
        self.ser.write(gripper_command_bytes)

    def ovr2ros_right_hand_inputs_callback(self, data):    
        self.controller_inputs_raw = data

    def shutdown(self):
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        self.ser.close()
        print("\n GRIPPER CONTROL SHUTDOWN! \n")

        return 0

if __name__ == "__main__":
    gripper = gripperControl()

    gripper.control_loop()