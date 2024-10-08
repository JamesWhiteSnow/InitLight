import cityflow
import pandas as pd
import os
import json
import math
import numpy as np
import itertools

class CityFlowEnvM(object):
    '''
    multi inersection cityflow environment
    '''

    def __init__(self,
                 lane_phase_info,
                 intersection_id,
                 num_step=2000,
                 thread_num=1,
                 cityflow_config_file='',
                 path_to_log='result'
                 ):
        self.eng = cityflow.Engine(cityflow_config_file, thread_num=thread_num)
        self.num_step = num_step
        self.intersection_id = intersection_id  
        self.state_size = None
        self.lane_phase_info = lane_phase_info  

        self.path_to_log = path_to_log
        self.current_phase = {}
        self.current_phase_time = {}
        self.start_lane = {}
        self.end_lane = {}
        self.phase_list = {}
        self.phase_startLane_mapping = {}
        self.intersection_lane_mapping = {} 

        for id_ in self.intersection_id:
            self.start_lane[id_] = self.lane_phase_info[id_]['start_lane']
            self.end_lane[id_] = self.lane_phase_info[id_]['end_lane']
            self.phase_startLane_mapping[id_] = self.lane_phase_info[id_]["phase_startLane_mapping"]

            self.phase_list[id_] = self.lane_phase_info[id_]["phase"]
            self.current_phase[id_] = self.phase_list[id_][0]
            self.current_phase_time[id_] = 0
        self.get_state()  

    def reset(self):
        self.eng.reset()

    def step(self, action):
        for id_, a in action.items():
            if self.current_phase[id_] == a:
                self.current_phase_time[id_] += 1
            else:
                self.current_phase[id_] = a
                self.current_phase_time[id_] = 1
            self.eng.set_tl_phase(id_, self.current_phase[id_])  
        self.eng.next_step()
        return self.get_state(), self.get_reward()

    def get_state(self):
        state = {id_: self.get_state_(id_) for id_ in self.intersection_id}
        return state

    def get_state_(self, id_):
        state = self.intersection_info(id_)
        start_vehicle_count = [state['start_lane_vehicle_count'][lane] for lane in self.start_lane[id_]]
        end_vehicle_count = [-state['end_lane_vehicle_count'][lane] for lane in self.end_lane[id_]]

        pressure = []
        start=[]
        end=[]
        for i in range(len(start_vehicle_count)):
            if i%3!=2:
                start.append(start_vehicle_count[i])
                end.append(end_vehicle_count[i])
        pressure.extend(start)
        pressure.extend(end)

        return_state = pressure
        return self.preprocess_state(return_state)

    def intersection_info(self, id_):
        state = {}
        state['lane_vehicle_count'] = self.eng.get_lane_vehicle_count()
        state['start_lane_vehicle_count'] = {lane: state['lane_vehicle_count'][lane] for lane in self.start_lane[id_]}
        state['end_lane_vehicle_count'] = {lane: state['lane_vehicle_count'][lane] for lane in self.end_lane[id_]}

        state['lane_waiting_vehicle_count'] = self.eng.get_lane_waiting_vehicle_count()
        state['start_lane_waiting_vehicle_count']={lane:state['lane_waiting_vehicle_count'][lane] for lane in self.start_lane[id_]}
        state['end_lane_waiting_vehicle_count']={lane:state['lane_waiting_vehicle_count'][lane] for lane in self.end_lane[id_]}

        state['current_phase'] = self.current_phase[id_]
        state['current_phase_time'] = self.current_phase_time[id_]
        return state

    def preprocess_state(self, state):
        return_state = np.array(state)
        if self.state_size is None:
            self.state_size = len(return_state.flatten())
        return_state = np.reshape(return_state, [1, self.state_size])
        return return_state

    def get_reward(self):
        reward = {id_: self.get_reward_(id_) for id_ in self.intersection_id}
        return reward

    def get_reward_(self, id_):
        pressure=0
        state = self.intersection_info(id_)
        start_vehicle_count = [state['start_lane_vehicle_count'][lane] for lane in self.start_lane[id_]]
        end_vehicle_count = [state['end_lane_vehicle_count'][lane] for lane in self.end_lane[id_]]
        for i in range(len(start_vehicle_count)):
            if i%3!=2:
                pressure+=(start_vehicle_count[i]-end_vehicle_count[i])
        reward=-abs(pressure)
        return reward