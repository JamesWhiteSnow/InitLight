import json
import os
from datetime import datetime
import pandas as pd
from time import *
import torch
from copy import deepcopy
import argparse

from cityflow_env import CityFlowEnvM
from utility import parse_roadnet, plot_data_lists
from ppo_agent import MPPOAgent

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  

class PPOConfig:
    def __init__(self) -> None:
        self.batch_size = 5
        self.gamma=0.99
        self.n_epochs = 4
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.gae_lambda=0.95
        self.policy_clip=0.2
        self.hidden_dim = 32
        self.update_fre = 40
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def main():
    date = datetime.now().strftime('%Y%m%d_%H%M%S')

    parser=argparse.ArgumentParser()
    parser.add_argument('--d',type=str,default='Syn1',help='dataset')
    args=parser.parse_args()

    dataset_path="Datasets/"+str(args.d)+"/"

    cityflow_config = {
        "interval": 1,
        "seed": 0,
        "laneChange": False,
        "dir": dataset_path,
        "roadnetFile": "roadnet.json",
        "flowFile": "flow.json",
        "rlTrafficLight": True,
        "saveReplay": False,
        "roadnetLogFile": "replayRoadNet.json",
        "replayLogFile": "replayLogFile.txt"
    }

    with open(os.path.join(dataset_path, "cityflow.config"), "w") as json_file:
        json.dump(cityflow_config, json_file)

    config = {
        'cityflow_config_file': dataset_path+"cityflow.config",
        'epoch': 200,
        'num_step': 3600,  
        'save_freq': 1,
        'phase_step': 10, 
        'model': 'PPO',
    }

    cfg=PPOConfig()

    cityflow_config = json.load(open(config['cityflow_config_file']))
    roadnetFile = cityflow_config['dir'] + cityflow_config['roadnetFile']
    config["lane_phase_info"] = parse_roadnet(roadnetFile)

    intersection_id = list(config['lane_phase_info'].keys())  
    config["intersection_id"] = intersection_id
    phase_list = {id_: config["lane_phase_info"][id_]["phase"] for id_ in intersection_id}
    config["phase_list"] = phase_list

    model_dir = "model/{}_{}".format(config['model'], date)
    result_dir = "result/{}_{}".format(config['model'], date)
    config["result_dir"] = result_dir

    if not os.path.exists("model"):
        os.makedirs("model")
    if not os.path.exists("result"):
        os.makedirs("result")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    env = CityFlowEnvM(config["lane_phase_info"],
                       intersection_id,
                       num_step=config["num_step"],
                       thread_num=8,
                       cityflow_config_file=config["cityflow_config_file"]
                       )

    config["state_size"] = env.state_size

    Magent = MPPOAgent(intersection_id,
                       state_size=config["state_size"],
                       cfg=cfg,
                       phase_list=config["phase_list"]
                       )
    Magent.load('')

    EPISODES = config['epoch']
    total_step = 0
    episode_rewards = {id_: [] for id_ in intersection_id}
    episode_travel_time = []


    i=0
    while i<EPISODES:
        env.reset()
        state = env.get_state()

        episode_length = 0
        episode_reward = {id_: 0 for id_ in intersection_id} 
        while episode_length < config['num_step']:
            action,prob,val = Magent.choose_action(state)  
            action_phase = {}
            for id_, a in action.items():
                action_phase[id_] = phase_list[id_][a]

            next_state, reward = env.step(action_phase)  

            for _ in range(config['phase_step'] - 1):
                next_state, reward_ = env.step(action_phase)

                episode_length += 1
                total_step += 1
                for id_ in intersection_id:
                    reward[id_] += reward_[id_]

            for id_ in intersection_id:
                episode_reward[id_] += reward[id_]

            episode_length += 1
            total_step += 1

            done = {}
            if episode_length==3600:
                done={id_: 1 for id_ in intersection_id}
            else:
                done={id_: 0 for id_ in intersection_id}

            Magent.remember(state, action, prob, val, reward_, done)
            if total_step % cfg.update_fre == 0:
                Magent.replay()

            state = next_state

        for id_ in intersection_id:
            episode_rewards[id_].append(episode_reward[id_])

        print('\n')
        print('Episode: {},travel time: {}'.format(i, env.eng.get_average_travel_time()))
        episode_travel_time.append(env.eng.get_average_travel_time())
        for id_ in intersection_id:
            episode_rewards[id_].append(episode_reward[id_])
        i+=1
    
    Magent.save(model_dir)

    df = pd.DataFrame(episode_rewards)
    df.to_csv(result_dir + '/rewards.csv', index=False)

    df = pd.DataFrame({"travel time": episode_travel_time})
    df.to_csv(result_dir + '/travel time.csv', index=False)

    plot_data_lists([episode_rewards[id_] for id_ in intersection_id], intersection_id,
                    figure_name=result_dir + '/rewards.pdf')
    plot_data_lists([episode_travel_time], ['travel time'], figure_name=result_dir + '/travel time.pdf')

if __name__ == '__main__':
    main()
