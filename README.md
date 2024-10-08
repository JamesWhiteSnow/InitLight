# InitLight
Due to repetitive trial-and-error style interactions between agents and a fixed traffic environment during the policy learning, existing Reinforcement Learning (RL)-based Traffic Signal Control (TSC) methods greatly suffer from long RL training time and poor adaptability of RL agents to other complex traffic environments. To address these problems, we propose a novel Adversarial Inverse Reinforcement Learning (AIRL)-based pretraining method named InitLight, which enables effective initial model generation for TSC agents. Unlike traditional RL-based TSC approaches that train a large number of agents simultaneously for a specific multi-intersection environment, InitLight pretrains only one single initial model based on multiple single-intersection environments together with their expert trajectories. Since the reward function learned by InitLight can recover ground-truth TSC rewards for different intersections at optimality, the pre-trained agent can be deployed at intersections of any traffic environments as initial models to accelerate subsequent overall global RL training. Comprehensive experimental results show that, the initial model generated by InitLight can not only significantly accelerate the convergence with much fewer episodes, but also own superior generalization ability to accommodate various kinds of complex traffic environments.

In this project, we open-source the source code of our InitLight approach. 

On Git Hub, we will introduce how to reproduce the results of our experiments in the paper.

For details of our method, please see our [original paper](https://dl.acm.org/doi/abs/10.24963/ijcai.2023/550) at 2023 International Joint Conference on Artificial Intelligence (IJCAI).

Welcome to cite our paper!

```
@inproceedings{ye2023initlight,
  title={InitLight: initial model generation for traffic signal control using adversarial inverse reinforcement learning},
  author={Ye, Yutong and Zhou, Yingbo and Ding, Jiepin and Wang, Ting and Chen, Mingsong and Lian, Xiang},
  booktitle ={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2023},
  pages={4949-4958},
  doi={10.24963/ijcai.2023/550}
}
```

## Requirements
Under the root directory, execute the following conda commands to configure the Python environment.
``conda create --name <new_environment_name> --file requirements.txt``

``conda activate <new_environment_name>``

### Simulator installation
Our experiments are implemented on top of the traffic simulator Cityflow. Detailed installation guide files can be found in https://cityflow-project.github.io/

#### 1. Install cpp dependencies
``sudo apt update && sudo apt install -y build-essential cmake``

#### 2. Clone CityFlow project from github
``git clone https://github.com/cityflow-project/CityFlow.git``

#### 3. Go to CityFlow project’s root directory and run
``pip install .``

#### 4. Wait for installation to complete and CityFlow should be successfully installed
``import cityflow``

``eng = cityflow.Engine``

## Run the code
#### Execute the following command to run the experiment over the specified dataset.
``python train.py --d <dataset_name>``

## Datasets
For the experiments, we used both synthetic and real-world traffic datasets provided by https://traffic-signal-control.github.io/dataset.html.
| Dataset Name | Dataset Type | # of intersections |
| :-----------: | :-----------: | :-----------: |
| Syn1 | Synthetic | 1×3 |
| Syn2 | Synthetic | 2×2 |
| Syn3 | Synthetic | 3×3 |
| Syn4 | Synthetic | 4×4 |
| Hangzhou1 | Real-world | 4×4 |
| Hangzhou2 | Real-world | 4×4 |
| Jinan1 | Real-world | 4×3 |
| Jinan2 | Real-world | 4×3 |
| Jinan3 | Real-world | 4×3 |
