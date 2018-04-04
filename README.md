# A Comparison of Deep Reinforcement Learning Algorithms
### Submission For:
- *Course:* CMPT 888 Computer Animation
- *Instructor:* Dr. KangKang Yin
- *Term:* Spring 2018
- *University:* Simon Fraser University


### Submission By:
- *Name:* Anmol Sharma
- *Email:* asa224@sfu.ca
- *University:* Simon Fraser University


# Instructions

### Folder Structure

- `logs` contains log files generated during of various DRL agents
- `modules` contains custom definitions of DRL agents
- `notebooks` contains practice/prototype notebooks for rapid experimentation and prototyping
- `weights` contains saved pre-trained model weights corresponding to each agent and their experiments

### How to Run (Individual Agents Training/Testing)

The file `main.py` is capable of loading, training and testing all the agents defined inside the `custom_agents` file in `modules`.

For example, if you want to train a DDPG agent on MountainCarContinuous-v0 environment from OpenAI Gym for 1000 episodes while visualizing the training process, you'd write the following:

```
python main.py --gym_id MountainCarContinuous-v0 --agent DDPG --episodes 1000 --exp exp_test
```

To test a pre-trained policy on an environment:

```
python main.py --gym_id MountainCarContinuous-v0 --agent DDPG --load ./weights/DDPG/<exp_name>/
```

### How to Run (Reproduce Experiments in Report)
The experiments are divided into three task:

| Experiment Name | Environment    | #Episodes |
|-----------------|----------------|-----------|
| exp_1           | HalfCheetah-v2 | 15000     |
| exp_2           | Walker2d-v2    | 15000     |
| exp_3           | Hopper-v2      | 15000     |

Three shell scripts by the names:

```
PPO_Experiments.sh
TRPO_Experiments.sh
VPG_Experiments.sh
```
are included in the folder. Running these scripts will run the three experiments corresponding to each agent and each supported environment according to the table above. 
