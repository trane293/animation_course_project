echo 'Experiment 3'
python main.py --gym_id HalfCheetah-v2 --agent PPO --load ./weights/PPO/exp_3/ -nm 5
echo 'Experiment 4'
python main.py --gym_id Walker2d-v2 --agent PPO --load ./weights/PPO/exp_4/ -nm 5
echo 'Experiment 5'
python main.py --gym_id Hopper-v2 --agent PPO --load ./weights/PPO/exp_5/ -nm 5
