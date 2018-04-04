echo 'Experiment 3'
python main.py --gym_id HalfCheetah-v2 --agent PPO --episodes 15000 --exp exp_3 --novisualize
echo 'Experiment 4'
python main.py --gym_id Walker2d-v2 --agent PPO --episodes 15000 --exp exp_4 --novisualize
echo 'Experiment 5'
python main.py --gym_id Hopper-v2 --agent PPO --episodes 15000 --exp exp_5 --novisualize
