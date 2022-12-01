# Constrained Model-Based RL Via Latent Nighmtares (Nightmare - Dreamer with constraints)




## Setup 

Install and setup safety gym or  Safe Panda GYM
```
git clone https://github.com/openai/safety-gym.git
pip3 install -U 'mujoco-py<2.2,>=2.1'
```

install  safety gym following instructions



## To run 
```bash
python test/mdp.py --env Safexp-PointGoal1-v0 --device cuda

or 


python test/mdp.py --env Safexp-PointGoal1-v0 --device cuda
```

Training graph already projects to Weights and baises but to start tensorboard
```bash
 tensorboard --logdir runs 
```