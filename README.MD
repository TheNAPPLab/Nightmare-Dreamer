# Constrained Model-Based RL Via Latent Nighmtares (Nightmare - Dreamer with constraints)


## Main ideas and modification
Add Cost to buffers
learn to predict cost along with rest of model
cost is added to model loss
so i need to learn gamma?
you never want to pass the cost

Adjust bffers to take 
## Setup 

Install and setup safety gym or  Safe Panda GYM
```
git clone https://github.com/openai/safety-gym.git
pip3 install -U 'mujoco-py<2.2,>=2.1'
```

install  safety gym following instructions