## Dreamer- v2 Pytorch

Pytorch implementation of [Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193)<br>

<p align="middle" >
  <img src="images/breakout.gif" title="breakout" width="200" />
  <img src="images/space_invaders.gif" title="space_invaders" width="200" /> 
  <img src="images/asterix.gif" title="asterix" width="200" />
  <img src="images/seaquest.gif" title="seaquest" width="200" /> 
</p>

## Installation

Dependencies:
* [Pytorch](https://pytorch.org/)
* [Gym](https://github.com/openai/gym)
* [MinAtar](https://github.com/kenjyoung/MinAtar)
* [Wandb](https://wandb.ai/) 

I have added requirements.txt using `conda list -e > requirements.txt` and environment.yml using `conda env export > environment.yml` from my own conda environment. <br>
I think it is easier to create a new conda environment(or venv etc.) and manually install the above listed few dependencies one by one.

## Running experiments
1) In tests folder, mdp.py and pomdp.py have been setup for experiments with MinAtar environments. All default hyper-parameters used are stored in a dataclass in [config.py](https://github.com/RajGhugare19/dreamerv2/blob/b6d65b8af7f91ae106c5b0cc11e29a2247dfa233/dreamerv2/training/config.py#L9). To run dreamerv2 with default HPs on POMDP breakout and cuda :
  ```
  python pomdp.py --env breakout --device cuda
  ``` 
  - Training curves are logged using wandb. 
  - A `results` folder will be created locally to store models while training:
  `test/results/env_name+'_'+env_id+'_'+pomdp/models`   

2) Experimenting on other environments(using gym-api) can be done by adding another hyper-parameter dataclass in [config.py](https://github.com/RajGhugare19/dreamerv2/blob/b6d65b8af7f91ae106c5b0cc11e29a2247dfa233/dreamerv2/training/config.py#L9). <br>

## Evaluating saved models

Trained models for all 5 games (mdp and pomdp version of each) are uploaded to the drive link: [link](https://drive.google.com/file/d/11200qHbwKRjfe3xRI_G6nyYaZRBFt8_k/view?usp=sharing) (64 MBs)<br>
Download and unzip the models inside /test directory.

Evaluate the saved model for POMDP version of breakout environment for 5 episodes, without rendering:
```
python eval.py --env breakout --eval_episode 5 --eval_render 0 --pomdp 1
```

## Evaluation Results

Average evaluation score(over 50 evaluation episodes) of models saved at every 0.1 million frames. Green curves correspond to agent which have access to complete information, while red curves correspond to agents trained with partial observability.

<img src="images/eval.png" width="5000" height="400">

In freeway, the agent gets stuck in a local maxima, wherein it learns to always move forward. The reason being that it is not penalised for crashing into cars. Probably due to policy entropy regularisation, its returns drop drastically around the 1 million frame mark, and gradually improve while maintaing the policy entropy.

## Training curves

All experiments were logged using wandb. Training runs for all MDP and POMDP variants of MinAtar environments can be found on the [wandb project](https://wandb.ai/raj19/mastering%20MinAtar%20with%20world%20models?workspace=user-raj19) page.

Please create an issue if you find a bug or have any queries.

## Code structure:
- `test`
  - `pomdp.py` run MinAtar experiments with partial observability.
  - `mdp.py` run MinAtar experiments with complete observability.
  - `eval.y` evaluate saved agents.
- `dreamerv2` dreamerv2 plus dreamerv1 and their combinations.
  - `models` neural network models.
    - `actor.py` discrete action model.
    - `dense.py` fully connected neural networks.
    - `pixel.py` convolutional encoder and decoder.
    - `rssm.py` recurrent state space model.
  - `training`
    - `config.py` hyper-parameter dataclass.
    - `trainer.py` training class, loss calculation.
    - `evaluator.py` evaluation class.
  - `utils`
    - `algorithm.py` lambda return function.
    - `buffer.py` replay buffers, batches of sequences.
    - `module.py` neural network parameters utils.
    - `rssm.py` recurrent state space model utils.
    - `wrapper.py` gym api and pomdp wrappers for MinAtar.     

## Hyper-Parameter description:

- `train_every`: number of frames to skip while training.
- `collect_intervals`: number of batches to be sampled from buffer, at every "train-every" iteration.
- `seq_len`: length of trajectory sequence to be sampled from buffer.
- `embedding_size`: size of embedding vector that is output by observation encoder.
- `rssm_type`: categorical or gaussian random variables for stochastic states.
- `rssm_node_size`: size of hidden layers of temporal posteriors and priors.
- `deter_size`: size of deterministic part of recurrent state.
- `stoch_size`: size of stochastic part of recurrent state.
- `class_size`: number of classes for each categorical random variable
- `category_size`: number of categorical random variables.
- `horizon`: horizon for imagination in future latent state space.
- `kl_balance_scale`: scale for kl balancing.
- `actor_entropy_scale`: scale for policy entropy regularization in latent state space.


## Acknowledgments
Awesome Environments used for testing:

- MinAtar by kenjyoung : [https://github.com/kenjyoung/MinAtar](https://github.com/kenjyoung/MinAtar)<br>
- qlan3's gym-games : [https://github.com/qlan3/gym-games](https://github.com/qlan3/gym-games)
- minigrid by maximecb : [https://github.com/maximecb/gym-minigrid](https://github.com/maximecb/gym-minigrid)<br>

This code is heavily inspired by the following works:

- danijar's Dreamer-v2 tensorflow implementation : [https://github.com/danijar/dreamerv2](https://github.com/danijar/dreamerv2)<br>
- juliusfrost's Dreamer-v1 pytorch implementation : [https://github.com/juliusfrost/dreamer-pytorch](https://github.com/juliusfrost/dreamer-pytorch)<br>
- yusukeurakami's Dreamer-v1 pytorch implementation: [https://github.com/yusukeurakami/dreamer-pytorch](https://github.com/yusukeurakami/dreamer-pytorch)<br>
- alec-tschantz's  PlaNet pytorch implementation : [https://github.com/alec-tschantz/planet](https://github.com/alec-tschantz/planet)<br>
- Kaixhin's PlaNet pytorch implementation : [https://github.com/Kaixhin/PlaNet](https://github.com/Kaixhin/PlaNet)
