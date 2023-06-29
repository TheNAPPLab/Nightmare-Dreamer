import wandb
import argparse
import os
import torch
import numpy as np
import gym
import sys
import safety_gymnasium

# from torch.utils.tensorboard import SummaryWriter


# sys.path.append('/Users/emma/dev/CMBRVLN/Safe-panda-gym')
# import panda_gym
from CMBR.utils.wrapper import GymMinAtar, OneHotAction, NormalizeActions, SafetyGymEnv
from CMBR.training.config import BaseSafeConfig
from CMBR.training.trainer import Trainer
from CMBR.training.evaluator import Evaluator

# def render(self, mode='human'):
#         if mode == 'rgb_array':
#             return self.env.state()
#         elif mode == 'human':
#             self.env.display_state(self.display_time)


def main(args):
    # tb = SummaryWriter()
    number_games = 0
    # def logTensorboard(data_dict,iter):
    #     for key, value in data_dict.items():
    #         tb.add_scalar(key, value, iter)

    wandb.login()
   
    env_name = args.env
    exp_id = args.id

    '''make dir for saving results'''
    result_dir = os.path.join('results', '{}_{}'.format(env_name, exp_id))
    model_dir = os.path.join(result_dir, 'models')  #dir to save learnt models
    os.makedirs(model_dir, exist_ok = True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #intialise Device
    if torch.cuda.is_available() and args.device:
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')

    print('using :', device)  
    # env = gym.make(env_name)
    env = safety_gymnasium.make(env_name) 
    action_space = env.action_space



    action_size = env.action_space.shape[0]
    obs_dtype = bool
    action_dtype = np.float32
    batch_size = args.batch_size
    seq_len = args.seq_len

 
        # obs_shape = env.observation_space.shape
    obs, info = env.reset()
    image_shape = obs.shape
        # image_shape = obs['vision'].shape
    config = BaseSafeConfig(
            env = env_name,
            pixel = False,
            obs_shape = image_shape,
            action_size = action_size,
            obs_dtype = np.float32,
            action_dtype = action_dtype,
            seq_len = seq_len,
            batch_size = batch_size,
            model_dir = model_dir, 
    )

    exploration_rate = 0.1
    config_dict = config.__dict__
    config_dict['experment'] ='Using primal dual method without a value function'
    trainer = Trainer(config, device)
    evaluator = Evaluator(config, device)

    with wandb.init(project='Safe RL via Latent world models', config = config_dict):
        """training loop"""
        print('...training...')
        train_metrics = {}
        trainer.collect_seed_episodes(env)
        obs, info  = env.reset()
        score, score_cost = 0, 0
        terminated, truncated = False, False
        prev_rssmstate = trainer.RSSM._init_rssm_state(1) # discrete hidden state
        prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)  
        episode_actor_ent = []   
        scores = []
        costs = []
        best_mean_score = 0
        best_mean_score_cost = 0
        best_combination_score = 0
        best_save_path = os.path.join(model_dir, 'models_best.pth')
        
        for iter in range(1, trainer.config.train_steps):  

            if iter % trainer.config.train_every == 0:
                train_metrics = trainer.train_batch(train_metrics)

            if iter % trainer.config.slow_target_update == 0:
                trainer.update_target()

            if iter % trainer.config.save_every == 0:
                trainer.save_model(iter)
                
            # action selection and roll out to obtain prosterior and prior
            with torch.no_grad():
                embed = trainer.ObsEncoder(torch.tensor(obs, dtype = torch.float32).unsqueeze(0).to(trainer.device))  
                _, posterior_rssm_state = trainer.RSSM.rssm_observe(embed, prev_action, not (terminated or truncated), prev_rssmstate)
                model_state = trainer.RSSM.get_model_state(posterior_rssm_state)
                action, action_dist = trainer.ActionModel(model_state, deter = not True)
                action = trainer.ActionModel.add_exploration(action, exploration_rate).detach()
                if iter % 400 == 0:
                    exploration_rate *= 0.99  # slowly decaying the noise
                action_ent = torch.mean(action_dist.entropy()).item()
                episode_actor_ent.append(action_ent)

            next_obs, reward, cost, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
            score_cost += cost
            score += reward
            done_ = terminated or truncated
            if done_ :
                number_games += 1
                trainer.buffer.add(obs, action.squeeze(0).cpu().numpy(), reward, cost, terminated)
                train_metrics['score_training'] = score
                train_metrics['number_games']  = number_games
                train_metrics['costs_score_training'] = score_cost
                train_metrics['action_ent_training'] =  np.mean(episode_actor_ent)
                wandb.log(train_metrics, step=iter)
                scores.append(score)
                costs.append(cost)
                if len(scores)>100:
                    scores.pop(0)
                    costs.pop(0)
                    current_average = np.mean(scores)
                    curr_avg_cost = np.mean(costs)
                    train_metrics['current_avg_score'] =  current_average
                    train_metrics['current_avg_cost'] =  curr_avg_cost

                    if current_average > best_mean_score:
                        best_mean_score = current_average 
                        print('saving best model with mean score : ', best_mean_score)
                        save_dict = trainer.get_save_dict()
                        torch.save(save_dict, best_save_path)
                
                obs, _ =  env.reset()
                score, score_cost = 0, 0
                terminated, truncated = False, False
                prev_rssmstate = trainer.RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
                episode_actor_ent = []
            else:
                trainer.buffer.add(obs, action.squeeze(0).detach().cpu().numpy(), reward, cost, terminated)
                obs = next_obs
                del next_obs
                prev_rssmstate = posterior_rssm_state
                prev_action = action


    '''evaluating probably best model'''
    evaluator.eval_saved_agent(env, best_save_path)

if __name__ == "__main__":

    """there are tonnes of HPs, if you want to do an ablation over any particular one, please add if here"""
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env", type=str,  default='SafetyRacecarGoal1Vision-v0', help='mini atari env name')
    parser.add_argument("--env", type=str,  default='SafetyPointGoal1-v0', help='mini atari env name')
#  parser.add_argument("--env", type=str,  default='PandaReachSafe-v2', help='mini atari env name')
    parser.add_argument("--is_use_vision", type = bool,  default = True, help='is it safe Panda gym')
    parser.add_argument("--id", type = str, default='0', help = 'Experiment ID')
    parser.add_argument('--seed', type=int, default=123, help = 'Random seed')
    parser.add_argument('--device', default = 'cpu', help = 'CUDA or CPU')
    parser.add_argument('--batch_size', type = int, default = 50, help='Batch size')
    parser.add_argument('--seq_len', type = int, default = 50, help='Sequence Length (chunk length)')
    args = parser.parse_args()
    main(args)
