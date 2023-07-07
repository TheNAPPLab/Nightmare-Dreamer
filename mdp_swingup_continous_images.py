import wandb
import argparse
import os
import torch
import numpy as np
import gym
import gymnasium 
# from dreamerv2.utils.wrapper import GymMinAtar, OneHotAction
from dreamerv2.training.config import MinAtarConfig
from dreamerv2.training.trainer_pendulum_continous_images import Trainer
from dreamerv2.training.evaluator import Evaluator
from helper import  eval_model_continous_images, get_image_obs 
from gymnasium.wrappers import PixelObservationWrapper


def main(args):
    wandb.login()
    env_name = args.env
    exp_id = args.id

    '''make dir for saving results'''
    result_dir = os.path.join('results', '{}_{}'.format(env_name, exp_id))
    model_dir = os.path.join(result_dir, 'models')                                                  #dir to save learnt models
    os.makedirs(model_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available() and args.device:
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')
    print('using :', device)  
    
    env = PixelObservationWrapper(gymnasium.make(env_name, render_mode="rgb_array"))
    obs_shape = (3, 64, 64)
    action_size = 1
    obs_dtype =  np.float32
    action_dtype = np.float32
    batch_size = args.batch_size
    seq_len = args.seq_len

    config = MinAtarConfig(
        env=env_name,
        pixel = True,
        actor_grad = 'dynamics', #reinforce
        obs_shape = obs_shape,
        action_size = action_size,
        obs_dtype = obs_dtype,
        action_dtype = action_dtype,
        seq_len = seq_len,
        batch_size = batch_size,
        model_dir = model_dir, 
    )
    number_games = 0
    config.actor['max_action'] = 2.0
    config.actor['dist'] = 'trunc_normal'
    config.expl['train_noise'] = 1.0
    config.expl['expl_min'] = 0.1
    config.expl['expl_decay'] = 150_000
    config.expl['decay_start'] = 30_000
    config.expl['expl_type'] = 'gaussian'
    config_dict = config.__dict__
    trainer = Trainer(config, device)
    evaluator = Evaluator(config, device)

    with wandb.init(project='Safe RL via Latent world models', config = config_dict):
        """training loop"""
        print('...training...')
        train_metrics = {}
        trainer.collect_seed_episodes(env)
        obs, _ = env.reset()
        score = 0
        terminated, truncated = False, False
        prev_rssmstate = trainer.RSSM._init_rssm_state(1)
        prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
        episode_actor_ent = []
        scores = []
        best_mean_score = 0
        best_save_path = os.path.join(model_dir, 'models_best.pth')
        for iter in range(1, trainer.config.train_steps):  
            if iter%trainer.config.train_every == 0:
                train_metrics = trainer.train_batch(train_metrics)
            if iter%trainer.config.slow_target_update == 0:
                trainer.update_target()                
            if iter%trainer.config.save_every == 0:
                trainer.save_model(iter)
            with torch.no_grad():
                embed = trainer.ObsEncoder(torch.tensor(get_image_obs(obs), dtype=torch.float32).unsqueeze(0).to(trainer.device)) 
                _, posterior_rssm_state = trainer.RSSM.rssm_observe(embed, prev_action, not terminated, prev_rssmstate)
                model_state = trainer.RSSM.get_model_state(posterior_rssm_state)
                action, action_dist = trainer.ActionModel(model_state, deter = False)
                action, expl_amount = trainer.ActionModel.add_exploration(iter, action, -config.actor['max_action'], config.actor['max_action'])
                action = action.detach()
                action_ent = torch.mean(action_dist.entropy()).item()
                episode_actor_ent.append(action_ent)
            next_obs, rew, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
            score += rew

            if terminated or truncated:
                number_games += 1
                trainer.buffer.add(get_image_obs(obs), action.squeeze(0).detach().cpu().numpy(), rew, terminated)
                train_metrics['train_rewards'] = score
                train_metrics['number_games']  = number_games
                train_metrics['action_ent'] =  np.mean(episode_actor_ent)
                train_metrics['Noise_Std'] = expl_amount
                if number_games % 100 == 0:
                    train_metrics['Eval_score'] = eval_model_continous_images(env, trainer)
                wandb.log(train_metrics, step=iter)
                scores.append(score)
                if len(scores)>100:
                    scores.pop(0)
                    current_average = np.mean(scores)
                    train_metrics['Last_100_avg_score'] =   np.mean(scores)
                    if current_average>best_mean_score:
                        best_mean_score = current_average 
                        train_metrics['current_best_avg_score'] =  current_average
                        print('saving best model with mean score : ', best_mean_score)
                        save_dict = trainer.get_save_dict()
                        torch.save(save_dict, best_save_path)
                obs, _ = env.reset()
                score = 0
                terminated, truncated = False, False
                prev_rssmstate = trainer.RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
                episode_actor_ent = []
            else:
                trainer.buffer.add(get_image_obs(obs), action.squeeze(0).detach().cpu().numpy(), rew, terminated)
                obs = next_obs
                prev_rssmstate = posterior_rssm_state
                prev_action = action

    '''evaluating probably best model'''
    evaluator.eval_saved_agent(env, best_save_path)

if __name__ == "__main__":

    """there are tonnes of HPs, if you want to do an ablation over any particular one, please add if here"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='Pendulum-v1',  help='mini atari env name')
    parser.add_argument("--id", type=str, default='0', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--device', default='cuda', help='CUDA or CPU')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=50, help='Sequence Length (chunk length)')
    args = parser.parse_args()
    main(args)
#pip install box2d pygame