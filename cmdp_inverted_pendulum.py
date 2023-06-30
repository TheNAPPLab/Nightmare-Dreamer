import wandb
import argparse
import os
import torch
import numpy as np
import gymnasium

from CMBR.utils.wrapper import GymMinAtar, OneHotAction, NormalizeActions, SafetyGymEnv
from CMBR.training.config import BaseSafeConfig
from CMBR.training.trainer_inverted_pendulum import Trainer
from CMBR.training.evaluator import Evaluator


def main(args):
    number_games = 0
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

    env = gymnasium.make(env_name) 

    action_size = env.action_space.shape[0]
    action_dtype = np.float32
    batch_size = args.batch_size
    seq_len = args.seq_len

    obs_shape = env.observation_space.shape



        # image_shape = obs['vision'].shape
    config = BaseSafeConfig(
            env = env_name,
            pixel = False,
            obs_shape = obs_shape,
            action_size = action_size,
            obs_dtype = np.float32,
            action_dtype = action_dtype,
            seq_len = seq_len,
            batch_size = batch_size,
            model_dir = model_dir, 
    )

    exploration_rate = 0.1
    config_dict = config.__dict__
    trainer = Trainer(config, device)
    evaluator = Evaluator(config, device)



    with wandb.init(project = 'Safe RL via Latent world models', config = config_dict):
        """training loop"""
        print('...training...')
        train_metrics = {}
        trainer.collect_seed_episodes(env)
        obs, _ = env.reset()
        score =  0
        terminated, truncated = False, False
        prev_rssmstate = trainer.RSSM._init_rssm_state(1)
        prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
        episode_actor_ent = []
        scores = []
        best_mean_score = 0
        best_save_path = os.path.join(model_dir, 'models_best.pth')

        def evaluate_deter():
            eval_obs, _ =  env.reset()
            eval_score =  0
            eval_terminated, eval_truncated = False, False
            eval_prev_rssmstate = trainer.RSSM._init_rssm_state(1)
            eval_prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)

            while not (eval_terminated or eval_truncated):
                with torch.no_grad():
                    eval_embed = trainer.ObsEncoder(torch.tensor(eval_obs, dtype = torch.float32).unsqueeze(0).to(trainer.device))  
                    _, eval_posterior_rssm_state = \
                        trainer.RSSM.rssm_observe(eval_embed, eval_prev_action, not eval_terminated, eval_prev_rssmstate)
                    eval_model_state = trainer.RSSM.get_model_state(eval_posterior_rssm_state)
                    eval_action, _= trainer.ActionModel(eval_model_state, deter = True)
                    eval_action = 3 * eval_action
                    eval_next_obs, eval_reward, eval_terminated, eval_truncated, _ = \
                        env.step(eval_action.squeeze(0).cpu().numpy())
                    eval_score += eval_reward
                    if eval_terminated or eval_truncated:
                        return eval_score
                    else:
                        eval_obs = eval_next_obs
                        eval_prev_rssmstate = eval_posterior_rssm_state
                        eval_prev_action = eval_action


        for iter in range(1, trainer.config.train_steps):  

            if iter % trainer.config.train_every == 0:
                train_metrics = trainer.train_batch(train_metrics)

            if iter % trainer.config.slow_target_update == 0:
                trainer.update_target()

            if iter % trainer.config.save_every == 0:
                trainer.save_model(iter)
        
                
            with torch.no_grad():
                embed = trainer.ObsEncoder(torch.tensor(obs, dtype = torch.float32).unsqueeze(0).to(trainer.device))  
                _, posterior_rssm_state = trainer.RSSM.rssm_observe(embed, prev_action, not terminated, prev_rssmstate)
                model_state = trainer.RSSM.get_model_state(posterior_rssm_state)
                action, action_dist = trainer.ActionModel(model_state, deter =  False)
                action = 3.0 * action
                action_ent = torch.mean(action_dist.entropy()).item()
                episode_actor_ent.append(action_ent)

            next_obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
            score += reward
            done_ = terminated or truncated
            if done_ :
                number_games += 1
                trainer.buffer.add(obs, action.squeeze(0).cpu().numpy(), reward, None, terminated)
                train_metrics['eval_rewards'] = evaluate_deter()
                train_metrics['train_rewards'] = score
                train_metrics['number_games']  = number_games
                train_metrics['action_ent'] =  np.mean(episode_actor_ent)
                wandb.log(train_metrics, step = iter)
                scores.append(score)
                if len(scores)>100:
                    scores.pop(0)
                    current_average = np.mean(scores)
                    train_metrics['current_avg_score'] =  current_average
                    if current_average > best_mean_score:
                        best_mean_score = current_average 
                        print('saving best model with mean score : ', best_mean_score)
                        save_dict = trainer.get_save_dict()
                        torch.save(save_dict, best_save_path)
                
                #reset operations
                obs, _ =  env.reset()
                score =  0
                terminated, truncated = False, False
                prev_rssmstate = trainer.RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
                episode_actor_ent = []
            else:
                trainer.buffer.add(obs, action.squeeze(0).detach().cpu().numpy(), reward, None, terminated)
                obs = next_obs
                del next_obs
                prev_rssmstate = posterior_rssm_state
                prev_action = action


    '''evaluating probably best model'''
    evaluator.eval_saved_agent(env, best_save_path)

if __name__ == "__main__":

    """there are tonnes of HPs, if you want to do an ablation over any particular one, please add if here"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str,  default='InvertedPendulum-v4', help='gym env name')
    parser.add_argument("--is_use_vision", type = bool,  default = False, help='is it safe Panda gym')
    parser.add_argument("--id", type = str, default='0', help = 'Experiment ID')
    parser.add_argument('--seed', type=int, default=123, help = 'Random seed')
    parser.add_argument('--device', default = 'CUDA', help = 'CUDA or CPU')
    parser.add_argument('--batch_size', type = int, default = 50, help='Batch size')
    parser.add_argument('--seq_len', type = int, default = 50, help='Sequence Length (chunk length)')
    args = parser.parse_args()
    print(args)
    main(args)
