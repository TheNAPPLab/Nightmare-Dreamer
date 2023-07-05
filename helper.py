import numpy as np
import torch

def control_to_onehot(control_values, num_bins = 15):
    # Convert control values to one-hot array
    control_value_x = control_values[0]
    control_value_y = control_values[1]

    index_x = int(((control_value_x + 1) / 2) * (num_bins - 1))
    index_y = int(((control_value_y + 1) / 2) * (num_bins - 1))

    index = index_y * num_bins + index_x

    onehot = np.zeros(num_bins * num_bins)
    onehot[index] = 1

    return onehot

def onehot_to_control(onehot, num_bins = 15):
    # Convert one-hot array to control values
    index = np.argmax(onehot)

    index_x = index % num_bins
    index_y = index // num_bins

    control_value_x = (index_x / (num_bins - 1) * 2) - 1
    control_value_y = (index_y / (num_bins - 1) * 2) - 1

    return np.array([control_value_x, control_value_y])

def calculate_epsilon(config, itr):
    expl_amount = config.expl['train_noise'] = 1.0
    expl_amount = expl_amount - itr/config.expl['expl_decay']
    return max(config.expl['expl_min'] , expl_amount)

def eval_model(env, trainer):
    obs_, _ = env.reset()
    score_ = 0
    terminated_, truncated_ = False, False
    prev_rssmstate_ = trainer.RSSM._init_rssm_state(1)
    prev_action_ = torch.zeros(1, trainer.action_size).to(trainer.device)
    scores_ = []

    for _ in range(5):
        with torch.no_grad():
            embed_ = trainer.ObsEncoder(torch.tensor(obs_, dtype=torch.float32).unsqueeze(0).to(trainer.device))  
            _, posterior_rssm_state_ = trainer.RSSM.rssm_observe(embed_, prev_action_, not terminated_, prev_rssmstate_)
            model_state_ = trainer.RSSM.get_model_state(posterior_rssm_state_)
            action_, _ = trainer.ActionModel(model_state_)
        next_obs_, rew_, terminated_, truncated_, _ = env.step(np.argmax(action_.cpu().numpy()))
        score_ += rew_
        if terminated_ or truncated_:
            trainer.buffer.add(obs_, action_.squeeze(0).cpu().numpy(), rew_, terminated_) # add high quality games also
            scores_.append(score_) 
            obs_, _ = env.reset()
            score_ = 0
            terminated_, truncated_ = False, False
            prev_rssmstate_ = trainer.RSSM._init_rssm_state(1)
            prev_action_ = torch.zeros(1, trainer.action_size).to(trainer.device)
        else:
            trainer.buffer.add(obs_, action_.squeeze(0).detach().cpu().numpy(), rew_, terminated_)
            obs_ = next_obs_
            prev_rssmstate_ = posterior_rssm_state_
            prev_action_ = action_
    return np.mean(scores_)