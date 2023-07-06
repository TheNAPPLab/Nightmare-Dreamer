import numpy as np
import torch
from scipy.signal import lfilter
from skimage.transform import resize

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
    scores_ = []
    for _ in range(5):
        obs_, _ = env.reset()
        score_ = 0
        terminated_, truncated_ = False, False
        prev_rssmstate_ = trainer.RSSM._init_rssm_state(1)
        prev_action_ = torch.zeros(1, trainer.action_size).to(trainer.device)
        while not terminated_ and not truncated_:
            with torch.no_grad():
                embed_ = trainer.ObsEncoder(torch.tensor(obs_, dtype=torch.float32).unsqueeze(0).to(trainer.device))  
                _, posterior_rssm_state_ = trainer.RSSM.rssm_observe(embed_, prev_action_, not terminated_, prev_rssmstate_)
                model_state_ = trainer.RSSM.get_model_state(posterior_rssm_state_)
                action_, _ = trainer.ActionModel(model_state_)
                action = action.detach()
        next_obs_, rew_, terminated_, truncated_, _ = env.step(np.argmax(action_.cpu().numpy()))
        score_ += rew_
        obs_ = next_obs_
        prev_rssmstate_ = posterior_rssm_state_
        prev_action_ = action_
    return np.mean(scores_)



def eval_model_continous(env, trainer):
    scores_ = []
    for _ in range(5):
        score_ = 0
        obs_, _ = env.reset()
        terminated_, truncated_ = False, False
        prev_rssmstate_ = trainer.RSSM._init_rssm_state(1)
        prev_action_ = torch.zeros(1, trainer.action_size).to(trainer.device)
        while not terminated_ and not truncated_:
            with torch.no_grad():
                embed_ = trainer.ObsEncoder(torch.tensor(obs_, dtype=torch.float32).unsqueeze(0).to(trainer.device))  
                _, posterior_rssm_state_ = trainer.RSSM.rssm_observe(embed_, prev_action_, not terminated_, prev_rssmstate_)
                model_state_ = trainer.RSSM.get_model_state(posterior_rssm_state_)
                action_, _ = trainer.ActionModel(model_state_, deter = True)
            next_obs_, rew_, terminated_, truncated_, _ = env.step(action_.squeeze(0).cpu().numpy())
            score_ += rew_
            obs_ = next_obs_
            prev_rssmstate_ = posterior_rssm_state_
            prev_action_ = action_
        scores_.append(score_) 
    return np.mean(scores_)

def eval_model_continous_images(env, trainer):
    scores_ = []
    for _ in range(5):
        score_ = 0
        obs_, _ = env.reset()
        terminated_, truncated_ = False, False
        prev_rssmstate_ = trainer.RSSM._init_rssm_state(1)
        prev_action_ = torch.zeros(1, trainer.action_size).to(trainer.device)
        while not terminated_ and not truncated_:
            with torch.no_grad():
                embed_ = trainer.ObsEncoder(torch.tensor(get_image_obs(obs_), dtype=torch.float32).unsqueeze(0).to(trainer.device))  
                _, posterior_rssm_state_ = trainer.RSSM.rssm_observe(embed_, prev_action_, not terminated_, prev_rssmstate_)
                model_state_ = trainer.RSSM.get_model_state(posterior_rssm_state_)
                action_, _ = trainer.ActionModel(model_state_, deter = True)
            next_obs_, rew_, terminated_, truncated_, _ = env.step(action_.squeeze(0).cpu().numpy())
            score_ += rew_
            obs_ = next_obs_
            prev_rssmstate_ = posterior_rssm_state_
            prev_action_ = action_
        scores_.append(score_) 
    return np.mean(scores_)

def exponential_decay(initial_value, decay_rate, current_step):
    decayed_value = initial_value * np.exp(-decay_rate * current_step)
    return decayed_value


class PinkNoiseGenerator:
    def __init__(self, length):
        self.length = length
        self.coefficients = None
        self.set_coefficients()  # Call set_coefficients during initialization

    def generate_noise(self, decay_factor):
        white_noise = np.random.randn(self.length)
        filtered_noise = lfilter(self.coefficients, 1.0, white_noise)
        pink_noise = filtered_noise / np.max(np.abs(filtered_noise))
        decayed_noise = decay_factor * pink_noise
        return decayed_noise

    def set_coefficients(self):
        order = self.length - 1
        self.coefficients = np.zeros(order + 1)
        self.coefficients[0] = 0.02109238
        self.coefficients[order] = 0.95861208
        for k in range(1, int(order / 2) + 1):
            denominator = 1 - (k / (order + 1)) ** 2
            self.coefficients[k] = self.coefficients[order - k] = np.sqrt(denominator)

def  get_image_obs(obs):
    image = obs['pixels'].transpose(2, 0, 1)
    return resize(image, (3, 64, 64)) / 255.0 


