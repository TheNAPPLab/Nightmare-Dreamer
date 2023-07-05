import torch 
import torch.nn as nn
import numpy as np
from torch import distributions as torchd

class DiscreteActionModel(nn.Module):
    def __init__(
        self,
        action_size,
        deter_size,
        stoch_size,
        embedding_size,
        actor_info,
        expl_info
    ):
        super().__init__()
        self.action_size = action_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.embedding_size = embedding_size
        self.layers = actor_info['layers']
        self.node_size = actor_info['node_size']
        self.act_fn = actor_info['activation']
        self.dist = actor_info['dist']
        self.act_fn = actor_info['activation']
        self.train_noise = expl_info['train_noise']
        self.decay_start = expl_info['decay_start']
        self.eval_noise = expl_info['eval_noise']
        self.expl_min = expl_info['expl_min']
        self.expl_decay = expl_info['expl_decay']
        self.expl_type = expl_info['expl_type']
        self.model = self._build_model()

    def _build_model(self):
        model = [nn.Linear(self.deter_size + self.stoch_size, self.node_size)]
        model += [self.act_fn()]
        for i in range(1, self.layers):
            model += [nn.Linear(self.node_size, self.node_size)]
            model += [self.act_fn()]

        if self.dist == 'one_hot':
            model += [nn.Linear(self.node_size, self.action_size)]
        else:
            raise NotImplementedError
        return nn.Sequential(*model) 

    def forward(self, model_state):
        action_dist = self.get_action_dist(model_state)
        action = action_dist.sample()
        action = action + action_dist.probs - action_dist.probs.detach()
        return action, action_dist

    def get_action_dist(self, modelstate):
        logits = self.model(modelstate)
        if self.dist == 'one_hot':
            return torch.distributions.OneHotCategorical(logits=logits)         
        else:
            raise NotImplementedError
            
    def add_exploration(self, action: torch.Tensor, itr: int, mode='train'):
        if mode == 'train':
            if iter <= self.decay_start:
                expl_amount = self.train_noise
            else:
                expl_amount = self.train_noise
                ir = itr - self.decay_start + 1
                expl_amount = expl_amount - ir/self.expl_decay
                expl_amount = max(self.expl_min, expl_amount)

        elif mode == 'eval':
            expl_amount = self.eval_noise
        else:
            raise NotImplementedError
            
        if self.expl_type == 'epsilon_greedy':
            if np.random.uniform(0, 1) < expl_amount:
                index = torch.randint(0, self.action_size, action.shape[:-1], device=action.device)
                action = torch.zeros_like(action)
                action[:, index] = 1
            return action, expl_amount

        raise NotImplementedError
    
class ContinousActionModel(nn.Module):
    def __init__(
        self,
        action_size,
        deter_size,
        stoch_size,
        embedding_size,
        actor_info,
        expl_info
    ):
        super().__init__()
        self.action_size = action_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.embedding_size = embedding_size
        self._min_std = actor_info['min_std']
        self.layers = actor_info['layers']
        self.node_size = actor_info['node_size']
        self.act_fn = actor_info['activation']
        self.dist = actor_info['dist']
        self.act_fn = actor_info['activation']
        self.train_noise = expl_info['train_noise']
        self.eval_noise = expl_info['eval_noise']
        self.expl_min = expl_info['expl_min']
        self.expl_decay = expl_info['expl_decay']
        self.expl_type = expl_info['expl_type']
        self.model = self._build_model()

    def _build_model(self):
        model = [nn.Linear(self.deter_size + self.stoch_size, self.node_size)]
        model += [self.act_fn()]
        for i in range(1, self.layers):
            model += [nn.Linear(self.node_size, self.node_size)]
            model += [self.act_fn()]

        if self.dist == 'trunc_normal':
            model += [nn.Linear(self.node_size, self.action_size * 2)]
        else:
            raise NotImplementedError
        return nn.Sequential(*model) 

    def forward(self, model_state, deter = False):
        out = self.model(model_state)
        mean, std = torch.chunk(out, 2, dim=-1) 
        mean = torch.tanh(mean)
        std = 2 * torch.sigmoid(std / 2) + self._min_std
        dist = SafeTruncatedNormal(mean, std, -1, 1)
        dist = ContDist(torchd.independent.Independent(dist, 1))

        if not deter:
            action = dist.sample()
            # action = 3.0 * action
            return action, dist
        else:
            action = dist.mode()
            # action = 3.0 * action
            return action, dist
        
    def add_exploration(self, action, action_min, action_max, noise_std = 0.1):
        # Scale the action based on the exploration schedule
        # exploration_action = action * exploration_schedule

        # Add exploration noise
        noise = torch.randn_like(action) * noise_std  # Assuming noise_std is a parameter or a constant
        action += noise

        # Clip the action within valid bounds (if needed)
        action = torch.clamp(action, action_min, action_max)

        return action
 

class SafeTruncatedNormal(torchd.normal.Normal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def rsample(self, sample_shape):
        event = super().rsample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event
    
class ContDist:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)
    
    def mean_(self):
        return self.mean

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        return self._dist.mean

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        return self._dist.log_prob(x)
    
class RequiresGrad:
    def __init__(self, model):
        self._model = model

    def __enter__(self):
        self._model.requires_grad_(requires_grad=True)

    def __exit__(self, *args):
        self._model.requires_grad_(requires_grad=False)