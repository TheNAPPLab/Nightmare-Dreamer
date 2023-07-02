import torch 
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal
import torch.distributions as distributions
from torch.distributions import constraints
import torch.nn.functional as F
from torch.distributions.transformed_distribution import TransformedDistribution

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
            expl_amount = self.train_noise
            expl_amount = expl_amount - itr/self.expl_decay
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
            return action

        raise NotImplementedError


# class ContinousActionModel(nn.Module):
#     def __init__(
#         self,
#         action_size,
#         deter_size,
#         stoch_size,
#         embedding_size,
#         actor_info,
#         expl_info,
#         min_std=1e-4,
#         init_std=5, 
#         mean_scale=5
#     ):
#         super().__init__()
#         self.action_size = action_size
#         self.deter_size = deter_size
#         self.stoch_size = stoch_size
#         self.embedding_size = embedding_size
#         self.layers = actor_info['layers']
#         self.node_size = actor_info['node_size']
#         self.act_fn = actor_info['activation']
#         self.dist = actor_info['dist']
#         self.act_fn = actor_info['activation']
#         self.train_noise = expl_info['train_noise']
#         self.eval_noise = expl_info['eval_noise']
#         self.expl_min = expl_info['expl_min']
#         self.expl_decay = expl_info['expl_decay']
#         self.expl_type = expl_info['expl_type']
       
#         self._min_std = min_std
#         self._init_std = init_std
#         self._mean_scale = mean_scale

#         self.model = self._build_model()

#     def _build_model(self):
#         model = [nn.Linear(self.deter_size + self.stoch_size, self.node_size)]
#         model += [self.act_fn()]
#         for _ in range(1, self.layers):
#             model += [nn.Linear(self.node_size, self.node_size)]
#             model += [self.act_fn()]

       
#         model += [nn.Linear(self.node_size, 2 * self.action_size)] # 2 heads std and mean
      
#         return nn.Sequential(*model) 

#     def forward(self, features, deter=False):

#         out = self.model(features)
#         mean, std = torch.chunk(out, 2, dim=-1) 

#         raw_init_std = np.log(np.exp(self._init_std)-1)
#         mean = self._mean_scale * torch.tanh(mean / self._mean_scale)
#         action_std = F.softplus(std + raw_init_std) + self._min_std

#         dist = distributions.Normal(mean, action_std)
       
#         dist = TransformedDistribution(dist, TanhBijector())
#         dist = distributions.independent.Independent(dist, 1)
#         dist = SampleDist(dist)

#         if deter:
#             return dist.mode(), dist
#         else:
#             return dist.rsample(), dist

#     def add_exploration(self, action, action_noise=0.3):

#         return torch.clamp(distributions.Normal(action, action_noise).rsample(), -1, 1)

        
class SafeTruncatedNormal(distributions.normal.Normal):

  def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
    super().__init__(loc, scale)
    self._low = low
    self._high = high
    self._clip = clip
    self._mult = mult

  def sample(self, sample_shape):
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

  def entropy(self):
    return self._dist.entropy()

  def mode(self):
    return self._dist.mean

  def sample(self, sample_shape=()):
    return self._dist.sample(sample_shape)

  def log_prob(self, x):
    return self._dist.log_prob(x)


class ContinousActionModel(nn.Module):
    def __init__(
        self,
        action_size,
        max_control,
        deter_size,
        stoch_size,
        embedding_size,
        actor_info,
        expl_info,
        min_std=1e-4,
        init_std=5, 
        mean_scale=5
    ):
        super().__init__()
        self.action_size = action_size
        self.deter_size = deter_size
        self.max_control  = max_control
        self.stoch_size = stoch_size
        self.embedding_size = embedding_size
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
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

        self.model = self._build_model()

    def _build_model(self):
        model = [nn.Linear(self.deter_size + self.stoch_size, self.node_size)]
        model += [self.act_fn()]
        for _ in range(1, self.layers):
            model += [nn.Linear(self.node_size, self.node_size)]
            model += [self.act_fn()]

       
        model += [nn.Linear(self.node_size, 2 * self.action_size)] # 2 heads std and mean
      
        return nn.Sequential(*model) 

    def forward(self, features, deter = False):

        out = self.model(features)
        if self.dist == 'trunc_normal':
            mean, std = torch.chunk(out, 2, dim=-1) 
            mean = torch.tanh(mean) #1 ,-1
            std = 2 * torch.sigmoid(std /2) + self._min_std
            dist = SafeTruncatedNormal(mean, std, -1, 1)
            dist = ContDist(distributions.independent.Independent(dist, 1))
            if deter: #not training
                cntrl =  torch.tensor(self.max_control).detach() * dist.mode()
                return cntrl, dist
            else:
                # cntrl = torch.tensor(self.max_control).detach() * dist.sample()
                cntrl = self.max_control * dist.sample()
                return cntrl , dist

        else:

            raw_init_std = np.log(np.exp(self._init_std)-1)
            mean = self._mean_scale * torch.tanh(mean / self._mean_scale)
            action_std = F.softplus(std + raw_init_std) + self._min_std

            dist = distributions.Normal(mean, action_std)
        
            dist = TransformedDistribution(dist, TanhBijector())
            dist = distributions.independent.Independent(dist, 1)
            dist = SampleDist(dist)

        if deter: #not training
                cntrl = self.max_control * dist.mode()
                return  cntrl, dist
        else:
                cntrl = self.max_control * dist.sample()
                return  cntrl, dist

    def add_exploration(self, action, action_noise=0.3):

        return torch.clamp(distributions.Normal(action, action_noise).rsample(), -1, 1)

        
class TanhBijector(distributions.Transform):

    def __init__(self):
        super().__init__()
        self.bijective = True
        self.domain = constraints.real
        self.codomain = constraints.interval(-1.0, 1.0)

    @property
    def sign(self): return 1.

    def _call(self, x): return torch.tanh(x)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def _inverse(self, y: torch.Tensor):
        y = torch.where(
            (torch.abs(y) <= 1.),
            torch.clamp(y, -0.99999997, 0.99999997),
            y)
        y = self.atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2. * (np.log(2) - x - F.softplus(-2. * x))


class SampleDist:

    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        sample = self._dist.rsample(self._samples)
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self._dist.sample()
