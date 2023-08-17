import torch
from torch import nn
import numpy as np
from collections import  deque
import torch.optim as optim
from torch import distributions as torchd
import ma_networks as networks
import ma_tools as tools
import torch.nn.functional as F
to_np = lambda x: x.detach().cpu().numpy()


class WorldModel(nn.Module):

  def __init__(self, step, config):
    super(WorldModel, self).__init__()
    self._step = step
    self._use_amp = True if config.precision==16 else False
    self._config = config
    self.encoder = networks.ConvEncoder(config.grayscale,
        config.cnn_depth, config.act, config.encoder_kernels)
    if config.size[0] == 64 and config.size[1] == 64:
      embed_size = 2 ** (len(config.encoder_kernels)-1) * config.cnn_depth
      embed_size *= 2 * 2
    else:
      raise NotImplemented(f"{config.size} is not applicable now")
    
    self.dynamics = networks.RSSM(
        config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
        config.dyn_input_layers, config.dyn_output_layers,
        config.dyn_rec_depth, config.dyn_shared, config.dyn_discrete,
        config.act, config.dyn_mean_act, config.dyn_std_act,
        config.dyn_temp_post, config.dyn_min_std, config.dyn_cell,
        config.num_actions, embed_size, config.device)
    
    self.heads = nn.ModuleDict()

    channels = (1 if config.grayscale else 3)
    shape = (channels,) + config.size

    if config.dyn_discrete:
      feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    else:
      feat_size = config.dyn_stoch + config.dyn_deter

    self.heads['image'] = networks.ConvDecoder(
        feat_size,  # pytorch version
        config.cnn_depth, config.act, shape, config.decoder_kernels,
        config.decoder_thin)
    self.heads['reward'] = networks.DenseHead(
        feat_size,  # pytorch version
        [], config.reward_layers, config.units, config.act)
    
    self.heads['cost'] = networks.DenseHead( #initalise model to learn cost
          feat_size,  # pytorch version
          [], config.cost_layers, config.units, config.act)
      
    if config.pred_discount:
      self.heads['discount'] = networks.DenseHead(
          feat_size,  # pytorch version
          [], config.discount_layers, config.units, config.act, dist='binary')
      
    for name in config.grad_heads:
      assert name in self.heads, name # check if imagination model, reward and cost are compulsorily intalised

    self._model_opt = tools.Optimizer(
        'model', self.parameters(), config.model_lr, config.opt_eps, config.grad_clip,
        config.weight_decay, opt = config.opt,
        use_amp = self._use_amp)
    
    
    #MOD
    self._scales = dict(
        reward = config.reward_scale, discount = config.discount_scale, cost = config.cost_scale)

  def _train(self, data):
    data = self.preprocess(data)

    with tools.RequiresGrad(self):
      with torch.cuda.amp.autocast(self._use_amp):
        embed = self.encoder(data)
        post, prior = self.dynamics.observe(embed, data['action'])
        kl_balance = tools.schedule(self._config.kl_balance, self._step)
        kl_free = tools.schedule(self._config.kl_free, self._step)
        kl_scale = tools.schedule(self._config.kl_scale, self._step)
        kl_loss, kl_value = self.dynamics.kl_loss(
            post, prior, self._config.kl_forward, kl_balance, kl_free, kl_scale)
        losses = {}
        likes = {}
        for name, head in self.heads.items():
          grad_head = (name in self._config.grad_heads)
          feat = self.dynamics.get_feat(post)
          feat = feat if grad_head else feat.detach()
          pred = head(feat)
          like = pred.log_prob(data[name])
          likes[name] = like
          # scales was applied here
          losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)
        model_loss = sum(losses.values()) + kl_loss

        # if self._config.learnable_lagrange:
        #   if self._config.update_lagrange_method == 0:
        #     self._update_lagrange_multiplier(torch.mean(torch.sum(data['cost'], dim = 1)))

        #   elif self._config.update_lagrange_method == 1:
        #     self._update_lagrange_multiplier(torch.max(torch.sum(data['cost'], dim = 1)))

        #lagrangian_loss = 
      metrics = self._model_opt(model_loss, self.parameters())

    metrics.update({f'{name}_loss': to_np(loss) for name, loss in losses.items()})
    metrics['kl_balance'] = kl_balance
    metrics['kl_free'] = kl_free
    metrics['kl_scale'] = kl_scale
    metrics['kl'] = to_np(torch.mean(kl_value))
    # if self._config.learnable_lagrange:
    #   metrics['lagrangian_multiplier'] = to_np(self._lagrangian_multiplier)
    with torch.cuda.amp.autocast(self._use_amp):
      metrics['prior_ent'] = to_np(torch.mean(self.dynamics.get_dist(prior).entropy()))
      metrics['post_ent'] = to_np(torch.mean(self.dynamics.get_dist(post).entropy()))
      context = dict(
          embed = embed, feat = self.dynamics.get_feat(post),
          kl = kl_value, postent = self.dynamics.get_dist(post).entropy())
    post = {k: v.detach() for k, v in post.items()}
    return post, context, metrics

  def preprocess(self, obs):
    # convert mdps to tensor and normalise image observation
    obs = obs.copy()
    obs['image'] = torch.Tensor(obs['image']) / 255.0 - 0.5

    if self._config.clip_rewards == 'tanh':
      obs['reward'] = torch.tanh(torch.Tensor(obs['reward'])).unsqueeze(-1)
    elif self._config.clip_rewards == 'identity':
      obs['reward'] = torch.Tensor(obs['reward']).unsqueeze(-1)
    else:
      raise NotImplemented(f'{self._config.clip_rewards} is not implemented')
    
    if self._config.clip_costs == 'tanh':
      obs['cost'] = torch.tanh(torch.Tensor(obs['cost'])).unsqueeze(-1)

    elif self._config.clip_costs == 'identity':
      obs['cost'] = torch.Tensor(obs['cost']).unsqueeze(-1)
    else:
      raise NotImplemented(f'{self._config.clip_costs} is not implemented')
    
    if 'discount' in obs:
      obs['discount'] *= self._config.discount
      obs['discount'] = torch.Tensor(obs['discount']).unsqueeze(-1)
    obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()} # convert obs to tensor
    return obs

  def video_pred(self, data):
    data = self.preprocess(data)
    truth = data['image'][:6] + 0.5
    embed = self.encoder(data)

    states, _ = self.dynamics.observe(embed[:6, :5], data['action'][:6, :5])
    recon = self.heads['image'](
        self.dynamics.get_feat(states)).mode()[:6]
    reward_post = self.heads['reward'](
        self.dynamics.get_feat(states)).mode()[:6]
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.dynamics.imagine(data['action'][:6, 5:], init)
    openl = self.heads['image'](
        self.dynamics.get_feat(prior)).mode()
    reward_prior = self.heads['reward'](
        self.dynamics.get_feat(prior)).mode()
    model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2

    return torch.cat([truth, model, error], 2)

class ImagBehavior(nn.Module):

  def __init__(self, config, world_model, stop_grad_actor=True, reward=None, cost = None):
    super(ImagBehavior, self).__init__()
    self._use_amp = True if config.precision==16 else False
    self._config = config
    self._world_model = world_model
    self._stop_grad_actor = stop_grad_actor
    self._reward = reward
    self._cost = cost
    self.cost_limit = self._config.limit_signal_prob if self._config.decay_cost else self._config.limit_signal_prob_decay_min
    if config.dyn_discrete:
      feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    else:
      feat_size = config.dyn_stoch + config.dyn_deter
    # intilaise actor, value network and target value network
    self.actor = networks.ActionHead(
        feat_size,  # pytorch version
        config.num_actions, config.actor_layers, config.units, config.act,
        config.actor_dist, config.actor_init_std, config.actor_min_std,
        config.actor_dist, config.actor_temp, config.actor_outscale)
    
    self.safe_actor = networks.ActionHead(
        feat_size,  # pytorch version
        config.num_actions, config.actor_layers, config.units, config.act,
        config.actor_dist, config.actor_init_std, config.actor_min_std,
        config.actor_dist, config.actor_temp, config.actor_outscale)
    
    self.value = networks.DenseHead(
        feat_size,  # pytorch version
        [], config.value_layers, config.units, config.act,
        config.value_head)
    

    self.cost_value = networks.DenseHead(
        feat_size,  # pytorch version
        [], config.value_layers, config.units, config.act,
        config.value_head)
    #discriminator network
    if self._config.learn_discriminator:
      self.discriminator = networks.Discriminator(
        feat_size +  config.num_actions,
        [], config.discriminator_layers, config.discriminator_units, config.act)
      
      self.discriminator_criterion = nn.BCELoss() # discriminator

    if config.slow_value_target or config.slow_actor_target:
      # target network
        self._slow_value = networks.DenseHead(
          feat_size,  # pytorch version
          [], config.value_layers, config.units, config.act)

        self._slow_cost_value = networks.DenseHead(
            feat_size,  # pytorch version
            [], config.value_layers, config.units, config.act)
        
        self._updates = 0

    kw = dict(wd = config.weight_decay, opt = config.opt, use_amp=self._use_amp) 
    # Actors Optimisers
    self._actor_opt = tools.Optimizer(
        'actor', self.actor.parameters(), config.actor_lr, config.opt_eps, config.actor_grad_clip,
        **kw)
    
    self._safe_actor_opt = tools.Optimizer(
        'safe_actor', self.safe_actor.parameters(), config.safe_actor_lr, config.opt_eps, config.actor_grad_clip,
        **kw)
    
    # Values Optimisers
    self._value_opt = tools.Optimizer(
        'value', self.value.parameters(), config.value_lr, config.opt_eps, config.value_grad_clip,
        **kw)
    
    self._cost_value_opt = tools.Optimizer(
          'cost_value', self.cost_value.parameters(), config.cost_value_lr, config.opt_eps, config.value_grad_clip,
          **kw)
    if self._config.learn_discriminator:
      self._discriminator_opt = tools.Optimizer(
        'discriminator', self.discriminator.parameters(), config.discrimiator_lr, config.opt_eps, config.discriminator_grad_clip,
          **kw
      )

    # lagrange parameters and pid and initalisation
    self._declare_lagrnagian()

  def _train(
        self, start, objective = None, constrain = None, action = None, \
        reward = None, cost = None, imagine = None, tape = None, repeats = None, mean_ep_cost = None, training_step = 0):
    objective = objective or self._reward

    #MOD
    constrain = constrain or self._cost

    self._update_slow_target()
    metrics = {}

    mets_lag = self._update_lag(training_step, mean_ep_cost)
    metrics.update(mets_lag)

    with tools.RequiresGrad(self.actor):
      with torch.cuda.amp.autocast(self._use_amp): #prcesion
        #imagination roll out
        imag_feat, imag_state, imag_action = self._imagine(
            start, self.actor, self._config.imag_horizon, repeats)

        reward = objective(imag_feat, imag_state, imag_action)

        actor_ent = self.actor(imag_feat).entropy()

        state_ent = self._world_model.dynamics.get_dist(
            imag_state).entropy()
        
        # Compute diffrent targets
        target, weights = self._compute_target(
            imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
            self._config.slow_actor_target)

        actor_loss, mets = self._compute_actor_loss(
            imag_feat, imag_state, imag_action, \
            target, actor_ent, state_ent, weights)

        metrics.update(mets)

        if self._config.slow_value_target != self._config.slow_actor_target:
          target, weights = self._compute_target(
              imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
              self._config.slow_value_target)
          
        value_input = imag_feat # inputs to value network

    with tools.RequiresGrad(self.value):
      with torch.cuda.amp.autocast(self._use_amp):
        value = self.value(value_input[:-1].detach())
        target = torch.stack(target, dim=1)
        value_loss = -value.log_prob(target.detach())
        if self._config.value_decay:
          value_loss += self._config.value_decay * value.mode()
        value_loss = torch.mean(weights[:-1] * value_loss[:,:,None])

    with tools.RequiresGrad(self.safe_actor):
      with torch.cuda.amp.autocast(self._use_amp): #prcesion

        safe_imag_feat, safe_imag_state, safe_imag_action = self._imagine(
              start, self.safe_actor, self._config.imag_horizon, repeats)
        #reward under safe policy
        reward_safe_policy = objective(safe_imag_feat, safe_imag_state, safe_imag_action)
        cost = constrain(safe_imag_feat, safe_imag_state, safe_imag_action)


        safe_actor_ent = self.safe_actor(safe_imag_feat).entropy()

        safe_state_ent = self._world_model.dynamics.get_dist(
            safe_imag_state).entropy()

        target_under_safe_policy, _ = self._compute_target(
            safe_imag_feat, safe_imag_state, safe_imag_action, reward_safe_policy, safe_actor_ent, safe_state_ent,
            self._config.slow_actor_target)
        
        target_cost = self._compute_target_cost(
                safe_imag_feat, safe_imag_state, safe_imag_action, cost, self._config.slow_actor_target)

        safe_actor_loss, mets = self._compute_safe_actor_loss( \
              safe_imag_feat, safe_imag_state, safe_imag_action, \
              target_cost, safe_actor_ent, safe_state_ent, weights,\
              imag_feat, imag_action, target_under_safe_policy)
        
        metrics.update(mets)
        safe_value_input = safe_imag_feat

    with tools.RequiresGrad(self.cost_value):
      with torch.cuda.amp.autocast(self._use_amp):
        cost_value = self.cost_value(safe_value_input[:-1].detach())
        target_cost = torch.stack(target_cost, dim=1)
        cost_value_loss = -cost_value.log_prob(target_cost.detach())
        # multi[ly by weights only if we wish to dsicount the value function
        cost_value_loss = torch.mean(weights[:-1] * cost_value_loss[:,:,None])
    if self._config.learn_discriminator:
      with tools.RequiresGrad(self.discriminator):
        with torch.cuda.amp.autocast(self._use_amp):
          discrimiator_loss = self._compute_discrimiator_loss(safe_imag_action, safe_imag_feat,\
                                    imag_action, imag_feat )
        

    metrics['reward_mean'] = to_np(torch.mean(reward))
    metrics['reward_std'] = to_np(torch.std(reward))

    #MOD
    metrics['cost_mean'] = to_np(torch.mean(cost))
    metrics['cost_std'] = to_np(torch.std(cost))


    metrics['actor_ent'] = to_np(torch.mean(actor_ent))
    metrics['safe_actor_ent'] = to_np(torch.mean(safe_actor_ent))
    metrics['mean_target'] = to_np(torch.mean(target.detach()))
    metrics['max_target'] = to_np(torch.max(target.detach()))
    metrics['std_target'] = to_np(torch.std(target.detach()))

    with tools.RequiresGrad(self):
      metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
      metrics.update(self._safe_actor_opt(safe_actor_loss, self.safe_actor.parameters()))
      metrics.update(self._value_opt(value_loss, self.value.parameters()))
      metrics.update(self._cost_value_opt(cost_value_loss, self.cost_value.parameters()))
      if self._config.learn_discriminator:
        metrics.update(self._discriminator_opt(discrimiator_loss, self.discriminator.parameters()))

    return imag_feat, imag_state, imag_action, weights, metrics

  def _imagine(self, start, policy, horizon, repeats=None):
    dynamics = self._world_model.dynamics
    if repeats:
      raise NotImplemented("repeats is not implemented in this version")
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}
    def step(prev, _):
      state, _, _ = prev
      feat = dynamics.get_feat(state)
      inp = feat.detach() if self._stop_grad_actor else feat
      action = policy(inp).sample()
      succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
      return succ, feat, action
    feat = 0 * dynamics.get_feat(start)
    action = policy(feat).mode()
    succ, feats, actions = tools.static_scan(
        step, [torch.arange(horizon)], (start, feat, action))
    states = {k: torch.cat([
        start[k][None], v[:-1]], 0) for k, v in succ.items()}
    if repeats:
      raise NotImplemented("repeats is not implemented in this version")

    return feats, states, actions

  def _compute_target(
      self, imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
      slow,):
    if 'discount' in self._world_model.heads:
      inp = self._world_model.dynamics.get_feat(imag_state)
      discount = self._world_model.heads['discount'](inp).mean
    else:
      discount = self._config.discount * torch.ones_like(reward)
    if self._config.future_entropy and self._config.actor_entropy() > 0:
      reward += self._config.actor_entropy() * actor_ent
    if self._config.future_entropy and self._config.actor_state_entropy() > 0:
      reward += self._config.actor_state_entropy() * state_ent
    if slow:
      value = self._slow_value(imag_feat).mode()
    else:
      value = self.value(imag_feat).mode()
    target = tools.lambda_return(
        reward[:-1], value[:-1], discount[:-1],
        bootstrap = value[-1], lambda_=self._config.discount_lambda, axis=0)
    weights = torch.cumprod(
        torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0).detach()
    return target, weights

  def _compute_target_cost(
      self, imag_feat, imag_state, imag_action, cost, slow):
    if 'discount' in self._world_model.heads:
      inp = self._world_model.dynamics.get_feat(imag_state)
      discount = self._world_model.heads['discount'](inp).mean
    else:
      discount = self._config.discount * torch.ones_like(cost)

    if slow:
      value_cost = self._slow_cost_value(imag_feat).mode()
    else:
      value_cost = self.cost_value(imag_feat).mode()

    target_cost = tools.lambda_return_cost(
        cost[:-1], value_cost[:-1], discount[:-1],
        bootstrap = value_cost[-1], lambda_ = self._config.discount_lambda, axis=0)
    # weights = torch.cumprod(
    #     torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0).detach()
    return target_cost

  def _compute_actor_loss(
      self, imag_feat, imag_state, imag_action, target, \
     actor_ent, state_ent, weights):
    metrics = {}
    inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
    policy = self.actor(inp)
    actor_ent = policy.entropy()
    target = torch.stack(target, dim=1)
    if self._config.imag_gradient == 'dynamics':
      actor_target = target

    elif self._config.imag_gradient == 'reinforce':
      actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (
          target - self.value(imag_feat[:-1]).mode()).detach()
      
    elif self._config.imag_gradient == 'both':
      actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (
          target - self.value(imag_feat[:-1]).mode()).detach()
      mix = self._config.imag_gradient_mix()
      actor_target = mix * target + (1 - mix) * actor_target
      metrics['imag_gradient_mix'] = mix

    else:
      raise NotImplementedError(self._config.imag_gradient)
    
    if not self._config.future_entropy and (self._config.actor_entropy() > 0):
      actor_target += self._config.actor_entropy() * actor_ent[:-1][:,:,None]
      
    if not self._config.future_entropy and (self._config.actor_state_entropy() > 0):
      actor_target += self._config.actor_state_entropy() * state_ent[:-1]


    actor_loss = -torch.mean(weights[:-1] * actor_target)
    return actor_loss, metrics

  def _compute_safe_actor_loss(
      self, safe_imag_feat, safe_imag_state, safe_imag_action,
      target_cost, safe_actor_ent, safe_state_ent, weights, 
      imag_feat, imag_action, target_under_safe_policy):
    metrics = {}
    inp = safe_imag_feat.detach() if self._stop_grad_actor else safe_imag_feat
    safe_policy = self.safe_actor(inp)
    target_cost =  torch.stack(target_cost, dim=1)
    target_under_safe_policy = torch.stack(target_under_safe_policy, dim = 1)
    safe_actor_target = 0
    penalty = 0
    if self._config.cost_imag_gradient == 'dynamics':
      penalty =  self._lambda_range_projection(self._lagrangian_multiplier).item() if self._config.learnable_lagrange else self._lagrangian_multiplier
      safe_actor_target += penalty *  target_cost

    elif self._config.cost_imag_gradient == 'reinforce':
      penalty =  self._lambda_range_projection(self._lagrangian_multiplier).item() if self._config.learnable_lagrange else self._lagrangian_multiplier
      safe_actor_target += penalty  * safe_policy.log_prob(safe_imag_action.detach())[:-1][:, :, None] * (
          target_cost - self.cost_value(safe_imag_feat[:-1]).mode()).detach()
      
    elif self._config.cost_imag_gradient == 'mix':
      reinforce = safe_policy.log_prob(safe_imag_action)[:-1][:, :, None] * (
          target_cost - self.cost_value(safe_imag_feat[:-1]).mode()).detach()
      
      mix = self._config.cost_imag_gradient_mix
      safe_actor_target += (1 - mix) * reinforce +  mix * target_cost

    #entropy term loss
    if self._config.cost_imag_gradient != "": #incase of pure cloninng
      if not self._config.future_entropy and (self._config.actor_entropy() > 0):
        safe_actor_target -= self._config.safe_actor_entropy * safe_actor_ent[:-1][:,:,None]
        
      if not self._config.future_entropy and (self._config.actor_state_entropy() > 0):
        safe_actor_target -= self._config.actor_state_entropy() * safe_state_ent[:-1]

    #behavior cloning loss
    if self._config.behavior_cloning == 'kl1':
      behavior_loss = self._action_kl_loss(self.actor(inp[:-1]), self.safe_actor(inp[:-1]))
      scaled_behavior_loss = self._config.actor_behavior_scale * behavior_loss
      safe_actor_target += scaled_behavior_loss

    elif self._config.behavior_cloning == 'kl2':
      '''
      inp is states from Safe policy
      inp_ is states from Control policy
      '''
      behavior_loss1 = self._action_kl_loss(self.actor(inp[:-1]), self.safe_actor(inp[:-1]))
      inp_ = imag_feat.detach() if self._stop_grad_actor else imag_feat
      behavior_loss2 = self._action_kl_loss(self.actor(inp_[:-1]), self.safe_actor(inp_[:-1])) # use control states
      behavior_loss = (behavior_loss1 + behavior_loss2) / 2
      scaled_behavior_loss = self._config.actor_behavior_scale * behavior_loss
      safe_actor_target += scaled_behavior_loss

    elif self._config.behavior_cloning == 'log_prob':
      inp_ = imag_feat.detach() if self._stop_grad_actor else imag_feat
      action_inp_ = imag_action.detach() if self._stop_grad_actor else imag_action
      safe_policy_ = self.safe_actor(inp_) # safe policy under control state
      behavior_loss =  -safe_policy_.log_prob(action_inp_)[:-1][:, :, None]
      if self._config.clamp_behavior_loss:
        behavior_loss = torch.clamp(behavior_loss, min = self._config.min_behavior_loss)
      scaled_behavior_loss = self._config.actor_behavior_scale * behavior_loss
      safe_actor_target += scaled_behavior_loss

    elif self._config.behavior_cloning == 'discriminator':
      behavior_loss = -self.discriminator(inp[:-1], safe_imag_action[:-1])
      scaled_behavior_loss = self._config.actor_behavior_scale  * behavior_loss
      safe_actor_target += scaled_behavior_loss

    elif self._config.behavior_cloning == 'discriminator_log':
      discriminator_predictions = self.discriminator(inp[:-1], safe_imag_action[:-1])
      output_shape = (safe_imag_action.shape[0]-1, safe_imag_action.shape[1], 1)
      control_labels = torch.ones(output_shape, device=self._config.device)
      behavior_loss = -F.binary_cross_entropy_with_logits(discriminator_predictions, control_labels)
      scaled_behavior_loss = self._config.actor_behavior_scale  * behavior_loss
      safe_actor_target += scaled_behavior_loss

    elif self._config.behavior_cloning == 'mse':
      cntrl_actions = imag_action.detach()[:-1]
      safe_actions = self.safe_actor(imag_feat.detach()).sample()[:-1] #safe actions given  states from control policy
      squared_diff = (safe_actions - cntrl_actions)**2
      behavior_loss = torch.mean(squared_diff, dim = 2)[:,:,None]
      scaled_behavior_loss = self._config.actor_behavior_scale  * behavior_loss
      safe_actor_target += scaled_behavior_loss

    safe_actor_target -= self._config.alpha1 * target_under_safe_policy

    if penalty > 1.0:
      safe_actor_target /= penalty

    if self._config.behavior_cloning != '':
      metrics['behavior_cloning_loss_mean'] = to_np(torch.mean(behavior_loss))
      metrics['behavior_cloning__loss_max'] = to_np(torch.max(behavior_loss))
      metrics['scaled_behavior_cloning_loss_mean'] = to_np(torch.mean(scaled_behavior_loss))
    else:
      metrics['behavior_cloning_loss_mean'] = 0
      metrics['behavior_cloning__loss_max'] = 0
      metrics['scaled_behavior_cloning_loss_mean'] = 0

    metrics['mean_target_under_safe_policy'] = to_np(torch.mean(target_under_safe_policy))
    metrics['max_target_under_safe_policy'] = to_np(torch.max(target_under_safe_policy))
    metrics['mean_target_cost'] = to_np(torch.mean(target_cost.detach()))
    metrics['max_target_cost'] = to_np(torch.max(target_cost.detach()))
    safe_actor_loss = torch.mean(weights[:-1] * safe_actor_target)
    return safe_actor_loss, metrics

  def _update_slow_target(self):
    if self._config.slow_value_target or self._config.slow_actor_target:
      if self._updates % self._config.slow_target_update == 0:
        mix = self._config.slow_target_fraction
        for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
          d.data = mix * s.data + (1 - mix) * d.data
      #mMOD
      if self._config.solve_cmdp:
        if self._updates % self._config.slow_target_update == 0:
          mix = self._config.slow_target_fraction
          for s, d in zip(self.cost_value.parameters(), self._slow_cost_value.parameters()):
            d.data = mix * s.data + (1 - mix) * d.data
      self._updates += 1

  def _action_kl_loss(self, control_policy, safe_policy):
    kld = torchd.kl.kl_divergence
    control_dist = control_policy._dist
    safe_dist = safe_policy._dist
   
    # value = kld(dist(lhs) if self._discrete else dist(lhs)._dist,
    #               dist(rhs) if self._discrete else dist(rhs)._dist)
    
    kl_value = kld(control_dist, safe_dist ).unsqueeze(-1)
    # loss = torch.mean(torch.maximum(kl_value, free))
    # loss = torch.mean(kl_value)
    return kl_value
  
  def _compute_lamda_loss(self, mean_cost, cost_limit):
    self._lagrangian_multiplier.requires_grad = True
    diff = mean_cost - cost_limit
    loss = -self._lagrangian_multiplier * diff
    return loss
  
  def _update_lagrange_multiplier(self, ep_costs, cost_limit):
    self._lamda_optimizer.zero_grad()
    lambda_loss = self._compute_lamda_loss(ep_costs, cost_limit)
    lambda_loss.backward()
    self._lamda_optimizer.step()
    if self._config.lamda_projection == 'relu':
      self._lagrangian_multiplier.data.clamp_(self._config.min_lagrangian)  # enforce: lambda in [0, inf]
      self._lagrangian_multiplier.data.clamp_max_(self._config.max_lagrangian) #prevent explosion
    else:
      self._lagrangian_multiplier.data.clamp_max_(self._config.max_lagrangian)

  def _pid_update(self, ep_cost_avg):
        metrics = {}
        delta = float(ep_cost_avg - self.cost_limit)  # ep_cost_avg: tensor
        self.pid_i = max(0., self.pid_i + delta * self.pid_Ki)
        if self.diff_norm:
            self.pid_i = max(0., min(1., self.pid_i))
        a_p = self.pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        a_d = self.pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(ep_cost_avg)
        pid_d = max(0., self._cost_d - self.cost_ds[0])
        pid_o = (self.pid_Kp * self._delta_p + self.pid_i +
                 self.pid_Kd * pid_d)
        self._lagrangian_multiplier = max(self._config.min_lagrangian, min(self._config.max_lagrangian, pid_o))
        if self.diff_norm:
            self._lagrangian_multiplier = min(1., self.cost_penalty)
        if not (self.diff_norm or self.sum_norm):
            self._lagrangian_multiplier = min(self.cost_penalty, self.penalty_max)
        self.cost_ds.append(self._cost_d)
        metrics['Proortional_term'] = self.pid_Kp * self._delta_p
        metrics['integral']  = self.pid_i
        metrics['derivative'] = self.pid_Kd * pid_d
        return metrics


  def _declare_lag_params(self):
    #max lag is self._config.max_lagrangian 0.75
    self.pid_d_delay = self._config.pid_d_delay
    self.cost_ds = deque(maxlen=self.pid_d_delay)
    self.cost_ds.append(0)
    self.pid_Kp = self._config.pid_Kp 
    self.pid_Ki = self._config.pid_Ki 
    self.pid_Kd = self._config.pid_Kd 
    self.pid_delta_p_ema_alpha = self._config.pid_delta_p_ema_alpha # 0 for hard update, 1 for no update
    self.pid_delta_d_ema_alpha = self._config.pid_delta_d_ema_alpha 
    self.sum_norm = True  # L = (J_r - lam * J_c) / (1 + lam); lam <= 0
    self.diff_norm = False  # L = (1 - lam) * J_r - lam * J_c; 0 <= lam <= 1
    self.pid_i = self._lagrangian_multiplier
    self._delta_p = 0
    self._cost_d = 0

  def _cost_limit(self, step):
    #  limit_signal_prob_decay_min:  12
    if not self._config.decay_cost:
      return self._config.limit_signal_prob_decay_min
    if step <= self._config.limit_decay_start:
        expl_amount = self._config.limit_signal_prob
    else:
        expl_amount =  self._config.limit_signal_prob
        ir = step  - self._config.limit_decay_start + 1
        expl_amount = expl_amount - ir/self._config.limit_signal_prob_decay
        expl_amount = max(self._config.limit_signal_prob_decay_min, expl_amount)
    return expl_amount

  def _update_lag(self, training_step, mean_ep_cost, target_cost = None):
    metrics = {}
    if not self._config.use_pid:
      metrics['lagrangian_multiplier'] = self._lagrangian_multiplier.detach().item() if self._config.learnable_lagrange else self._lagrangian_multiplier

      metrics['lagrangian_multiplier_projected'] = self._lambda_range_projection(self._lagrangian_multiplier).detach().item() if self._config.learnable_lagrange else self._lagrangian_multiplier
    # if training_step > self._config.limit_decay_start and training_step % 20_000 == 0 and (abs(self.cost_limit - mean_ep_cost) == 5 or mean_ep_cost <= self.cost_limit):
      if training_step > self._config.limit_decay_start and training_step % self._config.limit_decay_freq == 0 and  mean_ep_cost < self.cost_limit:
        # self.cost_limit = self._cost_limit(training_step)
        self.cost_limit = max(self._config.limit_signal_prob_decay_min, self.cost_limit-10)

      metrics["cost_limit"] = self.cost_limit

      if self._config.learnable_lagrange:
        if self._config.update_lagrange_metric == 'target_mean':
          self._update_lagrange_multiplier(torch.mean(target_cost.detach()),  self.cost_limit)
        elif self._config.update_lagrange_metric == 'target_max':
          self._update_lagrange_multiplier(torch.max(target_cost.detach()),  self.cost_limit)
        elif self._config.update_lagrange_metric == 'mean_ep_cost':
          self._update_lagrange_multiplier(mean_ep_cost,  self.cost_limit)
    else:
     
      metrics.update(self._pid_update( mean_ep_cost ))
      metrics['lagrangian_multiplier'] = self._lagrangian_multiplier
      metrics["cost_limit"] = self.cost_limit
    return metrics

  def _compute_discrimiator_loss(self, safe_actions, states_under_safe_policy,\
                                  control_action, states_under_control_policy ):
    statesGenerated_under_safe_policy = states_under_safe_policy.detach()
    safe_actions = safe_actions.detach()
    statesGenerated_under_control_policy = states_under_control_policy.detach()
    control_action = control_action.detach()

    safe_actions_under_GenratedControl_states =  self.safe_actor(statesGenerated_under_control_policy).sample().detach()

    control_actions_under_GenratedSafe_states =  self.actor(statesGenerated_under_safe_policy).sample().detach()

    pred_safe1 = self.discriminator(statesGenerated_under_safe_policy, safe_actions)
    pred_safe2 = self.discriminator(statesGenerated_under_control_policy, safe_actions_under_GenratedControl_states)

    pred_control1 = self.discriminator(statesGenerated_under_control_policy, control_action)
    pred_control2 = self.discriminator(statesGenerated_under_safe_policy, control_actions_under_GenratedSafe_states)

    output_shape = (safe_actions.shape[0], safe_actions.shape[1], 1)
    control_labels = torch.ones(output_shape, device=self._config.device)
    safe_labels = torch.zeros(output_shape, device=self._config.device)

    control_loss_pred = F.binary_cross_entropy_with_logits(pred_control1, control_labels) + F.binary_cross_entropy_with_logits(pred_control2, control_labels) 
    safe_loss_pred = F.binary_cross_entropy_with_logits(pred_safe1, safe_labels ) + F.binary_cross_entropy_with_logits(pred_safe2, safe_labels )

    loss = (control_loss_pred + safe_loss_pred)/4
    return loss

  def _declare_lagrnagian(self):
    if not self._config.use_pid:
      init_value = max(self._config.lagrangian_multiplier_init, 1e-5)


      self._lagrangian_multiplier = torch.nn.Parameter(
              torch.as_tensor( init_value ),
              requires_grad = True) if self._config.learnable_lagrange else self._config.lagrangian_multiplier_fixed
        
      if self._config.lamda_projection == 'relu':
          self._lambda_range_projection = torch.nn.ReLU()
      elif self._config.lamda_projection == 'sigmoid':
          self._lambda_range_projection = torch.nn.Sigmoid()
      elif self._config.lamda_projection == 'stretched_sigmoid':
          self._lambda_range_projection = tools.StretchedSigmoid(self._config.sigmoid_a)

      torch_opt = getattr(optim, 'Adam')
      self._lamda_optimizer = torch_opt([self._lagrangian_multiplier, ],
                                    lr = self._config.lambda_lr )  if self._config.learnable_lagrange else None
    else:
      init_value = min(self._config.lagrangian_multiplier_init, 1e-5)
      self._lagrangian_multiplier = init_value
      self._declare_lag_params()