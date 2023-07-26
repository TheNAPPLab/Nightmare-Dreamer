import torch
from torch import nn
import numpy as np
from PIL import ImageColor, Image, ImageDraw, ImageFont
import torch.optim as optim

import networks
import cmdp_tools as tools
to_np = lambda x: x.detach().cpu().numpy()


def target_ratio(b, max_target = 99.3429516957585 , max_cost = 1000):
  return b * max_target / max_cost

class WorldModel(nn.Module):

  def __init__(self, step, config):
    super(WorldModel, self).__init__()
    self._step = step
    self._cost_limit = config.cost_limit
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
    metrics['cost_limit'] = self._cost_limit
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

  # def _compute_lamda_loss(self, mean_ep_cost):
  #   if self._config.update_lagrange_method == 4:
  #     diff =  mean_ep_cost -  target_ratio(self._cost_limit)
  #   else:
  #     diff = mean_ep_cost - self._cost_limit
  #   return -self._lagrangian_multiplier *  diff

  # def _update_lagrange_multiplier(self, ep_costs):
  #       """ Update Lagrange multiplier (lambda)
  #           Note: ep_costs obtained from: self.logger.get_stats('EpCosts')[0]
  #           are already averaged across MPI processes.
  #       """
  #       self._lamda_optimizer.zero_grad()
  #       lambda_loss = self._compute_lamda_loss(ep_costs)
  #       lambda_loss.backward()
  #       self._lamda_optimizer.step()
  #       self._lagrangian_multiplier.data.clamp_(0)  # enforce: lambda in [0, inf]

class ImagBehavior(nn.Module):

  def __init__(self, config, world_model, stop_grad_actor=True, reward=None, cost = None):
    super(ImagBehavior, self).__init__()
    self._use_amp = True if config.precision==16 else False
    self._config = config
    self._world_model = world_model
    self._stop_grad_actor = stop_grad_actor
    self._reward = reward
    self._cost = cost
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
    
    self.value = networks.DenseHead(
        feat_size,  # pytorch version
        [], config.value_layers, config.units, config.act,
        config.value_head)
    
    if config.solve_cmdp:
      self.cost_value = networks.DenseHead(
        feat_size,  # pytorch version
        [], config.value_layers, config.units, config.act,
        config.value_head)
      

    if config.slow_value_target or config.slow_actor_target:
      # target network
      self._slow_value = networks.DenseHead(
          feat_size,  # pytorch version
          [], config.value_layers, config.units, config.act)
      #MOD
      #Cost Target network
      if config.solve_cmdp:
        self._slow_cost_value = networks.DenseHead(
            feat_size,  # pytorch version
            [], config.value_layers, config.units, config.act)
      self._updates = 0

    kw = dict(wd = config.weight_decay, opt = config.opt, use_amp=self._use_amp) #?
    self._actor_opt = tools.Optimizer(
        'actor', self.actor.parameters(), config.actor_lr, config.opt_eps, config.actor_grad_clip,
        **kw)
    self._value_opt = tools.Optimizer(
        'value', self.value.parameters(), config.value_lr, config.opt_eps, config.value_grad_clip,
        **kw)
    #MOD
    if config.solve_cmdp:
      self._cost_value_opt = tools.Optimizer(
          'cost_value', self.cost_value.parameters(), config.cost_value_lr, config.opt_eps, config.value_grad_clip,
          **kw)
          #MOD
    if self._config.solve_cmdp:
      # new Models and paramters include lagrange multiplier and Cost model 
      
      init_value = max(self._config.lagrangian_multiplier_init, 1e-5)

      self._lagrangian_multiplier = torch.nn.Parameter(
            torch.as_tensor( init_value ),
            requires_grad = True) if config.learnable_lagrange else self._config.lagrangian_multiplier_fixed
      
      if config.lamda_projection == 'relu':
        self._lambda_range_projection = torch.nn.ReLU()
      elif config.lamda_projection == 'sigmoid':
        self._lambda_range_projection = torch.nn.Sigmoid()
      elif config.lamda_projection == 'stretched_sigmoid':
        self._lambda_range_projection = tools.StretchedSigmoid(self._config.sigmoid_a)


      #MOD
      torch_opt = getattr(optim, 'Adam')
      self._lamda_optimizer = torch_opt([self._lagrangian_multiplier, ],
                                  lr = config.lambda_lr )  if config.learnable_lagrange else None
  
  def _train(
        self, start, objective = None, constrain = None, action = None, \
        reward = None, cost = None, imagine = None, tape = None, repeats = None):
    objective = objective or self._reward

    #MOD
    constrain = constrain or self._cost
    get_value_or_none = lambda x: x if self._config.solve_cmdp else None

    self._update_slow_target()
    metrics = {}

    with tools.RequiresGrad(self.actor):
      with torch.cuda.amp.autocast(self._use_amp): #prcesion
        #imagination roll out
        imag_feat, imag_state, imag_action = self._imagine(
            start, self.actor, self._config.imag_horizon, repeats)
        
        reward = objective(imag_feat, imag_state, imag_action)

        cost = constrain(imag_feat, imag_state, imag_action)

        actor_ent = self.actor(imag_feat).entropy()

        state_ent = self._world_model.dynamics.get_dist(
            imag_state).entropy()
        
        target, weights = self._compute_target(
            imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
            self._config.slow_actor_target)
        
        #MOD
        if self._config.solve_cmdp:
          target_cost = self._compute_target_cost(
              imag_feat, imag_state, imag_action, cost, actor_ent, state_ent,
              self._config.slow_actor_target)
          z = self._compute_z(cost, weights)

        actor_loss, mets = self._compute_actor_loss(
            imag_feat = imag_feat, imag_state = imag_state, imag_action = imag_action, \
            target = target, target_cost = target_cost if self._config.solve_cmdp  else None ,\
            actor_ent = actor_ent, state_ent = state_ent, weights = weights, z = z if self._config.solve_cmdp else None )
        
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

    #MOD
    if self._config.solve_cmdp:
      with tools.RequiresGrad(self.cost_value):
        with torch.cuda.amp.autocast(self._use_amp):
          cost_value = self.cost_value(value_input[:-1].detach())
          target_cost = torch.stack(target_cost, dim=1)
          cost_value_loss = -cost_value.log_prob(target_cost.detach())
          # multi[ly by weights only if we wish to dsicount the value function
          cost_value_loss = torch.mean(weights[:-1] * cost_value_loss[:,:,None])

    metrics['reward_mean'] = to_np(torch.mean(reward))
    metrics['reward_std'] = to_np(torch.std(reward))

    #MOD
    metrics['cost_mean'] = to_np(torch.mean(cost))
    metrics['cost_std'] = to_np(torch.std(cost))


    metrics['actor_ent'] = to_np(torch.mean(actor_ent))
    metrics['mean_target'] = to_np(torch.mean(target.detach()))
    metrics['max_target'] = to_np(torch.max(target.detach()))
    metrics['std_target'] = to_np(torch.std(target.detach()))

    with tools.RequiresGrad(self):
      metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
      metrics.update(self._value_opt(value_loss, self.value.parameters()))

      #MOD
      if self._config.solve_cmdp:
        metrics.update(self._cost_value_opt(cost_value_loss, self.cost_value.parameters()))

    #MOD
    if self._config.solve_cmdp:
      metrics['mean_target_cost'] = to_np(torch.mean(target_cost.detach()))
      metrics['max_target_cost'] = to_np(torch.max(target_cost.detach()))
      metrics['std_target_cost'] = to_np(torch.std(target_cost.detach()))
      metrics['min_target_cost'] = to_np(torch.min(target_cost.detach()))

    #log lagranian multipler
    if self._config.solve_cmdp and self._config.learnable_lagrange :
      metrics['lagrangian_multiplier'] = self._lagrangian_multiplier.item() if self._config.learnable_lagrange else self._lagrangian_multiplier

    #update lagrane multiplier if needed
    if self._config.solve_cmdp and self._config.learnable_lagrange:

      if self._config.update_lagrange_metric == 'target_mean':
        self._update_lagrange_multiplier(torch.mean(target_cost.detach()))
      elif self._config.update_lagrange_metric == 'target_max':
        self._update_lagrange_multiplier(torch.max(target_cost.detach()))


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
      self, imag_feat, imag_state, imag_action, cost, actor_ent, state_ent,
      slow,):
    if 'discount' in self._world_model.heads:
      inp = self._world_model.dynamics.get_feat(imag_state)
      discount = self._world_model.heads['discount'](inp).mean
    else:
      discount = self._config.discount * torch.ones_like(cost)
    if self._config.future_entropy and self._config.actor_entropy() > 0:
      cost += self._config.actor_entropy() * actor_ent
    if self._config.future_entropy and self._config.actor_state_entropy() > 0:
      cost += self._config.actor_state_entropy() * state_ent
    if slow:
      value_cost = self._slow_cost_value(imag_feat).mode()
    else:
      value_cost = self.cost_value(imag_feat).mode()

    target = tools.lambda_return(
        cost[:-1], value_cost[:-1], discount[:-1],
        bootstrap = value_cost[-1], lambda_=self._config.discount_lambda, axis=0)
    # weights = torch.cumprod(
    #     torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0).detach()
    return target

  # def _compute_target_cost(
  #     self, imag_feat, imag_state, imag_action, cost, actor_ent, state_ent,
  #     slow):
  #   if 'discount' in self._world_model.heads:
  #     inp = self._world_model.dynamics.get_feat(imag_state)
  #     discount = self._world_model.heads['discount'](inp).mean
  #   else:
  #     discount = self._config.discount * torch.ones_like(cost)

  #   if self._config.perturb_cost_entropy:
  #     if self._config.future_entropy and self._config.actor_entropy() > 0:
  #       cost += self._config.actor_entropy() * actor_ent

  #     if self._config.future_entropy and self._config.actor_state_entropy() > 0:
  #       cost += self._config.actor_state_entropy() * state_ent
    
  #   if slow:
  #     value_cost = self._slow_cost_value(imag_feat).mode()
  #   else:
  #     value_cost = self.cost_value(imag_feat).mode()

  #   target = tools.lambda_return_cost(
  #       cost[:-1], value_cost[:-1], discount[:-1],
  #       bootstrap = value_cost[-1], lambda_ = self._config.discount_lambda, axis = 0, 
  #       discount_value_cost = self._config.discount_value_cost)

    return target

  def _compute_actor_loss(
      self, imag_feat, imag_state, imag_action, target, \
      target_cost, actor_ent, state_ent, weights, z):
    metrics = {}
    inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
    policy = self.actor(inp)
    actor_ent = policy.entropy()
    target = torch.stack(target, dim=1)
    target_cost =  torch.stack(target_cost, dim=1) if target_cost else None
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

    if self._config.solve_cmdp:
      # Add a loss tp the objectiv
      penalty =  self._lambda_range_projection(self._lagrangian_multiplier).item() if self._config.learnable_lagrange else self._lagrangian_multiplier
      # penalty = 0.01
      if self._config.cost_imag_gradient =='dynamics':
        # cost_loss_term = penalty  * ( target_cost -  target_ratio(self._config.cost_limit) ) if self._config.reduce_target_cost else penalty * target_cost
        # cost_loss_term = min(penalty, 0.8) * target_cost
        cost_loss_term = penalty * target_cost
        actor_target -= cost_loss_term    # term will be negated and be an addition to the cost, so high target_cost means a higher actor loss
        # if penalty > 1.0:
        #   actor_target /= penalty

      elif self._config.cost_imag_gradient =='reinforce':
        cost_loss_term = policy.log_prob(imag_action)[:-1][:, :, None] * ( \
            target_cost - self.cost_value(imag_feat[:-1]).mode()).detach()
        cost_loss_term = penalty * cost_loss_term
        actor_target -= cost_loss_term    # term will be negated and be an addition to the cost, so high target_cost means a higher actor loss
        # actor_target /= penalty

      elif self._config.cost_imag_gradient =='z':
        c_tensor = torch.tensor(self._config.c, device=self._config.device)
        zero_tensor = torch.tensor(0.0, device = self._config.device)
        c = torch.where(z < 0, c_tensor, zero_tensor)
        actor_target -= c

      elif self._config.cost_imag_gradient =='z':
        c_tensor = torch.tensor(self._config.c, device=self._config.device)
        zero_tensor = torch.tensor(0.0, device = self._config.device)
        c = torch.where(z < 0, c_tensor, zero_tensor)
        actor_target -= c
    actor_loss = -torch.mean(weights[:-1] * actor_target)
    return actor_loss, metrics

  def _compute_z(self, cost, weights):
    cost_ = cost[:-1].detach()
    weights = weights[:-1]
    d = self._config.cost_limit - cost_.cumsum(dim=0)
    sum_of_discounted_costs = (weights * cost_).cumsum(dim=0)
    Z = d - sum_of_discounted_costs
    return Z
    # Z_t = torch.zeros_like(d)
    # H, B, _ = cost_.shape

    # for t in range(H):
    #   # sum_of_discounted_costs += (weights ** t) * cost_[t]
    #   sum_of_discounted_costs += weights * cost_[t]
    # for t in range(H):
    # # denominator = (gamma ** t * d_t[t])
    # # denominator += 1e-8
    #   Z_t[t] = (d[t] - sum_of_discounted_costs)


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

  def _compute_lamda_loss(self, mean_ep_cost):
    # diff =  mean_ep_cost -  target_ratio(self._config.cost_limit)
    diff = mean_ep_cost - self._config.cost_limit
    self._lagrangian_multiplier.requires_grad = True
    loss = -self._lagrangian_multiplier * diff
    return loss

  def _update_lagrange_multiplier(self, ep_costs):
        """ Update Lagrange multiplier (lambda)
            Note: ep_costs obtained from: self.logger.get_stats('EpCosts')[0]
            are already averaged across MPI processes.
        """
        self._lamda_optimizer.zero_grad()
        lambda_loss = self._compute_lamda_loss(ep_costs)
        lambda_loss.backward()
        self._lamda_optimizer.step()
        if self._config.lamda_projection != 'sigmoid':
          self._lagrangian_multiplier.data.clamp_(0)  # enforce: lambda in [0, inf]

 


