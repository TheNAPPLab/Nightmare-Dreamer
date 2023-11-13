import copy
import torch
from torch import nn
import numpy as np
from PIL import ImageColor, Image, ImageDraw, ImageFont
import torch.nn.functional as F
import torch.optim as optim
from torch import distributions as torchd
import ma_networksv3 as networks
import ma_toolsv3 as tools

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA(object):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.values = torch.zeros((2,)).to(device)
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        self.values = self.alpha * x_quantile + (1 - self.alpha) * self.values
        scale = torch.clip(self.values[1] - self.values[0], min=1.0)
        offset = self.values[0]
        return offset.detach(), scale.detach()
    

class CostEMA(object):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.values = torch.zeros((2,)).to(device)
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        self.values = self.alpha * x_quantile + (1 - self.alpha) * self.values
        scale = torch.clip(self.values[1] - self.values[0], min=1.0)
        offset = self.values[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_input_layers,
            config.dyn_output_layers,
            config.dyn_rec_depth,
            config.dyn_shared,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_temp_post,
            config.dyn_min_std,
            config.dyn_cell,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        if config.reward_head == "symlog_disc":
            self.heads["reward"] = networks.MLP(
                feat_size,  # pytorch version
                (255,),
                config.reward_layers,
                config.units,
                config.act,
                config.norm,
                dist=config.reward_head,
                outscale=0.0,
                device=config.device,
            )
        else:
            self.heads["reward"] = networks.MLP(
                feat_size,  # pytorch version
                [],
                config.reward_layers,
                config.units,
                config.act,
                config.norm,
                dist=config.reward_head,
                outscale=0.0,
                device=config.device,
            )
        #cost model
        if config.cost_head == "symlog_disc":
            self.heads["cost"] = networks.MLP(
                feat_size,  # pytorch version
                (255,),
                config.cost_layers,
                config.units,
                config.act,
                config.norm,
                dist=config.cost_head,
                outscale=0.0,
                device=config.device,
            )
        else:
            self.heads["reward"] = networks.MLP(
                feat_size,  # pytorch version
                [],
                config.cost_layers,
                config.units,
                config.act,
                config.norm,
                dist=config.cost_layers,
                outscale=0.0,
                device=config.device,
            )
        self.heads["cont"] = networks.MLP(
            feat_size,  # pytorch version
            [],
            config.cont_layers,
            config.units,
            config.act,
            config.norm,
            dist="binary",
            device=config.device,
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        self._scales = dict(reward=config.reward_scale, cost = config.cost_scale, cont=config.cont_scale)

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # cost (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    like = pred.log_prob(data[name])
                    losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)
                model_loss = sum(losses.values()) + kl_loss
            metrics = self._model_opt(model_loss, self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = obs.copy()
        obs["image"] = torch.Tensor(obs["image"]) / 255.0 - 0.5
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6] + 0.5
        model = model + 0.5
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)

class ImagBehavior(nn.Module):
    def __init__(self, config, world_model, stop_grad_actor=True, reward=None, cost=None):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        self._stop_grad_actor = stop_grad_actor
        self._reward = reward
        self._cost = cost
        self._initalise_safe_actor_buffer()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.ActionHead(
            feat_size,
            config.num_actions,
            config.actor_layers,
            config.units,
            config.act,
            config.norm,
            config.actor_dist,
            config.actor_init_std,
            config.actor_min_std,
            config.actor_max_std,
            config.actor_temp,
            outscale=1.0,
            unimix_ratio=config.action_unimix_ratio,
        )
        self.safe_actor = networks.ActionHead(
            feat_size,
            config.num_actions,
            config.actor_layers,
            config.units,
            config.act,
            config.norm,
            config.actor_dist,
            config.actor_init_std,
            config.actor_min_std,
            config.actor_max_std,
            config.actor_temp,
            outscale=1.0,
            unimix_ratio=config.action_unimix_ratio,
        )
        if config.value_head == "symlog_disc":
            self.value = networks.MLP(
                feat_size,
                (255,),
                config.value_layers,
                config.units,
                config.act,
                config.norm,
                config.value_head,
                outscale=0.0,
                device=config.device,
            )
        else:
            self.value = networks.MLP(
                feat_size,
                [],
                config.value_layers,
                config.units,
                config.act,
                config.norm,
                config.value_head,
                outscale=0.0,
                device=config.device,
            )
        if config.cost_value_head == "symlog_disc":
            self.cost_value = networks.MLP(
                feat_size,
                (255,),
                config.cost_value_layers,
                config.units,
                config.act,
                config.norm,
                config.cost_value_head,
                outscale=0.0,
                device=config.device,
            )
        else:
            self.cost_value = networks.MLP(
                feat_size,
                [],
                config.cost_value_layers,
                config.units,
                config.act,
                config.norm,
                config.cost_value_head,
                outscale=0.0,
                device=config.device,
            )
        if self._config.learn_discriminator:
            self.discriminator = networks.Discriminator(
                feat_size +  config.num_actions,
                [], config.discriminator_layers, config.discriminator_units)
        if config.slow_value_target:
            self._slow_value = copy.deepcopy(self.value)
            self._slow_cost_value = copy.deepcopy(self.cost_value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor_lr,
            config.ac_opt_eps,
            config.actor_grad_clip,
            **kw,
        )
        self._safe_actor_opt = tools.Optimizer(
            "safe_actor",
            self.safe_actor.parameters(),
            config.actor_lr,
            config.ac_opt_eps,
            config.actor_grad_clip,
            **kw,
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.value_lr,
            config.ac_opt_eps,
            config.value_grad_clip,
            **kw,
        )
        self._cost_value_opt = tools.Optimizer(
            "cost_value",
            self.value.parameters(),
            config.value_lr,
            config.ac_opt_eps,
            config.value_grad_clip,
            **kw,
        )
        if self._config.learn_discriminator:
            self._discriminator_opt = tools.Optimizer(
            'discriminator', self.discriminator.parameters(), config.discrimiator_lr, config.ac_opt_eps, config.discriminator_grad_clip,
            **kw
        )
        if self._config.reward_EMA:
            self.reward_ema = RewardEMA(device=self._config.device)
        if self._config.cost_EMA:
            self.cost_ema = CostEMA(device=self._config.device)
        self._declare_lagrnagian()
        self.cost_limit = self._config.init_cost_limit

    def _train(
        self,
        start,
        objective=None,
        constrain=None,
        action=None,
        reward=None,
        imagine=None,
        tape=None,
        repeats=None,
        mean_ep_cost = 0, 
        training_step = 0
    ):
        objective = objective or self._reward
        constrain = constrain or self._cost
        self._update_slow_target()
        metrics = {}
        

        mets_lag = self._update_lag(training_step, mean_ep_cost)
        if training_step > 80_000 and mean_ep_cost < self.cost_limit:
            self.cost_limit = max(12, self.cost_limit - 10)
        met_limit = {}
        met_limit["cost_limit"] = self.cost_limit

        metrics.update(mets_lag)
        metrics.update(met_limit)

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon, repeats
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled
                # slow is flag to indicate whether slow_target is used for lambda-return
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, imag_action, reward, actor_ent, state_ent
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_state,
                    imag_action,
                    target,
                    actor_ent,
                    state_ent,
                    weights,
                    base,
                )
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.slow_value_target:
                    value_loss = value_loss - value.log_prob(
                        slow_target.mode().detach()
                    )
                if self._config.value_decay:
                    value_loss += self._config.value_decay * value.mode()
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])
        
        if self._config.learn_safe_policy:
            with tools.RequiresGrad(self.safe_actor):
                with torch.cuda.amp.autocast(self._use_amp):
                    if not self._config.mpc:
                        safe_imag_feat, safe_imag_state, safe_imag_action = self._imagine(
                            start, self.safe_actor, self._config.imag_horizon, repeats
                        )
                        reward_safe = objective(imag_feat, imag_state, imag_action)
                        cost = constrain(safe_imag_feat, safe_imag_state, safe_imag_action)
                        safe_actor_ent = self.safe_actor(safe_imag_feat).entropy()
                        safe_state_ent = self._world_model.dynamics.get_dist(safe_imag_state).entropy()
                        # this target is not scaled
                        # slow is flag to indicate whether slow_target is used for lambda-return
                        target_cost, weights_safe, base_safe = self._compute_target_cost(
                            safe_imag_feat, safe_imag_state, safe_imag_action, cost, safe_actor_ent, safe_state_ent
                        )
                        safe_actor_loss, mets = self._compute_safe_actor_loss(
                            safe_imag_feat,
                            safe_imag_state,
                            safe_imag_action,
                            target_cost,
                            safe_actor_ent,
                            safe_state_ent,
                            weights_safe,
                            base_safe,
                        )
                        metrics.update(mets)
                        cost_value_input = safe_imag_feat
                    else:
                        safe_actor_loss, mets = self._compute_mpc_safe_actor_loss()

            if not self._config.mpc:
                with tools.RequiresGrad(self.cost_value):
                    with torch.cuda.amp.autocast(self._use_amp):
                        cost_value = self.cost_value(cost_value_input[:-1].detach())
                        target_cost = torch.stack(target_cost, dim=1)
                        # (time, batch, 1), (time, batch, 1) -> (time, batch)
                        cost_value_loss = -cost_value.log_prob(target_cost.detach())
                        slow_target_cost = self._slow_cost_value(cost_value_input[:-1].detach())
                        if self._config.slow_cost_value_target:
                            cost_value_loss = cost_value_loss - cost_value.log_prob(
                                slow_target_cost.mode().detach()
                            )
                        if self._config.cost_value_decay:
                            cost_value_loss += self._config.cost_value_decay * cost_value.mode()
                        # (time, batch, 1), (time, batch, 1) -> (1,)
                        cost_value_loss = torch.mean(weights_safe[:-1] * cost_value_loss[:, :, None])
        
            if self._config.learn_discriminator:
                with tools.RequiresGrad(self.discriminator):
                    with torch.cuda.amp.autocast(self._use_amp):
                        discrimiator_loss = self._compute_discrimiator_loss(safe_imag_action, safe_imag_feat,\
                                                    imag_action, imag_feat )
        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
       
        if self._config.learn_safe_policy and not self._config.mpc:
            metrics.update(tools.tensorstats(cost_value.mode(), "cost_value"))
            metrics.update(tools.tensorstats(target_cost, "target_cost"))
            metrics.update(tools.tensorstats(cost, "imag_cost"))
            metrics["safe_actor_entropy"] = to_np(torch.mean(safe_actor_ent))

        if self._config.actor_dist in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
            if self._config.learn_safe_policy:
                metrics.update(tools.tensorstats(safe_imag_action, "safe_imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
       
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))

            #safe actor parameters
            if self._config.learn_safe_policy:
                metrics.update(self._safe_actor_opt(safe_actor_loss, self.safe_actor.parameters()))
                if not self._config.mpc:
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

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")

        return feats, states, actions

    def _compute_target(
        self, imag_feat, imag_state, imag_action, reward, actor_ent, state_ent
    ):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        if self._config.future_entropy and self._config.actor_entropy > 0:
            reward += self._config.actor_entropy * actor_ent
        if self._config.future_entropy and self._config.actor_state_entropy > 0:
            reward += self._config.actor_state_entropy * state_ent
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]
    
    def _compute_target_cost(
        self, safe_imag_feat, safe_imag_state, safe_imag_action, cost, safe_actor_ent, safe_state_ent
    ):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(safe_imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(cost)
        if self._config.future_entropy and self._config.actor_entropy > 0:
            cost += self._config.safe_actor_entropy * safe_actor_ent
        if self._config.future_entropy and self._config.actor_state_entropy > 0:
            cost += self._config.safe_actor_state_entropy * safe_state_ent
        cost_value = self.cost_value(safe_imag_feat).mode()
        target_cost = tools.lambda_return(
            cost[1:],
            cost_value[:-1],
            discount[1:],
            bootstrap=cost_value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target_cost, weights, cost_value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_state,
        imag_action,
        target,
        actor_ent,
        state_ent,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
        policy = self.actor(inp)
        actor_ent = policy.entropy()
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            values = self.reward_ema.values
            metrics["EMA_005"] = to_np(values[0])
            metrics["EMA_095"] = to_np(values[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        if not self._config.future_entropy and self._config.actor_entropy > 0:
            actor_entropy = self._config.actor_entropy * actor_ent[:-1][:, :, None]
            actor_target += actor_entropy
        if not self._config.future_entropy and (self._config.actor_state_entropy > 0):
            state_entropy = self._config.actor_state_entropy * state_ent[:-1]
            actor_target += state_entropy
            metrics["actor_state_entropy"] = to_np(torch.mean(state_entropy))
        actor_loss = -torch.mean(weights[:-1] * actor_target)
        return actor_loss, metrics

    def _compute_safe_actor_loss(
        self,
        safe_imag_feat,
        safe_imag_state,
        safe_imag_action,
        target_cost,
        safe_actor_ent,
        safe_state_ent,
        weights,
        base,
    ):
        metrics = {}
        inp = safe_imag_feat.detach() if self._stop_grad_actor else safe_imag_feat
        safe_policy = self.safe_actor(inp)
        safe_actor_ent = safe_policy.entropy()
        # Q-val for actor is not transformed using symlog
        target_cost = torch.stack(target_cost, dim=1)
        if self._config.cost_EMA:
            offset, scale = self.cost_ema(target_cost)
            normed_target_cost = (target_cost - offset) / scale
            normed_base = (base - offset) / scale
            cost_adv = normed_target_cost - normed_base
            metrics.update(tools.tensorstats(normed_target_cost, "normed_target_cost"))
            cost_values = self.cost_ema.values
            metrics["Cost_EMA_005"] = to_np(cost_values[0])
            metrics["Cost_EMA_095"] = to_np(cost_values[1])
        safe_actor_target = 0 
        penalty =  self._lambda_range_projection(self._lagrangian_multiplier).item()
        if self._config.use_cost_adv:
            safe_actor_target = penalty * -cost_adv
        
        #safe_actor_target = penalty * -target_cost

        if self._config.behavior_cloning == 'discriminator':
            behavior_loss = self.discriminator(inp[:-1], safe_imag_action[:-1])
            scaled_behavior_loss = self._config.behavior_clone_scale  * behavior_loss
            safe_actor_target += scaled_behavior_loss
        elif self._config.behavior_cloning == 'kl1':
            behavior_loss = self._action_kl_loss(self.actor(inp[:-1]), self.safe_actor(inp[:-1]))
            scaled_behavior_loss = self._config.behavior_clone_scale * behavior_loss    
            safe_actor_target -= scaled_behavior_loss
  


        if not self._config.future_entropy and self._config.actor_entropy > 0:
            safe_actor_entropy = self._config.actor_entropy * safe_actor_ent[:-1][:, :, None]
            safe_actor_target += safe_actor_entropy
        if not self._config.future_entropy and (self._config.actor_state_entropy > 0):
            safe_state_entropy = self._config.actor_state_entropy * safe_state_ent[:-1]
            safe_actor_target += safe_state_entropy
            metrics["safe_actor_state_entropy"] = to_np(torch.mean(safe_state_entropy))

        if self._config.behavior_cloning != "":
            metrics.update(tools.tensorstats(behavior_loss, "behavior_loss"))
            metrics.update(tools.tensorstats(scaled_behavior_loss, "scaled_behavior_loss"))
        if penalty > 1.0:
            safe_actor_target /= penalty
        safe_actor_loss = -torch.mean(weights[:-1] * safe_actor_target)
        return safe_actor_loss, metrics

    def _update_slow_target(self):
        if self._config.slow_value_target:
            if self._updates % self._config.slow_target_update == 0:
                mix = self._config.slow_target_fraction
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data

                for s, d in zip(self.cost_value.parameters(), self._slow_cost_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1

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

    def _action_kl_loss(self, control_policy, safe_policy):
        kld = torchd.kl.kl_divergence
        control_dist = control_policy._dist
        safe_dist = safe_policy._dist
        kl_loss =  kld(control_dist, safe_dist ).unsqueeze(-1)
        return torch.max(kl_loss, torch.tensor(1, dtype=kl_loss.dtype, device=kl_loss.device))
    
    def _update_lag(self, training_step, mean_ep_cost, target_cost = None):
        metrics = {}
        metrics['lagrangian_multiplier'] = self._lagrangian_multiplier.detach().item()
        metrics['lagrangian_multiplier_projected'] = self._lambda_range_projection(self._lagrangian_multiplier).detach().item()
        # if training_step > self._config.start_cost_decay_step and training_step % self._config.cost_decay_freq == 0 and  mean_ep_cost < self.cost_limit:
        #     self.cost_limit = max(self._config.min_cost_budget, self.cost_limit-10)
        # metrics["cost_limit"] = self.cost_limit
        metrics['training_step'] = training_step
        self._update_lagrange_multiplier(mean_ep_cost,  self.cost_limit)
        return metrics
    
    def _update_lagrange_multiplier(self, ep_costs, cost_limit):
        self._lamda_optimizer.zero_grad()
        lambda_loss = self._compute_lamda_loss(ep_costs, cost_limit)
        lambda_loss.backward()
        self._lamda_optimizer.step()
        if self._config.lamda_projection == 'relu':
            self._lagrangian_multiplier.data.clamp_(self._config.min_lagrangian)  # enforce: lambda in [0, inf]
            self._lagrangian_multiplier.data.clamp_max_(self._config.max_lagrangian) #prevent explosion
       
    def _compute_lamda_loss(self, mean_cost, cost_limit):
        self._lagrangian_multiplier.requires_grad = True
        diff = mean_cost - cost_limit
        loss = -self._lagrangian_multiplier * diff
        return loss

    def _declare_lagrnagian(self):
        init_value = max(self._config.lagrangian_multiplier_init, 1e-5)
        self._lagrangian_multiplier = torch.nn.Parameter(
                torch.as_tensor( init_value ),
                requires_grad = True)
            
        if self._config.lamda_projection == 'relu':
            self._lambda_range_projection = torch.nn.ReLU()
        torch_opt = getattr(optim, 'Adam')
        self._lamda_optimizer = torch_opt([self._lagrangian_multiplier, ],
                                        lr = self._config.lambda_lr )

    def _compute_mpc_safe_actor_loss(self,):
        metrics = {}
        batch_size = 256 #256
        if self.curr_buffer_size >= batch_size:
            sample_indices = torch.randint(0, self.curr_buffer_size, (batch_size,))
            sampled_states = self.state_buffer[sample_indices]
            sampled_actions = self.action_buffer[sample_indices]
            safe_policy = self.safe_actor(sampled_states)

            safe_actor_ent = safe_policy.entropy()

            behavior_loss = -safe_policy.log_prob(sampled_actions)[:,:,None]
            actor_loss = behavior_loss

            actor_ent = self._config.safe_actor_entropy * safe_actor_ent[:,:,None]
            actor_loss -= actor_ent
            metrics.update(tools.tensorstats(behavior_loss, "behavior_loss"))
            metrics["safe_actor_entropy"] = to_np(torch.mean(actor_ent))
            return actor_loss, metrics




    @torch.no_grad()
    def _estimate_value(self, z, safe_action, model: WorldModel):
        reward_fn = lambda f, s, a: model.heads["reward"](f).mean()
        G, discount = 0, 1
        for t in range(self._config.horizon_mpc):
            feat = model.dynamics.get_feat(z)
            r_t = reward_fn(feat, None, None)
            z = model.dynamics.img_step(z, safe_action[t], sample = self._config.imag_sample)
            G += discount * r_t
            discount *= 0.99
        feat = model.dynamics.get_feat(z)
        V_t = self.value(feat).mode()
        return G + (discount * V_t)

    def _estimate_cost(self, z, safe_action, model: WorldModel):
        cost_fn = lambda f, s, a: model.heads["cost"](f).mean()
        total_cost = 0
        for t in range(self._config.horizon_mpc):
            feat = model.dynamics.get_feat(z)
            c_t = cost_fn(feat, None, None)
            z = model.dynamics.img_step(z, safe_action[t], sample = self._config.imag_sample)
            total_cost += c_t
        feat = model.dynamics.get_feat(z)
        return total_cost +  cost_fn(feat, None, None)
   
    @torch.no_grad()
    def get_safe_action(self, z, model : WorldModel, prev_mean=None ):
        state_ =  copy.deepcopy(model.dynamics.get_feat(z))
        # Sample policy trajectories
        if self._config.num_pi_trajs > 0:
            _z = {}
            safe_pi_action = torch.empty(self._config.horizon_mpc, self._config.num_pi_trajs, self._config.num_actions, device=self._config.device)
            _z['stoch'] = z['stoch'].repeat(self._config.num_pi_trajs, 1, 1 )
            _z['deter'] = z['deter'].repeat(self._config.num_pi_trajs, 1,)
            _z['logit'] = z['logit'].repeat(self._config.num_pi_trajs, 1,1 )

            for t in range(self._config.horizon_mpc-1):
                feat = model.dynamics.get_feat(_z)
                safe_pi_action[t] = self.safe_actor(feat).sample()
                _z = model.dynamics.img_step(_z, safe_pi_action[t], sample = self._config.imag_sample)

            feat = model.dynamics.get_feat(_z)
            safe_pi_action[-1] = self.safe_actor(feat).sample()
            
        #reinitalise z
        z['stoch'] = z['stoch'].repeat(self._config.num_samples, 1, 1 )
        z['deter'] = z['deter'].repeat(self._config.num_samples, 1)
        z['logit'] = z['logit'].repeat(self._config.num_samples, 1, 1 )
        mean = torch.zeros(self._config.horizon_mpc, self._config.num_actions, device=self._config.device)
        std = self._config.max_std*torch.ones(self._config.horizon_mpc, self._config.num_actions, device=self._config.device)
        actions = torch.empty(self._config.horizon_mpc, self._config.num_samples, self._config.num_actions, device=self._config.device)

        if prev_mean is not None: # not t0
            mean[:-1] = prev_mean[1:].cuda().float()

        if self._config.num_pi_trajs > 0:
            # actions[:, :self._config.num_pi_trajs] = safe_pi_action.view(self._config.imag_horizon,self._config.num_pi_trajs,-1)
            actions[:, :self._config.num_pi_trajs] = safe_pi_action

        # Iterate MPPI
        for i in range(self._config.mpc_iterations):

            # Sample actions
            actions[:, self._config.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * \
				torch.randn(self._config.horizon_mpc, self._config.num_samples-self._config.num_pi_trajs, self._config.num_actions, device=std.device)) \
				.clamp(-1, 1)

			# Compute elite actions
            value = self._estimate_value(z, actions, model).nan_to_num_(0)
            cost =  self._estimate_cost(z, actions, model).nan_to_num_(0)

            #prune cost rollouts that meet the cost requirement
            lowerCL_idxs = cost <= self._config.cost_threshold_mpc
            lowerCL_value, lowerCL_actions = value[lowerCL_idxs].view(-1, 1), actions[:, lowerCL_idxs.squeeze(), :]

            num_safe_candidates = torch.sum(lowerCL_idxs).item() 
            topk = self._config.num_elites if num_safe_candidates >= self._config.num_elites  else num_safe_candidates
            elite_idxs = torch.topk(lowerCL_value.squeeze(1), topk, dim=0).indices
            elite_value, elite_actions = lowerCL_value[elite_idxs], lowerCL_actions[:, elite_idxs]     

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self._config.temperature*(elite_value - max_value))
            score /= score.sum(0)
            mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)) \
                .clamp_(self._config.mpc_min_std, self._config.mpc_max_std)

        # Select action
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        action = actions[0].clamp_(-1, 1)
        self._add_buffer(state_, action)
        return action, mean

    def _initalise_safe_actor_buffer(self):
        self.buffer_size = 10000
        state_shape = (1, 1536)
        action_shape = (self._config.num_actions,)
        self.state_buffer = torch.empty((self.buffer_size,) + state_shape)
        self.action_buffer = torch.empty((self.buffer_size,) + action_shape)
        self.cursor = 0
        self.curr_buffer_size = 0
    
    def _add_buffer(self,state, action):
        self.action_buffer[self.cursor % self.buffer_size] = action
        self.state_buffer[self.cursor % self.buffer_size] = state
        self.curr_buffer_size += 1
        self.cursor += 1
        self.curr_buffer_size =  min(self.buffer_size, self.curr_buffer_size)
       