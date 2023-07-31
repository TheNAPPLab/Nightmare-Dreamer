import argparse
import collections
import functools
import os
import pathlib
import sys
import warnings
import wandb
if sys.platform == 'linux':
  os.environ['MUJOCO_GL'] = 'egl'

train_cost_lagrange = []
mean_eps_cost = 0
import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import ma_models as models
import ma_tools as tools
import cmdp_wrappers as wrappers

import torch
from torch import nn
from torch import distributions as torchd
to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):

  def __init__(self, config, logger, dataset):
    super(Dreamer, self).__init__()
    self._config = config
    self._logger = logger
    self._should_log = tools.Every(config.log_every)
    self._should_train = tools.Every(config.train_every)
    self._should_pretrain = tools.Once()
    self._should_reset = tools.Every(config.reset_every)
    self._should_expl = tools.Until( 
      int(config.expl_until / config.action_repeat)
    )
    self._metrics = {}
    self._step = count_steps(config.traindir)
    # Schedules.
    config.actor_entropy = (
        lambda x = config.actor_entropy: tools.schedule(x, self._step))
    
    config.actor_state_entropy = (
        lambda x = config.actor_state_entropy: tools.schedule(x, self._step))
    
    config.imag_gradient_mix = (
        lambda x=config.imag_gradient_mix: tools.schedule(x, self._step))
    
    self._dataset = dataset
    self._wm = models.WorldModel(self._step, config)
    self._task_behavior = models.ImagBehavior(
        config, self._wm, config.behavior_stop_grad)
    #inline function to get the reward prediction using world model, not sure why we need it though
    reward = lambda f, s, a: self._wm.heads['reward'](f).mean 
    self._expl_behavior = dict( # greedy which is using actor policy
        greedy = lambda: self._task_behavior,
        random = lambda: expl.Random(config),
        plan2explore = lambda: expl.Plan2Explore(config, self._wm, reward),
    )[config.expl_behavior]()

  def __call__(self, obs, reset, state = None, reward = None, cost = None, training = True):
    step = self._step
    if self._should_reset(step):
      state = None
    if state is not None and reset.any():
      mask = 1 - reset
      for key in state[0].keys():
        for i in range(state[0][key].shape[0]):
          state[0][key][i] *= mask[i]
      for i in range(len(state[1])):
        state[1][i] *= mask[i]
    if training and self._should_train(step):
      steps = (
          self._config.pretrain if self._should_pretrain()
          else self._config.train_steps)
      
      for _ in range(steps):
        self._train(next(self._dataset))

      if self._should_log(step):
        for name, values in self._metrics.items():
          self._logger.scalar(name, float(np.mean(values)))
          self._metrics[name] = []
        # openl = self._wm.video_pred(next(self._dataset))
        # self._logger.video('train_openl', to_np(openl))
        self._logger.write(fps=True)

    policy_output, state = self._policy(obs, state, training)

    if training:
      self._step += len(reset)
      self._logger.step = self._config.action_repeat * self._step
    return policy_output, state

  def _is_future_safety_violated(self, posterior_t):
    '''
    Starting from current state we roll out using learned model
    to see potential
    '''
    total_cost = 0
    cost_fn = lambda f, s, a: self._wm.heads['cost'](s).mode()
        # self._wm.dynamics.get_feat(s)  ).mode()
    with torch.no_grad():
        latent_state = posterior_t
        for _ in range(self._config.safety_look_ahead_steps):
            feat = self._wm.dynamics.get_feat(latent_state)
            total_cost += cost_fn(_, feat, _).item()
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
            latent_state = self._wm.dynamics.img_step(latent_state, action, sample = self._config.imag_sample)

    return total_cost > self._config.cost_threshold

  def _policy(self, obs, state, training):
    if state is None:
      batch_size = len(obs['image'])
      latent = self._wm.dynamics.initial(len(obs['image']))
      action = torch.zeros((batch_size, self._config.num_actions)).to(self._config.device)
    else:
      latent, action = state
    # check with actor to use
    embed = self._wm.encoder(self._wm.preprocess(obs))
    #latent_t = input( embed_t, action_{t-1}, latent_{t-1}
    latent, _ = self._wm.dynamics.obs_step(
        latent, action, embed, self._config.collect_dyn_sample)

    #begin roll out from here under control policy to check for violation, t -1 steps
    if self._config.eval_state_mean:
      latent['stoch'] = latent['mean']
    feat = self._wm.dynamics.get_feat(latent)

    constraint_violated = self._is_future_safety_violated(latent)
    if not training:
      actor =  self._task_behavior.safe_actor(feat) if constraint_violated \
              else self._task_behavior.actor(feat)
      action = actor.mode()

    elif self._should_expl(self._step):
      actor =  self._expl_behavior.safe_actor(feat) if constraint_violated \
              else self._expl_behavior.actor(feat)
      action = actor.sample()
    else:
      actor =  self._task_behavior.safe_actor(feat) if constraint_violated \
              else self._task_behavior.actor(feat)
      action = actor.sample()

    logprob = actor.log_prob(action)
    latent = {k: v.detach()  for k, v in latent.items()}
    action = action.detach()
    if self._config.actor_dist == 'onehot_gumble':
      action = torch.one_hot(torch.argmax(action, dim=-1), self._config.num_actions)
    action = self._exploration(action, training)
    policy_output = {'action': action, 'logprob': logprob}
    state = (latent, action)
    return policy_output, state

  def _exploration(self, action, training):
    amount = self._config.expl_amount if training else self._config.eval_noise
    if amount == 0:
      return action
    if 'onehot' in self._config.actor_dist:
      probs = amount / self._config.num_actions + (1 - amount) * action
      return tools.OneHotDist(probs=probs).sample()
    else:
      return torch.clip(torchd.normal.Normal(action, amount).sample(), -1, 1)
    raise NotImplementedError(self._config.action_noise)

  def _train(self, data):
    metrics = {}
    # train world model
    post, context, mets = self._wm._train(data)
    metrics.update(mets)
    start = post
    if self._config.pred_discount:  # Last step could be terminal.
      start = {k: v[:, :-1] for k, v in post.items()}
      context = {k: v[:, :-1] for k, v in context.items()}

    reward = lambda f, s, a: self._wm.heads['reward'](
        self._wm.dynamics.get_feat(s)).mode()
    cost = lambda f, s, a: self._wm.heads['cost'](
        self._wm.dynamics.get_feat(s)).mode()
    metrics.update(self._task_behavior._train(start, reward, cost)[-1])

    if self._config.expl_behavior != 'greedy':
      if self._config.pred_discount:
        data = {k: v[:, :-1] for k, v in data.items()}
      mets = self._expl_behavior.train(start, context, data)[-1]
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    #update training metrics for logs
    for name, value in metrics.items():
      if not name in self._metrics.keys():
        self._metrics[name] = [value]
      else:
        self._metrics[name].append(value)


def count_steps(folder):
  '''
  - find all files with extension .npz convert to string
  - COunt the file names wit that extension to know number of steps

  '''
  return sum(int(str(n).split('-')[-1][:-4]) - 1 for n in folder.glob('*.npz'))


def make_dataset(episodes, config):
  generator = tools.sample_episodes(
      episodes, config.batch_length, config.oversample_ends)
  dataset = tools.from_generator(generator, config.batch_size)
  return dataset


def make_env(config, logger, mode, train_eps, eval_eps):
  env = wrappers.SafetyGym(config.task, config.grayscale, action_repeat = config.action_repeat ) if not config.ontop else \
    wrappers.SafetyGym(config.task, config.grayscale, action_repeat = config.action_repeat, camera_name = 'fixednear' )
  
  env = wrappers.NormalizeActions(env)
  env = wrappers.TimeLimit(env, config.time_limit)
  env = wrappers.SelectAction(env, key='action')
  if (mode == 'train') or (mode == 'eval'):
    callbacks = [functools.partial(
        process_episode, config, logger, mode, train_eps, eval_eps)]
    env = wrappers.CollectDataset(env, callbacks)
  env = wrappers.RewardObs(env)
  env = wrappers.CostObs(env)
  return env


def process_episode(config, logger, mode, train_eps, eval_eps, episode):
  global train_cost_lagrange
  global mean_eps_cost
  directory = dict(train = config.traindir, eval = config.evaldir)[mode]
  cache = dict(train = train_eps, eval = eval_eps)[mode]
  filename = tools.save_episodes(directory, [episode])[0]
  length = len(episode['reward']) - 1
  score = float(episode['reward'].astype(np.float64).sum())
  score_cost = float(episode['cost'].astype(np.float64).sum())
  video = episode['image']
  if mode == 'eval':
    cache.clear()
  if mode == 'train' and config.dataset_size:
    total = 0
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
      if total <= config.dataset_size - length:
        total += len(ep['reward']) - 1
      else:
        del cache[key]
    logger.scalar('dataset_size', total + length)
  cache[str(filename)] = episode
  print(f'{mode.title()} episode has {length} steps, return {score:.1f} and cost {score_cost:.1f}.')
  if mode == 'train':
    train_cost_lagrange.append(score_cost)
    if len(train_cost_lagrange) > 100:
        train_cost_lagrange.pop(0)
    mean_eps_cost = np.mean(train_cost_lagrange)
    # print("Mean epsidode cost"  ,mean_eps_cost)
  logger.scalar(f'{mode}_cost_return', score_cost)
  logger.scalar(f'{mode}_return', score)
  logger.scalar(f'{mode}_length', length)
  logger.scalar(f'{mode}_episodes', len(cache))
  if mode == 'eval' or config.expl_gifs:
    logger.video(f'{mode}_policy', video[None])
  logger.write()


def set_test_paramters(config):
  # For testing on my mac to prevent high ram usage
  config.debug = True
  config.pretrain =  1
  config.prefill = 1
  config.train_steps = 1
  config.batch_size = 10
  config.batch_length = 20

def main(config):
  config_dict = config.__dict__
  config.task = 'SafetyPointCircle1-v0'
  config.steps = 300_000
  config.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  if sys.platform != 'linux': set_test_paramters(config)# if not zhuzun running so parameters for testing locally
  # print(config_dict)
  if sys.platform == 'linux': #not debugging on mac but running experiment

    # run =  wandb.init(project='Safe RL via Latent world models Setup mac', config = config_dict) \
    # if sys.platform != 'linux' else wandb.init(project='Safe RL via Latent world models Setup', config = config_dict)

    run = wandb.init(project='Safe RL via Latent world models Setup', config = config_dict)
  logdir = pathlib.Path(config.logdir).expanduser()
  config.traindir = config.traindir or logdir / 'train_eps'
  config.evaldir = config.evaldir or logdir / 'eval_eps'
  config.steps //= config.action_repeat
  config.eval_every //= config.action_repeat
  config.log_every //= config.action_repeat
  config.time_limit //= config.action_repeat
  config.act = getattr(torch.nn, config.act) #activation layer

  print('Logdir', logdir)
  logdir.mkdir(parents = True, exist_ok = True)
  config.traindir.mkdir(parents=True, exist_ok=True)
  config.evaldir.mkdir(parents=True, exist_ok=True)
  step = count_steps(config.traindir)
  logger = tools.Logger(logdir, config.action_repeat * step)

  print('Create envs.')
  if config.offline_traindir:
    directory = config.offline_traindir.format(**vars(config))
  else:
    directory = config.traindir
  train_eps = tools.load_episodes(directory, limit=config.dataset_size)

  if config.offline_evaldir:
    directory = config.offline_evaldir.format(**vars(config))
  else:
    directory = config.evaldir

  eval_eps = tools.load_episodes(directory, limit=1)
  make = lambda mode: make_env(config, logger, mode, train_eps, eval_eps)
  train_envs = [make('train') for _ in range(config.envs)]
  eval_envs = [make('eval') for _ in range(config.envs)]
  acts = train_envs[0].action_space
  config.num_actions = acts.n if hasattr(acts, 'n') else acts.shape[0]

  if not config.offline_traindir: 
    prefill = max(0, config.prefill - count_steps(config.traindir))
    print(f'Prefill dataset ({prefill} steps).')
    if hasattr(acts, 'discrete'):
      random_actor = tools.OneHotDist(torch.zeros_like(torch.Tensor(acts.low))[None])
    else:
      random_actor = torchd.independent.Independent(
          torchd.uniform.Uniform(torch.Tensor(acts.low)[None],
                                torch.Tensor(acts.high)[None]), 1)
    def random_agent(o, d, s, r, c):
      action = random_actor.sample()
      logprob = random_actor.log_prob(action)
      return {'action': action, 'logprob': logprob}, None
    tools.simulate(random_agent, train_envs, prefill)
    tools.simulate(random_agent, eval_envs, episodes=1)
    logger.step = config.action_repeat * count_steps(config.traindir)

  print('Simulate agent.')
  train_dataset = make_dataset(train_eps, config)
  eval_dataset = make_dataset(eval_eps, config)
  #intialise world models, and imgination(actor, critic)
  agent = Dreamer(config, logger, train_dataset).to(config.device)
  agent.requires_grad_(requires_grad = False)
  if (logdir / 'latest_model.pt').exists():
    agent.load_state_dict(torch.load(logdir / 'latest_model.pt'))
    agent._should_pretrain._once = False

  state = None
  while agent._step < config.steps:
    logger.write()
    print('Start evaluation.')
    video_pred = agent._wm.video_pred(next(eval_dataset))
    logger.video('eval_openl', to_np(video_pred))
    eval_policy = functools.partial(agent, training = False)
    tools.simulate(eval_policy, eval_envs, episodes = 1)
    print('Start training.')
    # this rolls out mdps, and adds state to the buffer inside the 
    # agent as well as get the action through the states
    state = tools.simulate(agent, train_envs, config.eval_every, state = state)
    torch.save(agent.state_dict(), logdir / 'latest_model.pt')
  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # parser.add_argument('--configs', nargs='+', required=True)
  parser.add_argument('--configs', nargs='+', default=['defaults', 'sgym'], required=False)
  args, remaining = parser.parse_known_args()
  configs = yaml.safe_load(
      (pathlib.Path(sys.argv[0]).parent / 'ma_configs.yaml').read_text())
  defaults = {}
  for name in args.configs:
    defaults.update(configs[name])
  parser = argparse.ArgumentParser()
  for key, value in sorted(defaults.items(), key = lambda x: x[0]):
    arg_type = tools.args_type(value)
    parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
  current_dir = os.path.dirname(os.path.abspath(__file__))
  logdir = os.path.join(current_dir, 'logdir', 'safecircle1', '0')
  existed_ns = [int(v) for v in os.listdir(os.path.join(current_dir, 'logdir', 'safecircle1'))]
  if len(existed_ns)>0:
    new_n = max(existed_ns)+1
    logdir = os.path.join(current_dir, 'logdir', 'safecircle1', str(new_n))
  parser.set_defaults(logdir = logdir)
  main(parser.parse_args(remaining))
