import numpy as np
import gym
import safety_gymnasium
import matplotlib.pyplot as plt

'''
The camera_name parameter can be chosen from:
human: The camera used for freely moving around and can get input from keyboard real time.

vision: The camera used for vision observation, which is fixed in front of the agent’s head.

track: The camera used for tracking the agent.

fixednear: The camera used for top-down observation.

fixedfar: The camera used for top-down observation, but is further than fixednear.
'''
class SafetyGym:
  def __init__(self, name, grayscale, size = (64, 64), action_repeat = 1, render_mode = 'rgb_array', camera_name = None) -> None:
    
    self._size = size
    self._grayscale = grayscale
    if not camera_name:
      self._env = safety_gymnasium.make(name, render_mode = render_mode, width = size[0], height = size[1])
    else:
      self._env = safety_gymnasium.make(name, render_mode = render_mode, width = size[0], height = size[1], camera_name = camera_name )
    # self._env.set_seed(0)
    self._action_repeat = action_repeat

  @property
  def observation_space(self):
    spaces = {}
    spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (3,), dtype = np.uint8)
    return gym.spaces.Dict(spaces)
  
  @property
  def action_space(self):
    return self._env.action_space
  
  def close(self):
    return self._env.close()

  def render(self):
    return self._env.render()
  
  def reset(self):
    self._env.reset()
    obs = {}
    obs['image'] = self.render()
    if self._grayscale:
        obs['image'] = obs['image'][..., None]
    return obs
  def step(self, action):
    timestep_reward = 0
    timestep_cost = 0
    for _ in range(self._action_repeat):
      o, reward, cost, terminated, truncated, info  = self._env.step(action)
      timestep_reward += reward or 0
      timestep_cost += cost or 0
      if terminated or truncated:
        break
    obs = {}
    obs['image'] = self.render()
    # plt.imshow( obs['image'])
    # plt.axis('off')
    # plt.show()
    if self._grayscale:
        obs['image'] = obs['image'][..., None]
    return obs, timestep_reward, timestep_cost, terminated, truncated, info


class NormalizeActions:

  def __init__(self, env):
    self._env = env
    self._mask = np.logical_and(
        np.isfinite(env.action_space.low),
        np.isfinite(env.action_space.high))
    self._low = np.where(self._mask, env.action_space.low, -1)
    self._high = np.where(self._mask, env.action_space.high, 1)

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    return gym.spaces.Box(low, high, dtype=np.float32)

  def step(self, action):
    original = (action + 1) / 2 * (self._high - self._low) + self._low
    original = np.where(self._mask, original, action)
    return self._env.step(original)


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    # obs, reward, done, info = self._env.step(action)
    # self._step += 1
    # if self._step >= self._duration:
    #   done = True
    #   if 'discount' not in info:
    #     info['discount'] = np.array(1.0).astype(np.float32)
    #   self._step = None
    # return obs, reward, done, info
    self._step += 1

    obs, reward, cost, terminated, truncated, info = self._env.step(action)
    done = terminated or truncated
    if done:
      if 'discount' not in info:
        info['discount'] = np.array(1.0).astype(np.float32)
    return obs, reward, cost, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()


class SelectAction:

  def __init__(self, env, key):
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    return self._env.step(action[self._key])


class CollectDataset:

  def __init__(self, env, callbacks=None, precision=32):
    self._env = env
    self._callbacks = callbacks or ()
    self._precision = precision
    self._episode = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    obs, reward, cost, done, info = self._env.step(action)
    obs = {k: self._convert(v) for k, v in obs.items()}
    transition = obs.copy()
    if isinstance(action, dict):
      transition.update(action)
    else:
      transition['action'] = action
    transition['reward'] = reward
    transition['cost'] = cost
    transition['discount'] = info.get('discount', np.array(1 - float(done)))
    self._episode.append(transition)
    if done:
      for key, value in self._episode[1].items():
        if key not in self._episode[0]:
          self._episode[0][key] = 0 * value
      episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
      episode = {k: self._convert(v) for k, v in episode.items()}
      info['episode'] = episode
      for callback in self._callbacks:
        callback(episode)
    return obs, reward, cost, done, info

  def reset(self):
    obs = self._env.reset()
    transition = obs.copy()
    # Missing keys will be filled with a zeroed out version of the first
    # transition, because we do not know what action information the agent will
    # pass yet.
    transition['reward'] = 0.0
    transition['cost'] = 0.0
    transition['discount'] = 1.0
    self._episode = [transition]
    return obs

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
      dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
    elif np.issubdtype(value.dtype, np.uint8):
      dtype = np.uint8
    else:
      raise NotImplementedError(value.dtype)
    return value.astype(dtype)


class RewardObs:

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = self._env.observation_space.spaces
    assert 'reward' not in spaces
    spaces['reward'] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
    return gym.spaces.Dict(spaces)

  def step(self, action):
    obs, reward, cost, done, info = self._env.step(action)
    obs['reward'] = reward
    return obs, reward, cost, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reward'] = 0.0
    return obs


class CostObs:

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = self._env.observation_space.spaces
    assert 'cost' not in spaces
    spaces['cost'] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
    return gym.spaces.Dict(spaces)

  def step(self, action):
    obs, reward, cost, done, info = self._env.step(action)
    obs['cost'] = cost
    return obs, reward, cost, done, info

  def reset(self):
    obs = self._env.reset()
    obs['cost'] = 0.0
    return obs

