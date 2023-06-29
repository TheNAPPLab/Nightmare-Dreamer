# import safety_gym
import gym
import numpy as np
# import minatar

class SafetyGymEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, env_name, display_time=50):
        self.display_time = display_time
        self.env_name = env_name
        self.env = gym.make('Safexp-PointGoal1-v0')
    
    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs['image'] = self.render().transpose(2, 0, 1).copy()
        return obs

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self.env.observation_spec().items():
          spaces[key] = gym.spaces.Box(
              -np.inf, np.inf, value.shape, dtype=np.float32)
        spaces['image'] = gym.spaces.Box(
            0, 255, (3,) + self._size , dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return gym.spaces.Box(self.env.action_space.low,self.env.action_spaceaction_space.high, dtype=np.float32)
    
    def step(self, action):
        time_step = self.env.step(action)
        obs = dict(time_step.observation)
        obs['image'] = self.render().transpose(2, 0, 1).copy()
        reward = time_step.reward or 0
        done = time_step.last()
        info = time_step.info
        info['discount': np.array(time_step.discount, np.float32)]
        return obs, reward, done, info
    
    
    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.env.state()
        elif mode == 'human':
            self.env.display_state(self.display_time)

    def close(self):
        if self.env.visualized:
            self.env.close_display()
        return 0
class NormalizeActions:

    def __init__(self, env):
        assert isinstance(env.action_space)
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
        low = np.where(self._mask,x -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self._env.step(original)

class GymMinAtar(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, env_name, display_time=50):
        self.display_time = display_time
        self.env_name = env_name
        self.env = minatar.Environment(env_name) 
        self.minimal_actions = self.env.minimal_action_set()
        h,w,c = self.env.state_shape()
        self.action_space = gym.spaces.Discrete(len(self.minimal_actions))
        self.observation_space = gym.spaces.MultiBinary((c,h,w))

    def reset(self):
        self.env.reset()
        return self.env.state().transpose(2, 0, 1)
    
    def step(self, index):
        '''index is the action id, considering only the set of minimal actions'''
        action = self.minimal_actions[index]
        r, terminal = self.env.act(action)
        self.game_over = terminal
        return self.env.state().transpose(2, 0, 1), r, terminal, {}

    def seed(self, seed='None'):
        self.env = minatar.Environment(self.env_name, random_seed=seed)
    
    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.env.state()
        elif mode == 'human':
            self.env.display_state(self.display_time)

    def close(self):
        if self.env.visualized:
            self.env.close_display()
        return 0

class breakoutPOMDP(gym.ObservationWrapper):
    def __init__(self, env):
        '''index 2 (trail) is removed, which gives ball's direction'''
        super(breakoutPOMDP, self).__init__(env)
        c,h,w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c-1,h,w))

    def observation(self, observation):
        return np.stack([observation[0], observation[1], observation[3]], axis=0)
    
class asterixPOMDP(gym.ObservationWrapper):
    '''index 2 (trail) is removed, which gives ball's direction'''
    def __init__(self, env):
        super(asterixPOMDP, self).__init__(env)
        c,h,w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c-1,h,w))
    
    def observation(self, observation):
        return np.stack([observation[0], observation[1], observation[3]], axis=0)
    
class freewayPOMDP(gym.ObservationWrapper):
    '''index 2-6 (trail and speed) are removed, which gives cars' speed and direction'''
    def __init__(self, env):
        super(freewayPOMDP, self).__init__(env)
        c,h,w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c-5,h,w))
    
    def observation(self, observation):
        return np.stack([observation[0], observation[1]], axis=0)    

class space_invadersPOMDP(gym.ObservationWrapper):
    '''index 2-3 (trail) are removed, which gives aliens' direction'''
    def __init__(self, env):
        super(space_invadersPOMDP, self).__init__(env)
        c,h,w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c-2,h,w))
    def observation(self, observation):
        return np.stack([observation[0], observation[1], observation[4], observation[5]], axis=0)

class seaquestPOMDP(gym.ObservationWrapper):
    '''index 3 (trail) is removed, which gives enemy and driver's direction'''
    def __init__(self, env):
        super(seaquestPOMDP, self).__init__(env)
        c,h,w = env.observation_space.shape
        self.observation_space = gym.spaces.MultiBinary((c-1,h,w))
        
    def observation(self, observation):
        return np.stack([observation[0], observation[1], observation[2], observation[4], observation[5], observation[6], observation[7], observation[8], observation[9]], axis=0)    

class ActionRepeat(gym.Wrapper):
    def __init__(self, env, repeat=1):
        super(ActionRepeat, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        done = False
        total_reward = 0
        total_cost = 0
        current_step = 0
        while current_step < self.repeat and not done:
            obs, reward, done, info = self.env.step(action)
            cost = info['cost']
            total_cost += cost
            total_reward += reward
            current_step += 1
        return obs, total_reward, total_cost, done, info

class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super(TimeLimit, self).__init__(env)
        self._duration = duration
        self._step = 0
    
    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            info['time_limit_reached'] = True
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()

class OneHotAction:

    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        shape = (self._env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        return space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
          raise ValueError(f'Invalid one-hot action:\n{action}')
        return self._env.step(index)

    def reset(self):
        return self._env.reset()

    def _sample_action(self):
        actions = self._env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference

