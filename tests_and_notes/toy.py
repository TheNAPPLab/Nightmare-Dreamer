# from wrappers import NormalizeActions
import safety_gymnasium
import gymnasium as gym

# env = safety_gymnasium.make('SafetyRacecarGoal1-v0')
# env = NormalizeActions(env=env)
# print("env", env)


env = gym.make("CartPole-v1", render_mode="rgb_array")
env = gym.experimental.wrappers.PixelObservationV0(env, pixels_only=True)
print(f'{env.observation_space=}')
env = gym.experimental.wrappers.ResizeObservationV0(env, (16, 16))
print(f'{env.observation_space=}')


