# from wrappers import NormalizeActions
import safety_gymnasium
import wrappers

# env = safety_gymnasium.make('SafetyRacecarGoal1-v0')
# env = NormalizeActions(env=env)
# print("env", env)


# env = gym.make("CartPole-v1", render_mode="rgb_array")
# env = gym.experimental.wrappers.PixelObservationV0(env, pixels_only=True)
# print(f'{env.observation_space=}')
# env = gym.experimental.wrappers.ResizeObservationV0(env, (16, 16))
# print(f'{env.observation_space=}')


genv = safety_gymnasium.make('SafetyPointCircle0-v0', render_mode ='human', width = 64, height = 64)
obs, _ = genv.reset()
image = genv.render()
print("yes")

# env = wrappers.SafetyGym(name = 'SafetyPointCircle0-v0')
# obs_space = env.observation_space
# action_space = env.action_space
#     # bound between 1 and -1
# env = wrappers.NormalizeActions()

  
# env = wrappers.SelectAction(env, key='action')




