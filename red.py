import gym
from numpngw import write_apng  # pip install numpngw or pip install panda-gym[extra]
import sys
sys.path.append('/Users/emma/dev/CMBRVLN')
import safety_gym
env = gym.make("Safexp-PointGoal1-v0")
images = []


obs = env.reset()
done = False
v = env.render(mode = "rgb_array",  camera_id = 1)
# images.append(env.render("rgb_array"))
images.append(v)

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    v = env.render(mode = "rgb_array")
    images.append(v)

env.close()

write_apng("reach-safe.png", images, delay=40)