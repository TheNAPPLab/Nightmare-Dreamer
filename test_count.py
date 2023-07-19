import safety_gymnasium

# env = safety_gymnasium.make("SafetyPointCircle1-v0", max_episode_steps=1000)
env = safety_gymnasium.make("SafetyPointGoal1-v0")
observation, info = env.reset(seed=0)

step  = 0
sum_cost = 0
while True:
    step+=1
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, cost, terminated, truncated, info = env.step(action)
    sum_cost += cost
    if terminated or truncated:
        break
print(step, sum_cost)

# n = 1000  # Number of steps
# gamma = 0.99  # Discount factor

# max_discounted_value = sum([gamma**i for i in range(n)])
# print("Maximum Discounted Value:", max_discounted_value)



# x = 25*100/1000
# print(x)