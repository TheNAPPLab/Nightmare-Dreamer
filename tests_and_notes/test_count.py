import safety_gymnasium

# env = safety_gymnasium.make("SafetyPointCircle1-v0", max_episode_steps=1000)
# env = safety_gymnasium.make("SafetyPointGoal1-v0")
# observation, info = env.reset(seed=0)

# step  = 0
# sum_cost = 0
# while True:
#     step+=1
#     action = env.action_space.sample()  # this is where you would insert your policy
#     observation, reward, cost, terminated, truncated, info = env.step(action)
#     sum_cost += cost
#     if terminated or truncated:
#         break
# print(step, sum_cost)

# n = 1000  # Number of steps
# gamma = 0.99  # Discount factor

# max_discounted_value = sum([gamma**i for i in range(n)])
# print("Maximum Discounted Value:", max_discounted_value)


# x = 25*100/1000
# print(x)


# def target_ratio(b, max_target = 99.3429516957585 , max_cost = 1000):
#   return b * max_target / max_cost

# print(target_ratio(25))

# import os

# # Get the current file's directory
# current_dir = os.path.dirname(os.path.abspath(__file__))
# print(current_dir)
import torch
budget = 9
a = (8 - (2 )) /(8)
b = (1 - (2/8))
print(a) 
print(b)
H = 14
gamma = 0.99
B = 2
cost_tensor = torch.ones(H, B, 1)
cost_tensor[:-3]  = 0
print('cost_tensor shape', cost_tensor.shape) #14 , B , 1
sum_of_discounted_costs = torch.zeros(B, 1)

# Compute the cumulative sum of costs over time (dim=0)
cumulative_sum_cost = cost_tensor.cumsum(dim=0)

# Compute d_t for each time step
d_t = budget - cumulative_sum_cost

print(d_t)
# Initialize Z_t with zeros
Z_t = torch.zeros_like(d_t)
for t in range(H):
    sum_of_discounted_costs += (gamma ** t) * cost_tensor[t]

# print(sum_of_discounted_costs)
for t in range(H):
    denominator = (gamma ** t * d_t[t])
    denominator += 1e-8
    Z_t[t] = (d_t[t] - sum_of_discounted_costs)

print("z_t", Z_t.shape)

print(Z_t)

batch_index = 0
print("Z_t for Batch {}:".format(batch_index))
print(Z_t[:, batch_index, :].squeeze().tolist())