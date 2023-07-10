import torch


def compute_return(
                reward: torch.Tensor,
                value: torch.Tensor,
                pcount: torch.Tensor,
                bootstrap: torch.Tensor,
                lambda_: float
            ):
    """
    Compute the discounted reward for a batch of data.
    reward, value, and discount are all shape [horizon - 1, batch, 1] (last element is cut off)
    Bootstrap is [batch, 1]
    """
    print('value', value)
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    print('next value',next_values)
    target = reward + pcount * next_values * (1 - lambda_)
    print(target)
    timesteps = list(range(reward.shape[0] - 1, -1, -1))
    print(timesteps)
    outputs = []
    accumulated_reward = bootstrap
    for t in timesteps:
        inp = target[t]
        discount_factor = pcount[t]
        accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
        outputs.append(accumulated_reward)
    returns = torch.flip(torch.stack(outputs), [0])
    return returns
    

    # Define test inputs
# reward = torch.tensor([[1.0], [2.0], [3.0]])
reward = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])

value = torch.tensor([[[0.5]], [[1.0]], [[1.5]]])


pcount = torch.tensor([[[0.9]], [[0.9**2]], [[0.9**3]]])

lambda_ = 0.5

    # Expected output
a = 1.0 + (0.9 *0.5 *1.0)
print(a)
b = 2.0 + (0.5*0.9*0.9 * 1.5)

print(b)
c= b + (0.9*0.9 * 1.5 * 0.5)
print(c)

d = a + (0.5*0.9 * c)
print(d)
    # Compute returns using the function
print(compute_return(reward[:-1], value[:-1], pcount[:-1], bootstrap = value[-1], lambda_ = 0.5))



