import torch
import pytest
from dreamerv2.utils.algorithm import compute_return 

def test_compute_return():
    # Define test inputs
    reward = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    value = torch.tensor([[[0.5]], [[1.0]], [[1.5]]])
    pcount = torch.tensor([[[0.9]], [[0.9**2]], [[0.9**3]]])

    # Expected output
    expected_returns = torch.tensor ([[[2.8967]], [[3.2150]]])
 

    # test V_t = r_t + y * (1-lamda) * value_{t+1} + y * lamda V_{t+1}
    # Compute returns using the function
    returns = compute_return(reward[:-1], value[:-1], pcount[:-1], bootstrap = value[-1], lambda_ = 0.5)


    # Compare the computed returns with the expected returns
    assert torch.allclose(returns, expected_returns, atol=1e-4)
