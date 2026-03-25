$$ \nabla_{a} \log p(a|s) \propto -\frac{\epsilon_\theta(a_k, s, k)}{\sigma_k} $$

$\epsilon_\theta$ 

 $$ \nabla_\phi \mathbb{E}[\log P_{\text{control}}(a_{\text{safe}}|s)] = \mathbb{E} \left[ \nabla_a \log P_{\text{control}}(a|s) \Big|{a=a{\text{safe}}} \cdot \nabla_\phi a_{\text{safe}}(\phi) \right] $$

As you pointed out in your 

math.md
, the score $\nabla_a \log P_{\text{control}}(a|s)$ is approximated by the diffusion model's noise prediction: $-\frac{\epsilon_\theta(a_k, s, k)}{\sigma_k}$.