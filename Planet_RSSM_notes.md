# PLanet's Reccurent State Space Model

In the PlaNet paper there are two state compnents to the latent dynamics ``Deterministic`` and ``Stochastic``,
The ``Deteministic`` state is defined by the hidden state of the ``GRU`` and the ``stcohastic`` states are the ``prosterior`` and the ``prior``, we show how both are computed and imporved later on.

The Prosterior distribution and Hidden state are meant to capture the current state and should be equivalent to the observation, and are computed from the observation, we improve prosterior and hidden state belief by tryin to reconstruct observation from these two beliefs.

The prior distribution aims to do what the posterior distribution providing the current state but pulrely with the hidden state and without the observation and would be used for planning, we imporve the prior distribution by making it similar to the prosterior distribution

## Considering the POMDP
- Transition function $s_t \sim p(s_t | s_{t - 1}, a_{t-1})$ 
  - Following the markovian property  the current state can be infered from the prvious time step state and action

- Observation function $o_t \sim p( o_t | s_t )$

- Reward function $r_t \sim p(r_t | s_t )$

<!-- - Policy function $a_t \sim p(a_t | o_{<=t}, a_{<t})$ -->
- Policy function $a_t \sim p(a_t | o_{t-1}, a_{t-1})$

Planet tries to learn the transition model via a GRU, observation model and reward model. and also a encoder $q(s_t | o_{t-1}, a_{t-1})$

---
## Computing Deterministic State

As mentioned earlier, the ``deterministic`` state, denoted as $h_t$, is defined by the hidden state of the GRU and is generated using the transition model $s_t \sim p(s_t | s_{t - 1}, a_{t-1})$. In the transition model, the current state $s_{t-1}$ (composed of the deterministic (hidden state) and stochastic (posterior/prior) components) and the current action $a_{t-1}$ are passed through a neural network. The resulting output is then combined with the current hidden state $h_t$ and fed into the GRU to compute the next hidden state $h_{t} = f(h_{t-1}, s_{t-1}, a_{t-1})$.

```
    def deterministic_state(self, h_t, s_t, a_t):
        """Returns the deterministic state given the previous states
        and action.
        """
        h = torch.cat([s_t, a_t], dim=-1)
        h = Relu(self.latent_activation_layer(h))
        return grucell(h, h_t)
```
----

## Posterior State and Its Computation
The posterior state plays a crucial role in the PlaNet algorithm as one of the stochastic states. It is responsible for computing the current state based on the current hidden state, denoted as $h_t$, and the observation $e_t$, which is actually the encoded observation. The posterior state serves as the input state to the transition model during model learning when the observation is available.

To compute the posterior state, both the encoded observation and the current hidden state are passed through a neural network layer. Subsequently, they are fed into separate neural network layers: one to calculate the mean, denoted as $\mu_t$, and the other to compute the standard deviation, denoted as $\sigma_t$. These calculations enable the posterior state to be sampled later on as a stochastic state. It is important to note that the state is defined as a diagonal Gaussian distribution, but further details will be discussed later on.

```
  def state_posterior(self, h_t, e_t, sample = False):
          """Returns the state prior given the deterministic state h_t and obs ( encoded state e_t )"""
          z = torch.cat([h_t, e_t], dim = -1)
          z = Relu(self.fc_posterior_1(z))
          m = self.fc_posterior_m(z)
          s = F.softplus(self.fc_posterior_s(z)) + 1e-1
          if sample: 
              return m + torch.randn_like(m) * s
          return m, s

```
----

## Prior State and its Computation
In this step, we focus on computing the prior state, which is an essential component for planning in a partially observable environment (POMDP) where we don't have direct access to the observation ``Encoder State``. The prior state is computed based on the current hidden state, also known as the deterministic state.


To compute the prior state, the current hidden state (deterministic state) is passed through a neural network layer. Subsequently, it is fed into two separate neural network layers: one to compute the mean ($\mu_t$) and the other to compute the standard deviation ($\sigma_t$). These parameters allow us to sample the prior state as a stochastic state. 

This provides us with the ability to plan completely in latent space.

Here is the Psudocode that illustrates the process:

```
    def state_prior(self, h_t, sample = False):
        """Returns the state prior given the deterministic state h_t """
        z = Relu(self.fc_prior_1(h_t))
        m = self.fc_prior_m(z)
        s = F.softplus(self.fc_prior_s(z)) + 1e-1
        if sample:
            return m + torch.randn_like(m) * s
        return m, s
```
--- 
## Predicting Reward

In this step, we aim to predict the current scalar reward based on the current stochastic state $s_t$ (posterior or prior) and the deterministic state $h_t$. The reward is modeled as a distribution, following the formulation in the paper: $r_t \sim p(r_t | h_t, s_t)$.


---

## Learning better Posterior, Prior, Representation, and Reward Signal


After the agent interacts with the environment, the MDP (Markov Decision Process) trajectory {$o_t$, $a_t$, $r_t$} is stored in the buffer. To enable better planning, we aim to improve our belief in the prior distribution as well as enhance the VAE to obtain a more accurate representation of the observations.

Here's a breakdown of how the major components of learning the model work:

1. We first sample the MDP trajectory from the buffer, which includes observations, actions, and rewards.
2. The observations are passed through the encoder to convert them into latent space representation.
3. The initial GRU states are initialized to obtain the first ``deterministic`` (hidden state) and ``stochastic`` state (posterior) since we have access to the encoded observations.
4. For each action, we compute the new deterministic state $h_t$ based on the previous deterministic state $h_{t-1}$, the previous stochastic state $s_{t-1}$, and the previous action $a_{t-1}$.
5. The prior distribution is computed using only the hidden state, while the posterior distribution is computed using both the hidden state and the encoded state.
6. In this step, we calculate the prior distribution as we aim to improve it later on to be accurate for planning without access to the observation.
7. Since we already have access to the current deterministic state to be used for next time step, to obtain the current stochastic state $s_t$, we simply sample from the posterior distribution.

Here is the Psudocode that illustrates the process:
```
observations, actions, reward = sample_batch()

e_t = rssm.encoder(observations)

h_t, s_t = rssm.get_init_state(e_t[0]) # get intial Deretminstic  and Stochastic state from current observation (Encoded Observation)

for i, a_t in enumerate(actions):
        h_t = rssm.deterministic_state_fwd(h_t, s_t, a_t)
        hidden_states.append(h_t)
        priors.append(rssm.state_prior(h_t))
        posteriors.append(rssm.state_posterior(h_t, e_t[i + 1]))
        posterior_samples.append(Normal(*posteriors[-1]).rsample())
        s_t = posterior_samples[-1]
```

---
## VAE
The encoder is implemented by a convolutional neural net, they define the encoder as $q(s_{1:T} | o_{1:T}, a_{1:T} ) = \prod_{t=1}^{T} q(s_t | s_{t-1}, a_{t-1}, o_t)$ [Proof](#proof-for-encoder-function) 
I guess they use q for Encoder as opposed to p, as its dijointed and generated from a diffrent distribution which is the observation 

## Improving Reconstruction, VAE and State Representation


In order to enhance the representation of observations or the current state, it is important to improve the posterior distribution obtained from the observation or the Observation Model. The posterior distribution is derived from the observation through an encoder, and we also aim to enhance the hidden state for similar reasons

When using a Variational Autoencoder (VAE), the objective is typically to make the decoded images or outputs as close as possible to the input observations. This is commonly achieved by utilizing the Mean Squared Error (MSE) loss function. In this case, we apply the MSE loss to the decoder of the VAE, comparing the output of the  ``decoder (using the hidden and stochastic state, i.e., the posterior )`` with the observations.

Although the paper states that the observation model is Gaussian with a mean parameterized by a deconvolutional neural network and identity covariance, the deconvolutional layer allows us to perform image reconstruction, and the assumption of identity covariance is implemented using a diagonal covariance matrix. However, in our implementation, we utilize a pixelwise error, rendering the specific form of the covariance matrix irrelevant."

```
reconstruction_loss = MSE(rssm.decoder(hidden_states, posterior), observations)
```

As mentioned earlier, the posterior is computed through an encoder, and neural networks are employed to predict the posterior and hidden states. By applying the MSE loss, the gradients can propagate backward through the network (thanks to autograd), thereby improving all the parameters involved in the process.

More on the full loss function later[Loss_proof](#the-loss-function).

## Improving the Prior Distribution

Intuitively, we aim to make the prior distribution resemble the posterior distribution discussed earlier. This can be achieved by utilizing the Kullback-Leibler (KL) divergence loss function, which measures the dissimilarity between the posterior and the prior distributions. By minimizing this loss, we encourage the prior distribution to align more closely with the posterior distribution, facilitating better representation and modeling of the underlying dynamics.

More on the full loss function later [Loss_proof](#the-loss-function).

## Improving Reward Prediction

Using a simple MSE loss we get our prediction of reward closer to the the value from the training environment ina pure supervised learning manner

## Latent Overshooting
Tbh i am not too sure whats going on here, however going over the source code from the ``google-reaserch`` , ``danijar`` and other source code.
After computing the prosterior, priors and hidden states, we use the usual loss for KL, and reconstruction only for the first time step after that we mask out some roll outs and compute the loss for each time step individually then average later on


## Proof for Encoder Function

$q(s_{1:T} | o_{1:T}, a_{1:T} ) = \prod_{t=1}^{T} q(s_t | s_{t-1}, a_{t-1}, o_t)$

$$
\text{Chain rule of probability:}\\ 
\\[1em]

P(A | B) = \frac{P(A \cap B)}{P(B)} \\[1em] 

P(A | B) = \frac{P(A, B)}{P(B)}\\[1em]

P(A, B) = P(A | B)  \cdot  P (B)\\[1em]

P(A, B, C) = P(A | B, C)  \cdot  P (B | C)  \cdot P(C) \\[1em]

P(A | B, C)  = \frac{P(A, B, C)}{ P (B | C)  \cdot P(C)}  \\[1em]

P(X_{1}, X_{2}....,X_{T}) = P(X_{1})  \cdot P(X_{2} | P(X_{1}) )  .... P(X_{n} | P(X_{1},P(X_{2}... P(X_{n-1} ) ) 

$$

### so we can expand  
$$
 q(s_{1:T} | o_{1:T}, a_{1:T} )  =  \frac{q(s_{1:T}, o_{1:T}, a_{1:T} )}{q(o_{1:T}, a_{1:T} )} \\[1em]
  \text{Note: We can rollout like this}\\[1em]

   q(s_{1:T}, o_{1:T}, a_{1:T} )  =  q(s_{1},s_{2}, s_{3}.... s_{T},  o_{1}, o_{2}, o_{3}.... o_{T},  a_{1},a_{2}, a_{3}.... a_{T})=  q(s_{1},s_{2:T},  o_{1}, o_{2:T},  a_{1}, a_{2_T} )\\[1em]

 \text{Express the numerator using the chain rule of probability as shown before:}\\[1em]
  q(s_{1:T}, o_{1:T}, a_{1:T} )  =  q(s_{T},s_{1:T-1},  o_{T}, o_{1:T-1},  a_{T}, a_{1:T-1} ) =    q(s_{T}, o_{T},  a_{T} | s_{1:T-1}, o_{1:T-1}, a_{1:T-1}) \cdot q( s_{1:T-1}, o_{1:T-1}, a_{1:T-1})\\[1em]

  \text{We can further do chainrule on the conditional probability:}\\[1em]
   q( s_{1:T-1}, o_{1:T-1}, a_{1:T-1})\\[1em]
  q( s_{1:T-1}, o_{1:T-1}, a_{1:T-1})= q(s_{T-1},s_{1:T-2},  o_{T-1}, o_{1:T-2},  a_{T-1}, a_{1:T-2} ) =    q(s_{T-1}, o_{T-1},  a_{T-1} | s_{1:T-2}, o_{1:T-2}, a_{1:T-2}) \cdot q( s_{1:T-2}, o_{1:T-2}, a_{1:T-2})\\[1em]

  \text{So we can recusrively recursively recursively recursively ... I will stop now 😂: roll out till T}\\[1em]
  q(s_{1:T}, o_{1:T}, a_{1:T} )  = q(s_{T}, o_{T},  a_{T} | s_{1:T-1}, o_{1:T-1}, a_{1:T-1}) \cdot   q(s_{T-1}, o_{T-1},  a_{T-1} | s_{1:T-2}, o_{1:T-2}, a_{1:T-2})  ...  q(s_{2}, o_{2}, a_{2} | s_1, o_1, a_1) \cdot q(s_{1}, o_{1}, a_{1}) \\[1em]

  
 \text{Now we express the Denominator using the chain rule of probability in a similar manner:}\\[1em]

 q(o_{1:T}, a_{1:T}) = q(o_T, a_T | o_{1:T-1}, a_{1:T-1}) \cdot q(o_{T-1}, a_{T-1} | o_{1:T-2}, a_{1:T-2}) \cdot \ldots \cdot q(o_{2}, a_{2} | o_1, a_1) \cdot q(o_1, a_1) \\[1em]

  q(s_{1:T} | o_{1:T}, a_{1:T}) = \frac{q(s_{T}, o_{T},  a_{T} | s_{1:T-1}, o_{1:T-1}, a_{1:T-1}) \cdot   q(s_{T-1}, o_{T-1},  a_{T-1} | s_{1:T-2}, o_{1:T-2}, a_{1:T-2})  ...  q(s_{2}, o_{2}, a_{2} | s_1, o_1, a_1) \cdot q(s_{1}, o_{1}, a_{1})}{q(o_T, a_T | o_{1:T-1}, a_{1:T-1}) \cdot q(o_{T-1}, a_{T-1} | o_{1:T-2}, a_{1:T-2}) \cdot \ldots \cdot q(o_{2}, a_{2} | o_1, a_1) \cdot q(o_1, a_1)}\\[1em]

 \text{Notice that some terms match in the fractions similar to:}\\[1em]
P(A | B) = \frac{P(A, B)}{P(B)}\\[1em]

  q(s_{1:T} | o_{1:T}, a_{1:T}) = \frac{q(s_{T}, o_{T},  a_{T} | s_{1:T-1}, o_{1:T-1}, a_{1:T-1})}{q(o_T, a_T | o_{1:T-1}, a_{1:T-1})} \cdot \frac{q(s_{T-1}, o_{T-1},  a_{T-1} | s_{1:T-2}, o_{1:T-2}, a_{1:T-2})}{q(o_{T-1}, a_{T-1} | o_{1:T-2}, a_{1:T-2})} \cdot \ldots \cdot \frac{q(s_{2}, o_{2}, a_{2} | s_1, o_1, a_1)}{q(o_{2}, a_{2} | o_1, a_1)} \cdot \frac{q(s_1, o_1, a_1)}{q(o_1, a_1)}\\[1em]

    \text{similar to :}\\[1em]
 P(A, B, C) = P(A | B, C)  \cdot  P (B | C)  \cdot P(C) \\[1em]

q(s_T, o_T, a_T | s_{1:T-1}, o_{1:T-1}, a_{1:T-1}) = q(s_T | o_T, a_T, s_{1:T-1}, o_{1:T-1}, a_{1:T-1}) \cdot q(o_T | a_T, s_{1:T-1}, o_{1:T-1}, a_{1:T-1}) \cdot q(a_T | s_{1:T-1}, o_{1:T-1}, a_{1:T-1})\\[1em]
 \text{And for the denominator :}\\[1em]
q(o_T, a_T | o_{1:T-1}, a_{1:T-1}) = q(o_T | a_T, o_{1:T-1}, a_{1:T-1}) \cdot q(a_T | o_{1:T-1}, a_{1:T-1})\\[1em]

   \text{ Now, observe that the terms involving }
    o_T 

   \text{ and }
    a_T
       \text{   cancel out in the numerator and denominator, resulting in: } \\[1em]

\frac{q(s_T, o_T, a_T | s_{1:T-1}, o_{1:T-1}, a_{1:T-1})}{q(o_T, a_T | o_{1:T-1}, a_{1:T-1})} = q(s_T | s_{1:T-1}, o_{1:T-1}, a_{1:T-1})\\[1em]

 q(s_{1:T} | o_{1:T}, a_{1:T}) = q(s_T | s_{1:T-1}, o_{1:T-1}, a_{1:T-1}) \cdot q(s_{T-1} | s_{1:T-2}, o_{1:T-2}, a_{1:T-2}) \cdot \ldots \cdot q(s_2 | s_1, o_1, a_1) \cdot q(s_1)\\[1em]
 \text{Finally Using some Markovian magic and assumption, the recent states, observations and actions are enough to describe the dynamics :}\\[1em]
 q(s_{1:T} | o_{1:T}, a_{1:T} ) = \prod_{t=1}^{T} q(s_t | s_{t-1}, a_{t-1}, o_t)\\[1em]
$$

## The Loss function


$$
P(A | B) = \frac{P(A \cap B)}{P(B)}  =  \frac{P(A) \cdot P(B | A)}{P(B)} \\[1em] 

\text{Total Probability Theorem :}\\[1em]

P(B) = \sum_{k=1}^{n} P(A_i) \cdot P(B | A_i) = \int_{A} P(B | A) \\[1em]

P(A | B) = \frac{P(A) \cdot P(B | A)}{ \int_{A} P(B | A) }  \\[1em] 


\ln p(o_{1:T} | a_{1:T})  = \ln  \int_{A} P(B | A)  \\[1em] 


\text{In the appendix they start with some description which i show below but couldnt sucesfully prove, I eventually skip this step but i welcome anyone to finish it: }\\[1em]
\ln p(o_{1:T} | a_{1:T}) \triangleq \ln\mathbb{E}_{p(s_{1:T} | a_{1:T})}  [\prod_{t=1}^{T} p(o_t |  s_t)] \\[1em]



\mathbb{E}_{p(s_{1:T} | a_{1:T})} [\prod_{t=1}^{T} p(o_t | s_t)] = \sum_{s_{1:T}} \left(\prod_{t=1}^{T} p(o_t | s_t)\right) p(s_{1:T} | a_{1:T})\\[1em]

\text{Taking the natural logarithm of both sides:: }\\[1em]

\ln \mathbb{E}_{p(s_{1:T} | a_{1:T})} [\prod_{t=1}^{T} p(o_t | s_t)] = \ln \sum_{s_{1:T}} \left(\prod_{t=1}^{T} p(o_t | s_t)\right) p(s_{1:T} | a_{1:T}) \\[1em]

\text{Now, let's expand the logarithm of the sum using the logarithmic properties: }\\[1em]

\ln \sum_{s_{1:T}} \left(\prod_{t=1}^{T} p(o_t | s_t)\right) p(s_{1:T} | a_{1:T}) = \ln \left(\sum_{s_{1:T}} \left(\prod_{t=1}^{T} p(o_t | s_t)\right) p(s_{1:T} | a_{1:T})\right) \\[1em]



\ln \sum_{s_{1:T}} \exp \left(\ln \left(\prod_{t=1}^{T} p(o_t | s_t)\right) p(s_{1:T} | a_{1:T})\right)\\[1em]

\ln \sum_{s_{1:T}} \exp \left(\ln \left(\prod_{t=1}^{T} p(o_t | s_t)\right) + \ln p(s_{1:T} | a_{1:T})\right)\\[1em]


\ln \sum_{s_{1:T}} \exp \left(\sum_{t=1}^{T} \ln p(o_t | s_t) + \ln p(s_{1:T} | a_{1:T})\right)\\[1em]


\text{Next, we use the equ   to simplify the expression: }
 \ln(\exp(x)) = x \\[1em]

 \sum_{s_{1:T}} \left(\sum_{t=1}^{T} \ln p(o_t | s_t) + \ln p(s_{1:T} | a_{1:T})\right)\\[1em]

 \text{Now, we can rearrange the terms inside the double summation: }\\[1em]

 \sum_{t=1}^{T} \sum_{s_{1:T}} \ln p(o_t | s_t) + \sum_{s_{1:T}} \ln p(s_{1:T} | a_{1:T})\\[1em]




$$

## Continuing Loss Proof

$$
\text{starting Proof from relationship stated in the appendix: }\\[1em]

\ln p(o_{1:T} | a_{1:T}) \triangleq \ln\mathbb{E}_{p(s_{1:T} | a_{1:T})}  [\prod_{t=1}^{T} p(o_t |  s_t)] \\[1em]
$$

<!-- \text{Jensen’s inequality: }\\[1em] -->