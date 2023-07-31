# Actor Loss


$$
actorloss =  \sum_{t=0}^{H-1} ( -V_{\lambda}^{t} - \eta H[a_t | z_t] -  \theta \ln P_{\psi}*Z_t)
$$
or 
$$
actorloss =  \sum_{t=0}^{H-1} ( -V_{\lambda}^{t} - \eta H[a_t | z_t] +  C = { C if Z_i < 0, 0, otherwise} )
$$


$$
Z_t = \frac{{d_t - \sum_{t=0}^{t} \gamma^t \cdot C_t(s_t)}}{{\gamma^t \cdot d_t}}
$$
d is budget left
$$
d_t = SafetyBudgetConstant - \sum_{t=0}^{t} C_t
$$
 <!-- + \theta * \ln P_{\psi}  -->

# Multi Agent NightMare

## Terms
$$
a_s = safeAction \\
a = Control action \\
p_s = states distribution under Safe state policy \\
p = states under Control policy \\
$$
## Safe Actor loss single roll out
$$

safeActorLoss = \ln \pi_{s}[a_s | s_{p}] * s.g(VC_{\lambda_{wrt \pi}}^{t} - Vc(s_p))  + KL (\pi[a | s_{p}],   \pi_{s}[a_s | s_{p}])
$$


## Safe Actor loss Double roll out
$$

safeActorLoss =  C_{\lambda_{x}}^{t} + KL (\pi[a | s_{p}], \pi_{s}[a_s | s_{p}])
$$