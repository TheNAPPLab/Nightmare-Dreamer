# Actor Loss


$$
actorloss =  \sum_{t=0}^{H-1} ( -V_{\lambda}^{t} - \eta H[a_t | z_t] -  \theta \ln P_{\psi}*Z_t)
$$

$$
Z_t = \frac{{d_t - \sum_{t=0}^{t} \gamma^t \cdot C_t(s_t)}}{{\gamma^t \cdot d_t}}
$$
d is budget left
$$
d_t = SafetyBudgetConstant - \sum_{t=0}^{t} C_t
$$
 <!-- + \theta * \ln P_{\psi}  -->