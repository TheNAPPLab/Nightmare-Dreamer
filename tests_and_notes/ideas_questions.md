# Algorithm Design question and possible equation
1. In the model loss obs + kl + b * reward + a * cost should we have any value `a` multiply the cost?
 b = 2 for dmc control for example for reward , shoudl we scale the cost as learning the cost might need to be of higher priority than the other part of the model?
2. Should we clip rewards? Some env use `tanh` some use `Identity`: But since Costs tends to be scarse and   we should probably just use `Identity`.
3. solve Safety DexterousHands
4. solve safe cheetah




# Questions about code base to check
1. whats difference with prefill and pretrain variable