step = 60_000
limit_decay_start = 5_000
limit_signal_prob = 100
limit_signal_prob_decay = 800   #1450
limit_signal_prob_decay_min = 12
if step <= limit_decay_start:
        expl_amount = limit_signal_prob
else:
        expl_amount =  limit_signal_prob
        ir = step  - limit_decay_start + 1
        expl_amount = expl_amount - ir/limit_signal_prob_decay
        expl_amount = max(limit_signal_prob_decay_min, expl_amount)
print( expl_amount)