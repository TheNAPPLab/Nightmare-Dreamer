expl_amount = 1.0
ir = 50000 - 35_000 + 1
expl_amount = expl_amount - ir/300_000
print( max(0.1, expl_amount))
# final_epsilon = 0.1 
# initial_epsilon = 1.0

# decay_rate = (final_epsilon - initial_epsilon) / (decay_end_step - decay_start_step)
# decayed_epsilon = initial_epsilon + decay_rate * (current_step - decay_start_step)
# return decayed_epsilon
# print(C)