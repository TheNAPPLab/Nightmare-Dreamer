# expl_amount = 1.0
# ir = 300_000 - 35_000 + 1
# expl_amount = expl_amount - ir/500_000
# print( max(0.1, expl_amount))
# # final_epsilon = 0.1 
# # initial_epsilon = 1.0

# # decay_rate = (final_epsilon - initial_epsilon) / (decay_end_step - decay_start_step)
# # decayed_epsilon = initial_epsilon + decay_rate * (current_step - decay_start_step)
# # return decayed_epsilon
# # print(C)action
# import math

# def exponential_decay(initial_value, decay_rate, current_step):
#     decayed_value = initial_value * math.exp(-decay_rate * current_step)
#     return decayed_value

# print(exponential_decay(1.0, 0.00001, 300_000 ))


value_target = 20
value_dist = 10
value_discount = 0.9
c = value_discount * value_dist - value_target
print(c)

a  = value_dist * value_discount
b = a - value_target
print(c)