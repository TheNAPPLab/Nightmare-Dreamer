import ma_dreamer as wrappers
size = [64, 64]
task = 'dmc_walker_walk'
action_repeat = 2

env = wrappers.DMGymnassium(task, action_repeat, size)
env = wrappers.NormalizeActions(env)
env = wrappers.TimeLimit(env, 1000)
env = wrappers.SelectAction(env, key='action')