from poissonTasks import Robot, distComponent, poissonTasks

fleet = [Robot([1, 2]), Robot([2, 1])]

incoming_dist = [distComponent(0, 60, 60, 10), distComponent(1, 50, 55, 5)]

n_types = 2

env = poissonTasks(fleet, incoming_dist, n_types)

num_steps = 50

obs = env.reset()

for step in range(num_steps):
    # take random action, but you can also do something more intelligent
    # action = my_intelligent_agent_fn(obs) 
    action = env.action_space.sample()
    
    if action[1] < env.n_types:
        print("attempting to assign robot", action[0], "to task", action[1])
    else:
        print("waiting")
    
    # apply the action
    obs, reward, done, info = env.step(action)
    
    print("delay of", obs[0])
    print("reward of", reward)
    
    # Render the env
    env.render()
    
    # If the epsiode is up, then start another one
    if done:
        env.reset()

# Close the env
env.close()