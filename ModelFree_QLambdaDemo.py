from poissonTasks import Robot, distComponent, poissonTasks
import gym
import numpy as np
import random

#Some of this code has been reused from Victor Bubis's Assignment 3 for CISC 856

def encode(tup, dims): #Precondition: tup and dims are of equal length
    output = tup[0]
    for i in range(1, len(dims)):
        output = output * dims[i] + tup[i]
    
    return output

def decode(code, dims):
    output = []
    for i in range(1, len(dims)):
        output.append(code // dims[-i])
        code = code % dims[-i]
    output.append(code)
    
    return tuple(output)

class SMDP_QLambda():
    def __init__(self, alpha, gamma, epsilon, lamb, env):
        #We use env to get state space and number of actions
        #lamb is lambda
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lamb = lamb
        
        #env in this case is the poissonTasks environment
        #We have had to hard code the dimensions of the parameters spaces in this case as this code is not designed
        # to be compatible with Gym; we did not have time to code this compatability
        self.action_dims = (2, 3)
        self.obs_dims = (2, 2, 1000, 1000)
        
        self.actions = np.arange(2*3)
        self.Q = np.zeros((2*2*1000*1000, 2*3))
        self.E = np.copy(self.Q) #Note, this has to be zeros; change this if Q is fuzzy initialized
    
    def epsilon_greedy(self, S):
        #Start with weighted binary choice between exploit and explore
        exploit = random.choices((True, False), (1-self.epsilon, self.epsilon))[0]
        if exploit:
            #We take the argmax greedy option
            return np.argmax(self.Q[S])
        else: #explore
            #For explore, use equally weighted options, including the greedy one
            return random.choice(self.actions) 
    
    def demo_episode(self, env, MAX_LOOPS = 1000):
        obs = env.reset()
        k, fleet_status, unassigned_counts = obs
        
        #Translate fleet_status and unassigned_counts into S_prime
        S = encode(fleet_status + unassigned_counts, self.obs_dims)
        for i in range(MAX_LOOPS):
            A = self.epsilon_greedy(S)
            
            #Translate A back into format
            A_d = decode(A, self.action_dims)
            
            if A_d[1] < env.n_types:
                print("attempting to assign robot", A_d[0], "to task", A_d[1])
            else:
                print("waiting")
            
            obs, R, done, _ = env.step(A_d) #R is reward
            
            print("delay of", obs[0])
            print("reward of", R)
            
            env.render()
            
            k, fleet_status, unassigned_counts = obs
            
            #Translate fleet_status and unassigned_counts into S_prime
            S_prime = encode(fleet_status + unassigned_counts, self.obs_dims)
            
            A_prime = np.argmax(self.Q[S]) #Greedy look-ahead
            
            delta = R + (self.gamma**k) * self.Q[S_prime][A_prime] - self.Q[S][A]
            self.E[S][A] = self.E[S][A] + 1 #We are accumulating traces
            #These are vectorized operations
            self.Q = self.Q + self.alpha * delta * self.E
            self.E = ((self.gamma * self.lamb)**k) * self.E
            #Apply updates for next loop
            S = S_prime
            if done:
                print("Max time passed, resetting simulation")
                print("===")
                env.reset()
                
        env.close()
        return done
    
fleet = [Robot([1, 2]), Robot([2, 1])]

incoming_dist = [distComponent(0, 60, 60, 10), distComponent(1, 50, 55, 5)]
n_types = 2
env = poissonTasks(fleet, incoming_dist, n_types, MAX_TIME = 500)
num_steps = 1000

Learner = SMDP_QLambda(0.01, 0.99, 0.25, 0.5, env)

Learner.demo_episode(env, num_steps)

print(Learner.Q)