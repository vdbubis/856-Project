import gym
from gym import spaces
import numpy as np
import bisect as bs
from itertools import repeat

rng = np.random.default_rng()

#Event schedule class
#An event is a tuple of either 'n' or 'c', followed by a non-negative integer
# ('n', N) is the creation of a new task from component N
# ('c', N) is robot N completing its task

class eventSchedule():
    def __init__(self, MAX_DELAY): #Event schedule must have an upper limit on the delay to ensure observation is bounded
        self.delays = np.array([], dtype='int')
        self.events = []
        self.MAX_DELAY = MAX_DELAY
        
    def add(self, delay, event): #Insort an event for both the np array and the list of events
        delay = np.clip(delay, 0, self.MAX_DELAY)
        index = bs.bisect(self.delays, delay)
        self.delays = np.insert(self.delays, index, delay)
        self.events.insert(index, event)
        
    def pop(self): #Return an event and time delay, update remaining delays
        delay = self.delays[0]
        self.delays = self.delays[1:] - delay
        event = self.events.pop(0)
        return delay, event
    
#The incoming distribution is a mixture of components. Each component in the basic version is just a type of task.
#Once locations are added, the component also includes a distribution for random location
#Future work: Include multiple types of tasks within a singe component

class distComponent():
    def __init__(self, task_type, difficulty, mean_delay, reward=1): #There is default reward, don't abuse this
        self.task_type = task_type
        self.difficulty = difficulty
        self.mean_delay = mean_delay
        self.reward = reward #FUTURE WORK: Reward can be replaced with a mean parameter, and rolled on get
        
    def getDelay(self):
        return rng.poisson(self.mean_delay, 1)
        
    def getTask(self):
        #With locations, we would be drawing from a distribution in this function
        return Task(self.task_type, self.difficulty, self.reward)
    
#A task has type, difficulty, and reward. In the more advanced version, it has a location as well.
class Task():
    def __init__(self, task_type, difficulty, reward):
        self.task_type = task_type
        self.difficulty = difficulty #Task difficulty
        self.reward = reward
        #self.location = location
        
    def __repr__(self):
        return "(" + str(self.task_type) + ", " + str(self.difficulty) + ", " + str(self.reward) + ")"
    
#A robot has fluencies as parameters, and in a state space tracks which task is assigned to it, if any.
#When locations are added, robots will also have their average velocity as a parameter and their current location as a state
class Robot():
    def __init__(self, fluencies): #Assigned task and fluencies. In future: average velocity and starting location
        self.fluencies = fluencies
        self.assigned = None #This should never be assigned to non-None at start of environment
        #Average velocity
        #Starting location
        
    def __repr__(self):
        return "Fluencies: " + str(self.fluencies) + ", assigned: " + str(self.assigned) 
        
    def assignTask(self, task):
        self.assigned = task
        #NOTE: We need a guard against fluency of 0
        return rng.poisson(task.difficulty / self.fluencies[task.task_type], 1)
    
    def completeTask(self): #Return reward and clear assignment
        reward = self.assigned.reward
        self.assigned = None
        return reward
        
    #In the future, we need to add a function for abandoning a task, which returns it to the unassigned tasks table

#This is the environment itself; extensive documentation is in our report.
class poissonTasks(gym.Env):
    def __init__(self, fleet, incoming_distribution, n_types, *, MAX_DELAY=1000, MAX_TASKS=1000, MAX_REWARD=1000, MAX_TIME=None):
        super(poissonTasks, self).__init__() #Python 2 invocation, to be safe
        
        self.fleet = fleet
        self.n_robots = len(fleet)
        self.incoming_distribution = incoming_distribution
        self.n_types = n_types
        
        self.MAX_DELAY = MAX_DELAY
        self.MAX_TASKS = MAX_TASKS
        self.MAX_REWARD = MAX_REWARD
        self.MAX_TIME = MAX_TIME
        
        self.reward_range = (0, MAX_REWARD)
        #Format of actions is (robot, task_type), representing robot assigned to that task.
        #If task_type is equal to n_types, then it is the equivalent of the 'wait' action. This is also why dimension is n_types + 1
        self.action_space = spaces.Tuple((spaces.Discrete(self.n_robots), spaces.Discrete(self.n_types + 1)))
        #"Wait" does not have an explicit formulation; it is implicit for non-legal actions
        #Format of observations is (delay, (fleet availability), (pending tasks by type))
        self.observation_space = spaces.Tuple((spaces.Discrete(self.MAX_DELAY),
                                             spaces.Tuple(tuple(repeat(spaces.Discrete(2), self.n_robots))),
                                             spaces.Tuple(tuple(repeat(spaces.Discrete(self.MAX_TASKS), self.n_types)))))
        
    def reset(self):
        for i in range(self.n_robots):
            self.fleet[i].assigned = None
        self.fleet_status = list(repeat(1, self.n_robots)) #1 means available, 0 means a task is in progress    
        
        self.unassigned_tasks = list(repeat([], self.n_types))
        self.unassigned_counts = list(repeat(0, self.n_types))
        
        #Sum of fleet_status is the number of idle robots
        #Sum of unassigned_counts is total number of unassigned tasks
        
        self.event_schedule = eventSchedule(self.MAX_DELAY)
        for i in range(len(self.incoming_distribution)):
            delay = self.incoming_distribution[i].getDelay()
            self.event_schedule.add(delay, ('n', i))
        self.time_elapsed = 0
        
        return self._current_observation()
       
    
    def _take_action(self, action):
        robot, task_type = action
        waiting = True #If we effectively take no action, we are waiting.
        
        #If the robot is available and there is such a task
        if task_type < self.n_types: #task_type == self.n_types means intentionally waiting
            if self.fleet_status[robot] and self.unassigned_counts[task_type] > 0:
                #Oldest task of that type is assigned
                task = self.unassigned_tasks[task_type].pop(0) #Remove task from unassigned pool
                self.unassigned_counts[task_type] = self.unassigned_counts[task_type] - 1 #Remove from count
            
                task_delay = self.fleet[robot].assignTask(task) #Assign the task
                self.fleet_status[robot] = 0 #Robot no longer available
            
                self.event_schedule.add(task_delay, ('c', action[0])) #Event of task completion added to schedule
                waiting = False #Legal action; we are no longer waiting
    
        return waiting
    
    def _current_observation(self): #observation without time passing
        return (0, tuple(self.fleet_status), tuple(self.unassigned_counts))
        
    def _next_observation(self): #go to next event, then observe
        
        reward = 0 #If nothing else happens, there is no reward
        step_delay, event = self.event_schedule.pop()
        self.time_elapsed = self.time_elapsed + step_delay
        
        if event[0] == 'n': #'N'ew task from incoming distribution; number specifies distribution component
            #First, roll the task and add it to the unassigned pool
            new_task = self.incoming_distribution[event[1]].getTask()
            #Effectively append; major bug source due to how aliasing works with nested lists
            self.unassigned_tasks[new_task.task_type] = self.unassigned_tasks[new_task.task_type] + [new_task] 
            self.unassigned_counts[new_task.task_type] = self.unassigned_counts[new_task.task_type] + 1
            
            #Next, schedule the next incoming task
            task_delay = self.incoming_distribution[event[1]].getDelay()
            self.event_schedule.add(task_delay, ('n', event[1]))
            #Possible TODO: Negative reward on incoming task in some conditions
            
        elif event[0] == 'c': #Robot 'C'ompleted assigned task; number specifies which robot completed task
            #First, the robot completes the task and gets reward
            reward = self.fleet[event[1]].completeTask()
            self.fleet_status[event[1]] = 1 #This robot is also idle now
            
        #else: #TODO: Add in error for invalid event
            
        return (step_delay, tuple(self.fleet_status), tuple(self.unassigned_counts)), reward
    
    def step(self, action):
        waiting = self._take_action(action)
        
        if waiting or sum(self.fleet_status) == 0 or sum(self.unassigned_counts) == 0:
            #We are waiting for the next event on the schedule
            #We might need to halt the simulation due to number of tasks exceeding the maximum
            obs, reward = self._next_observation()
            reward = np.clip(reward, 0, self.MAX_REWARD) #We ensure reward is within bounds
            
            done = (sum(self.unassigned_counts) >= self.MAX_TASKS)
            
            if self.MAX_TIME is not None:
                #We use this conditional assignment instead assigning the boolean to avoid overwriting our evaluation of MAX_TASKS
                if self.time_elapsed >= self.MAX_TIME: 
                    done = True
            
            return obs, reward, done, {}
        
        #We are not waiting; return observations with no reward, and we cannot be done due to no time passing or added tasks
        obs = self._current_observation()
        return obs, 0, False, {}
    
    def render(self):
        print("event schedule:", self.event_schedule.events)
        print("schedule delay:", self.event_schedule.delays)
        print("fleet:", self.fleet)
        print("unassigned tasks:", self.unassigned_tasks)
        print("===")