## Simulation Environment for Multi-Robot Task Assignment With a Central Task Assigner

This repository is our implementation for the project Multi-Robot Task Assignment With a Central Task Assigner. The goal of this project is to implement the environment for an Multi-Robot Task Assignment (MRTA) problem. We have chosen Python 3 with OpenAI Gym for this implementation.  

### Requirements

The project uses numpy and OpenAI Gym. They can be installed by running the following pip commands:
```
pip install numpy
pip install gym
```

### File Descriptions

There are three python files in this repository. These are explained below:
* poissonTasks.py: The implementation of our environment
* randomActionDemo.py: A simple task assigner that randomly assigns actions under in the poissonTasks environment.
* ModelFree_QLambdaDemo.py: Our implementation of a task assigner that uses the Q($\lambda$) algorithm for learning in the poissonTasks environment.

  

### Future Work

Due to Open AI Gym's handling of observation spaces, namely that it disallows observations of dynamic sizes, such as a list of unassigned task locations, the observation space in our proposed solution is incompatible with Gym's definition. Instead, we would have to create an observation space without using Gym's API. This is technically possible, as it is not strictly necessary to define action or observation spaces using Gym's API in order to inherit from the Env class, but it would mean that any sampling of such spaces would have to be implemented from scratch, which was a factor in the time constraints that we faced. By having a similar implementation for our action space, we could also make our state-action space representation much more efficient, since at present, many indices in the space represent invalid state-action pairs.
    
Beyond that, there are additional features that could be implemented for future work:

* Distribution components can be made more semantically specific if we allow the tasks to be of stochastic type, as well as including mixture distributions for location within a single component. Similar behavior is already possible by creating multiple distributions of the same location but of different task type or by creating multiple distributions of different locations but of the same task type. However, this similarity is inexact due to the dynamics of the underlying Poisson distributions being independent sequences.

* Currently, Open AI Gym's constraints on action space mean that with the central task assigner, we have a wait action for every robot, and all of these actions represent semantically identical behavior, while requiring far more visits to train. If we are manually creating an action space that is incompatible with Gym's API, then a priority would be to create a single action \emph{wait}, and possibly a dynamic action space in which robots are assigned to uniquely identifiable tasks.

* Currently, distribution components have fixed parameters for Poisson variable delays. However, since this parameter is an attribute, it could be feasible for us to parameterize the underlying distribution with respect to time, for a more complex problem.

* While we chose Poisson distributions for their useful sum property, which allows us to use the moment-generating function to easily estimate discounted rewards across sequential actions, nothing prevents the event schedule from tracking continuous time as well, similar to what is specified for a GSMDP. Therefore, we can modify event delays to draw from distributions besides Poisson, including continuous ones.

* We can also examine options for the partial observability of the state space, such as only allowing for indirect observation of unassigned tasks.

* Additionally, since we now have this environment, future work can explore various solutions to simulations of given parameters.

  
## License
This repository is licensed under [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
