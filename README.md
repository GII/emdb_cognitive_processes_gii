# e-MDB reference implementation for cognitive processes

This repository includes software packages for the cognitive process that manipulate the knowledge elements in the long-term memory (LTM) of the software implementation of the e-MDB cognitive architecture developed under the [PILLAR Robots project](https://pillar-robots.eu/)..

At the moment, there is just one process, which we have called "main cognitive loop". This process reads perceptions, calculate an activation value for each knowledge nugget (node) depending on those perceptions and the activation of the connected nodes (it detects the relevant contexts), and, finally, it executes an action [[1]](#1)

More cognitive processes will come, as those related to learning processes, necessary to generate and adapt world models, utility models, and policies.

Therefore, there is just on ROS 2 package right now that includes the implementation of the "main cognitive loop".

For more information about the cognitive architecture design, you can visit the [emdb_core](https://github.com/GII/emdb_core?tab=readme-ov-file#design) repository or the [PILLAR Robots official website](https://pillar-robots.eu/).

<a id="1">[1]</a> 
Duro, R. J., Becerra, J. A., Monroy, J., & Bellas, F. (2019). Perceptual generalization and context in a network memory inspired long-term memory for artificial cognition. _International Journal of Neural Systems, 29(06)_, 1850053.

## Table of Contents

- **[Dependencies](#dependencies)**
- **[Installation](#installation)**
- **[Configurate an experiment](#configurate-an-experiment)**
- **[Execution](#execution)**
- **[Results](#results)**

## Dependencies

These are the dependencies required to use this repository of the e-MDB cognitive architecture software:

- ROS 2 Humble
- Numpy 1.24.3
  
Other versions could work, but the indicated ones have proven to be functional.

## Installation

To install this package, it's necessary to clone this repository in a ROS workspace and build it with colcon.

```
colcon build --symlink-install
```
This respository only constitutes the main loop, the reference cognitive process of the e-MDB cognitive architecture. To get full functionality, it's required to add to the ROS workspace, at least, the [emdb_core](https://github.com/GII/emdb_core) repository, that constitutes the base of the architecture, and other packages that include the cognitive nodes, the experiment configuration and the interface that connects the architecture with a real or a simulated environment. Therefore, to use the first version of the architecture implemented by GII, these repositories need to be cloned into the workspace:
- [_emdb_core_]([https://github.com/GII/emdb_cognitive_nodes_gii](https://github.com/GII/emdb_core)). Core of the cognitive architecture.
- [_emdb_cognitive_nodes_gii_](https://github.com/GII/emdb_cognitive_nodes_gii). Reference implementation for the main cognitive nodes.
- [_emdb_discrete_event_simulator_gii_](https://github.com/GII/emdb_discrete_event_simulator_gii). Implementation of a discrete event simulator used in many experiments.
- [_emdb_experiments_gii_](https://github.com/GII/emdb_experiments_gii). Configuration files for experiments.

In these repositories is included an example experiment with the discrete event simulator in which the Policies, the Goal and the World Model are defined in the beginning, the objective being to create the corresponding PNodes and CNodes, which allow the Goal to be achieved effectively by the simulated robot. 

The Goal, called ObjectInBoxStandalone, consists of introducing a cylinder into a box correctly. For that, the robot can use, in a World Model called GripperAndLowFriction, the following policies:
- Grasp object: use one of the two grippers to grasp an object
- Grasp object with two hands: use both arms to grasp an object between their ends
- Change hands: move object from one gripper to the other 
- Sweep object: sweep an object to the central line of the table
- Ask nicely: ask experimenter, simulated in this case, to bring something to within reach
- Put object with robot: deposit an object to close to the robot base
- Put object in box: place an object in a receptacle
- Throw: throw an object to a position
  
The reward obtained could be 0.2, 0.3 or 0.6 if the robot with its action improves its situation to get the final goal. Finally, when the cylinder is introduced into the box, the reward obtained is 1.0. Thus, at the end of the experiment, seven PNodes and CNodes should be created, one per policy, except Put object with robot, which doesn't lead to any reward.

## Configurate an experiment

It's possible to configure the behavior of the main loop editing the experiment configuration file, stored in the [_emdb_experiments_gii_](https://github.com/GII/emdb_experiments_gii) repository (experiments/default_experiment.yaml), or one created by oneself:
```
Experiment:
    name: main_loop
    class_name: cognitive_processes.main_loop.MainLoop
    new_executor: False
    threads: 2
    parameters: 
        iterations: 6000
        trials: 50
        subgoals: False
Control:
    id: ltm_simulator
    control_topic: /main_loop/control
    control_msg: core_interfaces.msg.ControlMsg
    executed_policy_topic: /mdb/baxter/executed_policy
    executed_policy_msg: std_msgs.msg.String
```
As we can see, we can configure the number of iterations of the experiment, the number of trials that the robot will make before resetting the simulated world or the existence or not of subgoals. Also, it's possible to configure the param *new_executor* as True, so this will indicate to the Commander node that it has to create a new and dedicated execution node for each cognitive node that is created, with the number of threads indicated (2 in this case), although this has more to do with the [core](https://github.com/GII/emdb_core/blob/main/README.md#configurate-an-experiment) of the architecture.

Finally, there is the control part, which in the example case acts as a middleware between the cognitive architecture and the discrete event simulator, controlling the main loop the communications between both parts. In this case, the main loop publishes to the simulator some commands, such as the *reset world*, the current iteration and the active world model. Also, it indicates to the simulator where the policy to execute will be published. This can be adapted to another simulator or a real robot case.

## Execution

To execute the example experiment or another launch file, it's essential to source the ROS workspace:
```
source install/setup.bash
```
Afterwards, the experiment can be launched:
```
ros2 launch core example_launch.py
```
Once executed, it is possible to see the logs in the terminal, being able to follow the behavior of the experiment in real time.


## Results

Executing the example experiment, it will create two files by default: goodness.txt and pnodes_success.txt. 

In the first one, it is possible to observe important information, such as the policy executed and the reward obtained per iteration. It is possible to observe the learning process by seeing this file in real time with the following command:
```
tail -f goodness.txt
```
In the second file, it's possible to see the activation of the PNodes and if it was a point (True) or an anti-point (False).

When the execution is finished, it's possible to obtain statistics about reward and PNodes activations per 100 iterations by using the scripts available in the scripts directory of the core package (emdb_core/core/scripts):
```
python3 $ROS_WORKSPACE/src/emdb_core/core/scripts/generate_grouped_statistics -n 100 -f goodness.txt > goodness_grouped_statistics.csv

python3 $ROS_WORKSPACE/src/emdb_core/core/scripts/generate_grouped_success_statistics -n 100 -f pnodes_success.txt > pnodes_grouped_statistics.csv
```
To use these scripts it's necessary to have installed python-magic 0.4.27 dependency.

By plotting the data of these final files, it is possible to obtain a visual interpretation of the learning of the cognitive architecture.
