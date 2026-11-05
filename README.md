# e-MDB reference implementation for cognitive processes

This repository includes software packages for the cognitive processes that manipulate the knowledge elements in the long-term memory (LTM) of the software implementation of the e-MDB cognitive architecture developed under the [PILLAR Robots project](https://pillar-robots.eu/).

These are the available processes, in the *cognitive_processes* package:

- **MainLoop**: The main cognitive loop. It reads perceptions, calculates an activation value for each knowledge nugget (node) depending on those perceptions and the activation of the connected nodes (it detects the relevant contexts), and, finally, it executes an action [[1]](#1). It also handles goals, episodes and the updates of the LTM.
- **MainLoopLight**: A lightweight version of the main loop that only performs high level decision.
- **Deliberation**: Instantiated inside of the Utility Models, it selects actions using the predictions of the World Models and the Utility Models.
- **Reaction**: Instantiated inside of the learned policies, it executes a closed-loop reactive policy.

Their messages and services are defined in the *cognitive_processes_interfaces* package.

<a id="1">[1]</a> 
Duro, R. J., Becerra, J. A., Monroy, J., & Bellas, F. (2019). Perceptual generalization and context in a network memory inspired long-term memory for artificial cognition. _International Journal of Neural Systems, 29(06)_, 1850053.

## Dependencies

- ROS 2 Jazzy. Other distributions may work, but without the `EventsExecutor`, which is available from Jazzy on, the architecture uses much more CPU.
- Numpy, lower than version 2
- Scipy and xarray

## Installation

Clone this repository in a ROS workspace and build it with colcon:

```
colcon build --symlink-install
```

To get full functionality, these repositories are also required in the workspace:

- [_emdb_core_](https://github.com/GII/emdb_core). Core of the cognitive architecture.
- [_emdb_cognitive_nodes_gii_](https://github.com/GII/emdb_cognitive_nodes_gii). Reference implementation for the main cognitive nodes.
- [_emdb_discrete_event_simulator_gii_](https://github.com/GII/emdb_discrete_event_simulator_gii). Implementation of the discrete event simulators used in many experiments.
- [_emdb_experiments_gii_](https://github.com/GII/emdb_experiments_gii). Configuration files and launch files of the experiments.

## Execution

Source the ROS workspace and launch an experiment:

```
source install/setup.bash
ros2 launch experiments put_object_in_box_launch.py
```

The experiment is configured through its YAML file, in the *experiments* package of [_emdb_experiments_gii_](https://github.com/GII/emdb_experiments_gii). The output files are selected in that file too, and they are created in the current directory with a number before the extension, e.g. `goodness_0.txt`.

To follow the learning process in real time:

```
tail -f goodness_0.txt
```

## Documentation

You can find the API documentation of the processes in the [PILLAR Robots official documentation](https://docs.pillar-robots.eu/projects/emdb_cognitive_processes_gii/en/latest/), and how the architecture runs and how it is configured in the documentation of [_emdb_core_](https://docs.pillar-robots.eu/projects/emdb_core/en/latest/).
