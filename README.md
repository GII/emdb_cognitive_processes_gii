# e-MDB reference implementation for cognitive processes

Note:

***WORK IN PROGRESS***

The original repository has been split in 5 and we are refactoring everything, please, be patient while we move and rename everything.

These are the cognitive architecture repositories for PILLAR and their content:

- _emdb_core_. Essential elements of the cognitive architecture. These are necessary to run an experiment using the cognitive architecture.
- _emdb_cognitive_nodes_gii_. Reference implementation for the main cognitive nodes.
- _emdb_cognitive_processes_gii_. Reference implementation for the main cognitive processes.
- _emdb_discrete_event_simulator_gii_. Implementation of a discrete event simulator used in many experiments.
- _emdb_experiments_gii_. Configuration files for experiments.

This repository includes software packages for the cognitive process that manipulate the knowledge elements in the long-term memory (LTM).

At the moment, there is just one process, which we have called "main cognitive loop". This process reads perceptions, calculate an activation value for each knowledge nugget (node) depending on those perceptions and the activation of the connected nodes (it detects the relevant contexts), and, finally, it executes an action [[1]](#1)

More cognitive processes will come, as those related to learning processes, necessary to generate and adapt world models, utility models, and policies.

Therefore, there is just on ROS 2 package right now that includes the implementation of the "main cognitive loop".

<a id="1">[1]</a> 
Duro, R. J., Becerra, J. A., Monroy, J., & Bellas, F. (2019). Perceptual generalization and context in a network memory inspired long-term memory for artificial cognition. _International Journal of Neural Systems, 29(06)_, 1850053.
