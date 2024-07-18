This is part of the e-MDB architecture documentation. Main page [here](https://docs.pillar-robots.eu/en/latest/).

# e-MDB reference implementation for cognitive processes

This [repository](https://github.com/pillar-robots/emdb_cognitive_processes_gii) includes software packages for the cognitive process that manipulate the knowledge elements in the long-term memory (LTM) of the software implementation of the e-MDB cognitive architecture developed under the [PILLAR Robots project](https://pillar-robots.eu/).

At the moment, there is just one process, which we have called "main cognitive loop". This process reads perceptions, calculate an activation value for each knowledge nugget (node) depending on those perceptions and the activation of the connected nodes (it detects the relevant contexts), and, finally, it executes an action [[1]](#1).

More cognitive processes will come, as those related to learning processes, necessary to generate and adapt world models, utility models, and policies.

Therefore, there is just on ROS 2 package right now that includes the implementation of the "main cognitive loop".

In the section [API Documentation](cognitive_processes/main_loop.rst) you can find the documentation of the main cognitive loop module.


<a id="1">[1]</a> 
Duro, R. J., Becerra, J. A., Monroy, J., & Bellas, F. (2019). Perceptual generalization and context in a network memory inspired long-term memory for artificial cognition. _International Journal of Neural Systems, 29(06)_, 1850053.

```{toctree}
:caption: e-MDB Cognitive processes
:hidden:
:glob:

cognitive_processes/*
```