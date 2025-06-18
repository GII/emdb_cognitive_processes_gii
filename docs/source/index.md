This is part of the e-MDB architecture documentation. Main page [here](https://docs.pillar-robots.eu/en/latest/).

# e-MDB reference implementation for cognitive processes

This [repository](https://github.com/pillar-robots/emdb_cognitive_processes_gii) includes software packages for the cognitive process that manipulate the knowledge elements in the Long-Term Memory (LTM) of the software implementation of the e-MDB cognitive architecture developed under the [PILLAR Robots project](https://pillar-robots.eu/).

At the moment, there is just one process, which we have called **Main Cognitive Loop**, or simply Main Loop. This process reads perceptions, calculate an activation value for each knowledge nugget (node) depending on those perceptions and the activation of the connected nodes, and, finally, it executes an action [^1].

The behavior of the Main Loop can be seen in the following diagram:

<div style="width:100%; margin:auto; text-align:center;">

![Main Loop behavior](images/main_loop_behavior.svg)

*Main Loop behavior*
</div>

<!-- More cognitive processes will come, as those related to learning processes, necessary to generate and adapt world models, utility models, and policies. -->

Therefore, there is just on ROS 2 package right now that includes the implementation of the **Main Cognitive Loop**.

In the section [Main Loop API Documentation](cognitive_processes/main_loop.rst) you can find the API documentation of this process.

[^1]: Duro, R. J., Becerra, J. A., Monroy, J., & Bellas, F. (2019). Perceptual generalization and context in a network memory inspired long-term memory for artificial cognition. _International Journal of Neural Systems, 29(06)_, 1850053.

```{toctree}
:caption: e-MDB Cognitive processes
:hidden:
:glob:

cognitive_processes/*
```