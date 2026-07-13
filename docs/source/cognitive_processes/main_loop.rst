======================================
Cognitive Processes API Documentation
======================================

Here you can find a description of the Main Loop script, its specific topics, and the documentation of its methods.


++++++++++++++++++++
Cognitive Processes
++++++++++++++++++++

Base class for all cognitive processes in the architecture. It provides common functionalities and interfaces for different cognitive processes.

.. automodule:: cognitive_processes.cognitive_process
    :members:
    :show-inheritance:


+++++++++
Main loop
+++++++++

Python class that implements the Main Cognitive Loop of the system. The Main Loop is responsible for:

- Reading perceptions from the environment.
- Updating cognitive node activations.
- Selecting and executing policies.
- Evaluating rewards.
- Creating and managing cognitive nodes dynamically.
- Managing the world state and resetting it when necessary.

**Specific topics**

/{control_topic} => Publishes control messages with information about iterations, world state, and commands. The name of
the topic is defined in the experiment configuration file (usually main_loop/control).

/{episodes_topic} => Publishes episode data including perceptions, policies, actuations, and rewards for each execution cycle.
The name of the topic is defined in the experiment configuration file (usually main_loop/episodes).

.. automodule:: cognitive_processes.main_loop
    :members:
    :show-inheritance:


+++++++++++++
Deliberation
+++++++++++++

Python class that implements the Deliberation process of the cognitive architecture. The Deliberation process is responsible for selecting the most appropriate action or policy to execute based on the current state of the system, the predicted states and the current goal.
It is instantiated in the Utility Models.

.. automodule:: cognitive_processes.deliberation
    :members:
    :show-inheritance:

