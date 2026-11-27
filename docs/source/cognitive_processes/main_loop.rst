======================================
Cognitive Processes API Documentation
======================================

Here you can find a description of the cognitive processes, their specific topics and services, and the documentation of their methods.


++++++++++++++++++++
Cognitive Processes
++++++++++++++++++++

Base class for all cognitive processes in the architecture. It provides common functionalities and interfaces for different cognitive processes.

A process runs in its own thread: first setup() and then run(). The metaclass StartsAfterConstruction starts this thread after the constructor of the class
being instantiated, so subclasses don't have to start it. The event ready is set when setup() has finished, and call_when_ready() schedules a callback in the executor of a node
once it is set, without waiting.

From this thread, processes call services with own_client() and call_services(): the clients belong to the node of the process, several requests can be sent at once,
and only the thread of the process waits for the responses, so the executor goes on running callbacks meanwhile.

**Specific services**

/{name}/pause => Pauses or resumes the process, or asks it to stop.

**Specific topics**

/state => Subscribes to the changes in the LTM. The thread of the process applies them while waiting for activations.

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

/{episodes_topic} => Publishes episode data including perceptions, policies, actuations, and rewards for each execution cycle, as a core_interfaces/Container message.
It also subscribes to this topic, since policies and other processes publish episodes in it too, and it writes them to the output files from the thread of the loop.
The name of the topic is defined in the experiment configuration file (usually main_loop/episodes).

**Services used**

/{world_reset_service} => Resets the world. The name of the service is defined in the experiment configuration file (Control: world_reset_service).
The main loop also publishes a reset_world command in the control topic.

The environment variable EMDB_PERF_FILE=<path> makes the main loop record the wall-clock time at the end of each iteration in that file.

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


++++++++
Reaction
++++++++

Python class that implements the Reaction process of the cognitive architecture, which executes a closed-loop reactive policy using a learned model, instead of deliberating about candidate actions.
It is instantiated in the learned policies (PolicyLearned).

.. automodule:: cognitive_processes.reaction
    :members:
    :show-inheritance:

