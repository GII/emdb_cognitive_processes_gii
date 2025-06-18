===========================
Main Loop API Documentation
===========================

Here you can find a description of the Main Loop script, its specific topics, and the documentation of its methods.

+++++++++
Episode
+++++++++

The Episode class is used as the Short Term Memory (STM) by the Main Loop of the cognitive architecture. 

.. autoclass:: cognitive_processes.main_loop.Episode
    :members:
    :show-inheritance:

**Attributes**

* **old_perception** (dict): Previous perceptual data from the environment.
* **old_ltm_state** (dict): Previous state of the Long Term Memory.
* **policy** (str): Name of the policy being executed.
* **actuation** (dict): Values of actuators for the executed policy
* **perception** (dict): Current perceptual data from the environment.
* **ltm_state** (dict): Current state of the Long Term Memory.
* **reward_list** (dict): Dictionary mapping goals to their reward values.


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

.. autoclass:: cognitive_processes.main_loop.MainLoop
    :members:
    :show-inheritance:
