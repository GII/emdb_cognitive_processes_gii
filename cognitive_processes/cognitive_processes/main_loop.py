import sys
import rclpy
from rclpy.node import Node
from operator import attrgetter
import random
import yaml
import threading
import numpy
from copy import copy

from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from core.service_client import ServiceClient
from cognitive_node_interfaces.srv import (
    Execute,
    GetActivation,
    GetReward,
    GetInformation,
    AddPoint,
)
from cognitive_node_interfaces.msg import Perception
from core_interfaces.srv import GetNodeFromLTM, CreateNode
from core_interfaces.msg import ControlMsg

from core.utils import perception_dict_to_msg, perception_msg_to_dict, class_from_classname


class MainLoop(Node):
    """
    MainLoop class for managing the main loop of the system.

    This class handles the core logic of the system, including reading perceptions,
    selecting policies, and executing policies.
    """

    def __init__(self, name, **params):
        """
        Constructor for the MainLoop class.

        Initializes the MainLoop node and starts the main loop execution.
        """
        super().__init__(name)
        self.iteration = 0
        self.iterations = 0
        self.trials = 0
        self.trial = 0
        self.period = 1
        self.current_policy = None
        self.paused = False
        self.stop = False
        self.LTM_id = ""  # id of LTM currently being run by cognitive loop
        self.LTM_cache = (
            []
        )  # List of dics, like [{"name": "pnode1", "node_type": "PNode", "activation": 0.0}, {"name": "cnode1", "node_type": "CNode", "activation": 0.0}]
        self.default_class = {}
        self.perception_suscribers = {}
        self.perception_cache = {}
        self.reward_threshold = 0.9
        self.subgoals = False
        self.policies_to_test = []
        self.files = []
        self.default_class = {}
        self.random_seed = 0
        self.current_reward = 0
        self.current_world = None
        self.n_cnodes = 0
        self.sensorial_changes_val = False
        self.pnodes_success = {}

        self.cbgroup_perception = MutuallyExclusiveCallbackGroup()
        self.cbgroup_server = MutuallyExclusiveCallbackGroup()
        self.cbgroup_client = MutuallyExclusiveCallbackGroup()
        self.cbgroup_loop = MutuallyExclusiveCallbackGroup()

        self.node_clients = (
            {}
        )  # Keys are service name, values are service client object e.g. {'cognitive_node/policy0/get_activation: "Object: core.ServiceClient(Async)"'}

        self.control_publisher = self.create_publisher(ControlMsg, "main_loop/control", 10)
        # self.run_subscriber= self.create_subscription(Empty, 'main_loop/run', self.run, 1, callback_group=self.cbgroup_loop) #TODO: Check if it is possible to execute the loop as a callback

        for key, value in params.items():
            self.get_logger().debug("Setting atribute: " + str(key) + " with value: " + str(value))
            setattr(self, key, value)

        # Read LTM and configure perceptions
        self.setup()

        loop_thread = threading.Thread(target=self.run, daemon=True)
        loop_thread.start()

    def setup(self):
        self.rng = numpy.random.default_rng(self.random_seed)
        self.read_ltm()
        self.configure_perceptions()
        self.setup_files()
        self.setup_connectors()

    def read_ltm(self):
        """
        Makes an empty call for the LTM get_node service which returns all the nodes
        stored in the LTM. Then, a LTM cache dictionary is populated with the data.

        """
        self.get_logger().info("Reading nodes from LTM: " + self.LTM_id + "...")

        # Call get_node service from LTM
        service_name = "/" + str(self.LTM_id) + "/get_node"
        request = ""
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(GetNodeFromLTM, service_name)
        ltm_response = self.node_clients[service_name].send_request(name=request)

        # Process data string
        ltm_cache = yaml.safe_load(ltm_response.data)

        self.get_logger().debug(f"LTM Dump: {str(ltm_cache)}")

        for node_type in ltm_cache.keys():
            for node in ltm_cache[node_type].keys():
                self.LTM_cache.append(
                    {
                        "name": node,
                        "node_type": node_type,
                        "activation": ltm_cache[node_type][node]["activation"],
                    }
                )

        self.get_logger().debug(f"LTM Cache: {str(self.LTM_cache)}")

    def configure_perceptions(
        self,
    ):  # TODO(efallash): Add condition so that perceptions that are already included do not create a new suscription. For the case that new perceptions are added to the LTM and only some perceptions need to be configured
        """
        Reads the LTM cache and populates the perception subscribers and perception cache dictionaries.

        """
        self.get_logger().info("Configuring perceptions...")
        perceptions = [
            perception for perception in self.LTM_cache if perception["node_type"] == "Perception"
        ]

        self.get_logger().debug(f"Perception list: {str(perceptions)}")

        for perception_dict in perceptions:
            perception = perception_dict["name"]

            subscriber = self.create_subscription(
                Perception,
                f"/perception/{perception}/value",
                self.receive_perception_callback,
                1,
                callback_group=self.cbgroup_perception,
            )
            self.get_logger().debug(f"Subscription to: /perception/{perception}/value created")
            self.perception_suscribers[perception] = subscriber
            self.perception_cache[perception] = {}
            self.perception_cache[perception]["flag"] = threading.Event()
        # TODO check that all perceptions in the cache still exist in the LTM and destroy suscriptions that are no longer used
        self.get_logger().debug(f"Perception cache: {self.perception_cache}")

    def setup_files(self):
        if hasattr(self, "Files"):
            self.get_logger().info("Files detected, loading files...")
            for file_item in self.Files:
                self.add_file(file_item)
        else:
            self.get_logger().info("No files detected...")

    def add_file(self, file_item):
        """Process a file entry (create the corresponding object) in the configuration."""
        if "data" in file_item:
            new_file = class_from_classname(file_item["class"])(
                ident=file_item["id"],
                file_name=file_item["file"],
                data=file_item["data"],
                node=self,
            )
        else:
            new_file = class_from_classname(file_item["class"])(
                ident=file_item["id"], file_name=file_item["file"], node=self
            )
        self.files.append(new_file)

    def setup_connectors(self):
        if hasattr(self, "Connectors"):
            for connector in self.Connectors:
                self.default_class[connector["data"]] = connector.get("default_class")

    def publish_iteration(self):
        """
        Method for publishing execution data in the control topic in each iteration.

        """
        msg = ControlMsg()
        msg.command = ""
        msg.world = self.current_world
        msg.iteration = self.iteration
        self.control_publisher.publish(msg)

    def read_perceptions(self):
        """
        Reads the perception cache and returns the latest perception.

        This method iterates the perception cache dictionary. For each
        perception waits for the flag that signals that the value has
        been updated. Then, the value is copied in the sensing dictionary.

        When the whole cache is processed, the sensing dictionary is returned.

        :return: Latest sensing
        :rtype: dict
        """

        self.get_logger().info("Reading perceptions...")

        sensing = {}

        for (
            sensor
        ) in self.perception_cache.keys():  # TODO: Consider perception activation when reading
            self.perception_cache[sensor]["flag"].wait()
            sensing[sensor] = copy(self.perception_cache[sensor]["data"])
            self.perception_cache[sensor]["flag"].clear()
            self.get_logger().debug("Processing perception: " + str(sensor))

        self.get_logger().debug("DEBUG Read Perceptions: " + str(sensing))
        return sensing

    def receive_perception_callback(self, msg):
        """
        Receives a message from a perception value topic, processes the
        message and copies it to the perception cache. Finally sets the
        flag to signal that the value has been updated.

        :param msg: Message that contains the perception.
        :type msg: cognitive_node_interfaces.msg.Perception
        """
        perception_dict = perception_msg_to_dict(msg)

        for sensor in perception_dict.keys():
            if sensor in self.perception_cache:
                self.perception_cache[sensor]["data"] = copy(perception_dict[sensor])
                self.perception_cache[sensor]["flag"].set()
                self.get_logger().debug(
                    f'Receiving perception: {sensor} {self.perception_cache[sensor]["data"]} ...'
                )
            else:
                self.get_logger().error(
                    "Received sensor not registered in local perception cache!!!"
                )

    def ltm_change_callback(self):
        """
        PENDING METHOD: This method will read changes made in the LTM
        external to the cognitive process and update the LTM and perception
        caches accordingly.

        """
        self.get_logger().info("Processing change from LTM...")  # TODO(efallash): implement
        pass

    def select_policy(self, sensing):
        """
        Selects the policy with the higher activation based on the current sensing.

        :param sensing: The current sensing.
        :type sensing: dict
        :return: The selected policy.
        :rtype: str
        """
        self.update_activations(sensing)

        policy_activations = {}
        for node in self.LTM_cache:
            if node["node_type"] == "Policy":
                policy_activations[node["name"]] = node["activation"]

        self.get_logger().info("Select_policy - Activations: " + str(policy_activations))

        policy = max(zip(policy_activations.values(), policy_activations.keys()))[1]

        if not policy_activations[policy]:
            policy = self.random_policy()

        self.get_logger().info(f"Selecting a policy => {policy} ({policy_activations[policy]})")

        return policy

    def random_policy(self):
        """
        Selects a random policy.

        :return: The selected policy
        :rtype: str
        """

        if self.policies_to_test == []:
            self.policies_to_test = [
                node["name"] for node in self.LTM_cache if node["node_type"] == "Policy"
            ]

        policy = numpy.random.choice(self.policies_to_test)

        return policy

    def update_policies_to_test(self, policy=None):
        """
        Maintenance tasks on the pool of policies used to choose one randomly when needed.

        When no policy is passed, the method will fill the policies to test list with all
        available policies in the LTM.

        When a policy is passed, the policy will be removed from the policies to test list.

        :param policy: Optional policy name, defaults to None
        :type policy: str, optional
        """

        if policy:
            if policy in self.policies_to_test:
                self.policies_to_test.remove(policy)
        else:
            self.policies_to_test = [
                node["name"] for node in self.LTM_cache if node["node_type"] == "Policy"
            ]

    def sensorial_changes(self, sensing, old_sensing):
        """
        Return false if all perceptions have the same value as the previous step. True otherwise.

        :param sensing: Sensing in the current iteration.
        :type sensing: dict
        :param old_sensing: Sensing in the last iteration.
        :type old_sensing: dict
        :return: Boolean that indicates if there is a sensorial change.
        :rtype: bool
        """

        for sensor in sensing:
            for perception, perception_old in zip(sensing[sensor], old_sensing[sensor]):
                if isinstance(perception, dict):
                    for attribute in perception:
                        difference = abs(perception[attribute] - perception_old[attribute])
                        if difference > 0.01:
                            self.get_logger().debug("Sensorial change detected")
                            self.sensorial_changes_val = True
                            return True
                else:
                    if abs(perception[0] - perception_old[0]) > 0.01:
                        self.get_logger().debug("Sensorial change detected")
                        self.sensorial_changes_val = True
                        return True
        self.get_logger().debug("No sensorial change detected")
        self.sensorial_changes_val = False
        return False

    def update_activations(self, perception, new_sensings=True):
        """
        Requests a new activation to all nodes in the LTM Cache.

        :param perception: Perception used to calculate the activation
        :type perception: dict
        :param new_sensings: Indicates if a sensing change has ocurred, defaults to True
        :type new_sensings: bool, optional
        """

        self.get_logger().info("Updating activations...")

        for node in self.LTM_cache:
            if node["node_type"] == "PNode":
                if new_sensings:
                    activation = self.request_activation(node["name"], perception)
                    node["activation"] = activation if activation > 0.1 else 0
            elif node["node_type"] == "CNode":
                pass  # CNode activations are handled by the corresponding policies
            else:
                activation = self.request_activation(node["name"], perception)
                node["activation"] = activation if activation > 0.1 else 0

        self.get_logger().debug("DEBUG - LTM CACHE:" + str(self.LTM_cache))

    def request_activation(self, name, sensing):
        """
        This method calls the service to get the activation of a node.

        :param name: Name of the node.
        :type name: str
        :param sensing: Sensing used to calculate the activation.
        :type sensing: dict
        :return: Activation value returned
        :rtype: float
        """

        service_name = "cognitive_node/" + str(name) + "/get_activation"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(GetActivation, service_name)
        perception = perception_dict_to_msg(sensing)
        activation = self.node_clients[service_name].send_request(perception=perception)
        return activation.activation

    def request_neighbors(self, name):
        """
        This method calls the service to get the neighbors of a node.

        :param name: Node name.
        :type name: str
        :return: List of dictionaries with the information of each neighbor of the node. [{'name': <Neighbor Name>, 'node_type': <Neighbor type>},...]
        :rtype: list
        """

        service_name = "cognitive_node/" + str(name) + "/get_information"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(GetInformation, service_name)
        information = self.node_clients[service_name].send_request()

        neighbors_names = information.neighbors_name
        neighbors_types = information.neighbors_type

        neighbors = [
            {"name": node[0], "node_type": node[1]}
            for node in zip(neighbors_names, neighbors_types)
        ]

        self.get_logger().debug(f"REQUESTED NEIGHBORS: {neighbors}")

        return neighbors

    def execute_policy(self, policy):
        """
        Execute a policy.

        This method sends a request to the policy to be executed.

        :param policy: The policy to execute.
        :type policy: str
        :return: The response from executing the policy.
        :rtype: The executed policy.
        """
        self.get_logger().info("Executing policy " + str(policy) + "...")

        service_name = "policy/" + str(policy) + "/execute"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(Execute, service_name)
        policy_response = self.node_clients[service_name].send_request()
        return policy_response.policy

    def get_current_goal(self):
        """
        This method selects the goal with the highest activation in the LTM cache.

        :return: Goal with highest activation.
        :rtype: str
        """

        self.get_logger().info("Selecting goal with highest activation...")

        goal_activations = {}
        for node in self.LTM_cache:
            if node["node_type"] == "Goal":
                goal_activations[node["name"]] = node["activation"]

        self.get_logger().info("Selecting current goal - Activations: " + str(goal_activations))

        goal = max(zip(goal_activations.values(), goal_activations.keys()))[1]

        return goal

    def get_current_reward(self, old_sensing, sensing):
        """
        This method calls the get reward service of the current goal and
        returns the reward.

        :param old_sensing: Sensing prior to the performed action.
        :type old_sensing: dict
        :param sensing: Sensing after the performed action.
        :type sensing: dict
        :return: Current reward.
        :rtype: float
        """

        self.get_logger().info("Reading reward...")
        old_perception = perception_dict_to_msg(old_sensing)
        perception = perception_dict_to_msg(sensing)
        service_name = "goal/" + str(self.current_goal) + "/get_reward"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(GetReward, service_name)
        reward = self.node_clients[service_name].send_request(
            old_perception=old_perception, perception=perception
        )

        self.get_logger().info(f"Reading reward - Reward: {reward.reward}")

        return reward.reward

    def get_current_world_model(self):
        """
        This method selects the world model with the highest activation in the LTM cache.

        :return: World model with highest activation.
        :rtype: str
        """

        self.get_logger().info("Selecting world model with highest activation...")

        WM_activations = {}
        for node in self.LTM_cache:
            if node["node_type"] == "WorldModel":
                WM_activations[node["name"]] = node["activation"]

        self.get_logger().info(
            "Selecting current world model - Activations: " + str(WM_activations)
        )

        WM = max(zip(WM_activations.values(), WM_activations.keys()))[1]

        return WM

    def get_max_activation_node(self, node_type):  # TODO: Refactor
        # Pending to refactor all get_current_* into a general method
        raise NotImplementedError

    def update_pnodes_reward_basis(self, perception, policy, goal, reward):
        """
        This method creates or updates CNodes and PNodes according to the executed policy,
        current goal and reward obtained.

        The method follows these steps:
        1. Obtain the CNode(s) linked to the policy.
        -If there are CNodes linked to the policy, for each CNode:
        2. Obtain WorldModel, Goal and PNode activation
        3. Check if the WorldModel and Goal are active
        4. If there is a reward an antipoint is added,
        if there is no reward and the PNode is active, an antipoint is added.

        -If there are no CNodes connected to the policy a new CNode is created
        if there is reward.

        :param perception: Perception before the execution of the policy
        :type perception: dict
        :param policy: Policy executed
        :type policy: dict
        :param goal: Current goal
        :type goal: str
        :param reward: Reward obtained after the execution of the policy
        :type reward: float
        """

        self.get_logger().info("Updating p-nodes/c-nodes...")
        policy_neighbors = self.request_neighbors(policy)
        cnodes = [node["name"] for node in policy_neighbors if node["node_type"] == "CNode"]
        threshold = 0.1

        for cnode in cnodes:
            cnode_neighbors = self.request_neighbors(cnode)
            world_model = next(
                (
                    neighbor["name"]
                    for neighbor in cnode_neighbors
                    if neighbor["node_type"] == "WorldModel"
                )
            )
            goal = next(
                (
                    neighbor["name"]
                    for neighbor in cnode_neighbors
                    if neighbor["node_type"] == "Goal"
                )
            )
            pnode = next(
                (
                    neighbor["name"]
                    for neighbor in cnode_neighbors
                    if neighbor["node_type"] == "PNode"
                )
            )

            world_model_activation = next(
                (node["activation"] for node in self.LTM_cache if node["name"] == world_model)
            )
            goal_activation = next(
                (node["activation"] for node in self.LTM_cache if node["name"] == goal)
            )
            pnode_activation = next(
                (node["activation"] for node in self.LTM_cache if node["name"] == pnode)
            )

            if world_model_activation > threshold and goal_activation > threshold:
                if reward > threshold:
                    self.add_point(pnode, perception)
                    return None
                elif pnode_activation > threshold:
                    self.add_antipoint(pnode, perception)
                    return None

        if (not cnodes) and (reward > threshold):
            self.new_cnode(perception, goal, policy)
            return None
        self.get_logger().info("No update required in PNode/CNodes")

    def add_point(self, name, sensing):
        """
        Sends the request to add a point to a PNode.

        :param name: Name of the PNode
        :type name: str
        :param sensing: Sensorial data to be added as a point.
        :type sensing: dict
        :return: Success status received from the PNode
        :rtype: bool
        """

        service_name = "pnode/" + str(name) + "/add_point"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(AddPoint, service_name)

        perception = perception_dict_to_msg(sensing)
        response = self.node_clients[service_name].send_request(point=perception, confidence=1.0)
        self.get_logger().info(f"Added point in pnode {name}")
        self.get_logger().debug(f"POINT: {str(sensing)}")
        self.pnodes_success[name] = True
        return response.added

    def add_antipoint(self, name, sensing):
        """
        Sends the request to add an antipoint to a PNode.

        :param name: Name of the PNode
        :type name: str
        :param sensing: Sensorial data to be added as a point.
        :type sensing: dict
        :return: Success status received from the PNode
        :rtype: bool
        """

        service_name = "pnode/" + str(name) + "/add_point"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(
                AddPoint, service_name, callback_group=self.cbgroup_client
            )

        perception = perception_dict_to_msg(sensing)
        response = self.node_clients[service_name].send_request(point=perception, confidence=-1.0)
        self.get_logger().info(f"Added anti-point in pnode {name}")
        self.get_logger().debug(f"ANTI-POINT: {str(sensing)}")
        self.pnodes_success[name] = False
        return response.added

    def new_cnode(self, perception, goal, policy):
        """
        This method creates a new CNode/PNode pair.

        :param perception: Perception to be added as the first point in the PNode
        :type perception: dict
        :param goal: Goal that will be linked to the CNode
        :type goal: str
        :param policy: Policy that will be linked to the CNode
        :type policy: str
        """

        self.get_logger().info("Creating Cnode...")
        world_model = self.get_current_world_model()
        ident = f"{world_model}__{goal}__{policy}"

        space_class = self.default_class.get("Space")
        pnode_class = self.default_class.get("PNode")
        cnode_class = self.default_class.get("CNode")

        pnode_name = f"pnode_{ident}"
        pnode = self.create_node_client(
            name=pnode_name, class_name=pnode_class, parameters={"space_class": space_class}
        )
        # Update LTMCache with new CNode/PNode. This is a HACK, should be integrated with LTM's changes topic
        self.LTM_cache.append({"name": pnode_name, "node_type": "PNode", "activation": 0})

        if not pnode:
            self.get_logger().fatal(f"Failed creation of PNode {pnode_name}")
        self.add_point(pnode_name, perception)

        neighbors = {
            "neighbors": [
                node
                for node in self.LTM_cache
                if node["name"] in [world_model, goal, policy, pnode_name]
            ]
        }

        cnode_name = f"cnode_{ident}"
        cnode = self.create_node_client(
            name=cnode_name, class_name=cnode_class, parameters=neighbors
        )

        if not cnode:
            self.get_logger().fatal(f"Failed creation of CNode {cnode_name}")

        # Update LTMCache with new CNode/PNode. This is a HACK, should be integrated with LTM's changes topic
        self.LTM_cache.append({"name": cnode_name, "node_type": "CNode", "activation": 0})
        self.n_cnodes = self.n_cnodes + 1  # TODO: Consider the posibility of delete CNodes

    def create_node_client(self, name, class_name, parameters={}):
        """
        This method calls the add node service of the commander.

        :param name: Name of the node to be created.
        :type name: str
        :param class_name: Name of the class to be used for the creation of the node.
        :type class_name: str
        :param parameters: Optional parameters that can be passed to the node, defaults to {}
        :type parameters: dict, optional
        :return: Success status received from the commander
        :rtype: bool
        """

        self.get_logger().info("Requesting node creation")
        params_str = yaml.dump(parameters)
        service_name = "commander/create"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(CreateNode, service_name)
        response = self.node_clients[service_name].send_request(
            name=name, class_name=class_name, parameters=params_str
        )
        return response.created

    def reset_world(self):
        """
        Reset the world if necessary, according to the experiment parameters.
        """

        changed = False
        self.trial += 1
        if self.trial == self.trials or self.current_reward > 0.9 or self.iteration == 0:
            self.trial = 0
            changed = True
        if (self.iteration % self.period) == 0:
            # TODO: Implement periodic world changes
            pass
        if changed:
            self.get_logger().info("Asking for a world reset...")
            msg = ControlMsg()
            msg.command = "reset_world"
            msg.world = self.current_world
            msg.iteration = self.iteration
            self.control_publisher.publish(msg)
        return changed

    def update_status(self):
        """
        Method that writes the files with execution data.
        """

        self.get_logger().info("Writing files publishing status...")
        self.get_logger().debug(f"DEBUG: {self.pnodes_success}")
        for file in self.files:
            if file.file_object is None:
                file.write_header()
            file.write()

    def close_files(self):
        self.get_logger().info("Closing files...")
        for file in self.files:
            file.close()

    def run(self, _=None):
        """
        Run the main loop of the system.

        """

        self.get_logger().info("Running MDB with LTM:" + str(self.LTM_id))

        self.current_world = self.get_current_world_model()

        self.reset_world()
        sensing = self.read_perceptions()
        stm = []
        self.iteration = 1
        while (self.iteration <= self.iterations) and (not self.stop):

            if not self.paused:

                self.get_logger().info(
                    "*** ITERATION: " + str(self.iteration) + "/" + str(self.iterations) + " ***"
                )
                self.publish_iteration()

                self.current_policy = self.select_policy(sensing)
                self.execute_policy(self.current_policy)
                old_sensing, sensing = sensing, self.read_perceptions()
                self.get_logger().debug(
                    f"DEBUG PERCEPTION: \n old_sensing: {old_sensing} \n     sensing: {sensing}"
                )

                if not self.subgoals:
                    self.current_goal = self.get_current_goal()
                    self.current_reward = self.get_current_reward(old_sensing, sensing)
                    self.update_pnodes_reward_basis(
                        old_sensing, self.current_policy, self.current_goal, self.current_reward
                    )
                else:
                    raise NotImplementedError  # TODO: Implement prospection methods

                if self.reset_world():
                    reset_sensing = self.read_perceptions()

                    if self.current_reward > self.reward_threshold and self.subgoals:
                        raise NotImplementedError  # TODO: Implement prospection methods

                    sensing = reset_sensing

                self.update_policies_to_test(
                    policy=(
                        self.current_policy
                        if not self.sensorial_changes(sensing, old_sensing)
                        else None
                    )
                )

                self.update_status()
                self.iteration += 1

        self.close_files()


def main(args=None):
    rclpy.init()

    executor = MultiThreadedExecutor(num_threads=2)
    # executor=SingleThreadedExecutor()
    main_loop = MainLoop("main_loop")

    executor.add_node(main_loop)  # Test
    main_loop.get_logger().info("Runnning node")

    try:
        executor.spin()
    except KeyboardInterrupt:
        main_loop.destroy_node()


if __name__ == "__main__":
    main()
