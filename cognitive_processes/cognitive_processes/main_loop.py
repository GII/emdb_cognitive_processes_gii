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
from rclpy.time import Time

from core.service_client import ServiceClient
from cognitive_node_interfaces.srv import (
    Execute,
    GetActivation,
    GetReward,
    GetInformation,
    AddPoint,
    IsSatisfied
)
from cognitive_node_interfaces.msg import Perception
from core_interfaces.srv import GetNodeFromLTM, CreateNode
from cognitive_processes_interfaces.msg import ControlMsg
from cognitive_processes_interfaces.msg import Episode as EpisodeMsg

from core.utils import perception_dict_to_msg, perception_msg_to_dict, class_from_classname

class Episode():
    def __init__(self) -> None:
        self.old_perception={}
        self.old_ltm_state={}
        self.policy=''
        self.perception={}
        self.ltm_state={}
        self.reward_list={}


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
        self.stm = Episode()
        self.default_class = {}
        self.perception_suscribers = {}
        self.perception_cache = {}
        self.reward_threshold = 0.9
        self.activation_threshold = 0.01
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
        self.episode_publisher = self.create_publisher(EpisodeMsg, "main_loop/episodes", 10)

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

    def read_ltm(self, ltm_cache=None):
        """
        Makes an empty call for the LTM get_node service which returns all the nodes
        stored in the LTM. Then, a LTM cache dictionary is populated with the data.

        """
        self.get_logger().info("Reading nodes from LTM: " + self.LTM_id + "...")
        
        if not ltm_cache:
            ltm_cache, _ = self.request_ltm()
            self.get_logger().debug(f"LTM Dump: {str(ltm_cache)}")
        
        
        self.LTM_cache=[]
        for node_type in ltm_cache.keys():
            for node in ltm_cache[node_type].keys():
                activation = ltm_cache[node_type][node]["activation"]
                self.LTM_cache.append(
                    {
                        "name": node,
                        "node_type": node_type,
                        "activation": activation if activation > self.activation_threshold else 0, #Think about a proper threshold for activation. Is this even neccesary?
                        "activation_timestamp": ltm_cache[node_type][node]["activation_timestamp"]
                    }
                )

        self.get_logger().debug(f"LTM Cache: {str(self.LTM_cache)}")
    
    def request_ltm(self, timestamp=Time()):
        # Call get_node service from LTM
        service_name = "/" + str(self.LTM_id) + "/get_node"
        request = ""
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(GetNodeFromLTM, service_name)
        ltm_response = self.node_clients[service_name].send_request(name=request, timestamp=timestamp.to_msg())
        ltm = yaml.safe_load(ltm_response.data)
        updated = ltm_response.updated

        return ltm, updated

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
            self.perception_cache[sensor]["flag"].clear()
            self.get_logger().debug("Clearing flags: " + str(sensor))

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

        policy = self.rng.choice(self.policies_to_test)

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

    def update_activations(self, timestamp):
        """
        Requests a new activation to all nodes in the LTM Cache.

        :param perception: Perception used to calculate the activation
        :type perception: dict
        :param new_sensings: Indicates if a sensing change has ocurred, defaults to True
        :type new_sensings: bool, optional
        """

        self.get_logger().info("Updating activations...")
        updated=False
        while not updated:
            ltm, updated = self.request_ltm(timestamp=timestamp)

        self.read_ltm(ltm_cache=ltm)
    
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
    
    def get_goals(self, ltm_state):
        goals = []
        for node in ltm_state:
            if node["node_type"] == "Goal":
                if node["activation"] > 0.0:
                    goals.append(node['name'])

        self.get_logger().info(f"Active Goals: {goals}")
                    
        return goals
    
    def get_goals_reward(self, old_sensing, sensing):
        self.get_logger().info("Reading rewards...")
        rewards = {}
        old_perception = perception_dict_to_msg(old_sensing)
        perception = perception_dict_to_msg(sensing)

        for goal in self.active_goals:
            service_name = "goal/" + str(goal) + "/get_reward"
            if service_name not in self.node_clients:
                self.node_clients[service_name] = ServiceClient(GetReward, service_name)
            reward = self.node_clients[service_name].send_request(
                old_perception=old_perception, perception=perception
            )
            rewards[goal] = reward.reward

        self.get_logger().info(f"Reward_list: {rewards}")

        return rewards


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
    
    def get_needs(self):
        needs = []
        for node in self.LTM_cache:
            if node["node_type"] == "Need":
                if node["activation"] > 0.0:
                    needs.append(node['name'])

        self.get_logger().info(f"Active Needs: {needs}")
                    
        return needs
    
    def get_need_satisfaction(self, need_list, timestamp):
        self.get_logger().info("Reading satisfaction...")
        satisfaction = {}
        response=IsSatisfied.Response()
        for need in need_list:
            service_name = "need/" + str(need) + "/get_satisfaction"
            if service_name not in self.node_clients:
                self.node_clients[service_name] = ServiceClient(IsSatisfied, service_name)
            while not response.updated:
                response = self.node_clients[service_name].send_request(
                    timestamp=timestamp.to_msg()
                )
            satisfaction[need] = dict(satisfied=response.satisfied, need_type=response.need_type)
            response.updated = False

        self.get_logger().info(f"Satisfaction list: {satisfaction}")

        return satisfaction

    def get_max_activation_node(self, node_type):  # TODO: Refactor
        # Pending to refactor all get_current_* into a general method
        raise NotImplementedError
    
    def update_ltm(self, perception, policy, stm:Episode):
            self.update_pnodes_reward_basis(perception, policy, copy(stm.reward_list), stm.old_ltm_state)


    def update_pnodes_reward_basis(self, perception, policy, reward_list, ltm_cache):
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
        threshold = self.activation_threshold
        updates = False

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
                (node["activation"] for node in ltm_cache if node["name"] == world_model)
            )
            goal_activation = next(
                (node["activation"] for node in ltm_cache if node["name"] == goal)
            )
            pnode_activation = next(
                (node["activation"] for node in ltm_cache if node["name"] == pnode)
            )

            if world_model_activation > threshold and goal_activation > threshold:
                reward = reward_list.get(goal, 0.0)
                if reward > threshold:
                    reward_list.pop(goal)
                    self.add_point(pnode, perception)
                    updates = True
                elif pnode_activation > threshold:
                    self.add_antipoint(pnode, perception)
                    updates = True

        for goal, reward in reward_list.items():
            if reward > threshold:
                self.new_cnode(perception, goal, policy)
                updates = True

        if not updates:
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
                AddPoint, service_name
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
                {key:value for key, value in node.items() if (key != "activation" and key != "activation_timestamp")}
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
    


    def reset_world(self, check_finish=True):
        """
        Reset the world if necessary, according to the experiment parameters.
        """

        changed = False
        self.trial += 1
        if check_finish:
            finished = self.world_finished()
        else:
            finished=False

        if self.trial == self.trials or finished or self.iteration == 0:
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
    
    def world_finished(self):
        need_satisfaction = self.get_need_satisfaction(self.get_needs(), self.get_clock().now())
        if len(need_satisfaction)>0:
            finished = all((need_satisfaction[need]['satisfied'] for need in need_satisfaction if (need_satisfaction[need]['need_type'] == 'Operational')))
        else:
            finished=False
        return finished

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

    def publish_episode(self):
        msg=EpisodeMsg()
        msg.old_perception = perception_dict_to_msg(self.stm.old_perception)
        msg.old_ltm_state = yaml.dump(self.stm.old_ltm_state)
        msg.policy = self.stm.policy
        msg.perception = perception_dict_to_msg(self.stm.perception)
        msg.ltm_state = yaml.dump(self.stm.ltm_state)
        msg.reward_list = yaml.dump(self.stm.reward_list)
        msg.timestamp = self.get_clock().now().to_msg()
        self.episode_publisher.publish(msg)

    def run(self, _=None):
        """
        Run the main loop of the system.

        """

        self.get_logger().info("Running MDB with LTM:" + str(self.LTM_id))

        self.current_world = self.get_current_world_model()

        self.reset_world()
        #timestamp = self.get_clock().now()
        self.stm.perception = self.read_perceptions()
        self.update_activations(self.get_clock().now())
        self.active_goals = self.get_goals(self.LTM_cache)
        self.stm.reward_list= self.get_goals_reward(self.stm.old_perception, self.stm.perception)
        #self.stm.ltm_state = self.LTM_cache
        
        self.iteration = 1
        while (self.iteration <= self.iterations) and (not self.stop):

            if not self.paused:

                self.get_logger().info(
                    "*** ITERATION: " + str(self.iteration) + "/" + str(self.iterations) + " ***"
                )
                self.publish_iteration()
                self.update_activations(self.get_clock().now())
                self.current_policy = self.select_policy(self.stm.perception)
                self.current_policy = self.execute_policy(self.current_policy)
                self.stm.policy = self.current_policy
                #timestamp = self.get_clock().now()
                self.stm.old_perception, self.stm.perception = self.stm.perception, self.read_perceptions()
                self.stm.old_ltm_state=self.LTM_cache
                self.update_activations(self.get_clock().now())
                self.stm.ltm_state=self.LTM_cache

                self.get_logger().info(
                    f"DEBUG PERCEPTION: \n old_sensing: {self.stm.old_perception} \n     sensing: {self.stm.perception}"
                )


                self.active_goals = self.get_goals(self.stm.old_ltm_state)
                self.stm.reward_list= self.get_goals_reward(self.stm.old_perception, self.stm.perception)

                self.publish_episode()

                self.update_ltm(self.stm.old_perception, self.current_policy, self.stm)


                if self.reset_world():
                    #timestamp = self.get_clock().now()
                    reset_sensing = self.read_perceptions()
                    self.update_activations(self.get_clock().now())
                    self.stm.perception = reset_sensing
                    self.stm.ltm_state = self.LTM_cache

                self.update_policies_to_test(
                    policy=(
                        self.current_policy
                        if not self.sensorial_changes(self.stm.perception, self.stm.old_perception)
                        else None
                    )
                )
                #timestamp = self.get_clock().now()
                #self.get_logger().info(f'ITERATION END: {timestamp.seconds_nanoseconds()}')
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
