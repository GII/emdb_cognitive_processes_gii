import sys
import rclpy
from rclpy.node import Node
from operator import attrgetter
import random
import yaml
import threading
import numpy
import time
from copy import copy, deepcopy

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
from cognitive_node_interfaces.msg import PerceptionStamped, Activation
from core_interfaces.srv import GetNodeFromLTM, CreateNode, SetChangesTopic, UpdateNeighbor, StopExecution
from cognitive_processes_interfaces.msg import ControlMsg
from cognitive_processes_interfaces.msg import Episode as EpisodeMsg
from std_msgs.msg import String

from core.utils import perception_dict_to_msg, perception_msg_to_dict, actuation_dict_to_msg, actuation_msg_to_dict, class_from_classname

class Episode():
    def __init__(self) -> None:
        self.old_perception={}
        self.old_ltm_state={}
        self.policy=''
        self.actuation={}
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
        self.kill_on_finish = False
        self.LTM_id = ""  # id of LTM currently being run by cognitive loop
        self.LTM_cache = (
            {}
        )  # Nested dics, like {'CNode': {'CNode1': {'activation': 0.0}}, 'Drive': {...}, 'Goal': {...}, 'Need': {...}, 'Policy': {...}, 'Perception': {...},'PNode': {...}, 'UtilityModel': {...}, 'WorldModel': {...}}
        self.stm = Episode()
        self.default_class = {}
        self.perception_suscribers = {}
        self.perception_cache = {}
        self.activation_inputs = {}
        self.perception_time = 0
        self.activation_time = 0
        self.reward_threshold = 0.9
        self.activation_threshold = 0.01
        self.unlinked_drives=[]
        self.subgoals = False
        self.policies_to_test = []
        self.files = []
        self.default_class = {}
        self.random_seed = 0
        self.current_reward = 0
        self.current_world = None
        self.n_cnodes = 0
        self.n_goals = 0
        self.sensorial_changes_val = False
        self.softmax_selection = False
        self.softmax_temperature = 1
        self.pnodes_success = {}
        self.goal_count=0
        self.trials_data=[]
        self.last_reset=0

        self.cbgroup_perception = MutuallyExclusiveCallbackGroup()
        self.cbgroup_server = MutuallyExclusiveCallbackGroup()
        self.cbgroup_client = MutuallyExclusiveCallbackGroup()
        self.cbgroup_loop = MutuallyExclusiveCallbackGroup()

        self.node_clients = (
            {}
        )  # Keys are service name, values are service client object e.g. {'cognitive_node/policy0/get_activation: "Object: core.ServiceClient(Async)"'}


        for key, value in params.items():
            self.get_logger().debug("Setting atribute: " + str(key) + " with value: " + str(value))
            setattr(self, key, value)

        self.LTM_changes_client = ServiceClient(SetChangesTopic, f"{self.LTM_id}/set_changes_topic")

        # Read LTM and configure perceptions
        self.setup()

        loop_thread = threading.Thread(target=self.run, daemon=True)
        self.semaphore = threading.Semaphore()
        loop_thread.start()

    def setup(self):
        self.rng = numpy.random.default_rng(self.random_seed)
        self.read_ltm()
        self.configure_perceptions()
        self.setup_ltm_suscription()
        self.setup_files()
        self.setup_connectors()
        self.setup_control_channel()
        self.LTM_changes_client.send_request(changes_topic=True)
        self.kill_commander_client = ServiceClient(StopExecution, 'commander/kill')

    def read_ltm(self, ltm_dump=None):
        """
        Makes an empty call for the LTM get_node service which returns all the nodes
        stored in the LTM. Then, a LTM cache dictionary is populated with the data.

        """
        self.get_logger().info("Reading nodes from LTM: " + self.LTM_id + "...")
        
        #Get a LTM dump if not provided
        if not ltm_dump:
            ltm_dump = self.request_ltm()
            self.get_logger().debug(f"LTM Dump: {str(ltm_dump)}")
        
        #Add missing elements from LTM to LTM Cache
        for node_type in ltm_dump.keys():
            if self.LTM_cache.get(node_type, None) is None:
                self.LTM_cache[node_type] = {}
            for node in ltm_dump[node_type].keys():
                if self.LTM_cache[node_type].get(node, None) is None:
                    self.LTM_cache[node_type][node] = dict(activation = 0.0, activation_timestamp = 0, neighbors = ltm_dump[node_type][node]["neighbors"])
                    if "policy_params" in ltm_dump[node_type][node]:
                        self.get_logger().info(f"DEBUG - CNODE {node} policy_params: {ltm_dump[node_type][node]['policy_params']}")
                        if isinstance(ltm_dump[node_type][node]["policy_params"], dict):
                            self.LTM_cache[node_type][node]["policy_params"]=ltm_dump[node_type][node]["policy_params"]
                        else:
                            self.LTM_cache[node_type][node]["policy_params"]={}
                    self.create_activation_input(node, node_type)
                else: #If node exists update data (except activations)
                    node_data = ltm_dump[node_type][node]
                    del node_data["activation"]
                    del node_data["activation_timestamp"]
                    policy_params = node_data.pop("policy_params", None)
                    if policy_params is not None:
                        if isinstance(policy_params, dict):
                            self.LTM_cache[node_type][node]["policy_params"]=policy_params
                        else:
                            self.LTM_cache[node_type][node]["policy_params"]={}
                    self.LTM_cache[node_type][node].update(node_data) 
        
        #Remove elements in LTM Cache that were removed from LTM.
        for node_type in self.LTM_cache.keys():
            for node in self.LTM_cache[node_type]:
                if ltm_dump[node_type].get(node, None) is None:
                    del self.LTM_cache[node_type][node]
                    self.delete_activation_input(node)
                    
        #Check if there are any drives not linked to goals
        self.unlinked_drives=self.get_unlinked_drives()
        self.get_logger().debug(f"LTM Cache: {str(self.LTM_cache)}")
    
    def setup_ltm_suscription(self):
        self.ltm_suscription = self.create_subscription(String, "state", self.ltm_change_callback, 0, callback_group=self.cbgroup_client)

    def ltm_change_callback(self, msg):
        """
        PENDING METHOD: This method will read changes made in the LTM
        external to the cognitive process and update the LTM and perception
        caches accordingly.

        """
        self.semaphore.acquire()
        self.get_logger().info("Processing change from LTM...")
        ltm_dump = yaml.safe_load(msg.data)
        self.read_ltm(ltm_dump=ltm_dump)
        #self.configure_perceptions #CHANGE THIS SO THAT NEW PERCEPTIONS ARE ADDED AND OLD PERCEPTIONS ARE DELETED
        self.semaphore.release()
    
    def request_ltm(self):
        # Call get_node service from LTM
        service_name = "/" + str(self.LTM_id) + "/get_node"
        request = ""
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(GetNodeFromLTM, service_name)
        ltm_response = self.node_clients[service_name].send_request(name=request)
        ltm = yaml.safe_load(ltm_response.data)

        return ltm

    def configure_perceptions(
        self,
    ):  # TODO(efallash): Add condition so that perceptions that are already included do not create a new suscription. For the case that new perceptions are added to the LTM and only some perceptions need to be configured
        """
        Reads the LTM cache and populates the perception subscribers and perception cache dictionaries.

        """
        self.get_logger().info("Configuring perceptions...")
        perceptions = iter(self.LTM_cache['Perception'].keys())

        self.get_logger().debug(f"Perception list: {str(perceptions)}")

        for perception in perceptions:

            subscriber = self.create_subscription(
                PerceptionStamped,
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
    
    def setup_control_channel(self):
        control_msg=self.Control["control_msg"]
        episode_msg=self.Control["episodes_msg"]
        world_reset_msg=self.Control.get("world_reset_msg", None)
        world_reset_service=self.Control.get("world_reset_service", None)
        self.control_publisher = self.create_publisher(class_from_classname(control_msg), self.Control["control_topic"], 10)
        self.episode_publisher = self.create_publisher(class_from_classname(episode_msg), self.Control["episodes_topic"], 10)
        if world_reset_msg and world_reset_service:
            self.world_reset_client = ServiceClient(class_from_classname(world_reset_msg), world_reset_service)

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
        self.perception_time = self.get_clock().now().nanoseconds

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
        perception_dict = perception_msg_to_dict(msg.perception)

        for sensor in perception_dict.keys():
            if sensor in self.perception_cache:
                self.perception_cache[sensor]["data"] = copy(perception_dict[sensor])
                if Time.from_msg(msg.timestamp).nanoseconds > self.perception_time:
                    self.perception_cache[sensor]["flag"].set()
                self.get_logger().debug(
                    f'Receiving perception: {sensor} {self.perception_cache[sensor]["data"]} ...'
                )
            else:
                self.get_logger().error(
                    "Received sensor not registered in local perception cache!!!"
                )

    def select_policy(self, softmax=False):
        """
        Selects the policy with the higher activation based on the current sensing.

        :param sensing: The current sensing.
        :type sensing: dict
        :return: The selected policy.
        :rtype: str
        """
         
        if self.policies_to_test == []:
            self.policies_to_test = list(self.LTM_cache["Policy"].keys())

        # This is an UGLY HACK to avoid repetition of policies that yield no reward. Need to evaluate more options.
        policies_filtered = self.policies_to_test #Policies that have resulted in no perceptual change are filtered from this list
        policies= self.LTM_cache["Policy"].keys()

        policy_activations={}
        all_policy_activations={}
        for policy in policies_filtered:
            act=self.LTM_cache["Policy"][policy]["activation"]
            if act>self.activation_threshold: #Filters out non-activated policies
                policy_activations[policy]=act

        for policy in policies:
            all_policy_activations[policy]=self.LTM_cache["Policy"][policy]["activation"]
        self.get_logger().debug("Debug - All policy activations: " + str(all_policy_activations))
        self.get_logger().debug("Debug - Filtered policy activations: " + str(policy_activations))
        if not policy_activations:
            policy_pool = all_policy_activations
        else:
            policy_pool = policy_activations

        if softmax:
            selected = self.select_policy_softmax(policy_pool, self.softmax_temperature)
        else: 
            selected= self.select_max_policy(policy_pool)


        self.get_logger().info("Select_policy - Activations: " + str(all_policy_activations))
        self.get_logger().info("Discarded policies: " + str(set(policies)-set(policies_filtered)))

        if not policy_pool[selected]:
            selected = self.random_policy()

        self.get_logger().info(f"Selected policy => {selected} ({policy_pool[selected]})")

        return selected
    
    def select_max_policy(self, policy_activations:dict):
        selected= max(zip(policy_activations.values(), policy_activations.keys()))[1]
        return selected
    
    def select_policy_softmax(self, policy_activations:dict, temperature=1):
        # Convert activations to a numpy array for softmax computation
        activations = numpy.array(list(policy_activations.values()))
        policy_names = list(policy_activations.keys())

        # Compute softmax probabilities
        scaled_activations=activations/temperature
        exp_activations = numpy.exp(scaled_activations - numpy.max(scaled_activations))  # Subtract max for numerical stability
        probabilities = exp_activations / numpy.sum(exp_activations)
        policy_probabilities = {policy: prob for policy, prob in zip(policy_names, probabilities)}

        # Select a policy based on the probabilities
        selected = self.rng.choice(policy_names, p=probabilities)
        self.get_logger().info(f"DEBUG - Softmax selection: {selected}, Probabilities: {policy_probabilities}")
        return selected

    def random_policy(self):
        """
        Selects a random policy.

        :return: The selected policy
        :rtype: str
        """

        if self.policies_to_test == []:
            self.policies_to_test = list(self.LTM_cache["Policy"].keys())

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
            self.policies_to_test = list(self.LTM_cache["Policy"].keys())

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

    def update_activations(self):
        """
        Requests a new activation to all nodes in the LTM Cache.

        :param perception: Perception used to calculate the activation
        :type perception: dict
        :param new_sensings: Indicates if a sensing change has ocurred, defaults to True
        :type new_sensings: bool, optional
        """
        self.get_logger().info("Updating activations...")
        self.semaphore.acquire()
        self.activation_time=self.get_clock().now().nanoseconds
        for node in self.activation_inputs:
            self.activation_inputs[node]['flag'].clear()

        for node in self.activation_inputs:
            self.get_logger().debug(f"DEBUG: Waiting for activation: {node}")
            self.activation_inputs[node]['flag'].wait()
            self.activation_inputs[node]['flag'].clear()
        self.semaphore.release()
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
    
    def create_activation_input(self, name, node_type): #Adds a node from the activation inputs list.
        if name not in self.activation_inputs:
            subscriber=self.create_subscription(Activation, 'cognitive_node/' + str(name) + '/activation', self.read_activation_callback, 1, callback_group=self.cbgroup_server)
            flag=threading.Event()
            self.activation_inputs[name]=dict(node_type=node_type, subscriber=subscriber, flag=flag)
            self.get_logger().debug(f'Created new activation input: {name} of type {node_type}')
        else:
            self.get_logger().error(f'Tried to add {name} to activation inputs more than once')
    
    def delete_activation_input(self, name): #Deletes a node from the activation inputs list. By default reads activations.
        if name in self.activation_inputs:
            self.destroy_subscription(self.activation_inputs[name])
            self.activation_inputs.pop(name)

    def read_activation_callback(self, msg: Activation):
        node_name=msg.node_name
        node_type=msg.node_type
        activation=msg.activation
        timestamp=Time.from_msg(msg.timestamp).nanoseconds
        old_timestamp=self.LTM_cache[node_type][node_name]['activation_timestamp']
        self.LTM_cache[node_type][node_name]['activation']=activation
        self.LTM_cache[node_type][node_name]['activation_timestamp']=timestamp
        
        if timestamp > self.activation_time :            
            self.activation_inputs[node_name]['flag'].set()
        elif timestamp < old_timestamp:
            self.get_logger().error(f"JUMP BACK IN TIME DETECTED. ACTIVATION OF {node_type} {node_name}")
        
        act_file = getattr(self, "act_file", None) #CHANGE THIS
        if act_file is not None:
            act_file.receive_activation_callback(msg)

    def request_neighbors(self, name):
        """
        This method calls the service to get the neighbors of a node.

        :param name: Node name.
        :type name: str
        :return: List of dictionaries with the information of each neighbor of the node. [{'name': <Neighbor Name>, 'node_type': <Neighbor type>},...]
        :rtype: list
        """

        data_dict = self.get_node_data(name, self.LTM_cache)
        neighbors = data_dict["neighbors"]

        self.get_logger().debug(f"REQUESTED NEIGHBORS: {neighbors}")

        return neighbors
    
    def add_neighbor(self, node_name, neighbor_name):
        service_name=f"{self.LTM_id}/update_neighbor"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(UpdateNeighbor, service_name)
        response=self.node_clients[service_name].send_request(node_name=node_name, neighbor_name=neighbor_name, operation=True)
        return response.success

    def execute_policy(self, perception, policy):
        """
        Execute a policy.

        This method sends a request to the policy to be executed.

        :param policy: The policy to execute.
        :type policy: str
        :return: The response from executing the policy.
        :rtype: The executed policy.
        """
        

        service_name = "policy/" + str(policy) + "/execute"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(Execute, service_name)
        perc_msg=perception_dict_to_msg(perception)
        policy_response = self.node_clients[service_name].send_request(perception=perc_msg)
        action = policy_response.action
        self.get_logger().info("Executed policy " + str(policy_response.policy) + " with action " + str(action))
        return policy_response.policy, action 
    
    def get_goals(self, ltm_cache):
        goals = self.get_all_active_nodes("Goal", ltm_cache)
        return goals
    
    def get_goals_reward(self, old_sensing, sensing, ltm_cache):
        self.get_logger().info("Reading rewards...")
        rewards = {}
        old_perception = perception_dict_to_msg(old_sensing)
        perception = perception_dict_to_msg(sensing)

        for goal in self.active_goals:
            updated_reward=False
            while not updated_reward:
                service_name = "goal/" + str(goal) + "/get_reward"
                if service_name not in self.node_clients:
                    self.node_clients[service_name] = ServiceClient(GetReward, service_name)
                reward = self.node_clients[service_name].send_request(
                    old_perception=old_perception, perception=perception
                )
                rewards[goal] = reward.reward
                updated_reward=reward.updated

        #Add rewards obtained from unlinked drives
        if self.unlinked_drives:
            active_drives=[drive for drive in self.unlinked_drives if ltm_cache["Drive"][drive]["activation"]>self.activation_threshold]
            for drive in active_drives:
                updated_reward=False
                while not updated_reward:
                    service_name = "drive/" + str(drive) + "/get_reward"
                    if service_name not in self.node_clients:
                        self.node_clients[service_name] = ServiceClient(GetReward, service_name)
                    reward = self.node_clients[service_name].send_request()
                    rewards[drive] = reward.reward
                    updated_reward=reward.updated
        self.get_logger().info(f"Reward_list: {rewards}")
        return rewards

    def get_unlinked_drives(self):
        drives=self.LTM_cache.get("Drive", None)
        goals=self.LTM_cache.get("Goal", None)
        if drives:
            drives_list=list(drives.keys())
            for goal in goals:
                neighbors=goals[goal]["neighbors"]
                for neighbor in neighbors:
                    if neighbor["name"] in drives_list:
                        drives_list.remove(neighbor["name"])
            return drives_list
        else:
            return []

        

    def get_current_world_model(self):
        """
        This method selects the world model with the highest activation in the LTM cache.

        :return: World model with highest activation.
        :rtype: str
        """
        WM, WM_activations = self.get_max_activation_node("WorldModel")
        self.get_logger().info(f"Selecting world model with highest activation: {WM} ({WM_activations[WM]})")

        return WM
    
    def get_needs(self, ltm_cache):
        needs = self.get_all_active_nodes("Need", ltm_cache)

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

    def get_max_activation_node(self, node_type):
        node_activations = {}
        for node, data in self.LTM_cache[node_type].items():
            node_activations[node] = data["activation"]
        self.get_logger().info(f"Selecting most activated {node_type} - Activations: {node_activations}")
        selected = max(zip(node_activations.values(), node_activations.keys()))[1]
        return selected, node_activations
    
    def get_node_activations_by_type(self, node_type, ltm_cache):
        act_dict={}
        nodes=ltm_cache[node_type].keys()
        for node in nodes:
            act_dict[node]=ltm_cache[node_type][node]["activation"]
        #Sorts the dictionary by activation from more activated to less activated
        act_dict = {k: v for k, v in sorted(act_dict.items(), key=lambda item: item[1], reverse=True)}
        return act_dict
    
    def get_node_activations_by_list(self, node_list, ltm_cache):
        act_dict={}
        for node in node_list:
            act_dict[node]=self.get_node_data(node, ltm_cache)["activation"]
        #Sorts the dictionary by activation from more activated to less activated
        act_dict = {k: v for k, v in sorted(act_dict.items(), key=lambda item: item[1], reverse=True)}
        return act_dict

    def get_all_active_nodes(self, node_type, ltm_cache):
        nodes = [name for name in ltm_cache[node_type] if ltm_cache[node_type][name]["activation"] > self.activation_threshold]
        return nodes
    
    def get_node_data(self, node_name, ltm_cache):
        return next((nodes_dict[node_name] for nodes_dict in ltm_cache.values() if node_name in nodes_dict))

    def update_ltm(self, stm:Episode):
            self.update_pnodes_reward_basis(stm.old_perception, stm.perception, stm.policy, copy(stm.reward_list), stm.actuation, stm.old_ltm_state)


    def update_pnodes_reward_basis(self, old_perception, perception, policy, reward_list, actuation, ltm_cache):
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
        cnode_activations = self.get_node_activations_by_list(cnodes, ltm_cache)
        threshold = self.activation_threshold
        updates = False
        point_added = False

        for cnode in cnode_activations.keys():
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

            world_model_activation = self.get_node_data(world_model, ltm_cache)["activation"]
            goal_activation = self.get_node_data(goal, ltm_cache)["activation"]
            pnode_activation = self.get_node_data(pnode, ltm_cache)["activation"]
            cnode_data = self.get_node_data(cnode, ltm_cache)
            cnode_params = cnode_data['policy_params'].get('policy_params', [{}])[0]
            policy_params = actuation_msg_to_dict(actuation).get('policy_params', [{}])[0] 
             

            if world_model_activation > threshold and goal_activation > threshold and cnode_params == policy_params:
                reward = reward_list.get(goal, 0.0)
                if (reward > threshold):
                    reward_list.pop(goal)
                    if not point_added:
                        self.add_point(pnode, old_perception)
                        updates = True
                        point_added = True
                elif pnode_activation > threshold:
                    self.add_antipoint(pnode, old_perception)
                    updates = True

        for goal, reward in reward_list.items():
            if (reward > threshold) and (not point_added):
                if goal not in self.unlinked_drives:
                    self.new_cnode(old_perception, goal, policy, actuation)
                else:
                    drive = goal
                    goal = self.new_goal(perception, drive)
                    self.new_cnode(old_perception, goal, policy, actuation)
                point_added=True
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

    def new_cnode(self, perception, goal, policy, actuation):
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
        index=0
        while True:
            if f"cnode_{ident}_{index}" not in self.LTM_cache["CNode"]:
                break
            index+=1

        space_class = self.default_class.get("Space")
        pnode_class = self.default_class.get("PNode")
        cnode_class = self.default_class.get("CNode")

        pnode_name = f"pnode_{ident}_{index}"
        pnode = self.create_node_client(
            name=pnode_name, class_name=pnode_class, parameters={"space_class": space_class}
        )

        if not pnode:
            self.get_logger().fatal(f"Failed creation of PNode {pnode_name}")
        self.add_point(pnode_name, perception)

        neighbor_dict = {world_model: "WorldModel", pnode_name: "PNode", goal: "Goal"}
        neighbors = {
            "neighbors": [{"name": node, "node_type": node_type} for node, node_type in neighbor_dict.items()]
        }

        policy_parameter = {"policy_params": actuation_msg_to_dict(actuation)}
        params = {**neighbors, **policy_parameter}

        cnode_name = f"cnode_{ident}_{index}"
        cnode = self.create_node_client(
            name=cnode_name, class_name=cnode_class, parameters=params
        )
        #Add new C-Node as neighbor of the corresponding policy
        policy_success=self.add_neighbor(policy, cnode_name) 

        if not cnode or not policy_success:
            self.get_logger().fatal(f"Failed creation of CNode {cnode_name}")

        self.n_cnodes = self.n_cnodes + 1  # TODO: Consider the posibility of deleting CNodes
        return cnode_name

    def new_goal(self, perception, drive):
        self.get_logger().info("Creating Goal...")
        goal_name = f"goal_{self.n_goals}"
        goal_class = self.default_class.get("Goal")
        space_class = self.default_class.get("Space")
        neighbors= [{"name": drive, "node_type": "Drive"}]
        parameters = {
            "space_class": space_class, 
            "neighbors": neighbors,
            "history_size": 300, #TODO Pass this as parameter from yaml file
            "min_confidence": 0.94, #TODO Pass this as parameter from yaml file
            "ltm_id": self.LTM_id,
            "perception": perception
        }
        goal = self.create_node_client(
            name=goal_name, class_name=goal_class, parameters=parameters
        )
        self.n_goals+=1
        if not goal:
            self.get_logger().fatal(f"Failed creation of Goal {goal_name}")
        return goal_name

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
        params_str = yaml.dump(parameters, sort_keys=False)
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
            if self.iteration>0:
                iterations=self.iteration-self.last_reset
                self.trials_data.append((self.iteration, self.goal_count, iterations, finished))
                self.goal_count+=1
                self.last_reset=self.iteration

            if getattr(self, "world_reset_client", None):
                self.get_logger().info("Requesting world reset service...")
                self.world_reset_client.send_request(iteration=self.iteration, world=self.current_world)
            self.get_logger().info("Asking for a world reset...")
            msg = ControlMsg()
            msg.command = "reset_world"
            msg.world = self.current_world
            msg.iteration = self.iteration
            self.control_publisher.publish(msg)
        return changed
    
    def world_finished(self):
        need_satisfaction = self.get_need_satisfaction(self.get_needs(self.LTM_cache), self.get_clock().now())
        if len(need_satisfaction)>0:
            finished = any((need_satisfaction[need]['satisfied'] for need in need_satisfaction if (need_satisfaction[need]['need_type'] == 'Operational')))
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
        msg.policy = self.stm.policy
        msg.actuation = self.stm.actuation
        msg.perception = perception_dict_to_msg(self.stm.perception)
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
        self.stm.perception = self.read_perceptions()
        self.update_activations()
        self.active_goals = self.get_goals(self.LTM_cache)
        self.stm.reward_list= self.get_goals_reward(self.stm.old_perception, self.stm.perception, self.LTM_cache)
        self.iteration = 1
        
        while (self.iteration <= self.iterations) and (not self.stop):

            if not self.paused:

                self.get_logger().info(
                    "*** ITERATION: " + str(self.iteration) + "/" + str(self.iterations) + " ***"
                )
                self.publish_iteration()
                self.update_activations()
                self.stm.old_ltm_state=deepcopy(self.LTM_cache)
                self.current_policy = self.select_policy(softmax=self.softmax_selection)
                self.current_policy, self.stm.actuation = self.execute_policy(self.stm.perception, self.current_policy)
                self.stm.policy = self.current_policy
                self.stm.old_perception, self.stm.perception = self.stm.perception, self.read_perceptions()
                self.update_activations()
                self.stm.ltm_state=deepcopy(self.LTM_cache)

                self.get_logger().info(
                    f"DEBUG PERCEPTION: \n old_sensing: {self.stm.old_perception} \n     sensing: {self.stm.perception}"
                )


                self.active_goals = self.get_goals(self.stm.old_ltm_state)
                self.stm.reward_list= self.get_goals_reward(self.stm.old_perception, self.stm.perception, self.stm.old_ltm_state)

                self.publish_episode()

                self.update_ltm(self.stm)


                if self.reset_world():
                    reset_sensing = self.read_perceptions()
                    self.update_activations()
                    self.stm.perception = reset_sensing
                    self.stm.ltm_state = deepcopy(self.LTM_cache)

                self.update_policies_to_test(
                    policy=(
                        self.current_policy
                        if not self.sensorial_changes(self.stm.perception, self.stm.old_perception)
                        else None
                    )
                )
                self.update_status()
                self.iteration += 1

        self.close_files()
        if self.kill_on_finish:
            self.kill_commander_client.send_request()


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
