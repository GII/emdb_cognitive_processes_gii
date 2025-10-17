import threading
import numpy as np
import yaml
from copy import copy

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.time import Time

from cognitive_nodes.episode import Episode, Action, action_obj_to_msg
from core.service_client import ServiceClient
from core.utils import perception_dict_to_msg, perception_msg_to_dict, actuation_dict_to_msg, actuation_msg_to_dict, class_from_classname
from cognitive_nodes.episode import reward_dict_to_msg

from std_msgs.msg import String
from core_interfaces.srv import SetChangesTopic, GetNodeFromLTM, UpdateNeighbor, CreateNode
from cognitive_node_interfaces.msg import Episode as EpisodeMsg
from cognitive_node_interfaces.msg import PerceptionStamped, Activation
from cognitive_node_interfaces.srv import GetActivation, AddPoint, IsSatisfied, GetReward, Execute

class CognitiveProcess(Node):
    """
    CognitiveProcess class, base class for cognitive processes in the architecture.
    """
    def __init__(self, name, iterations = 0, trials = 1, LTM_id="", **params):
        """
        Constructor of the CognitiveProcess class.
        """
        super().__init__(name)
        # --- Loop control variables ---
        self.iteration = 0
        self.iterations = iterations
        self.trials = trials
        self.trial = 0
        self.period = 1
        self.paused = False
        self.stop = False

        # --- LTM and STM ---
        self.current_episode = Episode()
        self.LTM_id = LTM_id  # id of LTM currently being run by cognitive loop
        self.LTM_cache = (
            {}
        )  # Nested dics, like {'CNode': {'CNode1': {'activation': 0.0}}, ...}
        self.default_class = {}

        # --- Node/goal/drive management ---
        self.active_goals = []
        self.unlinked_drives = []

        # --- Perception handling ---
        self.perception_suscribers = {}
        self.perception_cache = {}
        self.perception_time = 0
        self.sensorial_changes_val = False

        # --- Activation handling ---
        self.activation_inputs = {}
        self.activation_time = 0
        self.activation_threshold = 0.01

        # --- Callback groups and service clients ---
        self.cbgroup_perception = MutuallyExclusiveCallbackGroup()
        self.cbgroup_server = MutuallyExclusiveCallbackGroup()
        self.cbgroup_client = MutuallyExclusiveCallbackGroup()
        self.cbgroup_loop = MutuallyExclusiveCallbackGroup()
        self.node_clients = (
            {}
        )  # Keys are service name, values are service client object

        self.LTM_changes_client = ServiceClient(SetChangesTopic, f"{self.LTM_id}/set_changes_topic")


    # =========================
    # SETUP
    # =========================

    def set_attributes_from_params(self, params):
        for key, value in params.items():
            self.get_logger().debug("Setting atribute: " + str(key) + " with value: " + str(value))
            setattr(self, key, value)

    def start_threading(self):
        self.loop_thread = threading.Thread(target=self.run, daemon=True)
        self.semaphore = threading.Semaphore()
        self.loop_thread.start()

    def setup(self):
        """
        Initial configuration of the MainLoop node.
        This method sets up the LTM, perceptions, files, connectors, control channel, etc.
        """
        self.rng = np.random.default_rng(self.random_seed)
        self.read_ltm()
        self.configure_perceptions()
        self.setup_ltm_suscription()
        self.setup_connectors()
        self.setup_control_channel()
        self.LTM_changes_client.send_request(changes_topic=True)

    def setup_ltm_suscription(self):
        """
        Sets up a subscription to the LTM state topic.
        """
        self.ltm_suscription = self.create_subscription(String, "state", self.ltm_change_callback, 0, callback_group=self.cbgroup_client)

    def setup_connectors(self):
        """
        Configures the default classes for the cognitive nodes.
        """
        if hasattr(self, "Connectors"):
            for connector in self.Connectors:
                self.default_class[connector["data"]] = connector.get("default_class")


    def setup_control_channel(self):
        """
        Configures the control channel.
        """
        control_msg=self.Control["control_msg"]
        episode_msg=self.Control["episodes_msg"]
        world_reset_msg=self.Control.get("world_reset_msg", None)
        world_reset_service=self.Control.get("world_reset_service", None)
        self.action_service = self.Control.get("executed_action_service", None)
        self.action_msg = self.Control.get("executed_action_msg", None)
        self.control_publisher = self.create_publisher(class_from_classname(control_msg), self.Control["control_topic"], 10)
        self.episode_publisher = self.create_publisher(class_from_classname(episode_msg), self.Control["episodes_topic"], 10)
        if world_reset_msg and world_reset_service:
            self.world_reset_client = ServiceClient(class_from_classname(world_reset_msg), world_reset_service)
        if self.action_service and self.action_msg:
            self.action_client = ServiceClient(class_from_classname(self.action_msg), self.action_service)

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
    
    

    # =========================
    # LTM READING AND PROCESSING
    # =========================

    def read_ltm(self, ltm_dump=None):
        """
        Makes an empty call for the LTM get_node service which returns all the nodes
        stored in the LTM. Then, a LTM cache dictionary is populated with the data.

        :param ltm_dump: Optional LTM dump to be used instead of requesting it from the current one.
        :type ltm_dump: dict
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
                    self.create_activation_input(node, node_type)
                else: #If node exists update data (except activations)
                    node_data = ltm_dump[node_type][node]
                    del node_data["activation"]
                    del node_data["activation_timestamp"]
                    self.LTM_cache[node_type][node].update(node_data) 
        
        #Remove elements in LTM Cache that were removed from LTM.
        for node_type in self.LTM_cache.keys():
            for node in self.LTM_cache[node_type]:
                if ltm_dump[node_type].get(node, None) is None:
                    del self.LTM_cache[node_type][node]
                    self.delete_activation_input(node)

        # Check if there are any drives not linked to goals
        self.unlinked_drives = self.get_unlinked_drives()
    
    def ltm_change_callback(self, msg):
        """
        PENDING METHOD: This method will read changes made in the LTM
        external to the cognitive process and update the LTM and perception
        caches accordingly.

        :param msg: Message containing the LTM dump.
        :type msg: std_msgs.msg.String
        """
        self.semaphore.acquire()
        self.get_logger().info("Processing change from LTM...")
        ltm_dump = yaml.safe_load(msg.data)
        self.read_ltm(ltm_dump=ltm_dump)
        #self.configure_perceptions #CHANGE THIS SO THAT NEW PERCEPTIONS ARE ADDED AND OLD PERCEPTIONS ARE DELETED
        self.semaphore.release()

    def request_ltm(self):
        """
        Requests the LTM dump from its service.

        :return: LTM dump as a dictionary.
        :rtype: dict
        """
        # Call get_node service from LTM
        service_name = "/" + str(self.LTM_id) + "/get_node"
        request = ""
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(GetNodeFromLTM, service_name)
        ltm_response = self.node_clients[service_name].send_request(name=request)
        ltm = yaml.safe_load(ltm_response.data)

        return ltm
    
    # =========================
    # ACTION EXECUTION
    # =========================
    def execute_action(self, perception: dict, action: Action):
        if action.policy_id != 0:
            # Execute a policy
            raise NotImplementedError("Implementation of ID to policy mapping is TBD")
            policy, action = self.execute_policy(perception, action.policy_id)
        else:
            return self.execute_actuation(action.actuation)

    def execute_policy(self, perception, policy):
        """
        Execute a policy.
        This method sends a request to the policy to be executed.

        :param perception: The perception to be used in the policy execution.
        :type perception: dict
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
        action= policy_response.action
        self.get_logger().info("Executed policy " + str(policy_response.policy) + "...")
        return policy_response.policy, action 
    
    def execute_actuation(self, actuation:dict):
        """
        Execute an action by sending it to the action service.

        :param actuation: The actuation dictionary containing the action details.
        :type actuation: dict
        :return: The response from the action service.
        :rtype: cognitive_node_interfaces.srv.Execute.Response
        """
        if self.action_client:
            actuation_msg = actuation_dict_to_msg(actuation)
            response = self.action_client.send_request(action=actuation_msg)
            self.get_logger().info("Executed action with response: " + str(response))
            return response
        else:
            self.get_logger().error("Action client not configured.")
            return False

    # =========================
    # PUBLISHING & STATUS
    # =========================

    def publish_episode(self):
        """
        Publish the current episode data to the episode topic.
        """
        msg=EpisodeMsg()
        msg.old_perception = perception_dict_to_msg(self.current_episode.old_perception)
        msg.parent_policy = self.current_episode.parent_policy
        msg.action = action_obj_to_msg(self.current_episode.action)
        msg.perception = perception_dict_to_msg(self.current_episode.perception)
        msg.reward_list = reward_dict_to_msg(self.current_episode.reward_list)
        msg.timestamp = self.get_clock().now().to_msg()
        self.episode_publisher.publish(msg)

    # =========================
    # PERCEPTION HANDLING
    # =========================
    def read_perceptions(self):
        """
        Reads the perception cache and returns the latest perception.

        This method iterates the perception cache dictionary. For each
        perception waits for the flag that signals that the value has
        been updated. Then, the value is copied in the sensing dictionary.

        When the whole cache is processed, the sensing dictionary is returned.

        :return: Latest sensing.
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

    def sensorial_changes(self, sensing, old_sensing, threshold=0.01):
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
                        if difference > threshold:
                            self.get_logger().debug("Sensorial change detected")
                            self.sensorial_changes_val = True
                            return True
                else:
                    if abs(perception[0] - perception_old[0]) > threshold:
                        self.get_logger().debug("Sensorial change detected")
                        self.sensorial_changes_val = True
                        return True
        self.get_logger().debug("No sensorial change detected")
        self.sensorial_changes_val = False
        return False

    # =========================
    # ACTIVATION HANDLING
    # =========================

    def update_activations(self):
        """
        This method updates the activations of the nodes in the LTM cache.
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
        :return: Activation value.
        :rtype: float
        """

        service_name = "cognitive_node/" + str(name) + "/get_activation"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(GetActivation, service_name)
        perception = perception_dict_to_msg(sensing)
        activation = self.node_clients[service_name].send_request(perception=perception)
        return activation.activation
    
    def create_activation_input(self, name, node_type): #Adds a node from the activation inputs list.
        """
        This method creates a new activation input for a node. 

        :param name: Name of the node to be added as an activation input.
        :type name: str
        :param node_type:  Type of the node to be added as an activation input.
        :type node_type: str
        """
        if name not in self.activation_inputs:
            subscriber=self.create_subscription(Activation, 'cognitive_node/' + str(name) + '/activation', self.read_activation_callback, 1, callback_group=self.cbgroup_server)
            flag=threading.Event()
            self.activation_inputs[name]=dict(node_type=node_type, subscriber=subscriber, flag=flag)
            self.get_logger().debug(f'Created new activation input: {name} of type {node_type}')
        else:
            self.get_logger().error(f'Tried to add {name} to activation inputs more than once')
    
    def delete_activation_input(self, name): #Deletes a node from the activation inputs list. By default reads activations.
        """
        This method deletes an activation input for a node.

        :param name: Name of the node to be deleted as an activation input.
        :type name: str
        """
        if name in self.activation_inputs:
            self.destroy_subscription(self.activation_inputs[name])
            self.activation_inputs.pop(name)


    def read_activation_callback(self, msg: Activation):
        """
        This method receives a message from an activation topic, processes the
        message and updates the activation in the LTM cache.

        :param msg: Message that contains the activation information.
        :type msg: cognitive_node_interfaces.msg.Activation
        """
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

    def read_activation_callback(self, msg: Activation):
        """
        This method receives a message from an activation topic, processes the
        message and updates the activation in the LTM cache.

        :param msg: Message that contains the activation information.
        :type msg: cognitive_node_interfaces.msg.Activation
        """
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
        

    # =========================
    # LTM UPDATES
    # =========================
    def update_ltm(self, stm:Episode):
        """
        Placeholder method to update the LTM with the current STM data.
        """
        raise NotImplementedError("This method should be implemented in the derived class.")
    
    def add_point(self, name, sensing):
        """
        Sends the request to add a point to a P-Node.

        :param name: Name of the P-Node.
        :type name: str
        :param sensing: Sensorial data to be added as a point.
        :type sensing: dict
        :return: Success status received from the P-Node.
        :rtype: bool
        """

        service_name = "pnode/" + str(name) + "/add_point"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(AddPoint, service_name)

        perception = perception_dict_to_msg(sensing)
        response = self.node_clients[service_name].send_request(point=perception, confidence=1.0)
        self.get_logger().info(f"Added point in pnode {name}")
        self.get_logger().debug(f"POINT: {str(sensing)}")
        return response.added

    def add_antipoint(self, name, sensing):
        """
        Sends the request to add an antipoint to a P-Node.

        :param name: Name of the P-Node.
        :type name: str
        :param sensing: Sensorial data to be added as a antipoint.
        :type sensing: dict
        :return: Success status received from the P-Node.
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
        return response.added

    def new_cnode(self, perception, goal, policy):
        """
        This method creates a new C-Node/P-Node pair.

        :param perception: Perception to be added as the first point in the P-Node.
        :type perception: dict
        :param goal: Goal that will be linked to the C-Node.
        :type goal: str
        :param policy: Policy that will be linked to the C-Node.
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

        if not pnode:
            self.get_logger().fatal(f"Failed creation of PNode {pnode_name}")
        self.add_point(pnode_name, perception)

        neighbor_dict = {world_model: "WorldModel", pnode_name: "PNode", goal: "Goal"}
        neighbors = {
            "neighbors": [{"name": node, "node_type": node_type} for node, node_type in neighbor_dict.items()]
        }

        cnode_name = f"cnode_{ident}"
        cnode = self.create_node_client(
            name=cnode_name, class_name=cnode_class, parameters=neighbors
        )
        #Add new C-Node as neighbor of the corresponding policy
        policy_success=self.add_neighbor(policy, cnode_name) 

        if not cnode or not policy_success:
            self.get_logger().fatal(f"Failed creation of CNode {cnode_name}")

        self.n_cnodes = self.n_cnodes + 1  # TODO: Consider the posibility of deleting CNodes
        return cnode_name

    def new_goal(self, perception, drive):
        """
        This method creates a new Goal node linked to a Drive.

        :param perception: Perception to be used in the Goal creation.
        :type perception: dict
        :param drive: Drive to which the Goal will be linked.
        :type drive: str
        :return: Name of the created Goal node.
        :rtype: str
        """
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
        :param parameters: Optional parameters that can be passed to the node, defaults to {}.
        :type parameters: dict
        :return: Success status received from the commander.
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

    def request_neighbors(self, name):
        """
        This method calls the service to get the neighbors of a node.

        :param name: Node name.
        :type name: str
        :return: List of dictionaries with the information of each neighbor of the node. [{'name': <Neighbor Name>, 'node_type': <Neighbor type>},...].
        :rtype: list
        """

        data_dict = self.get_node_data(name, self.LTM_cache)
        neighbors = data_dict["neighbors"]

        self.get_logger().debug(f"REQUESTED NEIGHBORS: {neighbors}")

        return neighbors
    
    def add_neighbor(self, node_name, neighbor_name):
        """
        This method adds a neighbor to a node in the LTM.

        :param node_name: Name of the node to which the neighbor will be added.
        :type node_name: str
        :param neighbor_name: Name of the neighbor to be added.
        :type neighbor_name: str
        :return: True if the neighbor was added successfully, False otherwise.
        :rtype: bool
        """
        service_name=f"{self.LTM_id}/update_neighbor"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(UpdateNeighbor, service_name)
        response=self.node_clients[service_name].send_request(node_name=node_name, neighbor_name=neighbor_name, operation=True)
        return response.success

    # =========================
    # LTM CACHE UTILITY METHODS
    # =========================

    def get_max_activation_node(self, node_type):
        """
        This method retrieves the node with the maximum activation of a given type from the LTM cache.

        :param node_type:  Type of the node to be selected (e.g., "WorldModel", "Goal", etc.).
        :type node_type: str
        :return: Tuple containing the name of the node with the maximum activation and a dictionary with all activations of that type.
        :rtype: tuple
        """
        node_activations = {}
        for node, data in self.LTM_cache[node_type].items():
            node_activations[node] = data["activation"]
        if not node_activations:
            return None, {}
        self.get_logger().info(f"Selecting most activated {node_type} - Activations: {node_activations}")
        selected = max(zip(node_activations.values(), node_activations.keys()))[1]
        return selected, node_activations
    
    def get_node_activations_by_type(self, node_type, ltm_cache):
        """
        This method retrieves the activations of all nodes of a given type from the LTM cache.

        :param node_type: Type of the nodes to be selected (e.g., "WorldModel", "Goal", etc.).
        :type node_type: str
        :param ltm_cache: LTM cache containing the nodes and their data.
        :type ltm_cache: dict
        :return: Dictionary with node names as keys and their activations as values, sorted by activation.
        :rtype: dict
        """
        act_dict={}
        nodes=ltm_cache[node_type].keys()
        for node in nodes:
            act_dict[node]=ltm_cache[node_type][node]["activation"]
        #Sorts the dictionary by activation from more activated to less activated
        act_dict = {k: v for k, v in sorted(act_dict.items(), key=lambda item: item[1], reverse=True)}
        return act_dict
    
    def get_node_activations_by_list(self, node_list, ltm_cache):
        """
        This method retrieves the activations of a list of nodes from the LTM cache.

        :param node_list: List of node names to retrieve activations for.
        :type node_list: list
        :param ltm_cache: LTM cache containing the nodes and their data.
        :type ltm_cache: dict
        :return: Dictionary with node names as keys and their activations as values, sorted by activation.
        :rtype: dicts
        """
        act_dict={}
        for node in node_list:
            act_dict[node]=self.get_node_data(node, ltm_cache)["activation"]
        #Sorts the dictionary by activation from more activated to less activated
        act_dict = {k: v for k, v in sorted(act_dict.items(), key=lambda item: item[1], reverse=True)}
        return act_dict

    def get_all_active_nodes(self, node_type, ltm_cache):
        """
        This method retrieves all active nodes of a given type from the LTM cache.

        :param node_type: Type of the nodes to be selected (e.g., "WorldModel", "Goal", etc.).
        :type node_type: str
        :param ltm_cache: LTM cache containing the nodes and their data.
        :type ltm_cache: dict
        :return: List of active nodes of the specified type.
        :rtype: list
        """
        nodes = [name for name in ltm_cache[node_type] if ltm_cache[node_type][name]["activation"] > self.activation_threshold]
        return nodes
    
    def get_node_data(self, node_name, ltm_cache):
        """
        This method retrieves the data of a node from the LTM cache.

        :param node_name: Name of the node to retrieve data for.
        :type node_name: str
        :param ltm_cache: LTM cache containing the nodes and their data.
        :type ltm_cache: dict
        :return: Data of the node as a dictionary.
        :rtype: dict
        """
        return next((nodes_dict[node_name] for nodes_dict in ltm_cache.values() if node_name in nodes_dict))
    
    def get_node_type(self, node_name, ltm_cache):
        """
        This method retrieves the type of a node from the LTM cache.

        :param node_name: Name of the node to retrieve the type for.
        :type node_name: str
        :param ltm_cache: LTM cache containing the nodes and their data.
        :type ltm_cache: dict
        :return: Data of the node as a dictionary.
        :rtype: dict
        """
        return next((node_type for node_type, nodes_dict in ltm_cache.items() if node_name in nodes_dict))
    
    # =========================
    # GOALS, REWARDS, NEEDS
    # =========================
    
    def get_goals(self, ltm_cache):
        """
        This method retrieves all active goals from the LTM cache.

        :param ltm_cache: LTM cache containing the nodes and their data.
        :type ltm_cache: dict
        :return: List of active goals.
        :rtype: list
        """
        goals = self.get_all_active_nodes("Goal", ltm_cache)
        return goals
    
    def get_goals_reward(self, old_sensing, sensing, ltm_cache):
        """
        This method retrieves the rewards for each active goal based on the old and current sensing.

        :param old_sensing: Old sensing data.
        :type old_sensing: dict
        :param sensing: Current sensing data.
        :type sensing: dict
        :param ltm_cache: LTM cache containing the nodes and their data.
        :type ltm_cache: dict
        :return: Dictionary with goal names as keys and their corresponding rewards as values.
        :rtype: dict
        """
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
            active_drives = [
                drive for drive in self.unlinked_drives
                if ltm_cache.get("Drive", {}).get(drive, {}).get("activation", 0) > self.activation_threshold
            ]
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
        """
        This method retrieves the drives that are not linked to any goal in the LTM cache.

        :return: List of unlinked drives. If there are no unlinked drives, it returns an empty list.
        :rtype: list
        """
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
        if WM is None:
            self.get_logger().info("No World Model found in LTM")
        else:
            self.get_logger().info(f"Selecting world model with highest activation: {WM} ({WM_activations[WM]})")
        return WM
    
    def get_purposes(self, ltm_cache):
        """
        This method retrieves all active purposes from the LTM cache.

        :param ltm_cache: LTM cache containing the nodes and their data.
        :type ltm_cache: dict
        :return: List of active purposes.
        :rtype: list
        """
        purposes = self.get_all_active_nodes("RobotPurpose", ltm_cache)

        self.get_logger().info(f"Active Purposes: {purposes}")
                    
        return purposes
    
    def get_purpose_satisfaction(self, purpose_list, timestamp):
        """
        This method retrieves the satisfaction of each purpose in the purpose_list.

        :param purpose_list: List of purposes.
        :type purpose_list: list
        :param timestamp: Timestamp to be used for the request.
        :type timestamp: rclpy.time.Time
        :return: Dictionary with purpose names as keys and their satisfaction status as values.
        :rtype: dict
        """
        self.get_logger().info("Reading satisfaction...")
        satisfaction = {}
        response=IsSatisfied.Response()
        for purpose in purpose_list:
            service_name = "robot_purpose/" + str(purpose) + "/get_satisfaction"
            if service_name not in self.node_clients:
                self.node_clients[service_name] = ServiceClient(IsSatisfied, service_name)
            while not response.updated:
                response = self.node_clients[service_name].send_request(
                    timestamp=timestamp.to_msg()
                )
            satisfaction[purpose] = dict(satisfied=response.satisfied, purpose_type=response.purpose_type, terminal=response.terminal)
            response.updated = False

        self.get_logger().info(f"Satisfaction list: {satisfaction}")

        return satisfaction
    
    # =========================
    # Process Execution
    # =========================

    def run(self):
        """
        Main loop of the cognitive process. This method is executed in a separate thread.
        It runs the cognitive process, reading perceptions, updating activations, and processing episodes.
        """
        raise NotImplementedError("This method should be implemented in the derived class.")


    
