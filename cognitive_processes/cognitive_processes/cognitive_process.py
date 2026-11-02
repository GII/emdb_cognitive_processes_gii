import threading
import time
import traceback
import numpy as np
import yaml

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.time import Time

from cognitive_nodes.episode import Episode, episode_obj_to_msg
from core.blocking_wait_check import check_blocking_wait
from core.service_client import SERVICE_CHECK_PERIOD
from core.utils import class_from_classname, compare_perceptions, resolve_seed
from core.container import Container, consolidate_containers

from std_msgs.msg import String
from core_interfaces.msg import Container as ContainerMsg
from core_interfaces.srv import SetChangesTopic, GetNodeFromLTM, UpdateNeighbor, CreateNode
from cognitive_node_interfaces.msg import Activation
from cognitive_node_interfaces.srv import GetActivation, AddPoints, IsSatisfied, GetReward, Execute, DuplicateNode, LogExecution
from cognitive_processes_interfaces.srv import Pause

class StartsAfterConstruction(type):
    """
    Metaclass of the cognitive processes: starts the process thread once the object
    has been fully constructed, that is, after the constructor of the class being
    instantiated, whatever its depth in the hierarchy. Subclasses inherit it, so
    they don't have to start the thread themselves.
    """
    def __call__(cls, *args, **kwargs):
        process = super().__call__(*args, **kwargs)  # Runs the whole chain of constructors
        process.start_process_thread()
        return process


class CognitiveProcess(Node, metaclass=StartsAfterConstruction):
    """
    CognitiveProcess class, base class for cognitive processes in the architecture.

    The process runs in its own thread (setup, then run), which starts automatically
    once the object has been fully constructed (see StartsAfterConstruction).
    """
    def __init__(self, name, iterations = 0, trials = 1, LTM_id="", **params):
        """
        Constructor of the CognitiveProcess class.
        """
        super().__init__(name)
        self.name = name
        # --- Loop control variables ---
        self.iteration = 0
        self.iterations = iterations
        self.trials = trials
        self.trial = 0
        self.period = 1
        self.paused = False
        self.stop = False
        self.ready = threading.Event()  # Set when setup has finished
        self.ready_lock = threading.Lock()  # Protects ready_callbacks and the change of ready
        self.ready_callbacks = []  # (node, callback) pairs to schedule when the process is ready (see call_when_ready)
        self.loop_thread = None  # Process thread, started after the constructor (see StartsAfterConstruction)

        # --- LTM and STM ---
        self.current_episode = Episode()
        self.LTM_id = LTM_id  # id of LTM currently being run by cognitive loop
        self.LTM_cache = (
            {}
        )  # Nested dics, like {'CNode': {'CNode1': {'activation': 0.0}}, ...}
        self.default_class = {}
        self.default_params = {}

        # --- Node/goal/drive management ---
        self.current_world = None
        self.active_goals = []
        self.unlinked_drives = []

        # --- Perception handling ---
        self.latest_perception = None
        self.perception_cache = {}
        self.perception_time = 0
        self.sensorial_changes_val = False

        # --- Activation handling ---
        self.activation_inputs = {}
        self.activation_time = 0
        self.activation_threshold = 0.01
        self.activation_logs = False
        self.activation_warning_period = 5.0  # Seconds between warnings while waiting for activations

        # --- Synchronization between the loop thread and the executor callbacks ---
        # Protects LTM_cache, activation_inputs and pending_ltm_dump. It is held only for short
        # operations, never while waiting, so the executor callbacks never block for long.
        self.activation_condition = threading.Condition()
        self.pending_ltm_dump = None  # Latest LTM dump received and not applied yet

        # --- Callback groups and service clients ---
        self.cbgroup_perception = MutuallyExclusiveCallbackGroup()
        self.cbgroup_server = MutuallyExclusiveCallbackGroup()
        self.cbgroup_client = MutuallyExclusiveCallbackGroup()
        self.node_clients = (
            {}
        )  # Keys are service name, values are service client object
        self.own_clients = {}  # Clients created on this node (see own_client), by service name

        self.pause_service = self.create_service(Pause, f"{self.name}/pause", self.pause_callback, callback_group=self.cbgroup_server)


    # =========================
    # SETUP
    # =========================

    def set_attributes_from_params(self, params):
        for key, value in params.items():
            self.get_logger().debug("Setting atribute: " + str(key) + " with value: " + str(value))
            setattr(self, key, value)

    def start_process_thread(self):
        """
        Starts the process (setup and run) in its own thread. The metaclass calls it
        once the object has been fully constructed. Further calls do nothing.
        """
        if self.loop_thread is not None:
            return
        self.loop_thread = threading.Thread(target=self.process_thread, daemon=True)
        self.loop_thread.start()

    def start_threading(self):
        """
        No longer needed: the process thread starts automatically once the object has
        been fully constructed. Kept so that processes that still call it from their
        constructor don't start the thread too early. It only logs a warning.
        """
        self.get_logger().warning(f"{type(self).__name__} calls start_threading(), which is no longer needed: the process thread starts automatically after the constructor")

    def process_thread(self):
        """
        Body of the process thread: configures the process (setup), signals that it
        is ready and runs its loop (run).
        """
        try:
            self.setup()
        except Exception:
            self.get_logger().fatal(f"Setup of {self.name} failed, the process will not run:\n{traceback.format_exc()}")
            raise
        with self.ready_lock:
            self.ready.set()
            ready_callbacks, self.ready_callbacks = self.ready_callbacks, []
        for node, callback in ready_callbacks:
            node.executor.create_task(callback)
        self.run()

    def call_when_ready(self, node, callback):
        """
        Schedules a callback in the executor of a node once the setup of the process
        has finished, or right away if it already has. Neither the caller nor the
        callback wait, so nodes can use it from their callbacks.

        :param node: Node whose executor runs the callback. It must already be spinning in
            an executor, so call this method from one of its callbacks.
        :type node: rclpy.node.Node
        :param callback: Function without arguments.
        :type callback: callable
        """
        with self.ready_lock:
            if not self.ready.is_set():
                self.ready_callbacks.append((node, callback))
                return
        node.executor.create_task(callback)

    def setup(self):
        """
        Initial configuration of the MainLoop node.
        This method sets up the LTM, perceptions, files, connectors, control channel, etc.
        """
        self.random_seed = resolve_seed(self.random_seed)
        self.get_logger().info(f"Cognitive process using random seed {self.random_seed}")
        self.rng = np.random.default_rng(self.random_seed)
        self.read_ltm()
        self.configure_perceptions()
        self.setup_ltm_suscription()
        self.setup_connectors()
        self.setup_control_channel()
        # Here, in the process thread: the constructor runs in an executor callback, where it must not wait for the service
        self.call_services([(self.own_client(SetChangesTopic, f"{self.LTM_id}/set_changes_topic"), SetChangesTopic.Request(changes_topic=True))])

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
                self.default_params[connector["data"]] = connector.get("parameters", {})


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
            self.world_reset_client = self.own_client(class_from_classname(world_reset_msg), world_reset_service)
        if self.action_service and self.action_msg:
            self.action_client = self.own_client(class_from_classname(self.action_msg), self.action_service)

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
                ContainerMsg,
                f"/perception/{perception}/value",
                self.receive_perception_callback,
                1,
                callback_group=self.cbgroup_perception,
            )
            data = None
            flag = threading.Event()
            self.get_logger().debug(f"Subscription to: /perception/{perception}/value created")
            self.perception_cache[perception] = dict(subscriber=subscriber, data=data, flag=flag)
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
        
        with self.activation_condition:
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
                for node in list(self.LTM_cache[node_type]):
                    if ltm_dump.get(node_type, {}).get(node, None) is None:
                        del self.LTM_cache[node_type][node]
                        self.delete_activation_input(node)
    
    def ltm_change_callback(self, msg):
        """
        Receives a change made in the LTM external to the cognitive process.

        The LTM dump is not applied here but stored, and the loop thread applies it
        (see apply_ltm_changes). This callback must not wait for the loop thread: the loop
        thread waits for activations that are received by callbacks of the same executor,
        which may have a single thread.

        :param msg: Message containing the LTM dump.
        :type msg: std_msgs.msg.String
        """
        self.get_logger().info("Received change from LTM...")
        ltm_dump = yaml.safe_load(msg.data)
        with self.activation_condition:
            # Each message contains the whole LTM, so only the latest one is needed
            self.pending_ltm_dump = ltm_dump
            self.activation_condition.notify_all()

    def apply_ltm_changes(self):
        """
        Applies the latest LTM dump received, if any, to the LTM and activation input caches.
        It must be called from the loop thread.
        """
        with self.activation_condition:
            ltm_dump = self.pending_ltm_dump
            self.pending_ltm_dump = None
            if ltm_dump is not None:
                self.get_logger().info("Processing change from LTM...")
                self.read_ltm(ltm_dump=ltm_dump)
                #self.configure_perceptions #CHANGE THIS SO THAT NEW PERCEPTIONS ARE ADDED AND OLD PERCEPTIONS ARE DELETED

    def request_ltm(self):
        """
        Requests the LTM dump from its service.

        :return: LTM dump as a dictionary.
        :rtype: dict
        """
        # Call get_node service from LTM
        service_name = "/" + str(self.LTM_id) + "/get_node"
        request = ""
        ltm_response = self.call_services([(self.own_client(GetNodeFromLTM, service_name), GetNodeFromLTM.Request(name=request))])[0]
        ltm = yaml.safe_load(ltm_response.data)

        return ltm
    
    # =========================
    # ACTION EXECUTION
    # =========================
    def execute_policy(self, perception, policy):
        """
        Execute a policy.
        This method sends a request to the policy to be executed.

        :param perception: The perception to be used in the policy execution.
        :type perception: core.container.Container
        :param policy: The policy to execute.
        :type policy: str
        :return: The response from executing the policy.
        :rtype: The executed policy.
        """
        service_name = "policy/" + str(policy) + "/execute"
        perc_msg=perception.to_msg()
        policy_response = self.call_services([(self.own_client(Execute, service_name), Execute.Request(perception=perc_msg))])[0]
        episode = policy_response.episode
        self.get_logger().info("Executed policy " + str(policy_response.policy) + "...")
        return policy_response.policy, episode 
    
    def execute_action(self, action):
        """
        Execute an action by sending it to the action service.

        :param action: The action to execute.
        :type action: core.container.Container
        :return: The response from the action service.
        :rtype: cognitive_node_interfaces.srv.Execute.Response
        """
        if self.action_client:
            actuation_msg = action.to_msg()
            response = self.call_services([(self.action_client, self.action_client.srv_type.Request(action=actuation_msg))])[0]
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
        msg=episode_obj_to_msg(self.current_episode)
        self.episode_publisher.publish(msg)

    # =========================
    # PERCEPTION HANDLING
    # =========================
    def read_perceptions(self):
        """
        Reads the perception cache and returns the latest perception.

        This method iterates the perception cache dictionary. For each
        perception waits for the flag that signals that the value has
        been updated. Then, the value is copied in the perception container.

        When the whole cache is processed, the perception container is returned.

        :return: Latest perception.
        :rtype: core.container.Container
        """

        self.get_logger().info("Reading perceptions...")

        self.perception_time = self.get_clock().now().nanoseconds

        # Clear all flags to get fresh readings
        for (
            sensor
        ) in self.perception_cache.keys():  # TODO: Consider perception activation when reading
            self.perception_cache[sensor]["flag"].clear()
            self.get_logger().debug("Clearing flags: " + str(sensor))
        
        # Wait for all perceptions to be updated
        for (
            sensor
        ) in self.perception_cache.keys():  # TODO: Consider perception activation when reading
            if not self.perception_cache[sensor]["flag"].wait(timeout=5.0):
                self.get_logger().warning(f"Timeout waiting for perception of sensor {sensor}")
                self.perception_cache[sensor]["flag"].wait()
            self.perception_cache[sensor]["flag"].clear()
            self.get_logger().debug("Processing perception: " + str(sensor))

        # Write the perceptions in the perception container and return it
        perception_data = [self.perception_cache[sensor]["data"] for sensor in self.perception_cache.keys()]
        self.latest_perception = consolidate_containers(perception_data, name="perception", container_type="perception")
        self.get_logger().debug("DEBUG Read Perceptions: " + str(self.latest_perception))
        return self.latest_perception

    def receive_perception_callback(self, msg):
        """
        Receives a message from a perception value topic, processes the
        message and copies it to the perception cache. Finally sets the
        flag to signal that the value has been updated.

        :param msg: Message that contains the perception.
        :type msg: core_interfaces.msg.Container
        """
        if msg.max_size>1:
            self.get_logger().error(f'Received perception with multiple readings: ({msg.name}). Perception messages should (currently) include only one reading!')
        elif msg.max_size==1:
            node_name=msg.name
            if node_name in self.perception_cache:
                if self.perception_cache[node_name]['data'] is None:
                    self.perception_cache[node_name]['data']=Container.from_msg(msg)
                else:
                    self.perception_cache[node_name]['data'].push_from_msg(msg)
                reading_time = msg.timestamps[-1]
                if reading_time > self.perception_time:
                    self.perception_cache[node_name]['flag'].set()
            else:
                self.get_logger().error(
                    "Received perception not registered in local perception cache!!!"
                )

        else:
            self.get_logger().warn("Empty perception recieved in P-Node")

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
        self.sensorial_changes_val = not compare_perceptions(sensing, old_sensing, threshold, label_mode="equal")
        return self.sensorial_changes_val

    # =========================
    # ACTIVATION HANDLING
    # =========================

    def update_activations(self, timeout=None):
        """
        This method updates the activations of the nodes in the LTM cache.

        It waits until every node in the activation inputs has published an activation
        computed after this method was called. LTM changes received meanwhile are applied
        while waiting: the method stops waiting for deleted nodes and starts waiting for
        new ones. While it waits, it periodically logs a warning with the pending nodes.

        :param timeout: Maximum time to wait in seconds. None (default) waits without limit.
        :type timeout: float
        :return: True if all the activations were received, False if the timeout expired.
        :rtype: bool
        """

        self.get_logger().info("Updating activations...")
        start = time.monotonic()
        next_warning = start + self.activation_warning_period
        with self.activation_condition:
            self.activation_time=self.get_clock().now().nanoseconds
            for node in self.activation_inputs:
                self.activation_inputs[node]['updated'] = False

            while True:
                self.apply_ltm_changes()
                pending = self.pending_activations()
                if not pending:
                    break
                now = time.monotonic()
                if timeout is not None and now - start >= timeout:
                    self.log_pending_activations(pending, now - start, timeout=True)
                    self.activation_logs = False
                    return False
                if now >= next_warning:
                    self.log_pending_activations(pending, now - start)
                    self.activation_logs = True
                    next_warning += self.activation_warning_period
                deadline = next_warning if timeout is None else min(next_warning, start + timeout)
                # Releases the lock while waiting, so the executor callbacks can run
                self.activation_condition.wait(timeout=deadline - now)
        self.activation_logs = False
        self.get_logger().debug("DEBUG - LTM CACHE:" + str(self.LTM_cache))
        return True

    def pending_activations(self):
        """
        Returns the nodes whose activation update_activations is still waiting for.

        :return: Names of the pending nodes.
        :rtype: list
        """
        skipped = self.skipped_activation_inputs()
        return [node for node, data in self.activation_inputs.items() if not data['updated'] and node not in skipped]

    def skipped_activation_inputs(self):
        """
        Returns the nodes whose activation update_activations does not wait for.
        Subclasses override it, for instance, for nodes that cannot publish their activation
        while the process runs.

        :return: Names of the nodes.
        :rtype: set
        """
        return set()

    def log_pending_activations(self, pending, waited, timeout=False):
        """
        Logs the nodes whose activation update_activations is still waiting for.

        :param pending: Names of the pending nodes.
        :type pending: list
        :param waited: Time waited in seconds.
        :type waited: float
        :param timeout: The wait was abandoned because the timeout expired.
        :type timeout: bool
        """
        details = []
        for node in pending:
            node_type = self.activation_inputs[node]['node_type']
            timestamp = self.LTM_cache.get(node_type, {}).get(node, {}).get('activation_timestamp', 0)
            age = (self.activation_time - timestamp) / 1e9 if timestamp else None
            last = f"last activation {age:.3f} s before the wait" if age is not None else "no activation received"
            details.append(f"{node_type} {node} ({last})")
        message = f"Waiting for the activation of {len(pending)} node(s) for {waited:.1f} s: " + "; ".join(details)
        if timeout:
            message = "Activation timeout. " + message
        self.get_logger().warning(message)

    def request_activation(self, name, perception):
        """
        This method calls the service to get the activation of a node.

        :param name: Name of the node.
        :type name: str
        :param perception: Perception used to calculate the activation.
        :type perception: core.container.Container
        :return: Activation value.
        :rtype: float
        """

        service_name = "cognitive_node/" + str(name) + "/get_activation"
        perception_msg = perception.to_msg()
        activation = self.call_services([(self.own_client(GetActivation, service_name), GetActivation.Request(perception=perception_msg))])[0]
        return activation
    
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
            self.activation_inputs[name]=dict(node_type=node_type, subscriber=subscriber, updated=False)
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
            self.destroy_subscription(self.activation_inputs[name]['subscriber'])
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
        with self.activation_condition:
            node_data = self.LTM_cache.get(node_type, {}).get(node_name, None)
            activation_input = self.activation_inputs.get(node_name, None)
            if node_data is None or activation_input is None:
                # The node was removed from the LTM, or it is not in the cache yet
                return
            old_timestamp=node_data['activation_timestamp']
            node_data['activation']=activation
            node_data['activation_timestamp']=timestamp

            if self.activation_logs:
                self.get_logger().debug(f'Activation received: {node_type} {node_name} - Activation: {activation} - Timestamp: {timestamp} / Activation Time: {self.activation_time}')
            
            if timestamp > self.activation_time :            
                activation_input['updated']=True
                self.activation_condition.notify_all()
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
    
    def add_point(self, name, perception, node_type="pnode", confidence=1.0):
        """
        Sends the request to add a point to a P-Node.

        :param name: Name of the P-Node.
        :type name: str
        :param perception: Perception data to be added as a point.
        :type perception: core.container.Container
        :param confidence: Confidence level for the point.
        :type confidence: float
        :return: Success status received from the P-Node.
        :rtype: bool
        """

        service_name = f"{node_type}/" + str(name) + "/add_points"
        perception_msg = perception.to_msg()
        response = self.call_services([(self.own_client(AddPoints, service_name), AddPoints.Request(points=perception_msg, confidences=[confidence]))])[0]
        added = getattr(response, 'added', False)
        if added:
            self.get_logger().info(f"Added point in {node_type} {name}")
            if node_type=="pnode":
                self.pnodes_success[name] = True
        else:
            self.get_logger().warning(f"Failed to add point in {node_type} {name}")
        self.get_logger().debug(f"POINT: {str(perception)} added={added}")
        return added

    def add_antipoint(self, name, perception, node_type="pnode", confidence=1.0):
        """
        Sends the request to add an antipoint to a P-Node.

        :param name: Name of the P-Node.
        :type name: str
        :param perception: Perception data to be added as a point.
        :type perception: core.container.Container
        :param confidence: Confidence level for the antipoint.
        :type confidence: float
        :return: Success status received from the P-Node.
        :rtype: bool
        """

        service_name = f"{node_type}/" + str(name) + "/add_points"
        perception = perception.to_msg()
        response = self.call_services([(self.own_client(AddPoints, service_name), AddPoints.Request(points=perception, confidences=[-1.0*confidence]))])[0]
        added = getattr(response, 'added', False)
        if added:
            self.get_logger().info(f"Added anti-point in {node_type} {name}")
            if node_type == "pnode":
                self.pnodes_success[name] = False
        else:
            self.get_logger().warning(f"Failed to add anti-point in {node_type} {name}")
        self.get_logger().debug(f"ANTIPOINT: {str(perception)} added={added}")
        return added

    def new_cnode(self, perception, goal, policy):
        """
        This method creates a new C-Node/P-Node pair.

        :param perception: Perception to be added as the first point in the P-Node.
        :type perception: core.container.Container
        :param goal: Goal that will be linked to the C-Node.
        :type goal: str
        :param policy: Policy that will be linked to the C-Node.
        :type policy: str
        """

        self.get_logger().info("Creating Cnode...")
        world_model = self.get_current_world_model()
        ident = f"{world_model}__{goal}__{policy}"

        space_class = self.default_class.get("Space") # NOTE: Space class is now passed through the P-Node's parameters in the yaml file
        space_params = self.default_params.get("Space", {}) #TODO: Pass any parameters from yaml file if needed
        pnode_class = self.default_class.get("PNode")
        pnode_params = self.default_params.get("PNode", {})
        cnode_class = self.default_class.get("CNode")
        cnode_params = self.default_params.get("CNode", {})

        pnode_name = f"pnode_{ident}"
        pnode = self.create_node_client(
            name=pnode_name, class_name=pnode_class, parameters={**pnode_params}
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
            name=cnode_name, class_name=cnode_class, parameters={**cnode_params, **neighbors}
        )
        #Add new C-Node as neighbor of the corresponding policy
        policy_success=self.add_neighbor(policy, cnode_name) 

        if not cnode or not policy_success:
            self.get_logger().fatal(f"Failed creation of CNode {cnode_name}")
        self.log_cnode_execution(cnode_name, success=True)

        self.n_cnodes = self.n_cnodes + 1  # TODO: Consider the posibility of deleting CNodes
        return cnode_name
    
    def log_cnode_execution(self, cnode_name, success):
        """
        This method logs the execution of a C-Node.

        :param cnode_name: Name of the C-Node that was executed.
        :type cnode_name: str
        :param success: Boolean indicating whether the execution was successful.
        :type success: bool
        """
        logging_service = f"cnode/{cnode_name}/log_execution"
        response = self.call_services([(self.own_client(LogExecution, logging_service), LogExecution.Request(success=success))])[0]
        return response.added

    def new_goal(self, perception, drive):
        """
        This method creates a new Goal node linked to a Drive.

        :param perception: Perception to be used in the Goal creation.
        :type perception: core.container.Container
        :param drive: Drive to which the Goal will be linked.
        :type drive: str
        :return: Name of the created Goal node.
        :rtype: str
        """
        self.get_logger().info("Creating Goal...")
        goal_name = f"goal_{self.n_goals}"
        goal_class = self.default_class.get("Goal")
        space_class = self.default_class.get("Space")
        domain = self.get_current_world_model()
        neighbor_dict = {drive: "Drive", domain: "WorldModel"}
        neighbors = {
            "neighbors": [{"name": node, "node_type": node_type} for node, node_type in neighbor_dict.items()]
        }
        goal_params = self.default_params.get("Goal", {})
        if goal_params.get("space_class", None) is None and space_class is not None:
            goal_params["space_class"] = space_class
        parameters = {**goal_params,
            **neighbors,
            "ltm_id": self.LTM_id,
        }
        goal = self.create_node_client(
            name=goal_name, class_name=goal_class, parameters=parameters
        )
        self.add_point(goal_name, perception, node_type="goal")
        self.n_goals+=1
        if not goal:
            self.get_logger().fatal(f"Failed creation of Goal {goal_name}")
        return goal_name
    
    def duplicate_node(self, node_name, perception):
        """
        Duplicate a cognitive node and add a point to the duplicate.

        :param node_name: Name of the node to be duplicated.
        :type node_name: str
        :param perception: Perception to be added as a point in the duplicated Goal.
        :type perception: core.container.Container
        :return: Name of the duplicated Goal node.
        :rtype: str
        """
        duplicate_service = f"cognitive_node/{node_name}/duplicate_node"
        response = self.call_services([(self.own_client(DuplicateNode, duplicate_service), DuplicateNode.Request(include_neighbors=True))])[0]
        new_goal_name = response.duplicate_node_name
        self.add_point(new_goal_name, perception, node_type="goal")
        return new_goal_name


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
        response = self.call_services([(self.own_client(CreateNode, service_name), CreateNode.Request(name=name, class_name=class_name, parameters=params_str))])[0]
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
        response = self.call_services([(self.own_client(UpdateNeighbor, service_name), UpdateNeighbor.Request(node_name=node_name, neighbor_name=neighbor_name, operation=True))])[0]
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
    
    def get_goals_reward(self, old_perception, perception, ltm_cache, update_space = True):
        """
        This method retrieves the rewards for each active goal based on the old and current sensing.

        :param old_perception: Old perception data.
        :type old_perception: core.container.Container
        :param perception: Current perception data.
        :type perception: core.container.Container
        :param ltm_cache: LTM cache containing the nodes and their data.
        :type ltm_cache: dict
        :return: Dictionary with goal names as keys and their corresponding rewards as values.
        :rtype: dict
        """
        self.get_logger().info("Reading rewards...")
        rewards = {}
        old_perception_msg = old_perception.to_msg() if old_perception else ContainerMsg()
        perception_msg = perception.to_msg() if perception else ContainerMsg()

        # The rewards of all the goals are requested at once. A goal whose reward is not updated is asked again in the next round
        self.request_rewards(
            [(goal, f"goal/{goal}/get_reward",
              GetReward.Request(old_perception=old_perception_msg, perception=perception_msg, update_space=update_space))
             for goal in self.active_goals],
            rewards,
        )

        # Check if there are any drives not linked to goals
        self.unlinked_drives = self.get_unlinked_drives()

        #Add rewards obtained from unlinked drives
        if self.unlinked_drives:
            active_drives = [
                drive for drive in self.unlinked_drives
                if ltm_cache.get("Drive", {}).get(drive, {}).get("activation", 0) > self.activation_threshold
            ]
            self.request_rewards(
                [(drive, f"drive/{drive}/get_reward", GetReward.Request(update_space=update_space)) for drive in active_drives],
                rewards,
            )
        self.get_logger().info(f"Reward_list: {rewards}")
        return rewards

    def request_rewards(self, requests, rewards):
        """
        Requests several rewards at once, repeating the requests whose reward is not updated, and
        stores them in rewards, in the order of the requests.

        :param requests: (name, service name, request) of each reward.
        :type requests: list
        :param rewards: Dictionary where the rewards are stored, by name.
        :type rewards: dict
        """
        pending = requests
        while pending:
            responses = self.call_services([(self.own_client(GetReward, service), request) for _, service, request in pending])
            for (name, _, _), response in zip(pending, responses):
                rewards[name] = response.reward
            pending = [item for item, response in zip(pending, responses) if not response.updated]

    def own_client(self, service_type, service_name):
        """
        Returns a client of a service created on this node, creating it the first time. A new client
        waits until it sees its service, logging a warning periodically. To be used from the process
        thread, together with call_services.

        :param service_type: Service type.
        :type service_type: type
        :param service_name: Service name.
        :type service_name: str
        :return: The client.
        :rtype: rclpy.client.Client
        """
        client = self.own_clients.get(service_name)
        if client is None:
            client = self.create_client(service_type, service_name, callback_group=self.cbgroup_client)
            check_blocking_wait(f"a new client waits for the service {service_name}")
            # Own loop instead of wait_for_service, which checks every 0.25 s
            start = time.monotonic()
            next_warning = start + self.activation_warning_period
            while not client.service_is_ready():
                time.sleep(SERVICE_CHECK_PERIOD)
                if time.monotonic() >= next_warning:
                    self.get_logger().warning(f"Waiting for the service {service_name} for {time.monotonic() - start:.1f} s")
                    next_warning += self.activation_warning_period
            self.own_clients[service_name] = client
        return client

    def call_services(self, calls):
        """
        Sends several service requests at once and waits, in the process thread, until all the
        responses have arrived. The clients belong to this node, so its executor, which runs in
        another thread, receives the responses meanwhile. While waiting, it periodically logs a
        warning with the missing responses. The wait has no limit.

        :param calls: (client, request) of each call. The clients must belong to this node (see own_client).
        :type calls: list
        :return: The responses, in the order of the calls.
        :rtype: list
        """
        if not calls:
            return []
        check_blocking_wait(f"the responses of {len(calls)} service calls")
        all_done = threading.Event()
        lock = threading.Lock()
        remaining = [len(calls)]

        def call_done(future):
            with lock:
                remaining[0] -= 1
                if remaining[0] == 0:
                    all_done.set()

        futures = []
        for client, request in calls:
            future = client.call_async(request)
            future.add_done_callback(call_done)
            futures.append(future)
        start = time.monotonic()
        while not all_done.wait(timeout=self.activation_warning_period):
            missing = [client.srv_name for (client, _), future in zip(calls, futures) if not future.done()]
            self.get_logger().warning(
                f"Waiting for {len(missing)} service responses for {time.monotonic() - start:.1f} s: {', '.join(missing)}"
            )
        return [future.result() for future in futures]

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
            for goal in self.active_goals:
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
        # The satisfaction of all the purposes is requested at once. A purpose whose satisfaction is not updated is asked again in the next round
        pending = list(purpose_list)
        while pending:
            responses = self.call_services([
                (self.own_client(IsSatisfied, f"robot_purpose/{purpose}/get_satisfaction"), IsSatisfied.Request(timestamp=timestamp.to_msg()))
                for purpose in pending
            ])
            for purpose, response in zip(pending, responses):
                if response.updated:
                    satisfaction[purpose] = dict(satisfied=response.satisfied, purpose_type=response.purpose_type, terminal=response.terminal)
            pending = [purpose for purpose, response in zip(pending, responses) if not response.updated]
        satisfaction = {purpose: satisfaction[purpose] for purpose in purpose_list}

        self.get_logger().info(f"Satisfaction list: {satisfaction}")

        return satisfaction
    
    def goal_has_cnode(self, goal, ltm_cache):
        """
        This method checks if a C-Node exists for a given goal in the LTM cache.

        :param goal: Name of the goal to check for a corresponding C-Node.
        :type goal: str
        :param ltm_cache: LTM cache containing the nodes and their data.
        :type ltm_cache: dict
        :return: True if a C-Node exists for the given goal, False otherwise.
        :rtype: bool
        """
        cnodes = ltm_cache["CNode"].keys()
        neighbors = []
        for cnode in cnodes:
            neighbors += [neighbor["name"] for neighbor in ltm_cache["CNode"][cnode]["neighbors"] if neighbor["node_type"] == "Goal"]
        return goal in neighbors
    
    def duplicate_connected(self, goal, cnodes, ltm_cache):
        """
        This method checks if a C-Node exists for any duplicate of a given goal that is connected to a specific policy in the LTM cache.

        :param goal: Name of the goal to check for duplicates.
        :type goal: str
        :param cnodes: List of C-Nodes to check for connections.
        :type cnodes: list
        :param ltm_cache: LTM cache containing the nodes and their data.
        :type ltm_cache: dict
        :return: True if a C-Node exists for any duplicate of the given goal that is connected to the specified policy, False otherwise.
        :rtype: bool
        """
        if "_dup_" in goal:
            base_goal = goal.split("_dup_")[0]
        else:
            base_goal = goal

        world_models = {cnode: [neighbor["name"] for neighbor in ltm_cache["CNode"][cnode]["neighbors"] if neighbor["node_type"] == "WorldModel"] for cnode in cnodes}
        
        for cnode in cnodes:

            duplicate = [base_goal in neighbor["name"] and self.current_world in world_models[cnode] for neighbor in ltm_cache["CNode"][cnode]["neighbors"] if neighbor["node_type"] == "Goal"]
            if any(duplicate):
                self.get_logger().info(f"Duplicate of {base_goal} found in CNode {cnode}")
                return True
        return False

    
    # =========================
    # Process Execution
    # =========================

    def pause_callback(self, request, response):
        """
        Callback function to handle pause requests for the cognitive process.

        :param request: The incoming request to pause the process.
        :type request: cognitive_node_interfaces.srv.Pause.Request
        :param response: The response object to be filled and returned.
        :type response: cognitive_node_interfaces.srv.Pause.Response
        :return: The response indicating whether the process was paused successfully.
        :rtype: cognitive_node_interfaces.srv.Pause.Response
        """
        self.paused = request.pause
        self.stop = request.stop
        if self.paused:
            self.get_logger().info("Cognitive process paused.")
        else:
            self.get_logger().info("Cognitive process resumed.")
        if self.stop:
            self.get_logger().info("Cognitive process stopped.")
            self.iterations = self.iteration # Store the current iteration count before stopping
        response.success = True
        return response

    def run(self):
        """
        Main loop of the cognitive process. This method is executed in a separate thread.
        It runs the cognitive process, reading perceptions, updating activations, and processing episodes.
        """
        raise NotImplementedError("This method should be implemented in the derived class.")


    
