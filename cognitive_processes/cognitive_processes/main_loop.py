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
from rclpy.time import Time

from core.service_client import ServiceClient
from cognitive_nodes.episode import Episode
from cognitive_node_interfaces.srv import (
    Execute,
    GetActivation,
    GetReward,
    GetInformation,
    AddPoint,
    IsSatisfied
)
from cognitive_processes.cognitive_process import CognitiveProcess
from cognitive_nodes.episode import reward_dict_to_msg
from core_interfaces.srv import CreateNode, SetChangesTopic, UpdateNeighbor, StopExecution
from cognitive_node_interfaces.msg import Activation
from cognitive_processes_interfaces.msg import ControlMsg
from cognitive_node_interfaces.msg import Episode as EpisodeMsg
from std_msgs.msg import String

from core.utils import perception_dict_to_msg, perception_msg_to_dict, actuation_dict_to_msg, actuation_msg_to_dict, class_from_classname


class MainLoop(CognitiveProcess):
    """
    MainLoop class for managing the main loop of the system.
    """

    # =========================
    # INITIALIZATION & SETUP
    # =========================
    def __init__(self, name, softmax_selection = False, softmax_temperature = 1, kill_on_finish = False, **params):
        """
        Constructor for the MainLoop class.
        Initializes the MainLoop node and starts the main loop execution.

        :param name: The name of the MainLoop node.
        :type name: str
        """
        super().__init__(name, **params)

        # --- STM ---
        self.stm = Episode()
        
        # --- Reward and policy selection ---
        self.reward_threshold = 0.9
        self.policies_to_test = []
        self.current_policy = None
        self.random_seed = 0
        self.current_reward = 0
        self.softmax_selection = softmax_selection
        self.softmax_temperature = softmax_temperature

        # --- Node/goal/drive management ---
        self.unlinked_drives = []
        self.current_world = None
        self.n_cnodes = 0
        self.n_goals = 0

        # --- File/output management ---
        self.files = []
        self.pnodes_success = {}

        # --- Experiment tracking ---
        self.goal_count = 0
        self.trials_data = []
        self.last_reset = 0
        self.kill_on_finish = kill_on_finish

        self.set_attributes_from_params(params)

        # Read LTM and configure perceptions
        self.setup()

        self.start_threading()


    # =========================
    # SETUP
    # =========================

    def setup(self):
        """
        Initial configuration of the MainLoop node.
        This method sets up the LTM, perceptions, files, connectors, control channel, etc.
        """
        super().setup()
        self.setup_files()
        self.kill_commander_client = ServiceClient(StopExecution, 'commander/kill')

    def read_ltm(self, ltm_dump=None):
        """
        Makes an empty call for the LTM get_node service which returns all the nodes
        stored in the LTM. Then, a LTM cache dictionary is populated with the data.

        :param ltm_dump: Optional LTM dump to be used instead of requesting it from the current one.
        :type ltm_dump: dict
        """
        super().read_ltm(ltm_dump=ltm_dump)

        # Check if there are any drives not linked to goals
        self.unlinked_drives = self.get_unlinked_drives()

    # =========================
    # File Handling
    # =========================

    def setup_files(self):
        """
        Configures the output files.
        """ 
        if hasattr(self, "Files"):
            self.get_logger().info("Files detected, loading files...")
            for file_item in self.Files:
                self.add_file(file_item)
        else:
            self.get_logger().info("No files detected...")

    def add_file(self, file_item):
        """
        Process a file entry (create the corresponding object) in the configuration.

        :param file_item: Dictionary with the file information.
        :type file_item: dict
        """
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
        """
        Close all files when execution is finished.
        """
        self.get_logger().info("Closing files...")
        for file in self.files:
            file.close()
    
    # =========================
    # PUBLISHING & STATUS
    # =========================
    def publish_iteration(self):
        """
        Method for publishing execution data in the control topic in each iteration.
        """
        msg = ControlMsg()
        msg.command = ""
        msg.world = self.current_world
        msg.iteration = self.iteration
        self.control_publisher.publish(msg)

    # =========================
    # POLICY SELECTION
    # =========================
    def select_policy(self, softmax=False):
        """
        Selects the policy with the higher activation.
        If softmax is True, it selects the policy using a softmax function.
        If no policy is selected, it selects a random policy.

        :param softmax: If True, selects the policy using a softmax function, defaults to False.
        :type softmax: bool
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
        """
        Selects the policy with the maximum activation.

        :param policy_activations: Dictionary with policy names as keys and their activations as values.
        :type policy_activations: dict
        :return: The name of the policy with the maximum activation.
        :rtype: str
        """
        selected= max(zip(policy_activations.values(), policy_activations.keys()))[1]
        return selected
    
    def select_policy_softmax(self, policy_activations:dict, temperature=1):
        """
        Selects a policy using the softmax function.

        :param policy_activations: Dictionary with policy names as keys and their activations as values.
        :type policy_activations: dict
        :param temperature: Temperature parameter for the softmax function, defaults to 1.
        :type temperature: int
        :return: The name of the selected policy.
        :rtype: str
        """
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

        :return: The selected policy.
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

        :param policy: Policy to be removed, defaults to None.
        :type policy: str
        """

        if policy:
            if policy in self.policies_to_test:
                self.policies_to_test.remove(policy)
        else:
            self.policies_to_test = list(self.LTM_cache["Policy"].keys())

    # =========================
    # ACTIVATION HANDLING
    # =========================            

    def read_activation_callback(self, msg: Activation):
        """
        This method receives a message from an activation topic, processes the
        message and updates the activation in the LTM cache.

        :param msg: Message that contains the activation information.
        :type msg: cognitive_node_interfaces.msg.Activation
        """
        super().read_activation_callback(msg)

        act_file = getattr(self, "act_file", None) #CHANGE THIS
        if act_file is not None:
            act_file.receive_activation_callback(msg)

    # =========================
    # POLICY EXECUTION
    # =========================
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
        self.get_logger().info(f"Selecting world model with highest activation: {WM} ({WM_activations[WM]})")

        return WM
    
    def get_needs(self, ltm_cache):
        """
        This method retrieves all active needs from the LTM cache.

        :param ltm_cache: LTM cache containing the nodes and their data.
        :type ltm_cache: dict
        :return: List of active needs.
        :rtype: list
        """
        needs = self.get_all_active_nodes("Need", ltm_cache)

        self.get_logger().info(f"Active Needs: {needs}")
                    
        return needs
    
    def get_need_satisfaction(self, need_list, timestamp):
        """
        This method retrieves the satisfaction of each need in the need_list.

        :param need_list: List of needs.
        :type need_list: list
        :param timestamp: Timestamp to be used for the request.
        :type timestamp: rclpy.time.Time
        :return: Dictionary with need names as keys and their satisfaction status as values.
        :rtype: dict
        """
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

    # =========================
    # LTM & STM UPDATES
    # =========================
    def update_ltm(self, stm:Episode):
        """
        This method updates the LTM with the perception changes, policy executed and reward obtained.

        :param stm: Episode object containing the information to update the LTM.
        :type stm: cognitive_processes.main_loop.Episode
        """
        self.update_pnodes_reward_basis(stm.old_perception, stm.perception, stm.parent_policy, copy(stm.reward_list), stm.old_ltm_state)


    def update_pnodes_reward_basis(self, old_perception, perception, policy, reward_list, ltm_cache):
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

        :param old_perception: Perception before the execution of the policy.
        :type old_perception: dict
        :param perception: Perception after the execution of the policy.
        :type perception: dict
        :param policy: Policy executed.
        :type policy: str
        :param reward_list: Dictionary with the rewards obtained for each goal after the execution of the policy.
        :type reward_list: dict
        :param ltm_cache: LTM cache containing the nodes and their data.
        :type ltm_cache: dict
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

            if world_model_activation > threshold and goal_activation > threshold:
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
                    self.new_cnode(old_perception, goal, policy)
                else:
                    drive = goal
                    goal = self.new_goal(perception, drive)
                    self.new_cnode(old_perception, goal, policy)
                point_added=True
                updates = True

        if not updates:
            self.get_logger().info("No update required in PNode/CNodes")

    
    # =========================
    # World Reset Management
    # =========================

    def reset_world(self, check_finish=True):
        """
        Reset the world if necessary, according to the experiment parameters.
        
        :param check_finish: If True, checks if the world has finished before deciding to reset. 
                    If False, only the trial/iteration count is considered.        
        :type check_finish: bool
        :return: True if the world was reset, False otherwise.
        :rtype: bool
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
            else:
                self.get_logger().info("Asking for a world reset...")
                msg = ControlMsg()
                msg.command = "reset_world"
                msg.world = self.current_world
                msg.iteration = self.iteration
                self.control_publisher.publish(msg)
        return changed
    
    def world_finished(self):
        """
        Check if the world has finished.

        :return: True if the world has finished, False otherwise.
        :rtype: bool
        """
        need_satisfaction = self.get_need_satisfaction(self.get_needs(self.LTM_cache), self.get_clock().now())
        if len(need_satisfaction)>0:
            finished = any((need_satisfaction[need]['satisfied'] for need in need_satisfaction if (need_satisfaction[need]['need_type'] == 'Operational')))
        else:
            finished=False
        return finished

    # =========================
    # MAIN LOOP
    # =========================
    def run(self):
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
                self.current_policy, self.stm.action.actuation = self.execute_policy(self.stm.perception, self.current_policy)
                self.stm.parent_policy = self.current_policy
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
                    self.stm.ltm_state = self.LTM_cache

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
