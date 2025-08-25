import numpy as np
import threading
import traceback
from copy import deepcopy, copy

from cognitive_nodes.episode import Episode, Action, episode_obj_list_to_msg_list, episode_msg_list_to_obj_list
from scipy.stats.qmc import LatinHypercube
from cognitive_processes.cognitive_process import CognitiveProcess
from core.service_client import ServiceClient

from cognitive_node_interfaces.srv import Predict


class Deliberation(CognitiveProcess):
    """
    Deliberation class: A cognitive process that allows the agent to deliberate on its actions and decisions.
    """
    def __init__(self, name, node, iterations=0, trials=1, LTM_id="", candidate_actions=5, softmax_selection = True, softmax_temperature=1.0, candidate_generation="latin", exploration_process=False, **params):
        """
        Constructor of the Deliberation class.
        """
        super().__init__(name, iterations, trials, LTM_id, **params)
        self.node = node
        self.candidate_actions = candidate_actions
        self.exploration_process = exploration_process

        self.start_flag = threading.Event()
        self.finished_flag = threading.Event()
        
        self.softmax_selection = softmax_selection
        self.softmax_temperature = softmax_temperature
        self.candidate_generation = candidate_generation
        self.reward_threshold = 0.1
        self.current_reward = 0.0
        
        # Read LTM and configure perceptions
        self.set_attributes_from_params(params)
        self.setup()
        self.start_threading()

    def setup(self):
        super().setup()
        self.configure_actuation()

    def configure_actuation(self):
        self.get_logger().info(f"Configuring actuation: {self.globals}")
        self.actuation_config = self.globals.get("actuation_config", None)
        self.actuation_dims = 0
        if self.actuation_config is None:
            raise ValueError(
                "Actuation configuration not found in globals. Please ensure it is set correctly on the YAML file."
            )
        else:
            self.actuation_dict = {}
            for actuator in self.actuation_config:
                self.actuation_dict[actuator] = [{}]
                for param in self.actuation_config[actuator]:
                    self.actuation_dict[actuator][0][param] = 0.0
                self.actuation_dims += len(self.actuation_config[actuator])
            self.get_logger().info(
                f"Actuation configuration {self.actuation_config} loaded with {self.actuation_dims} dimensions."
            )
        self.action_sampler = LatinHypercube(d=self.actuation_dims, rng=self.rng)

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
                if node == self.node.name:
                    self.activation_inputs[node]['flag'].set()
                self.get_logger().debug(f"DEBUG: Waiting for activation: {node}")
                self.activation_inputs[node]['flag'].wait()
                self.activation_inputs[node]['flag'].clear()
            self.semaphore.release()
            self.get_logger().debug("DEBUG - LTM CACHE:" + str(self.LTM_cache))

    def generate_candidate_actions(self, old_perception=None, algorithm = "latin"):
        """
        Generates a list of candidate Episode objects based on the configured actuation dimensions.
        Each Episode will have its old_perception set to the argument and action.actuation set to the candidate action.
        """
        if algorithm == "latin":
            candidate_matrix = self.action_sampler.random(n=self.candidate_actions)
        elif algorithm == "random":
            candidate_matrix = self.rng.random((self.candidate_actions, self.actuation_dims))
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        action_list = [deepcopy(self.actuation_dict) for _ in range(self.candidate_actions)]
        for i, action in enumerate(action_list):
            offset = 0
            for actuator, params in action.items():
                for j, param in enumerate(params[0]):
                    params[0][param] = candidate_matrix[i][j + offset]
                offset += len(params[0])
        # Create Episode objects for each candidate action
        episodes = []
        for i in range(self.candidate_actions):
            action = Action(actuation=action_list[i])
            episode = Episode(old_perception=old_perception, action=action)
            episodes.append(episode)
        self.get_logger().info(f"Generated {len(episodes)} candidate episodes")
        return episodes
    
    def predict_perceptions(self, world_model, input_episodes: list[Episode]) -> list[Episode]:
        """
        Predicts the expected utilities for the given input episodes using the Utility Model.
        
        :param world_model: The current world model.
        :type world_model: dict
        :param input_episodes: List of input episodes to predict utilities for.
        :type input_episodes: list[Episode]
        :return: List of predicted utilities.
        :rtype: list[float]
        """
        service_name = "world_model/" + str(world_model) + "/predict"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClient(Predict, service_name)
        # Use the World Model's predict method to get the predicted states
        predicted_episodes = self.node_clients[service_name].send_request(input_episodes=episode_obj_list_to_msg_list(input_episodes))
        return episode_msg_list_to_obj_list(predicted_episodes.output_episodes)
    
    def predict_utilities(self, input_episodes: list[Episode]) -> list[float]:
        """
        Predicts the expected utilities for the given input episodes using the Utility Model.
        
        :param input_episodes: List of input episodes to predict utilities for.
        :type input_episodes: list[Episode]
        :return: List of predicted utilities.
        :rtype: list[float]
        """
        return self.node.predict(input_episodes)
    
    def select_action(self, predicted_episodes, expected_utilities):
        """
        Selects an action probabilistically using softmax over the expected utilities.
        """
        softmax = getattr(self, "softmax_selection", True)
        if softmax:
            # Compute softmax probabilities
            utilities = np.array(expected_utilities)
            temp = self.softmax_temperature if hasattr(self, "softmax_temperature") else 1.0
            exp_utilities = np.exp((utilities - np.max(utilities)) / temp)
            probs = exp_utilities / np.sum(exp_utilities)
            probs = probs.reshape(-1)
            # Sample an index according to the probabilities
            selected_index = np.random.choice(len(predicted_episodes), p=probs)
            selected_episode = predicted_episodes[selected_index]
        else:
            selected_index = np.argmax(expected_utilities)
            selected_episode = predicted_episodes[selected_index]
        self.get_logger().info(f"Selected action: {selected_episode.action} with utility {expected_utilities[selected_index]}")
        return selected_episode.action
    
    def publish_episode(self):
        super().publish_episode()
        self.node.episodic_buffer.add_episode(self.current_episode, self.current_reward)

    def get_linked_goals(self):
        """
        Retrieves the goal linked to the parent node of the process
        """
        cnodes = [neighbor["name"] for neighbor in self.node.neighbors if neighbor["node_type"] == "CNode"]
        self.get_logger().info(f"Linked CNodes: {cnodes}")
        cnodes_neighbors = []
        for cnode in cnodes:
            cnodes_neighbors.extend(self.LTM_cache["CNode"][cnode]["neighbors"])
        self.get_logger().info(f"Linked CNodes neighbors: {cnodes_neighbors}")
        linked_goals = [neighbor["name"] for neighbor in cnodes_neighbors if neighbor["node_type"] == "Goal"]
        self.get_logger().info(f"Linked goals: {linked_goals}")
        return linked_goals

    def check_completion(self):
        self.get_logger().info("Checking if goals are completed")
        if not self.exploration_process:
            linked_goals = self.get_linked_goals()
        else:
            linked_goals = list(self.current_episode.reward_list.keys())
        self.get_logger().info(f"Linked goals: {linked_goals}")
        self.get_logger().info(f"Current rewards: {self.current_episode.reward_list}")
        rewards = [self.current_episode.reward_list[goal] for goal in linked_goals if goal in self.current_episode.reward_list]
        self.current_reward = max(rewards) if rewards else 0.0
        return any([reward > self.reward_threshold for reward in rewards])
    
    # =========================
    # LTM & STM UPDATES
    # =========================
    def update_ltm(self, stm:Episode):
        """
        This method updates the LTM with the perception changes, policy executed and reward obtained.

        :param stm: Episode object containing the information to update the LTM.
        :type stm: cognitive_processes.main_loop.Episode
        """
        if not self.exploration_process:
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
        cnodes = [node["name"] for node in self.node.neighbors if node["node_type"] == "CNode"]
        cnode_activations = self.get_node_activations_by_list(cnodes, ltm_cache)
        threshold = self.activation_threshold
        updates = False
        point_added = False

        for cnode in cnode_activations.keys():
            cnode_neighbors = self.request_neighbors(cnode)
            # world_model = next(
            #     (
            #         neighbor["name"]
            #         for neighbor in cnode_neighbors
            #         if neighbor["node_type"] == "WorldModel"
            #     )
            # )
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

            # world_model_activation = self.get_node_data(world_model, ltm_cache)["activation"]
            # goal_activation = self.get_node_data(goal, ltm_cache)["activation"]
            pnode_activation = self.get_node_data(pnode, ltm_cache)["activation"]

            # This assumes that when this method is executed, a trace has been created in the episodic buffer, either because a reward was obtained or because the max iterations were reached.
            # if world_model_activation > threshold and goal_activation > threshold:
            reward = reward_list.get(goal, 0.0)
            if (reward > threshold):
                reward_list.pop(goal)
                if not point_added:
                    trace = self.node.episodic_buffer.traces_buffer[-1]
                    self.add_pnode_trace(pnode, trace)
                    updates = True
                    point_added = True
            elif pnode_activation > threshold and len(self.node.episodic_buffer.antitraces_buffer) > 0:
                antitrace = self.node.episodic_buffer.antitraces_buffer[-1]
                self.add_pnode_antitrace(pnode, antitrace)
                updates = True

        # This section should not be necessary because Utility Models already have their C-Node created before executing 
        # for goal, reward in reward_list.items():
        #     if (reward > threshold) and (not point_added):
        #         if goal not in self.unlinked_drives:
        #             self.new_cnode(old_perception, goal, policy)
        #         else:
        #             drive = goal
        #             goal = self.new_goal(perception, drive)
        #             self.new_cnode(old_perception, goal, policy)
        #         point_added=True
        #         updates = True

        if not updates:
            self.get_logger().info("No update required in PNode/CNodes")

    def add_pnode_trace(self, pnode, trace):
        for episode, _ in trace:
            self.add_point(pnode, episode.old_perception)
    
    def add_pnode_antitrace(self, pnode, antitrace):
        for episode, _ in antitrace:
            self.add_antipoint(pnode, episode.old_perception)


    def deliberation_cycle(self):
        self.start_flag.wait()

        self.get_logger().info("Deliberation process started")
        self.current_world = self.get_current_world_model()
        self.current_episode.perception = self.read_perceptions()
        self.update_activations()
        self.active_goals = self.get_goals(self.LTM_cache)
        self.current_episode.reward_list= self.get_goals_reward(self.current_episode.old_perception, self.current_episode.perception, self.LTM_cache)
        self.iteration = 1
        finished = False

        while (self.iteration <= self.iterations) and (not self.stop) and not finished:

            if not self.paused:

                self.get_logger().info(
                    "*** DELIBERATION STEP: " + str(self.iteration) + "/" + str(self.iterations) + " ***"
                )
                self.update_activations()
                self.current_episode.old_ltm_state=deepcopy(self.LTM_cache)
                # GENERATE POSSIBLE ACTIONS
                candidate_actions = self.generate_candidate_actions(self.current_episode.perception, self.candidate_generation)
                # PREDICT EXPECTED PERCEPTIONS
                predicted_episodes = self.predict_perceptions(
                    self.current_world, candidate_actions
                )
                # GET EXPECTED UTILITIES
                predicted_utilities = self.predict_utilities(predicted_episodes)
                # SELECT ACTION
                self.current_episode.action = self.select_action(candidate_actions, predicted_utilities)
                # EXECUTE ACTION
                self.execute_action(self.current_episode.perception, self.current_episode.action)
                self.current_episode.parent_policy = self.node.name if not self.exploration_process else ""
                self.current_episode.old_perception, self.current_episode.perception = self.current_episode.perception, self.read_perceptions()
                self.update_activations()
                self.current_episode.ltm_state = deepcopy(self.LTM_cache)
                self.get_logger().info(
                    f"DEBUG PERCEPTION: \n old_sensing: {self.current_episode.old_perception} \n     sensing: {self.current_episode.perception}"
                )
                self.active_goals = self.get_goals(self.current_episode.old_ltm_state)
                self.current_episode.reward_list= self.get_goals_reward(self.current_episode.old_perception, self.current_episode.perception, self.current_episode.old_ltm_state)
                finished = self.check_completion()
                self.publish_episode()
                self.iteration += 1
        self.update_ltm(self.current_episode)
        self.finished_flag.set()
        self.start_flag.clear()
    

    def run(self):
        self.current_episode.perception = self.read_perceptions()
        self.node.episodic_buffer.configure_labels(self.current_episode)
        
        while True:
            try:
                self.deliberation_cycle()
            except Exception as e:
                self.get_logger().error(f"Exception in deliberation cycle: {e}")
                self.get_logger().error(traceback.format_exc())
                self.finished_flag.set()
                self.start_flag.clear()
                break




