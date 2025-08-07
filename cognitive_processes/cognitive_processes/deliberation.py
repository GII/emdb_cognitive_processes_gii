import threading
from copy import deepcopy

from cognitive_nodes.episode import Episode, Action, episode_obj_list_to_msg_list, episode_msg_list_to_obj_list
from scipy.stats.qmc import LatinHypercube
from cognitive_processes.cognitive_process import CognitiveProcess
from core.service_client import ServiceClient

from cognitive_node_interfaces.srv import Predict


class Deliberation(CognitiveProcess):
    """
    Deliberation class: A cognitive process that allows the agent to deliberate on its actions and decisions.
    """
    def __init__(self, node, iterations=0, trials=1, LTM_id="", candidate_actions=5, softmax_temperature=1.0, clear_buffer=True, **params):
        """
        Constructor of the Deliberation class.
        """
        super().__init__(node, iterations, trials, LTM_id, **params)
        self.node = node
        self.candidate_actions = candidate_actions

        self.start_flag = threading.Event()
        self.finished_flag = threading.Event()
        self.clear_buffer = clear_buffer
        
        self.softmax_temperature = softmax_temperature
        
        # Read LTM and configure perceptions
        self.set_attributes_from_params(params)
        self.setup()
        self.start_threading()

    def setup(self):
        super().setup()
        self.configure_actuation()

    def configure_actuation(self):
        self.node.get_logger().info(f"Configuring actuation: {self.globals}")
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
            self.node.get_logger().info(
                f"Actuation configuration {self.actuation_config} loaded with {self.actuation_dims} dimensions."
            )
        self.action_sampler = LatinHypercube(d=self.actuation_dims, rng=self.rng)

    def generate_candidate_actions(self, old_perception=None):
        """
        Generates a list of candidate Episode objects based on the configured actuation dimensions.
        Each Episode will have its old_perception set to the argument and action.actuation set to the candidate action.
        """
        candidate_matrix = self.action_sampler.random(n=self.candidate_actions)
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
        self.node.get_logger().info(f"Generated {len(episodes)} candidate episodes")
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
        import numpy as np
        # Compute softmax probabilities
        utilities = np.array(expected_utilities)
        temp = self.softmax_temperature if hasattr(self, "softmax_temperature") else 1.0
        exp_utilities = np.exp((utilities - np.max(utilities)) / temp)
        probs = exp_utilities / np.sum(exp_utilities)
        # Sample an index according to the probabilities
        selected_index = np.random.choice(len(predicted_episodes), p=probs)
        selected_episode = predicted_episodes[selected_index]
        self.node.get_logger().info(f"Selected action: {selected_episode.action} with utility {expected_utilities[selected_index]}")
        return selected_episode.action
    
    def publish_episode(self):
        super().publish_episode()
        self.node.episodic_buffer.add_episode(self.current_episode)

    def deliberation_cycle(self):
        self.start_flag.wait()

        self.node.get_logger().info("Deliberation process started")
        if self.clear_buffer:
            self.node.episodic_buffer.clear()
        self.current_world = self.get_current_world_model()
        self.current_episode.perception = self.read_perceptions()
        self.update_activations()
        self.active_goals = self.get_goals(self.LTM_cache)
        self.current_episode.reward_list= self.get_goals_reward(self.current_episode.old_perception, self.current_episode.perception, self.LTM_cache)
        self.iteration = 1
        
        while (self.iteration <= self.iterations) and (not self.stop):

            if not self.paused:

                self.node.get_logger().info(
                    "*** DELIBERATION STEP: " + str(self.iteration) + "/" + str(self.iterations) + " ***"
                )
                self.update_activations()
                self.current_episode.old_ltm_state=deepcopy(self.LTM_cache)
                # GENERATE POSSIBLE ACTIONS
                candidate_actions = self.generate_candidate_actions(self.current_episode.perception)
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
                self.current_episode.parent_policy = self.node.name
                self.current_episode.old_perception, self.current_episode.perception = self.current_episode.perception, self.read_perceptions()
                self.update_activations()
                self.current_episode.ltm_state = deepcopy(self.LTM_cache)
                self.node.get_logger().info(
                    f"DEBUG PERCEPTION: \n old_sensing: {self.current_episode.old_perception} \n     sensing: {self.current_episode.perception}"
                )
                self.active_goals = self.get_goals(self.current_episode.old_ltm_state)
                self.current_episode.reward_list= self.get_goals_reward(self.current_episode.old_perception, self.current_episode.perception, self.current_episode.old_ltm_state)
                self.publish_episode()
                self.iteration += 1

        self.finished_flag.set()
        self.start_flag.clear()
    

    def run(self):
        self.current_episode.perception = self.read_perceptions()
        self.node.episodic_buffer.add_episode(self.current_episode)
        while True:
            self.deliberation_cycle()







