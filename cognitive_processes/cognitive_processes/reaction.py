import threading
import traceback
from copy import deepcopy

from cognitive_nodes.episode import Episode, Action
from cognitive_processes.cognitive_process import CognitiveProcess


class Reaction(CognitiveProcess):
    """
    Reaction class: A cognitive process that executes a closed-loop reactive
    policy using a learned SAC model.

    Analogous to ``Deliberation`` but for direct reactive control: instead of
    generating candidate actions and predicting their utility, it queries the
    SAC model stored in its parent ``PolicyLearned`` node to obtain the next
    action at each step.

    The process is decoupled from the ``PolicyLearned`` node via a
    ``start_flag`` / ``finished_flag`` threading event pair, identical to the
    ``Deliberation`` <=> ``UtilityModel`` pattern:

    * ``PolicyLearned.execute_callback`` -> sets ``start_flag``, waits on
      ``finished_flag``.
    * ``Reaction.reaction_cycle`` -> waits on ``start_flag``, runs the loop,
      sets ``finished_flag``.

    The ``summary_episode`` attribute holds the result of the last completed
    cycle and is read by ``PolicyLearned.execute_callback`` to build its
    service response.

    Parameters
    ----------
    name : str
        ROS 2 node name for this cognitive process.
    node : PolicyLearned
        The parent policy node that owns the SAC model.
    iterations : int
        Maximum steps per reaction cycle (``max_steps``).
    trials : int
        Number of consecutive trials (passed to base class, usually 1).
    LTM_id : str
        Identifier of the Long-Term Memory node to connect to.
    reward_threshold : float
        Reward value above which a step is considered successful and the
        loop terminates early (default: 0.1).
    **params
        Remaining keyword arguments forwarded to ``CognitiveProcess`` (must
        include at least ``Control`` and ``globals`` from the YAML config).
    """

    def __init__(
        self,
        name,
        node,
        iterations=50,
        trials=1,
        LTM_id="",
        reward_threshold=0.1,
        **params,
    ):
        super().__init__(name, iterations, trials, LTM_id, **params)
        self.node = node
        self.reward_threshold = reward_threshold
        self.current_reward = 0.0

        self.start_flag = threading.Event()
        self.finished_flag = threading.Event()

        self.summary_episode = Episode()
        self.summary_episode.parent_policy = self.node.name

        self.set_attributes_from_params(params)
        self.setup()
        self.start_threading()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        super().setup()

    # ------------------------------------------------------------------
    # Activation (avoid deadlock with parent policy node)
    # ------------------------------------------------------------------

    def update_activations(self):
        """Override to auto-set the parent policy node's activation flag.

        The parent ``PolicyLearned`` node is blocked in ``execute_callback``
        waiting for ``finished_flag``, so it cannot respond to activation
        service calls.  We set its flag immediately (mirroring
        ``Deliberation.update_activations``) to prevent a deadlock.
        """
        self.get_logger().info("Updating activations...")
        self.semaphore.acquire()
        self.activation_time = self.get_clock().now().nanoseconds
        for node in self.activation_inputs:
            self.activation_inputs[node]['flag'].clear()

        for node in self.activation_inputs:
            if node == self.node.name:
                # Parent is blocked — set flag directly without a service call.
                self.activation_inputs[node]['flag'].set()
            self.get_logger().debug(f"DEBUG: Waiting for activation: {node}")
            self.activation_inputs[node]['flag'].wait()
            self.activation_inputs[node]['flag'].clear()
        self.semaphore.release()
        self.get_logger().debug("DEBUG - LTM CACHE:" + str(self.LTM_cache))

    # ------------------------------------------------------------------
    # Goal discovery
    # ------------------------------------------------------------------

    def get_linked_goals(self):
        """Return goal names linked to the parent policy via the CNode graph.

        Traverses: policy neighbours -> CNodes -> CNode neighbours -> Goals.
        This is topology-based, so it works regardless of current activation
        values.
        """
        cnodes = [
            neighbor["name"]
            for neighbor in self.node.neighbors
            if neighbor["node_type"] == "CNode"
        ]
        cnodes_neighbors = []
        for cnode in cnodes:
            if cnode in self.LTM_cache.get("CNode", {}):
                cnodes_neighbors.extend(self.LTM_cache["CNode"][cnode]["neighbors"])
        linked_goals = [
            neighbor["name"]
            for neighbor in cnodes_neighbors
            if neighbor["node_type"] == "Goal"
        ]
        self.get_logger().info(f"Linked goals: {linked_goals}")
        return linked_goals or self.get_goals(self.LTM_cache)

    # ------------------------------------------------------------------
    # Completion check
    # ------------------------------------------------------------------

    def check_completion(self):
        """Return True when any active goal reward exceeds ``reward_threshold``."""
        rewards = [
            self.current_episode.reward_list.get(goal, 0.0)
            for goal in self.active_goals
            if goal in self.current_episode.reward_list
        ]
        self.current_reward = max(rewards) if rewards else 0.0
        achieved = any(r > self.reward_threshold for r in rewards)
        self.get_logger().info(
            f"Reaction {self.get_name()}: completion check — "
            f"rewards={rewards} achieved={achieved}"
        )
        return achieved

    # ------------------------------------------------------------------
    # Core cycle
    # ------------------------------------------------------------------

    def reaction_cycle(self):
        """Run one closed-loop reaction cycle.

        Waits for ``start_flag``, executes up to ``self.iterations`` steps
        using the SAC model in ``self.node``, then sets ``finished_flag``.
        """
        self.start_flag.wait()

        self.get_logger().info(f"Reaction cycle started for {self.get_name()}")
        # Read initial perception and snapshot LTM / goals
        self.current_episode.perception = self.read_perceptions()
        self.summary_episode.old_perception = self.current_episode.perception
        self.update_activations()
        self.summary_episode.old_ltm_state = deepcopy(self.LTM_cache)
        self.active_goals = self.get_linked_goals()

        achieved = False
        step = 0

        while step < self.iterations and not self.stop and not achieved:
            if self.paused:
                step += 1
                continue

            self.get_logger().info(
                f"*** REACTION STEP: {step + 1}/{self.iterations} ***"
            )

            # Build observation vector from current perception
            obs = self.node._perception_dict_to_obs(self.current_episode.perception)

            # Query the SAC model for the next action
            with self.node._sac_lock:
                action_vec, _ = self.node._sac.predict(obs, deterministic=False)
            self.get_logger().info(
                f'[SAC step={step}] '
                f'obs=[{", ".join(f"{v:.4f}" for v in obs)}] | '
                f'action=[{", ".join(f"{v:.4f}" for v in action_vec)}]'
            )

            # Pack the action into an Action object
            actuation = self.node._action_vec_to_dict(action_vec)
            self.current_episode.action = Action(
                actuation=actuation, policy_id=0
            )
            self.current_episode.parent_policy = self.node.name

            # Execute the action
            self.execute_action(
                self.current_episode.perception, self.current_episode.action
            )

            # Read the resulting perception
            (
                self.current_episode.old_perception,
                self.current_episode.perception,
            ) = self.current_episode.perception, self.read_perceptions()

            # Obtain rewards from all active Goal nodes
            self.active_goals = self.get_linked_goals()
            self.current_episode.reward_list = self.get_goals_reward(
                self.current_episode.old_perception,
                self.current_episode.perception,
                self.LTM_cache,
            )

            # Publish episode and check for goal completion
            self.publish_episode()
            achieved = self.check_completion()

            self.get_logger().info(
                f"Reaction {self.get_name()}: step {step + 1} — "
                f"reward={self.current_reward:.3f} achieved={achieved}"
            )
            step += 1

        # Propagate final state to the summary episode read by PolicyLearned
        self.summary_episode.perception = self.current_episode.perception
        self.summary_episode.reward_list = self.current_episode.reward_list

        self.get_logger().info(
            f"Reaction cycle finished for {self.get_name()} "
            f"(steps={step}, achieved={achieved})"
        )

        self.finished_flag.set()
        self.start_flag.clear()

    # ------------------------------------------------------------------
    # Thread entry-point
    # ------------------------------------------------------------------

    def run(self):
        """Outer loop: keeps restarting ``reaction_cycle`` after each call."""
        self.current_episode.perception = self.read_perceptions()
        while True:
            try:
                self.reaction_cycle()
            except Exception as exc:
                self.get_logger().error(
                    f"Exception in reaction cycle: {exc}\n{traceback.format_exc()}"
                )
                self.finished_flag.set()
                self.start_flag.clear()
                break
