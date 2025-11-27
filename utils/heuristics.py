import numpy as np

class HeuristicTracker():
    def __init__(self, args, num_scenes):
        self.strategy = args.heuristic_strategy                         # Strategy to use for stuck agents
        assert self.strategy in ['none', 'base', 'probe'], "Invalid heuristic strategy, pick one of: none, base, probe"
        
        # limits
        self.stuck_limit_no_steps = args.stuck_limit_no_steps           # Number of steps with no movement, can rotate
        self.stuck_limit_no_turns = args.stuck_limit_no_turns           # Number of steps with no movemnet OR rotation
        self.heuristic_turn_steps = args.heuristic_turn_steps           # Number of steps to turn
        self.heuristic_forward_steps = args.heuristic_forward_steps     # Number of steps to move forward

        self.num_agents = num_scenes

        # variables to detect stuck agents
        self.num_steps_stuck_no_turns = np.zeros(self.num_agents, dtype=int)  # Counter for steps without
        self.num_steps_stuck_no_moves = np.zeros(self.num_agents, dtype=int)  # Counter for each agent
        self.previous_poses = np.ones((self.num_agents, 3)) * -np.inf         
        self.is_stuck = np.zeros(self.num_agents, dtype=bool)  
        self.performing_heuristic = np.zeros(self.num_agents, dtype=bool)  

        # variables for heuristic strategy
        self.steps_since_stuck = np.zeros(self.num_agents, dtype=int)  # Counter for steps since last stuck

        if self.strategy == 'probe':
            self.last_turn_direction = np.ones(self.num_agents, dtype=int)  # 0 for right, 1 for left
            self.last_turn_size = np.ones(self.num_agents, dtype=int) * self.heuristic_turn_steps  # number of turns in the current direction

        # print a concise summary of all parameters
        print(f"Heuristic Tracker initialized with strategy: {self.strategy}")
        print(f"  Stuck limits - No Moves: {self.stuck_limit_no_steps}, No Turns: {self.stuck_limit_no_turns}")
        print(f"  Heuristic steps - Turn: {self.heuristic_turn_steps}, Forward: {self.heuristic_forward_steps}\n")

    # reset all counters and previous poses
    def reset(self):
        self.num_steps_stuck_no_turns = np.zeros(self.num_agents, dtype=int)
        self.num_steps_stuck_no_moves = np.zeros(self.num_agents, dtype=int)
        self.previous_poses = np.ones((self.num_agents, 3)) * -np.inf
        self.is_stuck = np.zeros(self.num_agents, dtype=bool)

    # update the tracker with current local poses
    def update(self, local_poses):
        if self.strategy == 'none':
            pass

        current_poses = local_poses.cpu().numpy()  # Convert to NumPy array for comparison
            
        # Check for change in position (x, y)
        pos_unchanged = np.all(np.isclose(current_poses[:, :2], self.previous_poses[:, :2]), axis=1)

        # print("Current Poses: ", current_poses)
        # print("pos_uncahged: ", pos_unchanged)
        self.num_steps_stuck_no_moves = np.where(pos_unchanged, self.num_steps_stuck_no_moves + 1, 0)

        # check for changes in pos and orientation (x, y, o)
        pos_unchanged = np.all(np.isclose(current_poses, self.previous_poses), axis=1)
        self.num_steps_stuck_no_turns = np.where(pos_unchanged, self.num_steps_stuck_no_turns + 1, 0)

        # Set is_stuck boolean if counter reaches the limit
        self.is_stuck = np.logical_or(self.num_steps_stuck_no_moves >= self.stuck_limit_no_steps,
                                        self.num_steps_stuck_no_turns >= self.stuck_limit_no_turns)

        # dont interrupt if already performing heuristic
        self.is_stuck = np.logical_or(self.is_stuck, self.performing_heuristic)

        # print("stuck num moves: ", self.num_steps_stuck_no_moves)
        # print("stuck num turns: ", self.num_steps_stuck_no_turns)
        # print("Stuck Status: ", self.is_stuck)  
        # print("Heuristic Status: ", self.performing_heuristic)  

        return
    
    # return a heuristic action for each stuck agent, adjust internal flags and counters accordingly
    def get_heuristic_actions(self, local_poses):
        if np.any(self.is_stuck) == False:
            self.previous_poses = local_poses.cpu().numpy().copy()
            return [None] * self.num_agents  # No heuristic actions needed

        actions = [-1] * self.num_agents
        current_poses = local_poses.cpu().numpy()  # Convert to NumPy array for comparison

        for i in range(self.num_agents):
            if self.is_stuck[i]:
                self.performing_heuristic[i] = True # set to true, only disabled when heuristic done

                # keep turning and try moving forward until unstuck
                if self.strategy == "probe":
                    if self.steps_since_stuck[i] < self.last_turn_size[i]:
                        self.steps_since_stuck[i] += 1
                        
                        actions[i] = 3 if self.last_turn_direction[i] == 1 else 2   # turn in the opposite direction
                    
                    # Phase 2: Try moving forward for heuristic_forward_steps
                    elif self.steps_since_stuck[i] < self.last_turn_size[i] + self.heuristic_forward_steps:
                        # Check if position actually changed
                        # if not np.array_equal(local_poses[i][:2], self.previous_poses[i][:2]):
                        #     self.is_stuck[i] = False
                        # else:
                        self.steps_since_stuck[i] += 1
                        actions[i] = 1  # go straight
                            
                    # Phase 3: Cycle complete, repeat from turn phase
                    else:
                        if not np.all(np.isclose(current_poses[i][:2], self.previous_poses[i][:2])):
                            self.is_stuck[i] = False
                            self.performing_heuristic[i] = False

                            self.num_steps_stuck_no_moves[i] = 0
                            self.num_steps_stuck_no_turns[i] = 0
                            self.steps_since_stuck[i] = 0

                            self.last_turn_size[i] = self.heuristic_turn_steps
                            self.last_turn_direction[i] = 1 

                        else:   # increment turn counters and change direction
                            self.steps_since_stuck[i] = 0  

                            self.last_turn_direction[i] = 1 - self.last_turn_direction[i]
                            self.last_turn_size[i] += self.heuristic_turn_steps

                        actions[i] = 1  # go straight
                
                # turn to the right and step forward
                elif self.strategy == "base":
                    # turn to the right until the counter is > limit + heuristic steps
                    if self.steps_since_stuck[i] < self.heuristic_turn_steps:
                        self.steps_since_stuck[i] += 1  # Decrement counter to limit turning steps
                        
                        actions[i] = 3  # turn right

                    # move forward after turning enough steps
                    elif self.steps_since_stuck[i] < self.heuristic_turn_steps + self.heuristic_forward_steps:
                        self.steps_since_stuck[i] += 1  # Decrement counter to limit turning steps
                        
                        actions[i] = 1  # go straight

                    else:
                        self.is_stuck[i] = False  # Reset stuck status
                        self.performing_heuristic[i] = False

                        self.num_steps_stuck_no_moves[i] = 0
                        self.num_steps_stuck_no_turns[i] = 0
                        self.steps_since_stuck[i] = 0

                        actions[i] = 1  # go straight
                
                # no heuristic, reset stuck flag
                else:
                    self.is_stuck[i] = False
                    self.performing_heuristic[i] = False

        # # if unstuck, reset counters
        # for i in range(self.num_agents):
        #     if not self.is_stuck[i]:
        #         self.num_steps_stuck_no_moves[i] = 0
        #         self.num_steps_stuck_no_turns[i] = 0
        #         self.steps_since_stuck[i] = 0

        # Update previous poses for the next check
        self.previous_poses = current_poses.copy()

        return actions
    
    def _get_stuck(self):
        return self.is_stuck