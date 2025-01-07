import numpy as np
import sys

class QLearningAgent:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.1):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = np.zeros((state_space_size, state_space_size, action_space_size))

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_space_size)
        return np.argmax(self.q_table[state[0], state[1]])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
        td_target = reward + self.discount_factor * self.q_table[next_state[0], next_state[1], best_next_action]
        td_error = td_target - self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] += self.learning_rate * td_error

    def decay_exploration_rate(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def set_obstacle(self, state):
        self.q_table[state[0], state[1]] = -np.inf
        
def train_q_learning(agent, episodes, max_steps_per_episode):
    rewards_log = list()
    last_action = None
    #limits = get_limits((0, 0), None, agent.state_space_size, 5)
    for episode in range(episodes):
        state = (np.random.randint(agent.state_space_size), np.random.randint(agent.state_space_size))
        if is_obstacle(state, agent):
            continue
        #if not is_inside_limits(state, limits):
        #    continue
        accumulated_reward = 0
        for step in range(max_steps_per_episode):
            action = agent.choose_action(state)
            next_state = get_next_state(state, action, agent.state_space_size)
            reward = -1

            if is_obstacle(next_state, agent):
                reward -= 10
                next_state = state

            if last_action is not None:
                reward -= check_turn(last_action, action)
            
            #if not is_inside_limits(next_state, limits):
            #    reward -= 10
            #    next_state = state

            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            accumulated_reward += reward
            last_action = action
            if is_terminal_state(state):
                reward += 10
                agent.update_q_table(state, action, reward, next_state)
                break
        rewards_log.append(accumulated_reward)
        agent.decay_exploration_rate()
    return agent.q_table, rewards_log

# define a function that check if is a valid state
def is_valid_state(state, state_space_size):
    return 0 <= state[0] < state_space_size and 0 <= state[1] < state_space_size

# define a function that check the obstacle
def is_obstacle(state, agent):
    return np.sum(agent.q_table[state[0], state[1]]) == -np.inf

def check_turn(action, next_action):
    actions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    return np.arccos(np.dot(actions[action], actions[next_action]) / (np.linalg.norm(actions[action]) * np.linalg.norm(actions[next_action]))) #* 180 / np.pi

def get_next_state(state, action, state_space_size):
    moves = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    next_state = (state[0] + moves[action][0], state[1] + moves[action][1])
    next_state = (max(0, min(next_state[0], state_space_size - 1)), max(0, min(next_state[1], state_space_size - 1)))
    return next_state

def is_terminal_state(state):
    # Define your terminal state condition here
    return state == (9, 9)

# Create a function that check if the agent is inside the limits given the state and the limits
def is_inside_limits(state, limits):
    # limit = (x_min, x_max, y_min, y_max)
    # state = (x, y)
    return (limits[0] <= state[0] <= limits[1]) and (limits[2] <= state[1] <= limits[3])

# create a function that given a size of the state_space, returns the limits
def get_limits(state, last_action, state_space_size, n):
    # state = (x, y)
    # last_action = int
    # state_space_size = int
    # n = state_space_size
    n = n - 1
    
    x, y = state

    if last_action == 0:

        (x_min, x_max, y_min, y_max) = (max(0, x - n), x, max(0, y - n // 2), min(state_space_size, y + n // 2))
    
    elif last_action == 1:
        (x_min, x_max, y_min, y_max) = (max(0, x - n), x, y, min(state_space_size, y + n))
    
    elif last_action == 2:
        (x_min, x_max, y_min, y_max) = (max(0, x - n //2), min(state_space_size, x + n // 2), y, min(state_space_size, y + n))
    
    elif last_action == 3:
        (x_min, x_max, y_min, y_max) = (x, min(state_space_size, x + n), y, min(state_space_size, y + n))

    elif last_action == 4:
        (x_min, x_max, y_min, y_max) = (x, min(state_space_size, x + n), max(0, y - n//2), min(state_space_size, y + n // 2))
    
    elif last_action == 5:
        (x_min, x_max, y_min, y_max) = (x, min(state_space_size, x + n), max(0, y - n), y)
    
    elif last_action == 6:
        (x_min, x_max, y_min, y_max) = (max(0, x - n // 2), min(state_space_size, x + n // 2), max(0, y - n), y)
    
    elif last_action == 7:
        (x_min, x_max, y_min, y_max) = (max(0, x - n), x, max(0, y - n), y)

    else:
        (x_min, x_max, y_min, y_max) = (max(0, x - n//2), min(state_space_size, x + n//2), max(0, y - n // 2), min(state_space_size, y + n // 2))
    
    return (x_min, x_max, y_min, y_max)


# Create a function that given a QTabled, returns the best path
def get_best_path(q_table):
    state = (0, 0)
    path = [state]
    while not is_terminal_state(state):
        action = np.argmax(q_table[state[0], state[1]])
        state = get_next_state(state, action, q_table.shape[0])
        path.append(state)
    return path

# create a code that reads the config.yaml file and returns the state_space_size and action_space_size
def read_config():
    import yaml
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)    
    return config['q_learning'], config["environment"]

# Create a function that given a current continuous position on the grid, returns the current discrete position on the grid
def get_current_position_on_grid(current_position, cell_size):
    return (int(current_position[0] // cell_size), int(current_position[1] // cell_size))

# Create a function that given a list with the path and the cell_size, returns the path in the map
def get_path_on_map(path, cell_size, map_T_grid):
    path_on_map = []
    for state in path:
        state = np.array([state[0] + cell_size / 2, state[1] + cell_size / 2, 1.0]).reshape((3, 1))
        state_on_map = map_T_grid @ state
        path_on_map.append(tuple(state_on_map.flatten()[:2].tolist()))
    return path_on_map


if __name__ == "__main__":
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    
    # convert agr1 to an array knowing that it's a string with the format "x,y"
    arg1 = arg1.split(",")
    map_current_position = np.array([float(arg1[0]), float(arg1[1]), 1.0]).reshape((3, 1))
    print("Current position on the map:", map_current_position.flatten()[:2])

    # Read the config file
    config_agent, config_env = read_config()
    
    # Get the grid_T_map
    grid_T_map = config_env['grid_T_map']
    grid_T_map = np.array(grid_T_map).reshape((3, 3))
    grid_current_position = grid_T_map @ map_current_position
    print("Current position on the grid:", grid_current_position.flatten()[:2])

    # Get the current position on the grid in discrete form
    current_position = get_current_position_on_grid(grid_current_position.flatten()[:2], config_env['cell_size'])
    print("Current discrete position on the grid:", current_position)


    state_space_size = 10  # Example size, adjust as needed
    action_space_size = 8
    agent = QLearningAgent(state_space_size, action_space_size, learning_rate = config_agent['learning_rate'],
                                            discount_factor = config_agent['discount_factor'], exploration_rate = config_agent['exploration_rate'],
                                            exploration_decay=config_agent['exploration_decay'], min_exploration_rate=config_agent['min_exploration_rate'])
    #agent.set_obstacle((8, 8))
    q_table, rewards_log = train_q_learning(agent, episodes=config_agent['episodes'], max_steps_per_episode=config_agent['max_steps_per_episode'])


    # Get the path
    path = get_best_path(q_table)
    print(path)

    # Get the path on the map
    path_on_map = get_path_on_map(path, config_env['cell_size'], np.linalg.inv(grid_T_map))
    print(path_on_map)