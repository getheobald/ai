import sys
import time
import pickle
import numpy as np
from tqdm import tqdm
from vis_gym import *
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BOLD = '\033[1m'  # ANSI escape sequence for bold text
RESET = '\033[0m' # ANSI escape sequence to reset text formatting

# flag values set by command line args
train_flag = 'train' in sys.argv
gui_flag = 'gui' in sys.argv

setup(GUI=gui_flag)
env = game # Gym environment already initialized within vis_gym.py

env.render() # Uncomment to print game state info

def hash(obs):
	'''
    Compute a unique compact integer ID representing the given observation.

    Encoding scheme:
      - Observation fields:
          * player_health: integer in {0, 1,, 1}
          * guard_in_cell: optional identifier of a guard in the player’s cell (e.g. 'G1', 'G2', ...)

      - Each cell contributes a single digit (0–8) to a base-9 number:
          * If the cell is out of bounds → code = 8
          * Otherwise:
                tile_type = 
                    0 → empty
                    1 → trap
                    2 → heal
                    3 → goal
                has_guard = 1 if one or more guards present, else 0
                cell_value = has_guard * 4 + tile_type  # ranges from 0 to 7

        The 9 cell_values (row-major order: top-left → bottom-right) form a 9-digit base-9 integer `window_hash`.

      - The final state_id packs:
            * window_hash  → fine-grained local state
            * guard_index  → identity of guard in player’s cell (0 if none, 1–4 otherwise)
            * player_health → coarse health component

        Specifically:
            WINDOW_SPACE = 9 ** 9
            GUARD_SPACE  = WINDOW_SPACE       # for guard_index (0–4)
            HEALTH_SPACE = GUARD_SPACE * 5    # for health (0–2)

            state_id = (player_health * HEALTH_SPACE) 
                     + (guard_index * GUARD_SPACE) 
                     + window_hash

    Returns:
        int: A unique, compact integer ID suitable for tabular RL (e.g. as a Q-table key).
    '''
	health = int(obs.get('player_health', 0))
	window = obs.get('window', {})

	# Build cell values in a stable order: dx -1..1 (rows), dy -1..1 (cols)
	cell_values = []
	for dx in [-1, 0, 1]:
		for dy in [-1, 0, 1]:
			cell = window.get((dx, dy))
			if cell is None or not cell.get('in_bounds', False):
				cell_values.append(8)
				continue

			# Determine tile type
			if cell.get('is_trap'):
				tile_type = 1
			elif cell.get('is_heal'):
				tile_type = 2
			elif cell.get('is_goal'):
				tile_type = 3
			else:
				tile_type = 0

			has_guard = 1 if cell.get('guards') else 0
			cell_value = has_guard * 4 + tile_type
			cell_values.append(cell_value)

	# Pack into base-9 integer
	window_hash = 0
	base = 1
	for v in cell_values:
		window_hash += v * base
		base *= 9

	# Include guard identity when player is in the center cell.
	# guard_in_cell is a convenience field set by the environment (e.g. 'G1' or None).
	guard_in_cell = obs.get('guard_in_cell')
	if guard_in_cell:
		# map 'G1' -> 1, 'G2' -> 2, etc.
		try:
			guard_index = int(str(guard_in_cell)[-1])
		except Exception:
			guard_index = 0
	else:
		guard_index = 0

	# window_hash uses 9^9 space; reserve an extra multiplier for guard identity (0..4)
	WINDOW_SPACE = 9 ** 9
	GUARD_SPACE = WINDOW_SPACE  # one slot per guard id
	HEALTH_SPACE = GUARD_SPACE * 5  # 5 possible guard_id values (0 = none, 1-4 = guards)

	state_id = int(health) * HEALTH_SPACE + int(guard_index) * GUARD_SPACE + window_hash
	return state_id

'''
Complete the function below to do the following:

		1. Run a specified number of episodes of the game (argument num_episodes). An episode refers to starting in some initial
			 configuration and taking actions until a terminal state is reached.
		2. Maintain and update Q-values for each state-action pair encountered by the agent in a dictionary (Q-table).
		3. Use epsilon-greedy action selection when choosing actions (explore vs exploit).
		4. Update Q-values using the standard Q-learning update rule.

Important notes about the current environment and state representation

		- The environment is partially observable: observations returned by env.get_observation() include a centered 3x3
			"window" around the player plus the player's health. Each observation is a dict with these relevant keys:
					- 'player_position': (x, y)
					- 'player_health': integer (0=Critical, 1=Injured, 2=Full)
					- 'window': a dict keyed by (dx,dy) offsets in {-1,0,1} x {-1,0,1}. Each entry contains:
								{ 'guards': list or None, 'is_trap': bool, 'is_heal': bool, 'is_goal': bool, 'in_bounds': bool }
					- 'at_trap', 'at_heal', 'at_goal', and 'guard_in_cell' are convenience fields for the center cell.

		- To make a compact and consistent state hash for tabular Q-learning, encode the 3x3 window plus player health into a single integer.
			use the provided hash(obs) function above. Note that the player position is not included in the hash, as it is not needed for local decision-making.

		- Your Q-table should be a dict mapping state_id -> np.array of length env.action_space.n. Initialize arrays to zeros
			when you first encounter a state.

		- The actions available in this environment now include movement, combat, healing and waiting. The action indices are:
					0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: FIGHT, 5: HIDE, 6: HEAL, 7: WAIT

		- Remember to call obs, reward, done, info = env.reset() at the start of each episode.

		- Use a learning-rate schedule per (s,a) pair, i.e. eta = 1/(1 + N(s,a)) where N(s,a) is the
			number of updates applied to that pair so far.

Finally, return the dictionary containing the Q-values (called Q_table).

'''

def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):
	"""
	Run Q-learning algorithm for a specified number of episodes.

    Parameters:
    - num_episodes (int): Number of episodes to run.
    - gamma (float): Discount factor.
    - epsilon (float): Exploration rate.
    - decay_rate (float): Rate at which epsilon decays. Epsilon should be decayed as epsilon = epsilon * decay_rate after each episode.

    Returns:
    - Q_table (dict): Dictionary containing the Q-values for each state-action pair.
    """
	Q_table = {} # maps state_id to np.array of Q values (8, one for each action)
	N_updates = {} # maps state_id to np.array of 8 counters, one for each (s, a) pair
	n_actions = env.action_space.n # 8 actions
	episode_rewards = [] # used for plot

	for episode in tqdm(range(num_episodes), desc="Training"): # label for cli clarity
		obs, reward, done, info = env.reset() # start fresh ep and return first obs
		total_reward = 0

		while not done:
			state = hash(obs) # call hash func to get unique hash for current state

			# init q val and update count as zero if we haven't seen state before
			if state not in Q_table:
				Q_table[state] = np.zeros(n_actions)
				N_updates[state] = np.zeros(n_actions)

			# select action with epsilon-greedy policy
			# np.random.random() outputs num in [0,1)
			if np.random.random() < epsilon:
				action = env.action_space.sample() # take random action
			else:
				action = np.argmax(Q_table[state]) # take optimal action

			# actually take the action
			next_obs, reward, done, info = env.step(action)
			total_reward += reward

			next_state = hash(next_obs)

			# init next state if unseen
			# needed so np.max(Q_table[next_state]) doesn't crash
			if next_state not in Q_table:
				Q_table[next_state] = np.zeros(n_actions)
				N_updates[next_state] = np.zeros(n_actions)

			# q-learning update equation

			# calc eta based on num updates to this (s,a) pair
			eta = 1.0 / (1.0 + N_updates[state][action])

			# optimal q-val from next state
			# V_opt(s')
			best_next = np.max(Q_table[next_state])

			bracket = reward + gamma * best_next

			# this is the q-learning equation!!!!
			Q_table[state][action] = (1 - eta) * Q_table[state][action] + eta * bracket
			N_updates[state][action] += 1 # increment counter for this (s,a) pair
			obs = next_obs # advance state

		episode_rewards.append(total_reward)

		# decay epsilon after each ep
		epsilon *= decay_rate

	# after all episodes, plot rewards
	# use rolling average to smooth the curve so the trend is visible
	window_size = max(1, num_episodes // 100)  # smooth over 1% of episodes
	smoothed = np.convolve(
		episode_rewards,
		np.ones(window_size) / window_size,
		mode='valid'
	)

	fig, ax = plt.subplots(figsize=(12, 5))
	ax.plot(episode_rewards, color='steelblue', alpha=0.25, linewidth=0.6, label='Episode reward')
	ax.plot(
		range(window_size - 1, len(episode_rewards)),
		smoothed,
		color='darkorange', linewidth=2,
		label=f'Rolling avg (window={window_size})'
	)
	ax.set_xlabel('Episode', fontsize=14)
	ax.set_ylabel('Total Reward', fontsize=14)
	ax.set_title(
		f'Q-Learning Training Rewards\n'
		f'(episodes={num_episodes}, decay={decay_rate})',
		fontsize=15
	)
	ax.legend(fontsize=12)
	ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
	ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{int(y):,}'))
	plt.tight_layout()
	plt.savefig(
		f'rewards_{num_episodes}_{decay_rate}.png',
		dpi=200, bbox_inches='tight'
	)
	plt.close()


	return Q_table

'''
Specify number of episodes and decay rate for training and evaluation.
'''

num_episodes = 1000
decay_rate = 0.999

'''
Run training if train_flag is set; otherwise, run evaluation using saved Q-table.
'''

if train_flag:
	Q_table = Q_learning(num_episodes=num_episodes, gamma=0.9, epsilon=1, decay_rate=decay_rate) # Run Q-learning

	# Save the Q-table dict to a file
	with open('Q_table_'+str(num_episodes)+'_'+str(decay_rate)+'.pickle', 'wb') as handle:
		pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)


'''
Evaluation mode: play episodes using the saved Q-table. Useful for debugging/visualization.
Based on autograder logic used to execute actions using uploaded Q-tables.
'''

def softmax(x, temp=1.0):
	e_x = np.exp((x - np.max(x)) / temp)
	return e_x / e_x.sum(axis=0)

if not train_flag:
	
	rewards = []

	filename = 'Q_table_'+str(num_episodes)+'_'+str(decay_rate)+'.pickle'
	input(f"\n{BOLD}Currently loading Q-table from "+filename+f"{RESET}.  \n\nPress Enter to confirm, or Ctrl+C to cancel and load a different Q-table file.\n(set num_episodes and decay_rate in Q_learning.py).")
	Q_table = np.load(filename, allow_pickle=True)

	# tracking variables
	n_actions = env.action_space.n
	action_names = env.actions       # ['UP','DOWN','LEFT','RIGHT','FIGHT','HIDE','HEAL','WAIT']
	episode_lengths  = []            # steps taken per episode
	unseen_states    = set()         # states seen in eval but not in Q-table
	qtable_actions   = 0             # actions taken from known states
	random_actions   = 0             # actions taken because state was unseen
	total_actions    = 0

	# counts of each action taken in either heal state or sharing cell with each guard
	action_counts = {
		'HEAL': np.zeros(n_actions, dtype=float),
		'G1':   np.zeros(n_actions, dtype=float),
		'G2':   np.zeros(n_actions, dtype=float),
		'G3':   np.zeros(n_actions, dtype=float),
		'G4':   np.zeros(n_actions, dtype=float),
	}

	start_time = time.time()

	for episode in tqdm(range(10000), desc="Eval"): # label for cli clarity
		obs, reward, done, info = env.reset()
		total_reward = 0
		steps = 0
		
		while not done:
			state = hash(obs)
			total_actions += 1
			steps += 1

			try:
				action = np.random.choice(env.action_space.n, p=softmax(Q_table[state]))  # select action using softmax over Q-values
				qtable_actions += 1
			except KeyError:
				action = env.action_space.sample()  # random action, if state not in Q-table
				random_actions += 1
				unseen_states.add(state)
			
			# record action for heatmap
			if obs.get('at_heal'):
				action_counts['HEAL'][action] += 1
			guard = obs.get('guard_in_cell')
			if guard in action_counts:
				action_counts[guard][action] += 1
			
			obs, reward, done, info = env.step(action)
			
			total_reward += reward
			if gui_flag:
				refresh(obs, reward, done, info, delay=.1)  # update game screen if using GUI

		#print("Total reward:", total_reward) lol don't uncomment this unless you wanna spam terminal asf
		rewards.append(total_reward)
		episode_lengths.append(steps)
	avg_reward = sum(rewards)/len(rewards)

	elapsed = time.time() - start_time

	# print metrics
	print(f"\n{'='*58}")
	print(f"  EVALUATION RESULTS  |  episodes={num_episodes}, decay={decay_rate}")
	print(f"{'='*58}")
	print(f"  Unique states in Q-table           : {len(Q_table):,}")
	print(f"  Average reward (10k eval episodes) : {avg_reward:.2f}")
	print(f"  Average episode length             : {np.mean(episode_lengths):.2f} steps")
	print(f"  Total user time (10k episodes)     : {elapsed:.2f} s")
	print(f"  Unseen states during evaluation    : {len(unseen_states):,}")
	print(f"  % actions from Q-table             : {100*qtable_actions/total_actions:.2f}%")
	print(f"  % random actions (unseen states)   : {100*random_actions/total_actions:.2f}%")
	print(f"{'='*58}\n")

	# action distribution heatmap
	rows      = ['HEAL', 'G1', 'G2', 'G3', 'G4']
	heat_data = np.zeros((len(rows), n_actions))
	for i, key in enumerate(rows):
		total = action_counts[key].sum()
		if total > 0:
			heat_data[i] = action_counts[key] / total

	fig, ax = plt.subplots(figsize=(12, 5))
	im = ax.imshow(heat_data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)

	ax.set_xticks(range(n_actions))
	ax.set_xticklabels(action_names, fontsize=12)
	ax.set_yticks(range(len(rows)))
	ax.set_yticklabels(rows, fontsize=12)
	ax.set_xlabel('Action', fontsize=13)
	ax.set_ylabel('Context', fontsize=13)
	ax.set_title(
		f'Normalised Action Distribution by Context\n'
		f'episodes={num_episodes}, decay={decay_rate}',
		fontsize=14
	)

	# annotate each cell with its value
	for i in range(len(rows)):
		for j in range(n_actions):
			val = heat_data[i, j]
			ax.text(j, i, f'{val:.2f}', ha='center', va='center',
			        fontsize=10, color='black' if val < 0.6 else 'white')

	cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
	cbar.set_label('Normalised frequency', fontsize=11)
	plt.tight_layout()
	plt.savefig(
		f'action_dist_{num_episodes}_{decay_rate}.png',
		dpi=200, bbox_inches='tight'
	)
	plt.close()
	print(f"Heatmap saved → action_dist_{num_episodes}_{decay_rate}.png")