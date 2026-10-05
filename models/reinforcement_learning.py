import random
import numpy as np
from sklearn.linear_model import SGDRegressor

GRID = [
    "Aooo#ooooo",
    "o#oo#ooo#o",
    "o#o#oo#o#o",
    "ooooDDo#oo",
    "o#ooDDoo#o",
    "ooooDD#ooo",
    "o#ooDDo#oo",
    "ooooDDo#oo",
    "o#o#oo#o#o",
    "oooo#ooooT",
]

ROWS = 10
COLUMNS = 10
START = (0, 0)
GOAL = (9, 9)

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
NUMBER_OF_ACTIONS = len(ACTIONS)
NUMBER_OF_STATES = ROWS * COLUMNS
NUMBER_OF_FEATURES = NUMBER_OF_STATES * NUMBER_OF_ACTIONS

REWARDS = {
    "normal": -1,
    "out_of_bounds": -3,
    "wall": -5,
    "danger": -10,
    "goal": 100,
}

MAX_STEPS = 200
GAMMA = 0.95
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995
DEFAULT_EPISODES = 1000


def step(state, action):
    row = state[0] + ACTIONS[action][0]
    column = state[1] + ACTIONS[action][1]

    if not (0 <= row < ROWS and 0 <= column < COLUMNS):
        return state, REWARDS["out_of_bounds"], False, "Out of bounds"

    cell = GRID[row][column]

    if cell == "#":
        return state, REWARDS["wall"], False, "Wall"

    next_state = (row, column)

    if cell == "T":
        return next_state, REWARDS["goal"], True, "Target"

    if cell == "D":
        return next_state, REWARDS["danger"], False, "Danger Zone"

    return next_state, REWARDS["normal"], False, "Path"


def encode(state, action):
    features = np.zeros(NUMBER_OF_FEATURES, dtype=float)
    state_index = state[0] * COLUMNS + state[1]
    feature_index = state_index * NUMBER_OF_ACTIONS + action
    features[feature_index] = 1.0
    return features


def predict_q_values(model, state):
    features = np.array([
        encode(state, action)
        for action in range(NUMBER_OF_ACTIONS)
    ])
    return model.predict(features)


def train(episodes=DEFAULT_EPISODES):
    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    rng = random.Random(42)
    gamma = GAMMA
    epsilon = INITIAL_EPSILON

    model = SGDRegressor(
        loss="squared_error",
        penalty=None,
        fit_intercept=False,
        learning_rate="constant",
        eta0=0.1,
        random_state=42,
    )

    model.partial_fit(
        np.zeros((1, NUMBER_OF_FEATURES)),
        np.array([0.0]),
    )

    successes = 0
    rewards = []
    best_path = None
    best_reward = float("-inf")

    for _ in range(episodes):
        state = START
        total = 0
        path = [state]

        for _ in range(MAX_STEPS):
            if rng.random() < epsilon:
                action = rng.randrange(NUMBER_OF_ACTIONS)
            else:
                q_values = predict_q_values(model, state)
                best_actions = np.flatnonzero(q_values == q_values.max()).tolist()
                action = rng.choice(best_actions)

            next_state, reward, terminated, _ = step(state, action)

            if terminated:
                target = float(reward)
                successes += 1
            else:
                next_q_values = predict_q_values(model, next_state)
                target = reward + gamma * float(next_q_values.max())

            features = encode(state, action).reshape(1, -1)
            model.partial_fit(features, np.array([target]))

            state = next_state
            total += reward
            path.append(state)

            if terminated:
                break

        rewards.append(total)

        if total > best_reward:
            best_reward = total
            best_path = list(path)

        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

    state = START
    path = [state]
    steps = []

    for number in range(1, MAX_STEPS + 1):
        q_values = predict_q_values(model, state)
        action = int(np.argmax(q_values))
        next_state, reward, terminated, cell_type = step(state, action)

        steps.append({
            "number": number,
            "state": state,
            "action": ACTION_NAMES[action],
            "next_state": next_state,
            "cell_type": cell_type,
            "reward": reward,
        })
        path.append(next_state)
        state = next_state

        if terminated:
            break

    reached_goal = state == GOAL
    total_reward = sum(s["reward"] for s in steps)

    q_table = []
    for row in range(ROWS):
        for column in range(COLUMNS):
            position = (row, column)
            cell = GRID[row][column]
            if cell != "#" and position != GOAL:
                q_table.append({
                    "state": position,
                    "action_values": [
                        round(float(v), 2)
                        for v in predict_q_values(model, position).tolist()
                    ],
                })

    recent = rewards[-100:] if len(rewards) >= 100 else rewards
    average_reward = round(sum(recent) / len(recent), 2)
    success_percentage = round((successes / episodes) * 100, 2)

    return {
        "episodes": episodes,
        "successes": successes,
        "success_percentage": success_percentage,
        "final_epsilon": round(epsilon, 4),
        "average_reward": average_reward,
        "reached_goal": reached_goal,
        "path": path,
        "best_path": best_path,
        "steps": steps,
        "q_table": q_table,
        "movements": len(steps),
        "total_reward": total_reward,
    }