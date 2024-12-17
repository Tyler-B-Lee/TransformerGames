import logging
logging.basicConfig(
    filename='file.log',
    format="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
    filemode='w',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

import models.tictactoe_model as tictactoe_model
import environments.tictactoe_env as tictactoe_env

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import random
import numpy as np

# let's train the model
def train_tictactoe_model(seed=0):
    # set the seed
    torch.manual_seed(seed)

    # create the environment
    env = make_vec_env(tictactoe_env.TicTacToeEnv, n_envs=1, seed=seed)

    # create the PPO agent
    agent = PPO(tictactoe_model.TicTacToePolicy, env, verbose=1)

    # train the agent
    agent.learn(total_timesteps=10000)

    # save the model
    agent.save("tictactoe_model")

    return agent

def print_top_actions(action_probs):
    top5_action_idx = np.argsort(-action_probs)[:5]
    top5_actions = action_probs[top5_action_idx]
    logger.debug(f"Top 5 actions: {[str(i) + ': ' + str(round(a,4))[:5] for i,a in zip(top5_action_idx, top5_actions)]}")

def play_against_random(agent: PPO):
    env = tictactoe_env.TicTacToeEnv()
    obs, _ = env.reset()
    done = False

    agent_turn = 0 if (random.random() < 0.5) else 1
    logger.debug(f"Agent plays as player {agent_turn + 1}")

    env.render()

    while not done:
        if env.current_player_num == agent_turn:
            with torch.no_grad():
                obs = obs.unsqueeze(0)
                action_probs = agent.policy.action_probability(obs)[0].numpy()
                value = agent.policy.predict_values(obs)[0].item()
                wr = (value + 1) * 50
                logger.debug(f"Agent thinks it has a {wr:.2f}% chance of winning")
                print_top_actions(action_probs)
                action = np.random.choice(len(action_probs), p=action_probs)
        else:
            # get indices of legal actions
            legal_actions = env.legal_actions
            legal_actions = torch.where(legal_actions > 0.5)[0]
            action = random.choice(legal_actions)

        logger.debug(f"Player {env.current_player_num + 1} plays action {action}")
        obs, reward, done, a, b = env.step(action)
        env.render()


if __name__ == "__main__":
    # train_tictactoe_model()

    # load the model and play a game
    agent = PPO.load("tictactoe_model")
    play_against_random(agent)