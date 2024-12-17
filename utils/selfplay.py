import numpy as np
import random

from utils.files import load_model, load_all_models, get_best_model_name
from utils.agents import Agent

import logging
logger = logging.getLogger(__name__)

def selfplay_wrapper(env):
    class SelfPlayEnv(env):
        # wrapper over the normal single player env, but loads the best self play model
        def __init__(self, opponent_type, verbose):
            super(SelfPlayEnv, self).__init__(verbose)
            self.opponent_type = opponent_type
            self.opponent_models = load_all_models(self)
            self.best_model_name = get_best_model_name(self.name)

        def setup_opponents(self):
            if self.opponent_type == 'rules':
                self.opponent_agent = Agent('rules')
            else:
                # incremental load of new model
                best_model_name = get_best_model_name(self.name)
                if self.best_model_name != best_model_name:
                    self.opponent_models.append(load_model(self, best_model_name))
                    self.best_model_name = best_model_name

                if self.opponent_type == 'random':
                    start = 0
                    end = len(self.opponent_models) - 1
                    i = random.randint(start, end)
                    self.opponent_agent = Agent('ppo_opponent', self.opponent_models[i]) 

                elif self.opponent_type == 'best':
                    self.opponent_agent = Agent('ppo_opponent', self.opponent_models[-1])  

                elif self.opponent_type == 'mostly_best':
                    j = random.uniform(0,1)
                    if j < 0.8:
                        self.opponent_agent = Agent('ppo_opponent', self.opponent_models[-1])  
                    else:
                        start = 0
                        end = len(self.opponent_models) - 1
                        i = random.randint(start, end)
                        self.opponent_agent = Agent('ppo_opponent', self.opponent_models[i])  

                elif self.opponent_type == 'base':
                    self.opponent_agent = Agent('base', self.opponent_models[0])  

            self.agent_player_num = np.random.choice(self.n_players)
            self.agents = [self.opponent_agent] * self.n_players
            self.agents[self.agent_player_num] = None
            try:
                #if self.players is defined on the base environment
                logger.debug(f'Agent plays as Player {self.players[self.agent_player_num].id}')
            except:
                pass

        def reset(self, seed=None):
            super().reset(seed=seed)
            self.setup_opponents()

            if self.current_player_num != self.agent_player_num:   
                self.continue_game()

            return self.observation, {}

        @property
        def current_agent(self):
            return self.agents[self.current_player_num]

        def continue_game(self):
            observation = None
            reward = None
            terminated, truncated = False, False

            while self.current_player_num != self.agent_player_num:
                self.render()
                action = self.current_agent.choose_action(self, choose_best_action = False, mask_invalid_actions = False)
                observation, reward, terminated, truncated, _ = super(SelfPlayEnv, self).step(action)
                logger.debug(f'Rewards: {reward}')
                logger.debug(f'Terminated: {terminated}')
                logger.debug(f'Truncated: {truncated}')
                if terminated or truncated:
                    break

            return observation, reward, terminated, truncated, None

        def step(self, action):
            self.render()
            observation, reward, terminated, truncated, _ = super(SelfPlayEnv, self).step(action)
            logger.debug(f'Action played by agent: {action}')
            logger.debug(f'Rewards: {reward}')
            logger.debug(f'Terminated: {terminated}')
            logger.debug(f'Truncated: {truncated}')

            done = terminated or truncated

            if not done:
                package = self.continue_game()
                if package[0] is not None:
                    observation, reward, terminated, truncated, _ = package

            agent_reward = reward[self.agent_player_num]
            logger.debug(f'\nReward To Agent: {agent_reward}')

            if terminated or truncated:
                self.render()

            return observation, agent_reward, terminated, truncated, {}

    return SelfPlayEnv