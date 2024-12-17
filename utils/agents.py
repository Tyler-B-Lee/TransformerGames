import logging
logger = logging.getLogger(__name__)
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import random
import string
import torch

def sample_action(action_probs):
    action = np.random.choice(len(action_probs), p = action_probs)
    return action

def mask_actions(legal_actions, action_probs):
    masked_action_probs = action_probs * legal_actions
    masked_action_probs = masked_action_probs / masked_action_probs.sum()
    return masked_action_probs

class Agent():
  def __init__(self, name, model = None):
      self.name = name
      self.id = self.name + '_' + ''.join(random.choice(string.ascii_lowercase) for x in range(5))
      self.model = model
      self.points = 0

  def print_top_actions(self, action_probs):
    top5_action_idx = np.argsort(-action_probs)[:5]
    top5_actions = action_probs[top5_action_idx]
    logger.debug(f"Top 5 actions: {[str(i) + ': ' + str(round(a,2))[:5] for i,a in zip(top5_action_idx, top5_actions)]}")

  def choose_action(self, env, choose_best_action, mask_invalid_actions):
      if self.name == 'rules':
        action_probs = np.array(env.rules_move())
        value = None
      elif self.name == 'random':
        legal_actions = env.legal_actions.numpy()
        action_probs = legal_actions / legal_actions.sum()  # Uniform distribution over legal actions
        value = None
      else:
        with torch.no_grad():
          obs_tensor = env.observation.unsqueeze(0)
          action_probs = self.model.policy.action_probability(obs_tensor)[0].numpy()
          value = self.model.policy.predict_values(obs_tensor)[0].item()
          logger.debug(f'Value {value:.2f}')
          wr = (value + 1) * 50
          logger.debug(f'Agent thinks it has a {wr:.2f}% chance of winning')

      self.print_top_actions(action_probs)
      
      if mask_invalid_actions and self.name != 'random':
        action_probs = mask_actions(env.legal_actions.numpy(), action_probs)
        logger.debug('Masked ->')
        self.print_top_actions(action_probs)
        
      action = np.argmax(action_probs)
      logger.debug(f'Best action {action}')

      if not choose_best_action and self.name != 'random':
          action = sample_action(action_probs)
          logger.debug(f'Sampled action {action} chosen')

      return action



