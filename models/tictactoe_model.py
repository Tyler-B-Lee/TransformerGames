from gymnasium import Space
import torch
import torch.nn as nn
from torch.distributions.utils import logits_to_probs
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs


OBS_LENGTH = 18
TOKEN_LENGTH = 2
ACTIONS = 9

TRANSFORMER_FEATURE_SIZE = 64
TRANSFORMER_FF_DIM = 128
TRANSFORMER_NHEAD = 8
TRANSFORMER_NUM_LAYERS = 3

POLICY_FEATURE_SIZE = 16
VALUE_FEATURE_SIZE = 16

COMPUTE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py
class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).to(COMPUTE_DEVICE)
        embeddings = self.embeddings_table[final_mat].to(COMPUTE_DEVICE)

        return embeddings
    

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=18):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.inv_freq = 1. / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.inv_freq = self.inv_freq.to(COMPUTE_DEVICE)

    def forward(self, x):
        sinusoid_inp = torch.einsum('i , j -> i j', torch.arange(x.shape[1]).float(), self.inv_freq)
        pos_emb = torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=-1)
        
        return pos_emb


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 2

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
        self.rotary_pos_emb = RotaryPositionalEmbedding(hid_dim)

    def apply_rotary_pos_emb(self, x, pos_emb):
        half_dimension = x.shape[-1] // 2
        x1, x2 = x[..., :half_dimension], x[..., half_dimension:]
        pos_emb1, pos_emb2 = pos_emb[..., :half_dimension], pos_emb[..., half_dimension:]
        return torch.cat((x1 * pos_emb1.cos() + x2 * pos_emb1.sin(), x2 * pos_emb2.cos() - x1 * pos_emb2.sin()), dim=-1)

    def forward(self, query, key, value, mask = None):
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        batch_size = query.shape[0]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        pos_emb = self.rotary_pos_emb(query)

        query = self.apply_rotary_pos_emb(query, pos_emb)
        key = self.apply_rotary_pos_emb(key, pos_emb)
        value = self.apply_rotary_pos_emb(value, pos_emb)

        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) / self.scale

        #r_q1 = [batch size, n heads, query len, head dim]
        #r_k1 = [batch size, n heads, key len, head dim]
        #attn = [batch size, n heads, query len, key len]

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim = -1))

        #attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        x = torch.matmul(attn, r_v1)
        
        #r_v1 = [batch size, n heads, value len, head dim]
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x


class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = nn.Sequential(
            nn.Linear(hid_dim, pf_dim),
            nn.ReLU(),
            nn.Linear(pf_dim, hid_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, src len]
                
        #self attention
        _src = self.self_attn_layer_norm(src)
        
        #_src = [batch size, src len, hid dim]
        
        src = self.self_attention(_src, _src, _src, src_mask)
        
        #src = [batch size, src len, hid dim]
        
        #dropout, residual connection and layer norm
        src = self.dropout(src)
        
        src = src + _src
        
        src = self.ff_layer_norm(src)
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #_src = [batch size, src len, hid dim]
        
        #dropout, residual and layer norm
        _src = self.dropout(_src)
        
        src = src + _src
        
        src = self.ff_layer_norm(src)
        
        #src = [batch size, src len, hid dim]
        
        return src


class CustomTransformerExtractor(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_dim, TRANSFORMER_FEATURE_SIZE)
        self.layers = nn.ModuleList([MyTransformerEncoderLayer(TRANSFORMER_FEATURE_SIZE, TRANSFORMER_NHEAD, TRANSFORMER_FF_DIM, 0.1, COMPUTE_DEVICE) for _ in range(TRANSFORMER_NUM_LAYERS)])
        # self.fc = nn.Linear(TRANSFORMER_FEATURE_SIZE, TRANSFORMER_FEATURE_SIZE)
        # self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, None)
        # x = self.fc(x)
        x = x.mean(dim=1)
        return x


class CustomValueHead(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, VALUE_FEATURE_SIZE * 2),
            nn.ReLU(),
            nn.Linear(VALUE_FEATURE_SIZE * 2, VALUE_FEATURE_SIZE),
            nn.ReLU(),
            nn.Linear(VALUE_FEATURE_SIZE, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.seq(x)


class CustomPolicyHead(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, POLICY_FEATURE_SIZE * 2),
            nn.ReLU(),
            nn.Linear(POLICY_FEATURE_SIZE * 2, POLICY_FEATURE_SIZE),
            nn.ReLU(),
            nn.Linear(POLICY_FEATURE_SIZE, ACTIONS)
        )

    def forward(self, x, legal_actions):
        mask = (1 - legal_actions) * -1e8
        # print(f"Mask: {mask} | shape: {mask.shape}")
        # print(f"seq(x): {self.seq(x)} | shape: {self.seq(x).shape}")
        return self.seq(x) + mask


class FeatureMaskExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Space, features_dim: int, device: torch.device):
        super().__init__(observation_space, features_dim)
        self.extractor = CustomTransformerExtractor(input_dim = TOKEN_LENGTH).to(device)
        self.device = device
    
    def forward(self, x):
        # input size: (batch_size, OBS_LENGTH + ACTIONS)
        x = x.to(self.device)
        obs, legal_actions = torch.split(x, [OBS_LENGTH, ACTIONS], dim=1)
        # reshape obs to (batch_size, 9, 2)
        obs = obs.view(-1, 9, 2)
        features = self.extractor(obs)
        return features, legal_actions


class CustomMLPExtractor(nn.Module):
    def __init__(
        self,
        last_layer_dim_pi: int = 8,
        last_layer_dim_vf: int = 8,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"> Loading Extractor Heads on device: {self.device}")

        # Policy network
        self.policy_net = CustomPolicyHead(input_dim=TRANSFORMER_FEATURE_SIZE).to(self.device)
        # Value network
        self.value_net = CustomValueHead(input_dim=TRANSFORMER_FEATURE_SIZE).to(self.device)

    def forward(self, features):
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features, legal_actions):
        features = features.to(self.device)
        legal_actions = legal_actions.to(self.device)
        return self.policy_net(features, legal_actions)

    def forward_critic(self, features):
        features = features.to(self.device)
        return self.value_net(features)


class TicTacToePolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(TicTacToePolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            features_extractor_class=FeatureMaskExtractor,
            features_extractor_kwargs=dict(features_dim=TRANSFORMER_FEATURE_SIZE, device=device),
            **kwargs,
        )

    def _build_mlp_extractor(self):
        self.mlp_extractor = CustomMLPExtractor()

    def forward(self, obs, deterministic=False):
        obs = obs.to(self.device)
        # print(f"Obs: {obs} | shape: {obs.shape}")
        features, legal_actions = self.features_extractor(obs)
        # print(f"Features: {features} | shape: {features.shape}")
        # print(f"Legal Actions: {legal_actions} | shape: {legal_actions.shape}")
        policy_logits = self.mlp_extractor.forward_actor(features, legal_actions)
        values = self.mlp_extractor.forward_critic(features)
        dist = CategoricalDistribution(ACTIONS)
        dist.proba_distribution(policy_logits)

        if deterministic:
            actions = torch.argmax(policy_logits, dim=1)
        else:
            actions = dist.sample()
        
        log_prob = dist.log_prob(actions)
        return actions, values, log_prob
    
    def _predict(self, obs, deterministic=False):
        obs = obs.to(self.device)
        features, legal_actions = self.features_extractor(obs)
        policy_logits = self.mlp_extractor.forward_actor(features, legal_actions)
        dist = CategoricalDistribution(ACTIONS)
        dist.proba_distribution(policy_logits)

        if deterministic:
            actions = torch.argmax(policy_logits, dim=1)
        else:
            actions = dist.sample()
        
        return actions
    
    def evaluate_actions(self, obs, actions):
        obs = obs.to(self.device)
        features, legal_actions = self.features_extractor(obs)
        policy_logits = self.mlp_extractor.forward_actor(features, legal_actions)
        values = self.mlp_extractor.forward_critic(features)
        dist = CategoricalDistribution(ACTIONS)
        dist.proba_distribution(policy_logits)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        obs = obs.to(self.device)
        features, legal_actions = self.features_extractor(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return latent_vf
    
    def action_probability(self, obs: PyTorchObs):
        "Return the action probabilities for each action according to the current policy given the observations."
        obs = obs.to(self.device)
        features, legal_actions = self.features_extractor(obs)
        policy_logits = self.mlp_extractor.forward_actor(features, legal_actions)
        return logits_to_probs(policy_logits)


if __name__ == "__main__":
    # create dummy input of a tictactoe position
    import torch
    import gymnasium as gym
    obs = [
        [1,0, 0,0, 0,1, 0,0, 1,0, 0,0, 1,0, 0,0, 0,1,
            0,1,0,1,0,1,0,1,0],
        [0,1, 0,0, 0,1, 0,0, 1,0, 0,0, 0,0, 0,0, 0,1,
            0,1,0,1,0,1,1,1,0],
    ]
    obs = torch.tensor(obs, dtype=torch.float32)
    print(f"Obs: {obs}")
    policy = TicTacToePolicy(gym.spaces.Box(low=0, high=1, shape=(OBS_LENGTH + ACTIONS,)), gym.spaces.Discrete(ACTIONS), lr_schedule=lambda x: 1e-4)
    actions, values, log_prob = policy(obs)
    print(f"Actions: {actions}")
    print(f"Values: {values}")
    print(f"Log Prob: {log_prob}")
    print(f"\n\nObs: {obs}")
    values = policy.predict_values(obs)
    print(f"Values: {values}")
    action_probs = policy.action_probability(obs)
    print(f"Action Probs: {action_probs}")