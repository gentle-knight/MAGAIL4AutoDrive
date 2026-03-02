import torch
import torch.nn as nn


def _get_activation(name: str):
    name = (name or "elu").lower()
    mapping = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leakyrelu": nn.LeakyReLU,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation: {name}")
    return mapping[name]()


class ActorCritic(nn.Module):
    """Minimal HBBC ActorCritic for inference-only deployment."""

    def __init__(
        self,
        num_actor_obs=18,
        num_critic_obs=18,
        num_actions=2,
        latent_c_dim=4,
        latent_eps_dim=6,
        use_style_latent=True,
        actor_hidden_dims=None,
        activation="elu",
    ):
        super().__init__()
        _ = num_critic_obs  # kept for checkpoint compatibility
        if actor_hidden_dims is None:
            actor_hidden_dims = [512, 256, 128]

        act_fn = _get_activation(activation)
        self.latent_c_dim = int(latent_c_dim)
        self.latent_eps_dim = int(latent_eps_dim)
        self.use_style_latent = bool(use_style_latent)

        layers = [nn.Linear(num_actor_obs, actor_hidden_dims[0]), act_fn]
        for i in range(len(actor_hidden_dims) - 1):
            layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
            layers.append(_get_activation(activation))
        self.actor_trunk = nn.Sequential(*layers)
        self.actor_head = nn.Linear(actor_hidden_dims[-1], num_actions)

        if self.use_style_latent:
            self.style_trunk = nn.Sequential(
                nn.Linear(self.latent_eps_dim, 512),
                _get_activation(activation),
                nn.Linear(512, 256),
                _get_activation(activation),
                nn.Linear(256, 128),
                _get_activation(activation),
            )
            self.style_head = nn.Linear(128, self.latent_eps_dim)
            self.style_activation = torch.tanh

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        if self.use_style_latent:
            obs = observations[..., :-(self.latent_c_dim + self.latent_eps_dim)]
            eps = observations[..., -self.latent_c_dim - self.latent_eps_dim:-self.latent_c_dim]
            c = observations[..., -self.latent_c_dim:]
            eps = self.style_activation(self.style_head(self.style_trunk(eps)))
            observations = torch.cat([obs, eps, c], dim=-1)
        embedding = self.actor_trunk(observations)
        return self.actor_head(embedding)
