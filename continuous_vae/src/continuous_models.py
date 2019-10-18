from src.discrete_models import init_mlp
from src.discrete_models import GenerativeModel as DiscreteGenerativeModel
from src.discrete_models import InferenceNetwork as DiscreteInferenceNetwork
import torch
import torch.nn as nn


class MultilayerPerceptronNormal(nn.Module):
    def __init__(self, dims, non_linearity):
        """
        Args:
            dims: list of ints
            non_linearity: differentiable function

        Returns: nn.Module which represents an MLP with architecture

            x -> Linear(dims[0], dims[1]) -> non_linearity ->
            ...
            Linear(dims[-3], dims[-2]) ->
            non_linearity -> Linear(dims[-2], dims[-1]) -> mu
                          -> Linear(dims[-2], dims[-1]) -> exp -> std

        """

        super(MultilayerPerceptronNormal, self).__init__()
        self.dims = dims
        self.non_linearity = non_linearity
        self.linear_modules = nn.ModuleList()
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.linear_modules.append(nn.Linear(in_dim, out_dim))
        self.logsigma = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        temp = x
        for linear_module in self.linear_modules[:-1]:
            temp = self.non_linearity(linear_module(temp))
        mu = self.linear_modules[-1](temp)
        sig = torch.exp(self.logsigma(temp))
        return mu, sig


def init_normal_mlp(in_dim, out_dim, hidden_dim, num_layers, non_linearity=nn.Tanh()):
    """Initializes a MultilayerPerceptronNormal.

    Args:
        in_dim: int
        out_dim: int
        hidden_dim: int
        num_layers: int
        non_linearity: differentiable function

    Returns: a MultilayerPerceptron with the architecture

        x -> Linear(in_dim, hidden_dim) -> non_linearity ->
        ...
        Linear(hidden_dim, hidden_dim) -> non_linearity ->
        Linear(hidden_dim, out_dim) -> y

        where num_layers = 0 corresponds to

        x -> Linear(in_dim, out_dim) -> mu
          -> Linear(in_dim, out_dim) -> exp -> std
        """
    dims = [in_dim] + [hidden_dim for _ in range(num_layers)] + [out_dim]
    return MultilayerPerceptronNormal(dims, non_linearity)


class GenerativeModel(DiscreteGenerativeModel):
    def __init__(self,
                 num_stochastic_layers=2,
                 num_deterministic_layers=2,
                 latent_dim=50,
                 hidden_dim=200,
                 obs_dim=784,
                 train_obs_mean=None,
                 learn_prior=False,
                 device=torch.device('cpu')):
        """
        Args:
        num_stochastic_layers: int; 1 corresponds to a p(z)p(x | z)
        num_deterministic_layers: int; 0 corresponds to linear layers
        latent_dim: int; default = 200
        obs_dim: int; default = 784
        train_obs_mean: tensor of shape [obs_dim]
        device: torch.device('cpu') or torch.device('cuda')
        latent_dims: if specified, overrides num_stochastic_layers and
            latent_dim; latent_dims[0] is the dimensionality of the
            latent layer that is furthest from obs"""

        super().__init__(module_init_only=True)
        self.num_stochastic_layers = num_stochastic_layers
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.latent_dims = [latent_dim for _ in range(num_stochastic_layers)]

        self.num_deterministic_layers = num_deterministic_layers
        self.obs_dim = obs_dim

        self.device = device
        self.learn_prior = learn_prior

        # We don't learn the prior
        if self.learn_prior:
            print("Learning prior")
            self.latent_param_mu = nn.Parameter(torch.zeros(self.latent_dims[0], device=device, dtype=torch.float))
            self.latent_param_sig = nn.Parameter(torch.ones(self.latent_dims[0], device=device, dtype=torch.float))
        else:
            self.latent_param_mu  = torch.zeros(self.latent_dims[0], device=device, dtype=torch.float)
            self.latent_param_sig = torch.ones(self.latent_dims[0], device=device, dtype=torch.float)

        self.decoders = nn.ModuleDict()

        for i in range(1, self.num_stochastic_layers):
            self.decoders[str(i)] = init_normal_mlp(in_dim=self.latent_dims[i - 1],
                                                    out_dim=self.latent_dims[i],
                                                    hidden_dim=self.hidden_dim,
                                                    num_layers=num_deterministic_layers,
                                                    non_linearity=nn.Tanh())

        # This is the mlp from discrete.py that doesn't produce a sigma
        self.decoder_to_obs = init_mlp(in_dim=self.latent_dims[-1],
                                       out_dim=obs_dim,
                                       hidden_dim=self.hidden_dim,
                                       num_layers=num_deterministic_layers,
                                       non_linearity=nn.Tanh())

    def get_latent_layer_param(self, layer_idx, previous_latent_layer=None):
        """Returns params of distribution for single latent layer.

        Args:
            layer_idx: 0 means the layer furthest away from obs
            previous_latent_layer: tensor of shape [batch_size, latent_dim]
                (only applicable if layer_idx > 0)

        Returns:
            mu for Normal latent of shape [batch_size, latent_dim]
            sig for Normal latent of shape [batch_size, latent_dim]

            if layer_idx is 0, shape [latent_dim],
            otherwise [batch_size, latent_dim]"""

        if layer_idx == 0:
            return self.latent_param_mu, self.latent_param_sig
        else:
            return self.decoders[str(layer_idx)](previous_latent_layer)

    def get_latent_layer_dist(self, layer_idx, previous_latent_layer=None):
        """Returns distribution for single latent layer.

        Args:
            layer_idx: 0 means the layer furthest away from obs
            previous_latent_layer: tensor of shape [batch_size, latent_dim]
                (only applicable if layer_idx > 0)

        Returns: Normal distribution with event_shape [latent_dim]
            if layer_idx is 0, batch_shape is [],
            otherwise [batch_size]"""

        mu, sig = self.get_latent_layer_param(layer_idx, previous_latent_layer)

        return torch.distributions.Independent(torch.distributions.Normal(mu, sig), reinterpreted_batch_ndims=1)

    def get_obs_param(self, latent):
        """
        Args:
            latent: tuple (latent_0, ..., latent_N) where each
                latent_n is a tensor of shape [batch_size, latent_dim]
                with values in {0, 1}

        Returns: logits of Bernoulli likelihood of shape
            [batch_size, obs_dim]
        """
        latent_layer = latent[-1]
        return self.decoder_to_obs(latent_layer)

    def get_obs_dist(self, latent):
        """Args:
            latent: tuple (latent_0, ..., latent_N) where each
                latent_n is a tensor of shape [batch_size, latent_dim]
                with values in {0, 1}

        Returns: Bernoulli distribution of batch_shape [batch_size] and
            event_shape [obs_dim]
        """
        return torch.distributions.Independent(torch.distributions.Bernoulli(logits=self.get_obs_param(latent)),
                                               reinterpreted_batch_ndims=1)

    def sample_latent_and_obs(self, num_samples=1, reparam=True):
        """Args:
            num_samples: int

        Returns:
            latent: tuple (latent_0, ..., latent_N) where each
                latent_n is a tensor of shape [batch_size, latent_dim]
                with values in {0, 1}
            obs: tensor of shape [num_samples, obs_dim]
        """
        latent_dist = self.get_latent_dist()

        if reparam:
            latent = latent_dist.rsample((num_samples, ))
        else:
            latent = latent_dist.sample((num_samples, ))

        obs_dist = self.get_obs_dist(latent)
        obs = obs_dist.sample()

        return latent, obs


class InferenceNetwork(DiscreteInferenceNetwork):
    def __init__(self,
                 num_stochastic_layers=2,
                 num_deterministic_layers=2,
                 latent_dim=50,
                 hidden_dim=200,
                 obs_dim=784,
                 train_obs_mean=None,
                 device=torch.device('cpu')):
        """Args:
            num_stochastic_layers: int; 1 corresponds to a p(z)p(x | z)
            num_deterministic_layers: int; 0 corresponds to linear layers
            latent_dim: int; default = 200
            obs_dim: int; default = 784
            train_obs_mean: tensor of shape [obs_dim]
            device: torch.device('cpu') or torch.device('cuda')
            latent_dims: if specified, overrides num_stochastic_layers and
                latent_dim; latent_dims[0] is the dimensionality of the
                latent layer that is furthest from obs
        """
        super().__init__(module_init_only=True)
        self.num_stochastic_layers = num_stochastic_layers
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.latent_dims = [latent_dim for _ in range(num_stochastic_layers)]

        self.num_deterministic_layers = num_deterministic_layers
        self.obs_dim = obs_dim
        self.device = device

        self.encoder_to_obs = init_normal_mlp(in_dim=obs_dim,
                                              out_dim=self.latent_dims[-1],
                                              hidden_dim=self.hidden_dim,
                                              num_layers=num_deterministic_layers,
                                              non_linearity=nn.Tanh())
        self.encoders = nn.ModuleDict()

        for i in reversed(range(self.num_stochastic_layers - 1)):
            self.encoders[str(i)] = init_normal_mlp(in_dim=self.latent_dims[i + 1],
                                                    out_dim=self.latent_dims[i],
                                                    hidden_dim=self.hidden_dim,
                                                    num_layers=num_deterministic_layers,
                                                    non_linearity=nn.Tanh())

    def get_latent_layer_param(self, layer_idx, previous_latent_layer=None, obs=None):
        """Returns params of distribution for single latent layer.

        Args:
            layer_idx: 0 means the layer furthest away from obs
            previous_latent_layer: tensor of shape [batch_size, latent_dim]
                (only applicable if layer_idx < num_stochastic_layers - 1)
            obs: tensor of shape [batch_size, obs_dim] of values in {0, 1}
                (only applicable if layer_idx = num_stochastic_layers - 1)

        Returns:
            mu for Normal latent of shape [batch_size, latent_dim]
            sig for Normal latent of shape [batch_size, latent_dim]
            """
        if layer_idx == self.num_stochastic_layers - 1:
            return self.encoder_to_obs(obs)
        else:
            return self.encoders[str(layer_idx)](previous_latent_layer)

    def get_latent_layer_dist(self, layer_idx, previous_latent_layer=None, obs=None):
        """Returns distribution for single latent layer.

        Args:
            layer_idx: 0 means the layer furthest away from obs
            previous_latent_layer: tensor of shape [batch_size, latent_dim]
                (only applicable if layer_idx < num_stochastic_layers - 1)
            obs: tensor of shape [batch_size, obs_dim] of values in {0, 1}
                (only applicable if layer_idx = num_stochastic_layers - 1)

        Returns: Normal distribution with event_shape [latent_dim]
            and batch_shape is [batch_size]"""

        mu, sig = self.get_latent_layer_param(layer_idx, previous_latent_layer, obs=obs)

        return torch.distributions.Independent(torch.distributions.Normal(mu, sig), reinterpreted_batch_ndims=1)

    def sample_from_latent_dist(self, latent_dist, num_samples, reparam=True):
        """Samples from q(latent | obs)

        Args:
            latent_dist: distribution with batch shape [batch_size] and event
                shape [latent_dim]
            num_samples: int

        Returns:
            latent: tensor of shape [num_samples, batch_size, num_mixtures]
        """
        if reparam:
            return latent_dist.rsample((num_samples, ))
        else:
            return latent_dist.sample((num_samples, ))
