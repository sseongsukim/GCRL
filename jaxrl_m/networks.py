"""Common networks used in RL.

This file contains nn.Module definitions for common networks used in RL. It is divided into three sets:

1) Common Networks: MLP
2) Common RL Networks:
    For discrete action spaces: DiscreteCritic is a Q-function
    For continuous action spaces: Critic, ValueCritic, and Policy provide the Q-function, value function, and policy respectively.
    For ensembling: ensemblize() provides a wrapper for creating ensembles of networks (e.g. for min-Q / double-Q)
3) Meta Networks for vision tasks:
    WithEncoder: Combines a fully connected network with an encoder network (encoder may come from jaxrl_m.vision)
    ActorCritic: Same as WithEncoder, but for possibly many different networks (e.g. actor, critic, value)
"""

"""
Adapted from 
1) https://github.com/dibyaghosh/jaxrl_m/blob/main/jaxrl_m/networks.py
2) https://github.com/seohongpark/ogbench/blob/master/impls/utils/networks.py
3) https://github.com/seohongpark/ogbench/blob/master/impls/utils/flax_utils.py
"""

from jaxrl_m.typing import *

import flax.linen as nn
import jax.numpy as jnp

import distrax
import jax
import flax.linen as nn
import jax.numpy as jnp


class ModuleDict(nn.Module):
    """A dictionary of modules.

    This allows sharing parameters between modules and provides a convenient way to access them.

    Attributes:
        modules: Dictionary of modules.
    """

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        """Forward pass.

        For initialization, call with `name=None` and provide the arguments for each module in `kwargs`.
        Otherwise, call with `name=<module_name>` and provide the arguments for that module.
        """
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f"When `name` is not specified, kwargs must contain the arguments for each module. "
                    f"Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}"
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


###############################
#
#  Common Networks
#
###############################
class Identity(nn.Module):

    @nn.compact
    def __call__(self, x):
        return x


class LengthNormalize(nn.Module):

    @nn.compact
    def __call__(self, x):
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True) * jnp.sqrt(x.shape[-1])


def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    activate_final: int = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    def setup(self):
        self.layers = [
            nn.Dense(size, kernel_init=self.kernel_init) for size in self.hidden_dims
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers) or self.activate_final:
                x = self.activations(x)
        return x


class LayerNormMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    activate_final: int = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                x = nn.LayerNorm()(x)
        return x


###############################
#
#
#  Common RL Networks
#
###############################


class DiscreteCritic(nn.Module):
    hidden_dims: Sequence[int]
    n_actions: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        return MLP((*self.hidden_dims, self.n_actions), activations=self.activations)(
            observations
        )


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1), activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """
    Useful for making ensembles of Q functions (e.g. double Q in SAC).

    Usage:

        critic_def = ensemblize(Critic, 2)(hidden_dims=hidden_dims)

    """
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)


class Critic(nn.Module):
    hidden_dims: tuple = (256, 256)
    use_layer_norm: bool = False
    activate_final: bool = False

    @nn.compact
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], -1)
        if self.use_layer_norm:
            module = LayerNormMLP
        else:
            module = MLP
        q = module(
            (*self.hidden_dims, 1),
            activate_final=self.activate_final,
            activations=nn.gelu,
        )(x).squeeze(-1)
        return q


class EnsembleCritic(nn.Module):
    hidden_dims: tuple = (256, 256)
    use_layer_norm: bool = False
    activate_final: bool = False
    ensemble_size: int = 2

    @nn.compact
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], -1)
        if self.use_layer_norm:
            module = LayerNormMLP
        else:
            module = MLP
        module = ensemblize(module, self.ensemble_size)
        q1, q2 = module(
            (*self.hidden_dims, 1),
            activate_final=self.activate_final,
            activations=nn.relu,
        )(x).squeeze(-1)
        return q1, q2


class ImplicitPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 1e-2

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
    ):
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
        )(observations)
        means = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)
        actions = jnp.tanh(means)
        return actions


class Policy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2.0
    tanh_squash_distribution: bool = False
    state_dependent_std: bool = True
    final_fc_init_scale: float = 1e-2

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
        plan: bool = False,
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
        )(observations)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)
        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )(outputs)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        if plan:
            return means, jnp.exp(log_stds)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        if self.tanh_squash_distribution:
            distribution = TransformedWithMode(
                distribution, distrax.Block(distrax.Tanh(), ndims=1)
            )
        return distribution


class DiscretePolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 1e-2

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
        )(observations)

        logits = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)

        distribution = distrax.Categorical(
            logits=logits / jnp.maximum(1e-6, temperature)
        )

        return distribution


class TransformedWithMode(distrax.Transformed):
    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())


def softplus(x):
    return jnp.logaddexp(x, 0)


def soft_clamp(x, _min, _max):
    x = _max - softplus(_max - x)
    x = _min + softplus(x - _min)
    return x


class EnsembleLinear(nn.Module):
    input_dim: int
    output_dim: int
    num_ensemble: int
    weight_decay: float

    def setup(self):
        self.weight = self.param(
            "kernel",
            nn.initializers.glorot_normal(),
            (self.num_ensemble, self.input_dim, self.output_dim),
        )
        self.bias = self.param(
            "bias",
            nn.initializers.glorot_normal(),
            (self.num_ensemble, 1, self.output_dim),
        )

    def __call__(self, x: jnp.ndarray):
        x = jnp.einsum("nbi,nij->nbj", x, self.weight)
        x = x + self.bias
        return x

    def get_decay_loss(self):
        decay_loss = self.weight_decay * (0.5 * ((self.weight**2).sum()))
        return decay_loss


class EnsembleDyanmics(nn.Module):
    obs_dim: int
    action_dim: int
    hidden_dims: Sequence[int]
    weight_decays: Sequence[int]
    num_ensemble: int
    pred_reward: bool

    def setup(self):
        hidden_dims = [self.obs_dim + self.action_dim] + list(self.hidden_dims)
        self.layers = [
            EnsembleLinear(
                input_dim=input_dim,
                output_dim=output_dim,
                num_ensemble=self.num_ensemble,
                weight_decay=weight_decay,
            )
            for input_dim, output_dim, weight_decay in zip(
                hidden_dims[:-1], hidden_dims[1:], self.weight_decays
            )
        ]
        output_dim = self.obs_dim + 1 if self.pred_reward else self.obs_dim
        self.final_layers = EnsembleLinear(
            input_dim=hidden_dims[-1],
            output_dim=output_dim * 2,
            num_ensemble=self.num_ensemble,
            weight_decay=self.weight_decays[-1],
        )
        self.min_logvar = self.param(
            "min_logvar", nn.initializers.constant(-10.0), (output_dim,)
        )
        self.max_logvar = self.param(
            "max_logvar", nn.initializers.constant(0.5), (output_dim,)
        )
        self.output_dim = output_dim

    def __call__(self, obs_action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = obs_action
        for layer in self.layers:
            x = layer(x)
            x = nn.swish(x)
        x = self.final_layers(x)
        mean, logvar = x[:, :, : self.output_dim], x[:, :, self.output_dim :]
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar

    def get_max_logvar_sum(self):
        return self.max_logvar.sum()

    def get_min_logvar_sum(self):
        return self.min_logvar.sum()

    def get_total_decay_loss(self):
        decay_loss = 0
        for layer in self.layers:
            decay_loss += layer.get_decay_loss()
        decay_loss += self.final_layers.get_decay_loss()
        return decay_loss


###############################
#
#
#   Meta Networks for Encoders
#
###############################


def get_latent(
    encoder: nn.Module, observations: Union[jnp.ndarray, Dict[str, jnp.ndarray]]
):
    """

    Get latent representation from encoder. If observations is a dict
        a state and image component, then concatenate the latents.

    """
    if encoder is None:
        return observations

    elif isinstance(observations, dict):
        return jnp.concatenate(
            [encoder(observations["image"]), observations["state"]], axis=-1
        )

    else:
        return encoder(observations)


class WithEncoder(nn.Module):
    encoder: nn.Module
    network: nn.Module

    def __call__(self, observations, *args, **kwargs):
        latents = get_latent(self.encoder, observations)
        return self.network(latents, *args, **kwargs)


class ActorCritic(nn.Module):
    """Combines FC networks with encoders for actor, critic, and value.

    Note: You can share encoder parameters between actor and critic by passing in the same encoder definition for both.

    Example:

        encoder_def = ImpalaEncoder()
        actor_def = Policy(...)
        critic_def = Critic(...)
        # This will share the encoder between actor and critic
        model_def = ActorCritic(
            encoders={'actor': encoder_def, 'critic': encoder_def},
            networks={'actor': actor_def, 'critic': critic_def}
        )
        # This will have separate encoders for actor and critic
        model_def = ActorCritic(
            encoders={'actor': encoder_def, 'critic': copy.deepcopy(encoder_def)},
            networks={'actor': actor_def, 'critic': critic_def}
        )
    """

    encoders: Dict[str, nn.Module]
    networks: Dict[str, nn.Module]

    def actor(self, observations, **kwargs):
        latents = get_latent(self.encoders["actor"], observations)
        return self.networks["actor"](latents, **kwargs)

    def critic(self, observations, actions, **kwargs):
        latents = get_latent(self.encoders["critic"], observations)
        return self.networks["critic"](latents, actions, **kwargs)

    def value(self, observations, **kwargs):
        latents = get_latent(self.encoders["value"], observations)
        return self.networks["value"](latents, **kwargs)

    def __call__(self, observations, actions):
        rets = {}
        if "actor" in self.networks:
            rets["actor"] = self.actor(observations)
        if "critic" in self.networks:
            rets["critic"] = self.critic(observations, actions)
        if "value" in self.networks:
            rets["value"] = self.value(observations)
        return rets


###############################
#
#
#   Goal-conditioned Networks
#
###############################


class GCEncoder(nn.Module):
    state_encoder: nn.Module = None
    concat_encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations, goals=None, goal_encoded=False):
        reps = []
        if self.state_encoder is not None:
            reps.append(self.state_encoder(observations))

        if goals is not None:
            if goal_encoded:
                reps.append(goals)
            else:
                if self.concat_encoder is not None:
                    x = jnp.concatenate([observations, goals], axis=-1)
                    reps.append(self.concat_encoder(x))
        reps = jnp.concatenate(reps, axis=-1)
        return reps


class GCEnsembleValue(nn.Module):
    hidden_dims: Sequence[int]
    use_layer_norm: bool
    gc_encoder: nn.Module = None

    def setup(self):
        module = LayerNormMLP if self.use_layer_norm else MLP
        module = ensemblize(module, 2)
        self.value_layer = module(
            hidden_dims=(*self.hidden_dims, 1),
            activate_final=False,
        )

    def __call__(self, observations, goals=None, actions=None):
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v1, v2 = self.value_layer(inputs).squeeze(-1)
        return v1, v2


class GCDiscreteActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    use_layer_norm: bool
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        module = LayerNormMLP if self.use_layer_norm else MLP
        self.actor_layer = module(
            hidden_dims=self.hidden_dims,
            activate_final=True,
        )
        self.logit_layer = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )

    def __call__(self, observations, goals=None, goal_encoded=False, temperature=1.0):
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(
                observations,
                goals,
                goal_encoded,
            )
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, -1)
        outputs = self.actor_layer(inputs)
        logits = self.logit_layer(outputs) / jnp.maximum(1e-6, temperature)
        dist = distrax.Categorical(
            logits=logits,
        )
        return dist


class GCContinuousActor(nn.Module):
    hidden_dims: Sequence[int]
    use_layer_norm: bool
    action_dim: int
    log_std_min: Optional[float] = -5.0
    log_std_max: Optional[float] = 2.0
    tanh_squash_distribution: bool = False
    state_dependent_std: bool = False
    constant_std: bool = True
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        module = LayerNormMLP if self.use_layer_norm else MLP
        self.actor_layer = module(
            hidden_dims=self.hidden_dims,
            activate_final=True,
        )
        self.mean_layer = nn.Dense(
            self.action_dim,
            kernel_init=default_init(self.final_fc_init_scale),
        )
        if self.state_dependent_std:
            self.log_std_layer = nn.Dense(
                self.action_dim,
                kernel_init=default_init(self.final_fc_init_scale),
            )
        else:
            if not self.constant_std:
                self.log_stds = self.param(
                    "log_stds",
                    nn.initializers.zeros,
                    (self.action_dim,),
                )

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(
                observations,
                goals,
                goal_encoded,
            )
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_layer(inputs)

        means = self.mean_layer(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_layer(outputs)
        else:
            if self.constant_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        dist = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        if self.tanh_squash_distribution:
            dist = TransformedWithMode(
                distribution=dist, bijector=distrax.Block(distrax.Tanh(), ndims=-1)
            )
        return dist


###############################
#
#
#   QUasimetic Networks
#
###############################


class GCMRNValue(nn.Module):
    hidden_dims: Sequence[int]
    latent_dim: int
    use_layer_norm: bool
    encoder: nn.Module = None

    def setup(self):
        module = LayerNormMLP if self.use_layer_norm else MLP
        self.phi = module(
            hidden_dims=(*self.hidden_dims, self.latent_dim),
            activate_final=False,
        )

    def __call__(self, observations, goals, is_phi=False, info=False):
        if is_phi:
            phi_s, phi_g = observations, goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        sym_s = phi_s[..., : self.latent_dim // 2]
        sym_g = phi_g[..., : self.latent_dim // 2]

        asym_s = phi_s[..., self.latent_dim // 2 :]
        asym_g = phi_g[..., self.latent_dim // 2 :]

        squared_dist = ((sym_s - sym_g) ** 2).sum(axis=-1)
        quasi = nn.relu((asym_s - asym_g).max(axis=-1))

        v = jnp.sqrt(jnp.maximum(squared_dist, 1e-12)) + quasi

        if info:
            return v, phi_s, phi_g
        else:
            return v


class Param(nn.Module):
    init_value: float = 0.0

    @nn.compact
    def __call__(self):
        return self.param("value", init_fn=lambda key: jnp.full((), self.init_value))


class LogParam(nn.Module):
    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param(
            "log_value", init_fn=lambda key: jnp.full((), jnp.log(self.init_value))
        )
        return jnp.exp(log_value)


class GCIQEValue(nn.Module):
    hidden_dims: Sequence[int]
    latent_dim: int
    dim_per_component: int
    use_layer_norm: bool
    encoder: nn.Module = None

    def setup(self):
        module = LayerNormMLP if self.use_layer_norm else MLP
        self.phi = module(
            hidden_dims=(*self.hidden_dims, self.latent_dim),
            activate_final=False,
        )
        self.alpha = Param()

    def __call__(self, observations, goals, is_phi=False, info=False):
        alpha = jax.nn.sigmoid(self.alpha())
        if is_phi:
            phi_s, phi_g = observations, goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)
        x = jnp.reshape(phi_s, (*phi_s.shape[:-1], -1, self.dim_per_component))
        y = jnp.reshape(phi_g, (*phi_g.shape[:-1], -1, self.dim_per_component))
        valid = x < y
        xy = jnp.concatenate(jnp.broadcast_arrays(x, y), axis=-1)
        ixy = xy.argsort(axis=-1)
        sxy = jnp.take_along_axis(xy, ixy, axis=-1)
        neg_inc_copies = jnp.take_along_axis(
            valid,
            ixy % self.dim_per_component,
            axis=-1,
        ) * jnp.where(ixy < self.dim_per_component, -1, 1)
        neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
        neg_f = -1.0 * (neg_inp_copies < 0)
        neg_incf = jnp.concatenate(
            [neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]],
            axis=-1,
        )
        components = (sxy * neg_incf).sum(axis=-1)
        v = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(axis=-1)
        if info:
            return v, phi_s, phi_g
        else:
            return v
