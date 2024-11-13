from typing import Any
from functools import partial

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from jaxrl_m.vision import encoders
from jaxrl_m.common import TrainState
from jaxrl_m.networks import *


class QRLAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    network: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    def value_loss(self, batch, network_params):
        d_neg = self.network.select("value")(
            batch["observations"],
            batch["value_goals"],
            params=network_params,
        )
        d_pos = self.network.select("value")(
            batch["observations"],
            batch["next_observations"],
            params=network_params,
        )
        lam = self.network.select("lam")(params=network_params)

        d_neg_loss = (100 * jax.nn.softplus(5 - d_neg / 100)).mean()
        d_pos_loss = (jax.nn.relu(d_pos - 1) ** 2).mean()

        value_loss = d_neg_loss + d_pos_loss * jax.lax.stop_gradient(lam)
        lam_loss = lam * (self.config["eps"] - jax.lax.stop_gradient(d_pos_loss))
        total_loss = value_loss + lam_loss
        value_info = {
            "value/total_loss": total_loss,
            "value/value_loss": value_loss,
            "value/lambda_loss": lam_loss,
            "value/d_neg_loss": d_neg_loss,
            "value/lam": lam,
        }
        return total_loss, value_info

    def dynamics_loss(self, batch, network_params):
        _, obs_rep, next_obs_rep = self.network.select("value")(
            batch["observations"],
            batch["next_observations"],
            info=True,
            params=network_params,
        )
        pred_next_obs_reps = obs_rep + self.network.select("dynamics")(
            jnp.concatenate([obs_rep, batch["actions"]], axis=-1),
            params=network_params,
        )
        distance1 = self.network.select("value")(
            next_obs_rep,
            pred_next_obs_reps,
            is_phi=True,
            params=network_params,
        )
        distance2 = self.network.select("value")(
            pred_next_obs_reps,
            next_obs_rep,
            is_phi=True,
            params=network_params,
        )
        dynamics_loss = (distance1 + distance2).mean() / 2
        dynamics_info = {
            "dynamics/dynamics_loss": dynamics_loss,
            "dynamics/distance1": distance1.mean(),
            "dynamics/distance2": distance2.mean(),
        }
        return dynamics_loss, dynamics_info

    def actor_loss(self, batch, network_params, rng=None):
        if self.config["actor_loss"] == "awr":
            v = -self.network.select("value")(
                batch["observations"],
                batch["actor_goals"],
            )
            nv = -self.network.select("value")(
                batch["next_observations"],
                batch["actor_goals"],
            )
            advantage = nv - v
            exp_a = jnp.minimum(jnp.exp(advantage * self.config["alpha"]))

            dist = self.network.select("actor")(
                batch["observations"],
                batch["actor_goals"],
                params=network_params,
            )
            log_prob = dist.log_prob(batch["actions"])
            actor_loss = -(exp_a * log_prob).mean()
            actor_info = {
                "actor/actor_loss": actor_loss,
                "actor/advantage_mean": advantage.mean(),
                "actor/log_prob_mean": log_prob.mean(),
            }
            if not self.config["discrete"]:
                actor_info.update(
                    {
                        "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
                        "std": jnp.mean(dist.scale_diag),
                    }
                )
            return actor_loss, actor_info
        elif self.config["actor_loss"] == "ddpgbc":
            assert not self.config["discrete"]
            dist = self.network.select("actor")(
                batch["observations"],
                batch["actor_goals"],
                params=network_params,
            )
            if self.config["const_std"]:
                q_actions = jnp.clip(dist.mode(), -1.0, 1.0)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1.0, 1.0)

            _, obs_reps, goal_reps = self.network.select("value")(
                batch["observations"],
                batch["actor_goals"],
                info=True,
            )
            pred_network_obs_reps = obs_reps + self.network.select("dynamics")(
                jnp.concatenate([obs_reps, q_actions], axis=-1)
            )
            q = -self.network.select("value")(
                pred_network_obs_reps,
                goal_reps,
                is_phi=True,
            )

            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(batch["actions"])
            bc_loss = -(self.config["alpha"] * log_prob).mean()
            actor_loss = q_loss + bc_loss
            actor_info = {
                "actor/actor_loss": actor_loss,
                "actor/q_loss": q_loss,
                "actor/bc_loss": bc_loss,
                "actor/q_mean": q.mean(),
                "actor/log_prob_mean": log_prob.mean(),
                "actor/mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
                "actor/std": jnp.mean(dist.scale_diag),
            }
            return actor_loss, actor_info
        else:
            raise ValueError

    @jax.jit
    def total_loss(self, batch, network_params, rng=None):
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(
            batch=batch,
            network_params=network_params,
        )

        if self.config["actor_loss"] == "ddpgbc":
            dynamics_loss, dynamics_info = self.dynamics_loss(
                batch=batch,
                network_params=network_params,
            )
        else:
            dynamics_loss, dynamics_info = 0.0, {}

        rng, actor_key = jax.random.split(rng, 2)
        actor_loss, actor_info = self.actor_loss(
            batch=batch,
            network_params=network_params,
            rng=actor_key,
        )
        loss = actor_loss + value_loss + dynamics_loss
        return loss, {**value_info, **actor_info, **dynamics_info}

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch, pmap_axis: str = None):
        new_rng, rng = jax.random.split(self.rng, 2)

        def loss_fn(network_params):
            return self.total_loss(batch, network_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(
            loss_fn=loss_fn,
            has_aux=True,
            pmap_axis=pmap_axis,
        )
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        dist = self.network.select("actor")(
            observations,
            goals,
            temperature=temperature,
        )
        actions = dist.sample(seed=seed)
        if not self.config["discrete"]:
            actions = jnp.clip(actions, -1.0, 1.0)
        return actions


def create_learner(
    seed: int,
    observations: np.ndarray,
    actions: np.ndarray,
    config: Dict,
):
    rng = jax.random.PRNGKey(seed=seed)
    rng, model_key = jax.random.split(rng, 2)

    latents = np.zeros((observations.shape[0], config["latent_dim"]), dtype=np.float32)
    if config["discrete"]:
        actions_dim = actions.max() + 1
    else:
        actions_dim = actions.shape[-1]

    encoders = dict()
    if config["encoder"] is not None:
        encoder_module = encoders[config["encoder"]]
        encoders["value"] = encoder_module()
        encoders["actor"] = GCEncoder(concat_encoder=encoder_module())

    if config["quasimetric_type"] == "mrn":
        value_def = GCMRNValue(
            hidden_dims=config["value_hidden_dims"],
            latent_dim=config["latent_dim"],
            use_layer_norm=config["value_layer_norm"],
            encoder=encoders.get("value"),
        )
    elif config["quasimetric_type"] == "iqe":
        value_def = GCIQEValue(
            hidden_dims=config["value_hidden_dims"],
            latent_dim=config["latent_dim"],
            dim_per_component=8,
            use_layer_norm=config["value_layer_norm"],
            encoder=encoders.get("value"),
        )
    else:
        raise ValueError

    if config["actor_loss"] == "ddpgbc":
        dynamics_module = LayerNormMLP if config["dynamics_layer_norm"] else MLP
        dynamics_def = dynamics_module(
            hidden_dims=(*config["value_hidden_dims"], config["latent_dim"]),
            activate_final=False,
        )

    if config["discrete"]:
        actor_def = GCDiscreteActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=actions_dim,
            use_layer_norm=config["actor_layer_norm"],
            gc_encoder=encoders.get("actor"),
        )
    else:
        actor_def = GCContinuousActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=actions_dim,
            use_layer_norm=config["actor_layer_norm"],
            gc_encoder=encoders.get("actor"),
            state_dependent_std=False,
            constant_std=config["const_std"],
        )

    lambda_def = LogParam()
    network_info = dict(
        value=(value_def, (observations, observations)),
        actor=(actor_def, (observations, observations)),
        lam=(lambda_def, ()),
    )
    if config["actor_loss"] == "ddpgbc":
        network_info.update(
            dynamics=(dynamics_def, np.concatenate([latents, actions], axis=-1)),
        )
    networks = {k: v[0] for k, v in network_info.items()}
    networks_args = {k: v[1] for k, v in network_info.items()}

    network_def = ModuleDict(networks)
    network_tx = optax.adam(learning_rate=config["lr"])
    network_params = network_def.init(model_key, **networks_args)["params"]
    network = TrainState.create(network_def, network_params, tx=network_tx)
    return QRLAgent(rng=rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            algo_name="qrl",
            # Agent hyperparameters.
            lr=3e-4,  # Learning rate.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            quasimetric_type="iqe",  # Quasimetric parameterization type ('iqe' or 'mrn').
            latent_dim=512,  # Latent dimension for the quasimetric value function.
            actor_layer_norm=False,
            value_layer_norm=True,
            dynamics_layer_norm=True,
            discount=0.99,  # Discount factor (unused by default; can be used for geometric goal sampling in GCDataset).
            eps=0.05,  # Margin for the dual lambda loss.
            actor_loss="ddpgbc",  # Actor loss type ('awr' or 'ddpgbc').
            alpha=0.0003,  # Temperature in AWR or BC coefficient in DDPG+BC.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(
                str
            ),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class="GCDataset",  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=1.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(
                int
            ),  # Number of frames to stack.
        )
    )
    return config
