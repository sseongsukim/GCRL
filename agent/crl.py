from functools import partial
from typing import Any

import flax.struct

from jaxrl_m.typing import *
from jaxrl_m.networks import (
    GCBilinearValue,
    GCDiscreteBilinearCritic,
    GCDiscreteActor,
    GCContinuousActor,
    ModuleDict,
    GCEncoder,
)
from jaxrl_m.vision import encoders as encoder_modules
from jaxrl_m.common import TrainState, nonpytree_field

import flax.struct
import copy
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax


class CRLAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    network: TrainState
    config: dict = nonpytree_field()

    def contrastive_loss(
        self,
        batch,
        network_params,
        module_name="critic",
    ):
        batch_size = batch["observations"].shape[0]
        if module_name == "critic":
            actions = batch["actions"]
        else:
            actions = None
        v, phi, psi = self.network.select(module_name)(
            batch["observations"],
            batch["value_goals"],
            actions=actions,
            info=True,
            params=network_params,
        )
        if len(phi.shape) == 2:
            phi = phi[None, ...]
            psi = psi[None, ...]
        logits = jnp.einsum("eik, ejk->ije", phi, psi) / jnp.sqrt(phi.shape[-1])
        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda x: optax.sigmoid_binary_cross_entropy(logits=x, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(logits).mean()
        logits = logits.mean(axis=-1)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        infos = {
            "contrastive_loss": contrastive_loss,
            "v_mean": v.mean(),
            "binary_accuracy": jnp.mean((logits > 0) == I),
            "categrical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logits_mean": logits.mean(),
        }
        contrastive_info = {}
        for k, v in infos.items():
            contrastive_info[f"contrastive_{module_name}/{k}"] = v
        return contrastive_loss, contrastive_info

    def actor_loss(self, batch, network_params, rng=None):
        if self.config["actor_log_q"]:

            def value_transform(x):
                return jnp.log(jnp.maximum(x, 1e-6))

        else:

            def value_transform(x):
                return x

        if self.config["actor_loss"] == "awr":
            v = value_transform(
                self.network.select("value")(
                    batch["observations"],
                    batch["actor_goals"],
                )
            )
            q1, q2 = value_transform(
                self.network.select("critic")(
                    batch["observations"],
                    batch["actor_goals"],
                    batch["actions"],
                )
            )
            q = jnp.minimum(q1, q2)
            advantage = q - v
            exp_a = jnp.minimum(jnp.exp(advantage * self.config["alpha"]), 100.0)

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
                "actor/log_prob": log_prob.mean(),
            }
            if not self.config["discrete"]:
                actor_info.update(
                    {
                        "actor/mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
                        "actor/std": jnp.mean(dist.scale_diag),
                    }
                )
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
            q1, q2 = value_transform(
                self.network.select("critic")(
                    batch["observations"],
                    batch["actor_goals"],
                    q_actions,
                )
            )
            q = jnp.minimum(q1, q2)

            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(batch["actions"])
            bc_loss = -(self.config["alpha"] * log_prob).mean()
            actor_loss = bc_loss + q_loss
            actor_info = {
                "actor/actor_loss": actor_loss,
                "actor/log_prob": log_prob.mean(),
                "actor/bc_loss": bc_loss,
                "actor/q_loss": q_loss,
                "actor/q_mean": q.mean(),
                "actor/mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
                "actor/std": jnp.mean(dist.scale_diag),
            }
            return actor_loss, actor_info
        else:
            raise ValueError

    @jax.jit
    def total_loss(self, batch, network_params, rng=None):
        rng = rng if rng is not None else self.rng
        critic_loss, critic_info = self.contrastive_loss(
            batch,
            network_params,
            "critic",
        )
        if self.config["actor_loss"] == "awr":
            value_loss, value_info = self.contrastive_loss(
                batch,
                network_params,
                "value",
            )
        else:
            value_loss, value_info = 0.0, {}

        rng, actor_key = jax.random.split(rng, 2)
        actor_loss, actor_info = self.actor_loss(
            batch,
            network_params,
            actor_key,
        )
        loss = critic_loss + actor_loss + value_loss
        return loss, {**actor_info, **critic_info, **value_info}

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng, 2)

        def loss_fn(network_params):
            return self.total_loss(batch, network_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(
            loss_fn=loss_fn,
            has_aux=True,
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
    seed,
    observations,
    actions,
    config,
):
    rng = jax.random.PRNGKey(seed)
    rng, model_key = jax.random.split(rng, 2)

    if config["discrete"]:
        action_dim = actions.max() + 1
    else:
        action_dim = actions.shape[-1]
    encoders = dict()
    if config["encoder"] is not None:
        encoder_module = encoder_modules[config["encoder"]]
        encoders["critic_state"] = encoder_module()
        encoders["critic_goal"] = encoder_module()
        encoders["actor"] = GCEncoder(concat_encoder=encoder_module())
        if config["actor_loss"] == "awr":
            encoders["value_state"] = encoder_module()
            encoders["value_goal"] = encoder_module()

    if config["discrete"]:
        critic_def = GCDiscreteBilinearCritic(
            hidden_dims=config["value_hidden_dims"],
            latent_dim=config["latent_dim"],
            use_layer_norm=config["value_layer_norm"],
            ensemble=True,
            value_exp=True,
            state_encoder=encoders.get("critic_state"),
            goal_encoder=encoders.get("critic_goal"),
            action_dim=action_dim,
        )
    else:
        critic_def = GCBilinearValue(
            hidden_dims=config["value_hidden_dims"],
            latent_dim=config["latent_dim"],
            use_layer_norm=config["value_layer_norm"],
            ensemble=True,
            value_exp=True,
            state_encoder=encoders.get("critic_state"),
            goal_encoder=encoders.get("critic_goal"),
        )
    if config["actor_loss"] == "awr":
        value_def = GCBilinearValue(
            hidden_dims=config["value_hidden_dims"],
            latent_dim=config["latent_dim"],
            use_layer_norm=config["value_layer_norm"],
            ensemble=True,
            value_exp=False,
            state_encoder=encoders.get("value_state"),
            goal_encoder=encoders.get("value_goal"),
        )
    if config["discrete"]:
        actor_def = GCDiscreteActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            use_layer_norm=config["actor_layer_norm"],
            gc_encoder=encoders.get("actor"),
        )
    else:
        actor_def = GCContinuousActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            use_layer_norm=config["actor_layer_norm"],
            constant_std=config["const_std"],
            gc_encoder=encoders.get("actor"),
        )
    network_info = dict(
        critic=(critic_def, (observations, observations, actions)),
        actor=(actor_def, (observations, observations)),
    )
    if config["actor_loss"] == "awr":
        network_info.update(value=(value_def, (observations, observations)))
    networks = {k: v[0] for k, v in network_info.items()}
    network_args = {k: v[1] for k, v in network_info.items()}

    network_def = ModuleDict(networks)
    network_tx = optax.adam(learning_rate=config["lr"])
    network_params = network_def.init(model_key, **network_args)["params"]
    network = TrainState.create(network_def, network_params, tx=network_tx)
    return CRLAgent(rng=rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            algo_name="crl",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            latent_dim=512,  # Latent dimension for phi and psi.
            value_layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,
            discount=0.99,  # Discount factor.
            actor_loss="ddpgbc",  # Actor loss type ('awr' or 'ddpgbc').
            alpha=0.03,  # Temperature in AWR or BC coefficient in DDPG+BC.
            actor_log_q=True,  # Whether to maximize log Q (True) or Q itself (False) in the actor loss.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(
                str
            ),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class="GCDataset",  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
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
