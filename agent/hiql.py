from functools import partial
from typing import Any

from jaxrl_m.typing import *
from jaxrl_m.networks import *
from jaxrl_m.vision import encoders
from jaxrl_m.common import TrainState

import flax.struct
import copy
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax


def expectile_loss(adv, diff, expectile):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)


class HIQLAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    network: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    def compute_value_loss(self, batch, network_params):
        next_tv1, next_tv2 = self.network.select("target_value")(
            batch["next_observations"],
            batch["value_goals"],
        )
        next_tv = jnp.minimum(next_tv1, next_tv2)
        q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_tv

        tv1, tv2 = self.network.select("target_value")(
            batch["observations"],
            batch["value_goals"],
        )
        tv = (tv1 + tv2) / 2

        advantage = q - tv

        q1 = batch["rewards"] + self.config["discount"] * batch["masks"] * next_tv1
        q2 = batch["rewards"] + self.config["discount"] * batch["masks"] * next_tv2
        v1, v2 = self.network.select("value")(
            batch["observations"],
            batch["value_goals"],
            params=network_params,
        )
        v = (v1 + v2) / 2

        value_loss1 = expectile_loss(
            advantage, q1 - v1, self.config["expectile"]
        ).mean()
        value_loss2 = expectile_loss(
            advantage, q2 - v2, self.config["expectile"]
        ).mean()
        value_loss = value_loss1 + value_loss2

        value_info = {
            "value/value_loss": value_loss,
            "value/v_mean": v.mean(),
            "value/advantage_mean": advantage.mean(),
            "value/q": q.mean(),
            "value/target_v_mean": tv.mean(),
            "value/next_target_v_mean": ((next_tv1 + next_tv2) / 2).mean(),
        }

        return value_loss, value_info

    def compute_actor_loss(self, batch, network_params):
        v1, v2 = self.network.select("value")(
            batch["observations"],
            batch["low_actor_goals"],
        )
        v = (v1 + v2) / 2
        nv1, nv2 = self.network.select("value")(
            batch["next_observations"],
            batch["low_actor_goals"],
        )
        nv = (nv1 + nv2) / 2

        advantage = nv - v
        exp_a = jnp.minimum(jnp.exp(advantage * self.config["low_alpha"]), 100.0)

        goal_reps = self.network.select("goal_rep")(
            jnp.concatenate(
                [batch["observations"], batch["low_actor_goals"]],
                axis=-1,
            ),
            params=network_params,
        )
        if not self.config["low_actor_rep_grad"]:
            goal_reps = jax.lax.stop_gradient(goal_reps)

        dist = self.network.select("low_actor")(
            batch["observations"],
            goal_reps,
            goal_encoded=True,
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
                    "actor/mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
                    "actor/std": jnp.mean(dist.scale_diag),
                }
            )
        return actor_loss, actor_info

    def compute_high_loss(self, batch, network_params):
        v1, v2 = self.network.select("value")(
            batch["observations"],
            batch["high_actor_goals"],
        )
        v = (v1 + v2) / 2
        nv1, nv2 = self.network.select("value")(
            batch["high_actor_targets"],
            batch["high_actor_goals"],
        )
        nv = (nv1 + nv2) / 2

        advantage = nv - v
        exp_a = jnp.minimum(jnp.exp(advantage * self.config["high_alpha"]), 100.0)

        dist = self.network.select("high_actor")(
            batch["observations"],
            batch["high_actor_goals"],
            params=network_params,
        )
        target = self.network.select("goal_rep")(
            jnp.concatenate(
                [batch["observations"], batch["high_actor_targets"]],
                axis=-1,
            )
        )
        log_prob = dist.log_prob(target)
        high_loss = -(exp_a * log_prob).mean()

        high_info = {
            "high_actor/high_loss": high_loss,
            "high_actor/advantage_mean": advantage.mean(),
            "high_actor/log_prob_mean": log_prob.mean(),
            "high_actor/mse_loss": jnp.mean((dist.mode() - target) ** 2),
            "high_actor/std": jnp.mean(dist.scale_diag),
        }
        return high_loss, high_info

    @jax.jit
    def total_loss(self, batch, network_params):
        value_loss, value_info = self.compute_value_loss(batch, network_params)
        actor_loss, actor_info = self.compute_actor_loss(batch, network_params)
        high_loss, high_info = self.compute_high_loss(batch, network_params)
        loss = value_loss + actor_loss + high_loss
        return loss, {**value_info, **actor_info, **high_info}

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng, 2)

        def loss_fn(network_params):
            return self.total_loss(batch, network_params)

        new_network, update_info = self.network.apply_loss_fn(
            loss_fn=loss_fn,
            has_aux=True,
        )

        new_target_params = jax.tree.map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params["modules_value"],
            self.network.params["modules_target_value"],
        )
        new_network.params["modules_target_value"] = new_target_params

        return self.replace(network=new_network, rng=new_rng), update_info

    @jax.jit
    def sample_actions(
        self,
        observations: np.ndarray,
        goals: np.ndarray = None,
        seed: PRNGKey = None,
        temperature: float = 1.0,
    ):
        rng, high_key, low_key = jax.random.split(seed, 3)
        high_dist = self.network.select("high_actor")(
            observations,
            goals,
            temperature=temperature,
        )
        goal_reps = high_dist.sample(seed=high_key)
        goal_reps = (
            goal_reps
            / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True)
            * jnp.sqrt(goal_reps.shape[-1])
        )

        low_dist = self.network.select("low_actor")(
            observations, goal_reps, goal_encoded=True, temperature=temperature
        )
        actions = low_dist.sample(seed=low_key)
        if not self.config["discrete"]:
            actions = jnp.clip(actions, -1.0, 1.0)
        return actions


def create_learner(
    seed: int,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    config: Dict,
):

    rng = jax.random.PRNGKey(seed)
    rngs, model_key = jax.random.split(rng, 2)

    action_dim = actions.max() + 1 if config["discrete"] else actions.shape[-1]

    goal_rep_seq = []
    if config["encoder"] is not None:
        encoder_module = encoders[config["encoder"]]
        goal_rep_seq = [encoder_module()]

    module = LayerNormMLP if config["value_layer_norm"] else MLP

    goal_rep_seq.append(
        module(
            hidden_dims=(*config["value_hidden_dims"], config["rep_dim"]),
            activate_final=False,
        )
    )

    goal_rep_seq.append(LengthNormalize())
    goal_rep_def = nn.Sequential(goal_rep_seq)

    if config["encoder"] is None:
        value_encoder_def = GCEncoder(
            state_encoder=Identity(),
            concat_encoder=goal_rep_def,
        )
        target_value_encoder_def = GCEncoder(
            state_encoder=Identity(),
            concat_encoder=goal_rep_def,
        )
        low_actor_encoder_def = GCEncoder(
            state_encoder=Identity(),
            concat_encoder=goal_rep_def,
        )
        high_actor_encoder_def = None
    else:
        value_encoder_def = GCEncoder(
            state_encoder=encoder_module(),
            concat_encoder=goal_rep_def,
        )
        target_value_encoder_def = GCEncoder(
            state_encoder=encoder_module(),
            concat_encoder=goal_rep_def,
        )
        low_actor_encoder_def = GCEncoder(
            state_encoder=encoder_module(),
            concat_encoder=goal_rep_def,
        )
        high_actor_encoder_def = GCEncoder(
            concat_encoder=encoder_module(),
        )

    # Value network
    value_def = GCEnsembleValue(
        hidden_dims=(),
        use_layer_norm=config["value_layer_norm"],
        gc_encoder=value_encoder_def,
    )
    target_def = GCEnsembleValue(
        hidden_dims=(),
        use_layer_norm=config["value_layer_norm"],
        gc_encoder=target_value_encoder_def,
    )

    # Low actor network
    if config["discrete"]:
        low_actor_def = GCDiscreteActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            use_layer_norm=config["actor_layer_norm"],
            gc_encoder=low_actor_encoder_def,
        )
    else:
        low_actor_def = GCContinuousActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            use_layer_norm=config["actor_layer_norm"],
            gc_encoder=low_actor_encoder_def,
            constant_std=config["const_std"],
        )

    # High actor network
    high_actor_def = GCContinuousActor(
        hidden_dims=config["actor_hidden_dims"],
        action_dim=config["rep_dim"],
        constant_std=config["const_std"],
        use_layer_norm=config["actor_layer_norm"],
        gc_encoder=high_actor_encoder_def,
    )

    network_info = dict(
        goal_rep=(
            goal_rep_def,
            (jnp.concatenate([observations, observations], axis=-1)),
        ),
        value=(
            value_def,
            (observations, observations),
        ),
        target_value=(
            target_def,
            (observations, observations),
        ),
        low_actor=(
            low_actor_def,
            (observations, observations),
        ),
        high_actor=(
            high_actor_def,
            (observations, observations),
        ),
    )
    networks = {k: v[0] for k, v in network_info.items()}
    network_args = {k: v[1] for k, v in network_info.items()}

    network_def = ModuleDict(networks)
    network_tx = optax.adam(learning_rate=config["lr"])
    network_params = network_def.init(model_key, **network_args)["params"]

    network = TrainState.create(network_def, network_params, tx=network_tx)

    params = network.params
    params["modules_target_value"] = params["modules_value"]

    return HIQLAgent(rng=rngs, network=network, config=flax.core.FrozenDict(**config))


# def get_config():
#     config = ml_collections.ConfigDict(
#         dict(
#             # Agent hyperparameters.
#             algo_name="hiql",  # Agent name.
#             lr=3e-4,  # Learning rate.
#             actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
#             value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
#             value_layer_norm=True,  # Whether to use layer normalization.
#             actor_layer_norm=False,
#             discount=0.99,  # Discount factor.
#             tau=0.005,  # Target network update rate.
#             expectile=0.7,  # IQL expectile.
#             low_alpha=3.0,  # Low-level AWR temperature.
#             high_alpha=3.0,  # High-level AWR temperature.
#             subgoal_steps=25,  # Subgoal steps.
#             rep_dim=10,  # Goal representation dimension.
#             low_actor_rep_grad=False,  # Whether low-actor gradients flow to goal representation (use True for pixels).
#             const_std=True,  # Whether to use constant standard deviation for the actors.
#             discrete=False,  # Whether the action space is discrete.
#             encoder=None,  # Visual encoder name (None, 'impala_small', etc.).
#             # Dataset hyperparameters.
#             dataset_class="HGCDataset",  # Dataset class name.
#             value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
#             value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
#             value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
#             value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
#             actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
#             actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
#             actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
#             actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
#             gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
#             p_aug=0.0,  # Probability of applying image augmentation.
#             frame_stack=ml_collections.config_dict.placeholder(
#                 int
#             ),  # Number of frames to stack.
#         )
#     )
#     return config


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            algo_name="hiql",  # Agent name.
            lr=3e-4,  # Learning rate.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            value_layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.7,  # IQL expectile.
            low_alpha=3.0,  # Low-level AWR temperature.
            high_alpha=3.0,  # High-level AWR temperature.
            subgoal_steps=10,  # Subgoal steps.
            rep_dim=10,  # Goal representation dimension.
            low_actor_rep_grad=True,  # Whether low-actor gradients flow to goal representation (use True for pixels).
            const_std=True,  # Whether to use constant standard deviation for the actors.
            discrete=False,  # Whether the action space is discrete.
            encoder="impala_small",  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class="HGCDataset",  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.5,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(
                int
            ),  # Number of frames to stack.
        )
    )
    return config
