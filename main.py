from absl import app, flags
from ml_collections import config_flags
from functools import partial

from jaxrl_m.evaluation import supply_rng, evaluate
from jaxrl_m.wandb import setup_wandb, default_wandb_config, get_flag_dict
from jaxrl_m.dataset import GCDataset, HGCDataset
from agent import algos
from src import env_utils
import tensorflow as tf

import os
from datetime import datetime
from tqdm import tqdm
import wandb
import pickle
import numpy as np
import random
import flax
import jax
import jax.numpy as jnp
from flax.jax_utils import pad_shard_unpad, replicate, unreplicate

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "antmaze-large-navigate-v0", "Environment name.")
flags.DEFINE_string("save_dir", None, "Logging dir (if not None, save params).")
flags.DEFINE_string("run_group", "DEBUG", "")
flags.DEFINE_integer("num_episodes", 50, "")
flags.DEFINE_integer("num_videos", 2, "")
flags.DEFINE_integer("log_steps", 1000, "")
flags.DEFINE_integer("eval_steps", 100000, "")
flags.DEFINE_integer("save_steps", 250000, "")
flags.DEFINE_integer("total_steps", 1000000, "")
flags.DEFINE_integer("wandb_offline", 1, "")
flags.DEFINE_integer("multi_gpu", 1, "")
flags.DEFINE_integer("batch_size", 512, "")

seed = np.random.randint(low=0, high=10000000)
flags.DEFINE_integer("seed", seed, "")

config_flags.DEFINE_config_file("agent", "agent/hiql.py", lock_config=False)


def prepare_batch(batch, num_devices):

    sharded_batch = jax.tree.map(
        lambda x: x.reshape(num_devices, x.shape[0] // num_devices, *x.shape[1:]),
        batch,
    )
    return sharded_batch


def main(_):
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(prepare_batch, num_devices=len(devices))

    tf.config.set_visible_devices([], "GPU")

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    # Wandb & Log
    wandb_config = default_wandb_config()
    wandb_config.update(
        {
            "project": "GCRL",
            "group": f"{FLAGS.agent.algo_name}",
            "name": f"{FLAGS.agent.algo_name}_{FLAGS.env_name}_{FLAGS.seed}",
        }
    )
    config_flags.DEFINE_config_dict("wandb", wandb_config, lock_config=False)

    start_time = int(datetime.now().timestamp())
    FLAGS.wandb["name"] += f"_{start_time}"
    flag_dict = get_flag_dict()
    if FLAGS.save_dir is not None:
        FLAGS.save_dir = os.path.join(
            FLAGS.save_dir,
            wandb.run.project,
            wandb.config.exp_prefix,
            wandb.config.experiment_id,
        )
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        print(f"Saving config to {FLAGS.save_dir}/config.pkl")
        with open(os.path.join(FLAGS.save_dir, "config.pkl"), "wb") as f:
            pickle.dump(flag_dict, f)

    setup_wandb({**flag_dict}, offline=FLAGS.wandb_offline, **FLAGS.wandb)
    # Env & Dataset
    config = FLAGS.agent
    env, train_dataset, val_dataset = env_utils.make_env_and_datasets(
        dataset_name=FLAGS.env_name
    )
    dataset_class = {
        "GCDataset": GCDataset,
        "HGCDataset": HGCDataset,
    }[config["dataset_class"]]
    train_dataset = dataset_class(train_dataset, config)
    if val_dataset is not None:
        val_dataset = dataset_class(val_dataset, config)
    # Agent
    learner = algos[FLAGS.agent.algo_name]
    example_batch = train_dataset.sample(FLAGS.batch_size)
    if config["discrete"]:
        example_batch["actions"] = np.full_like(
            example_batch["actions"], env.action_space.n - 1
        )
    agent = learner(
        seed=FLAGS.seed,
        observations=example_batch["observations"],
        actions=example_batch["actions"],
        config=config,
        multi_gpu=FLAGS.multi_gpu,
    )
    if FLAGS.multi_gpu:
        FLAGS.batch_size *= len(devices)
        agent = jax.device_put(jax.tree.map(jnp.array, agent), sharding.replicate())

    for step in tqdm(range(1, FLAGS.total_steps + 1), smoothing=0.1, desc="train"):
        batch = train_dataset.sample(FLAGS.batch_size)

        if FLAGS.multi_gpu:
            batch = shard_fn(batch)
        agent, update_info = agent.update(batch)

        if step % FLAGS.log_steps == 0:
            wandb.log(update_info)


if __name__ == "__main__":
    app.run(main)
