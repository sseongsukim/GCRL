from absl import app, flags
from ml_collections import config_flags
from functools import partial
from collections import defaultdict

from jaxrl_m.evaluation import supply_rng, multi_tasks_evaluate
from jaxrl_m.wandb import (
    setup_wandb,
    default_wandb_config,
    get_flag_dict,
    get_wandb_video,
)
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
flags.DEFINE_integer("wandb_offline", 0, "")
flags.DEFINE_integer("multi_gpu", 0, "")
flags.DEFINE_integer("batch_size", 512, "")

flags.DEFINE_integer("eval_tasks", None, "")
flags.DEFINE_float("eval_temperature", 0.0, "")
flags.DEFINE_float("eval_gaussian", None, "")
flags.DEFINE_integer("video_frame_skip", 3, "")

seed = np.random.randint(low=0, high=10000000)
flags.DEFINE_integer("seed", seed, "")
config_flags.DEFINE_config_file("agent", "agent/qrl.py", lock_config=False)


def prepare_batch(batch, num_devices, multi_gpu):
    if multi_gpu:
        sharded_batch = jax.tree.map(
            lambda x: x.reshape(num_devices, x.shape[0] // num_devices, *x.shape[1:]),
            batch,
        )
        return sharded_batch
    else:
        return batch


def main(_):
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    prepare_fn = partial(
        prepare_batch, num_devices=len(devices), multi_gpu=FLAGS.multi_gpu
    )

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
    )

    if FLAGS.multi_gpu:
        FLAGS.batch_size *= len(devices)
        agent = jax.device_put(jax.tree.map(jnp.array, agent), sharding.replicate())

    for step in tqdm(range(FLAGS.total_steps + 1), smoothing=0.1, desc="train"):
        batch = prepare_fn(train_dataset.sample(FLAGS.batch_size))
        agent, update_info = agent.update(batch)

        if step % FLAGS.eval_steps == 0:
            policy_fn = supply_rng(
                agent.sample_actions,
                rng=jax.random.PRNGKey(np.random.randint(0, 2**32)),
            )

            task_infos = (
                env.unwrapped.task_infos
                if hasattr(env.unwrapped, "task_infos")
                else env.task_infos
            )
            num_tasks = (
                FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
            )
            renders = []
            eval_metrics = {}
            all_metrics = defaultdict(list)
            for task_id in tqdm(range(1, num_tasks + 1), desc="subtask", smoothing=0.1):
                task_name = task_infos[task_id - 1]["task_name"]
                eval_info, trajectories, videos = multi_tasks_evaluate(
                    policy_fn=policy_fn,
                    env=env,
                    task_id=task_id,
                    num_episodes=FLAGS.num_episodes,
                    num_videos=FLAGS.num_videos,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                    discrete=config["discrete"],
                )
                renders.extend(videos)
                metrics_name = ["success"]
                eval_metrics.update(
                    {
                        f"eval/{task_name}_{k}": v
                        for k, v in eval_info.items()
                        if k in metrics_name
                    }
                )
                for k, v in eval_info.items():
                    if k in metrics_name:
                        all_metrics[k].append(v)
            for k, v in all_metrics.items():
                eval_metrics[f"eval/overall_{k}"] = np.mean(v)
            if FLAGS.num_videos > 0:
                video = get_wandb_video(
                    renders=renders,
                    n_cols=num_tasks,
                )
                eval_metrics["video"] = video

            update_info.update(eval_metrics)

        if step % FLAGS.log_steps == 0:
            if val_dataset is not None:
                val_batch = prepare_fn(val_dataset.sample(FLAGS.batch_size))
                _, val_info = agent.total_loss(val_batch, network_params=None)

            for k, v in val_info.items():
                update_info[f"val/{k}"] = v

            wandb.log(update_info)


if __name__ == "__main__":
    app.run(main)
