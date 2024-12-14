# GCRL
Implement offline Goal-conditioned reinforcement learning algorithms based on [OGBench](https://github.com/seohongpark/ogbench/tree/master)
### Software Environment
1. OS: Ubuntu 20.04
2. Python: 3.9.*
3. CUDA: 12.6 / CUDNN: 8.9.6
--- 
### Algorithms
1. [HIQL](https://arxiv.org/abs/2307.11949)
2. [CRL](https://arxiv.org/abs/2206.07568)
3. [QRL](https://arxiv.org/abs/2304.01203)
---
### Train
If you want to use wandb, set --wandb_offline to 0; otherwise, set it to 1.

Additionally, if you want to use multi-GPU, set --multi_gpu to 1; otherwise, set it to 0.


Please refer to [OGBench](https://github.com/seohongpark/ogbench/tree/master) for the available environments.
1. HIQL
    ```
    python main.py --agent=agent/hiql.py --env_name=antmaze-large-navigate-v0
    ```
2. CRL
    ```
    python main.py --agent=agent/crl.py --env_name=antmaze-large-navigate-v0
    ```
3. QRL
    ```
    python main.py --agent=agent/qrl.py --env_name=antmaze-large-navigate-v0
    ```

In Pixel-based environment
1. HIQL
    ```
    python main.py --env_name=visual-antmaze-teleport-navigate-v0 --agent=agent/hiql.py --agent.encoder=impala_small --agent.low_actor_rep_grad=True
    ```
2. CRL 
    ```
    python main.py --env_name=visual-antmaze-teleport-navigate-v0 --agent=agent.crl.py --agent.encoder=impala_small
    ```
3. QRL
    ```
    python main,py --env_name=visual-antmaze-teleport-navigate-v0 --agent=agent.qrl.py --agent.encoder=impala_small
    ```
