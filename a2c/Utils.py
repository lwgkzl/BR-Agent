import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Union, Callable, Optional

from a2c.Collect import MyCollector
from tianshou.policy import BasePolicy


def test_episode(
    policy: BasePolicy,
    collector: MyCollector,
    test_fn: Optional[Callable[[int, Optional[int]], None]],
    epoch: int,
    n_episode: Union[int, List[int]],
    writer: Optional[SummaryWriter] = None,
    global_step: Optional[int] = None,
    name: str = "test/"
) -> Dict[str, float]:
    """A simple wrapper of testing policy in collector."""
    collector.reset_env()
    collector.reset_buffer()
    policy.eval()
    if test_fn:
        test_fn(epoch, global_step)
    if collector.get_env_num() > 1 and isinstance(n_episode, int):
        n = collector.get_env_num()
        n_ = np.zeros(n) + n_episode // n
        n_[:n_episode % n] += 1
        n_episode = list(n_)
    result = collector.collect(n_episode=n_episode)
    if writer is not None and global_step is not None:
        for k in result.keys():
            if 'class' not in k:
                writer.add_scalar(name + k, result[k], global_step=global_step)
    return result



def gather_info(
    start_time: float,
    train_c: MyCollector,
    test_c: MyCollector,
    best_reward: float,
    best_reward_std: float,
    best_rate: float,
    best_mate_num: float,
    best_len: float,
) -> Dict[str, Union[float, str]]:
    """A simple wrapper of gathering information from collectors.

    :return: A dictionary with the following keys:

        * ``train_step`` the total collected step of training collector;
        * ``train_episode`` the total collected episode of training collector;
        * ``train_time/collector`` the time for collecting frames in the \
            training collector;
        * ``train_time/model`` the time for training models;
        * ``train_speed`` the speed of training (frames per second);
        * ``test_step`` the total collected step of test collector;
        * ``test_episode`` the total collected episode of test collector;
        * ``test_time`` the time for testing;
        * ``test_speed`` the speed of testing (frames per second);
        * ``best_reward`` the best reward over the test results;
        * ``duration`` the total elapsed time.
    """
    duration = time.time() - start_time
    model_time = duration - train_c.collect_time - test_c.collect_time
    train_speed = train_c.collect_step / (duration - test_c.collect_time)
    test_speed = test_c.collect_step / test_c.collect_time
    return {
        "train_step": train_c.collect_step,
        "train_episode": train_c.collect_episode,
        "train_time/collector": f"{train_c.collect_time:.2f}s",
        "train_time/model": f"{model_time:.2f}s",
        "train_speed": f"{train_speed:.2f} step/s",
        "test_step": test_c.collect_step,
        "test_episode": test_c.collect_episode,
        "test_time": f"{test_c.collect_time:.2f}s",
        "test_speed": f"{test_speed:.2f} step/s",
        "best_reward": best_reward,
        'best_rate': best_rate,
        "best_mate_num": best_mate_num,
        "best_len": best_len,
        "best_result": f"{best_reward:.2f} ± {best_reward_std:.2f}",
        "duration": f"{duration:.2f}s",
    }
