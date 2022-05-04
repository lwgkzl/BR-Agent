import time
import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Union, Callable, Optional

from a2c.Collect import MyCollector
from tianshou.policy import BasePolicy
from tianshou.utils import tqdm_config, MovAvg
from a2c.Utils import test_episode, gather_info

def Myonpolicy_trainer(
    policy: BasePolicy,
    train_collector: MyCollector,
    test_collector: MyCollector,
    max_epoch: int,
    step_per_epoch: int,
    collect_per_step: int,
    repeat_per_collect: int,
    episode_per_test: Union[int, List[int]],  # 每一次测试测试几个episode
    batch_size: int,
    train_fn: Optional[Callable[[int, int], None]] = None,
    test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
    stop_fn: Optional[Callable[[float], bool]] = None,
    save_fn: Optional[Callable[[BasePolicy], None]] = None,
    writer: Optional[SummaryWriter] = None,
    log_interval: int = 1,
    verbose: bool = True,
    test_in_train: bool = True,
    test_probs: bool = False,
) -> Dict[str, Union[float, str]]:
    """A wrapper for on-policy trainer procedure.

    The "step" in trainer means a policy network update.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class.
    :param train_collector: the collector used for training.
    :type train_collector: :class:`~tianshou.data.Collector`
    :param test_collector: the collector used for testing.
    :type test_collector: :class:`~tianshou.data.Collector`
    :param int max_epoch: the maximum of epochs for training. The training
        process might be finished before reaching the ``max_epoch``.
    :param int step_per_epoch: the number of step for updating policy network 在每一个epoch最多更新多少次网络
        in one epoch.
    :param int collect_per_step: the number of episodes the collector would  在一个step要进行收集多少个数据
        collect before the network update. In other words, collect some
        episodes and do one policy network update.
    :param int repeat_per_collect: the number of repeat time for policy
        learning, for example, set it to 2 means the policy needs to learn each
        given batch data twice.
    :param episode_per_test: the number of episodes for one policy evaluation.
    :type episode_per_test: int or list of ints
    :param int batch_size: the batch size of sample data, which is going to
        feed in the policy network.
    :param function train_fn: a function receives the current number of epoch
        and step index, and performs some operations at the beginning of
        training in this poch.
    :param function test_fn: a function receives the current number of epoch
        and step index, and performs some operations at the beginning of
        testing in this epoch.
    :param function save_fn: a function for saving policy when the undiscounted
        average mean reward in evaluation phase gets better.
    :param function stop_fn: a function receives the average undiscounted
        returns of the testing result, return a boolean which indicates whether
        reaching the goal.
    :param torch.utils.tensorboard.SummaryWriter writer: a TensorBoard
        SummaryWriter.
    :param int log_interval: the log interval of the writer.
    :param bool verbose: whether to print the information.
    :param bool test_in_train: whether to test in the training phase.
    :param bool test_probs: 在测试集使用多个精度.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    env_step, gradient_step = 0, 0
    best_epoch, best_reward, best_reward_std = -1, -1.0, 0.0
    best_rate = 0.
    best_mate_num = 0.
    best_avg_len = 0.
    stat: Dict[str, MovAvg] = {}
    start_time = time.time()

    train_collector.reset_stat()
    test_collector.reset_stat()
    test_in_train = test_in_train and train_collector.policy == policy

    for epoch in range(1, 1 + max_epoch):
        # train

        policy.train()
        with tqdm.tqdm(
            total=step_per_epoch, desc=f"Epoch #{epoch}", **tqdm_config
        ) as t:
            while t.n < t.total:
                if train_fn:
                    train_fn(epoch, env_step)
                result = train_collector.collect(n_episode=collect_per_step)
                env_step += int(result["n/st"])
                data = {
                    "env_step": str(env_step),
                    "rew": f"{result['rew']:.2f}",
                    "len": str(int(result["len"])),
                    "n/ep": str(int(result["n/ep"])),
                    "n/st": str(int(result["n/st"])),
                    "v/ep": f"{result['v/ep']:.2f}",
                    "v/st": f"{result['v/st']:.2f}",
                    "rate": f"{result['hit_rate']:.2f}",
                }
                if writer and env_step % log_interval == 0:
                    for k in result.keys():
                        if "class" not in k:
                            writer.add_scalar(
                                "train/" + k, result[k], global_step=env_step)
                if test_in_train and stop_fn and stop_fn(result["rew"]):
                    test_result = test_episode(
                        policy, test_collector, test_fn,
                        epoch, episode_per_test, writer, env_step)
                    if stop_fn(test_result["rew"]):
                        if save_fn:
                            save_fn(policy)
                        for k in result.keys():
                            data[k] = f"{result[k]:.2f}"
                        t.set_postfix(**data)
                        return gather_info(
                            start_time, train_collector, test_collector,
                            test_result["rew"], test_result["rew_std"])
                    else:
                        policy.train()

                losses = policy.update(
                    0, train_collector.buffer,
                    batch_size=batch_size, repeat=repeat_per_collect)

                train_collector.reset_buffer()
                step = max([1] + [
                    len(v) for v in losses.values() if isinstance(v, list)])
                gradient_step += step
                for k in losses.keys():
                    if stat.get(k) is None:
                        stat[k] = MovAvg()
                    stat[k].add(losses[k])
                    data[k] = f"{stat[k].get():.6f}"
                    if writer and gradient_step % log_interval == 0:
                        writer.add_scalar(
                            k, stat[k].get(), global_step=gradient_step)
                t.update(step)
                t.set_postfix(**data)
            if t.n <= t.total:
                t.update()
        # test

        start_time = time.time()
        result = test_episode(policy, test_collector, test_fn, epoch,
                              episode_per_test, writer, env_step)
        end_time = time.time()
        print("total_time: ", (end_time-start_time)/60, time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        if test_probs: # 测试第二个阈值
            # policy.actor.threshold = 0.9
            test_probs_result = test_episode(policy, test_collector, test_fn, epoch,
                                episode_per_test, writer, env_step, name='test_prob1/')
            print(result['hit_rate'], test_probs_result['hit_rate'])
            best_rate = max(best_rate, test_probs_result['hit_rate'])
            # policy.actor.threshold = 0.95
        test_hit_rate = result['hit_rate']
        # best_rate = max(best_rate, test_hit_rate)
        best_flag = 0
        if best_epoch == -1 or best_rate < test_hit_rate:
            best_reward, best_reward_std = result["rew"], result["rew_std"]
            best_rate = test_hit_rate
            best_epoch = epoch
            best_mate_num = result['mate_num']
            best_flag = 1
            best_avg_len = result['len']
        if best_rate == test_hit_rate and result['mate_num'] > best_mate_num:
            best_mate_num = result['mate_num']
            best_avg_len = result['len']
            best_flag = 1
        if save_fn and best_flag == 1:
            save_fn(policy)
        if verbose:
            print(f"Epoch #{epoch}: test_reward: {result['rew']:.6f} ± "
                  f"{result['rew_std']:.6f}, best_reward: {best_reward:.6f} ± "
                  f"{best_reward_std:.6f}"
                  f"  hit_rate: {test_hit_rate}:.3f"
                  f" mate_num:  {result['mate_num']}"
                  f" avg_len:  {result['len']}"
                  )
            print(
                f"  best_rate: {best_rate}:.3f"
                f"  best_mate_num: {best_mate_num}:.3f"
                f"  best_len: {best_avg_len}:.3f"
                f"  in: #{best_epoch}:.3f"
            )

        if stop_fn and stop_fn(best_reward):
            break
    return gather_info(start_time, train_collector, test_collector,
                       best_reward, best_reward_std, best_rate,best_mate_num,best_avg_len)
