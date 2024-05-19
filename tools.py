import copy
import random
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback

import os
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization

from pruning import tmp_remove_parametrization, set_masks, is_pruned, \
    prune_mlp_remove_parametrization, prune_cnn_remove_parametrization
import quantization


def form_log_name(args):
    if args.alg == 'dqn':
        if args.res is True:
            res = f'{args.alg}_{args.env}_resnet'
        else:
            res = f'{args.alg}_{args.env}_cnn'
    else:
        res = f'{args.alg}_{args.env}_[{args.actor_layers}_{args.actor_layer_dim}]'

    res += f'_sp={str(args.target_sparsity) + "_it" + str(args.pruning_iterations) if args.prune else "No_prune"}_er={args.er}' \
           f'_qw={args.quantize_weights}_pc={args.quantize_per_channels}_asym={args.asymmetric}_qa={args.quantize_activations}_lr_{str(args.lr).replace(".", "_")}_seed={args.seed}'

    return res


def seed_everything(seed, using_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if using_cuda:
        torch.cuda.manual_seed(seed)
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_random_seed(seed)


def print_tensor_info(tensor):
    print('Size:', tensor.weight().size())
    print(f'Number of unique parameters: {torch.dequantize(tensor.weight()).unique().numel()}')
    # add to all modules
    print(f'Zero point: {tensor.weight().q_zero_point()}')
    print(f'Scale: {tensor.weight().q_scale()}')
    print('Int_repr:', tensor.weight().int_repr().unique(return_counts=True))
    # print('Unique:', quantized_model.model_fp32[0].weight.unique().numel())


def erk_sparsity(dims, target_sparsity):
    erk_coeffs = []
    parameters_num = []

    for i in range(len(dims) - 1):
        # for DQN
        if type(dims[i]) is tuple:
            # n_l * n_{l+1} * kernel_w * kernel_h
            if len(dims[i]) == 2:
                parameters_num.append(dims[i][0] * dims[i][1])
            else:
                parameters_num.append(dims[i][0] * dims[i][1] * dims[i][2] * dims[i][3])

            erk_coeffs.append(1 - (sum(dims[i])) / parameters_num[-1])

        else:
            parameters_num.append(dims[i] * dims[i + 1])
            erk_coeffs.append(1 - (dims[i] + dims[i + 1]) / parameters_num[-1])

    erk_coeffs = np.array(erk_coeffs)
    parameters_num = np.array(parameters_num)

    # дважды перпроверил
    k = np.sum(parameters_num) * target_sparsity / np.sum(parameters_num * erk_coeffs)

    return erk_coeffs * k


# for DQB
def get_params(module):
    print(module)

    if type(module) is nn.Conv2d:
        return (module.in_channels, module.out_channels, module.kernel_size[0], module.kernel_size[1])
    elif type(module) is nn.Linear:
        return (module.in_features, module.out_features)
    else:
        print('Undefined')
        raise Exception()


from stable_baselines3.common.callbacks import BaseCallback


class ModelSaveCallback(EvalCallback):
    # def __init__(self, args, verbose: int = 1):
    def __init__(self,
                 args,
                 postprocess,
                 original,
                 eval_env,
                 callback_on_new_best=None,
                 callback_after_eval=None,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path=None,
                 best_model_save_path=None,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1,
                 warn: bool = True
                 ):
        super().__init__(eval_env, callback_on_new_best, callback_after_eval,
                         n_eval_episodes, eval_freq, log_path, best_model_save_path,
                         deterministic, render, verbose, warn)

        self.args = args

        self.alg = args.alg
        self.quantize_weights = args.quantize_weights
        self.quantize_activations = args.quantize_activations
        self.quantize_start = args.quantize_start
        self.total_steps = args.total_steps
        self.pruning_end = args.pruning_end
        self.prune = args.prune
        self.quantizing_started = False


        self.postprocess = postprocess
        self.original = copy.deepcopy(original)

        '''if self.prune:
            if self.quantize_weights:
                # Prune and quantization
                self.start_saving = int(max(self.quantize_start * self.total_steps,
                                            self.pruning_end * self.total_steps))
            else:
                # Only pruning
                self.start_saving= int(self.pruning_end * self.total_steps)
        else:
            if self.quantize_weights:
                # Only quantization
                self.start_saving = int(self.quantize_start * self.total_steps)
            else:
                # No prune, no quant
                self.start_saving = 0'''
        # мб np.inf
        max_step = int(self.total_steps * (1 + args.additional_steps)) + 1000

        if self.prune:
            self.start_saving_nq = int(self.pruning_end * self.total_steps)
        else:
            self.start_saving_nq = 0


        if self.quantize_weights:
            self.start_saving_quantized = int(self.quantize_start * self.total_steps)
        else:
            self.start_saving_quantized = max_step


        print(f'Quantized saving start: {self.start_saving_quantized}')
        print(f'Not quantized (prune/dense) saving start: {self.start_saving_nq}')

    def _on_step(self) -> bool:
        continue_training = True

        # if self.n_calls % 500 == 0:
        #     with torch.no_grad():
        #         print('>>>>> tmp cnn:', self.model.q_net.features_extractor.cnn[0].weight.sum())
        #         print('>>>>> tmp linear:', self.model.q_net.features_extractor.linear[0].weight.sum())
        #         print('>>>>> tmp q_net:', self.model.q_net.q_net[0].weight.sum())
        #         print('>>>>> tmp q_net bias:', self.model.q_net.q_net[0].bias.sum())

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if self.start_saving_nq <= self.num_timesteps:
                if self.num_timesteps <= self.start_saving_quantized:
                    self.logger.record("nq/mean", float(mean_reward))
                    self.logger.record("nq/std", float(std_reward))
                else:
                    self.logger.record("qat/mean", float(mean_reward))
                    self.logger.record("qat/std", float(std_reward))

            if self.quantizing_started is False and self.start_saving_quantized < self.num_timesteps:
                print('Quantizing step achieved')
                print('Drop the best_mean_reward')
                self.quantizing_started = True
                self.best_mean_reward = -np.inf

            # My code Add the second condition
            if mean_reward > self.best_mean_reward:
                print("New best mean reward!")
                print('Checking the saving possibility...', end=' ')

            if mean_reward > self.best_mean_reward and \
                    (self.start_saving_quantized < self.num_timesteps or self.start_saving_nq < self.num_timesteps):
                print('Yes')
                print('Saving...')

                if self.best_model_save_path is not None:

                    print(f'Number of completed timesteps: {self.num_timesteps}')
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    # Add smart saving

                    if self.num_timesteps > self.start_saving_nq:
                        if self.num_timesteps < self.start_saving_quantized:
                            # сохраняем dense/pruned
                            print('Saving not quantized model...')
                            # self.model.save(os.path.join(self.best_model_save_path, "best_nq_model_full"))
                            self.smart_saver('best_nq_model')

                        else:
                            # сохраняем квантизованный dense/prunes
                            print('Saving quantized model...')
                            # self.model.save(os.path.join(self.best_model_save_path, "best_q_model_full"))
                            self.smart_saver('best_q_model')
                    else:
                        if self.num_timesteps >= self.start_saving_quantized:
                            print('ERROR: Strange situation for saving quantized before finishing pruning... ')

                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()
            else:
                print('No')

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def smart_saver(self, model_name):
        if self.quantize_activations is not True:
            # вспомогательные модули сохраняем как есть
            # модули основной сети обрабатываемч
            print('No activations => Saving')
            saving_dir = os.path.join(self.best_model_save_path, model_name)
            os.makedirs(saving_dir, exist_ok=True)

            # model_copy = copy.deepcopy(self.model)
            already_pruned = is_pruned(self.alg, self.model)
            already_quantized = quantization.is_quantized(self.alg, self.model)
            already_quantized = False
            masks = []

            if self.alg == 'sac':
                modules = dict(
                    latent_pi=copy.deepcopy(self.model.policy.actor.latent_pi),
                    mu=copy.deepcopy(self.model.policy.actor.mu),
                    log_std=copy.deepcopy(self.model.policy.actor.log_std)
                )

                print(f'Saving critic...')
                torch.save(self.model.critic.state_dict(), os.path.join(saving_dir, 'critic'))
                print(f'Saving critic_target...')
                torch.save(self.model.critic_target.state_dict(), os.path.join(saving_dir, 'critic_target'))
                # там же еще сети есть!!!


            elif self.alg == 'dqn':
                if quantization.is_quantized(self.alg, self.model) is False and self.args.res is False:
                    print('>>>>> tmp cnn:', self.model.q_net.features_extractor.cnn[0].weight.sum())
                    print('>>>>> tmp linear:', self.model.q_net.features_extractor.linear[0].weight.sum())
                    print('>>>>> tmp q_net:', self.model.q_net.q_net[0].weight.sum())


                modules = dict(
                    cnn=copy.deepcopy(self.model.q_net.features_extractor.cnn),
                    linear=copy.deepcopy(self.model.q_net.features_extractor.linear),
                    q_net=copy.deepcopy(self.model.q_net.q_net)
                )


                # print(f'Saving target value to {os.path.join(saving_dir, "target")}...')
                print(f'Saving target net...')
                # print(self.model.q_net_target.state_dict())
                torch.save(self.model.q_net_target.state_dict(), os.path.join(saving_dir, 'target'))
            else:
                modules = {}

            for name in modules.keys():
                print(f'Processing module {name}')
                if already_pruned:
                    if name == 'cnn':
                        masks.append(prune_cnn_remove_parametrization(modules[name]))
                    else:
                        masks.append(prune_mlp_remove_parametrization(modules[name]))

                if already_quantized:
                    pass

                # torch.save(modules[name], os.path.join(saving_dir, name))
                torch.save(modules[name].state_dict(), os.path.join(saving_dir, name))

            # if masks is not None:
            if already_pruned:
                print('Saving masks...')
                torch.save(masks, os.path.join(saving_dir, 'masks'))

            # print('Restore masks...') не надо так-как мы работаем с копиями


def smart_loader(model, best_model_save_path, model_name, alg, quantize_activations, apply_masks_pemanent=False):
    if quantize_activations is True:
        print('Impossible to load best weights due to activations_quantization')
        return model

    print('Smart loader...')

    directory = os.path.join(best_model_save_path, model_name)

    already_pruned = is_pruned(alg, model)
    already_quantized = quantization.is_quantized(alg, model)
    already_quantized = False

    # search masks
    if os.path.isfile(os.path.join(directory, 'masks')):
        print('Masks are found')
        masks = torch.load(os.path.join(directory, 'masks'), map_location=model.device)
    else:
        masks = None

    if alg == 'sac':
        modules = dict(
            latent_pi=model.policy.actor.latent_pi,
            mu=model.policy.actor.mu,
            log_std=model.policy.actor.log_std
        )

        print(f'Loading critic...')
        model.critic.load_state_dict(torch.load(os.path.join(directory, 'critic')))
        print(f'Loading critic_target...')
        model.critic_target.load_state_dict(torch.load(os.path.join(directory, 'critic_target')))

    elif alg == 'dqn':
        modules = dict(
            cnn=model.q_net.features_extractor.cnn,
            linear=model.q_net.features_extractor.linear,
            q_net=model.q_net.q_net
        )

        print(f'Loading target net...')
        # print(f'Loading target value to {os.path.join(directory, "target")}')
        # print(torch.load(os.path.join(directory, 'target')))
        # print(model.q_net_target)
        model.q_net_target.load_state_dict(torch.load(os.path.join(directory, 'target')))
    else:
        modules = {}

    if alg == 'sac':
        print(f'Control sum latent_pi: {model.policy.actor.latent_pi[0].weight.sum()}')

    for name in modules.keys():
        print(f'Processing module {name}. Prune: {already_pruned}')

        if already_pruned:
            print(f'Already pruned -> remove parametrization')
            if name == 'cnn':
                prune_cnn_remove_parametrization(modules[name])
            else:
                prune_mlp_remove_parametrization(modules[name])

        modules[name].load_state_dict(torch.load(os.path.join(directory, name)))
        if name == 'latent_pi':
            print(f'Control sum latent_pi: {modules[name][0].weight.sum()}')

    if alg == 'sac':
        print(f'Control sum latent_pi: {model.policy.actor.latent_pi[0].weight.sum()}')

    print(model.device)

    if masks is not None:
        set_masks(alg, model, masks, quant=False)

    if apply_masks_pemanent is True:
        print('Remove masks permanently')
        for name in modules.keys():
            print(f'Processing module {name}. Prune: {already_pruned}')

            # if already_pruned:
                # print(f'Already pruned -> remove parametrization')
            if name == 'cnn':
                prune_cnn_remove_parametrization(modules[name])
            else:
                prune_mlp_remove_parametrization(modules[name])

    return model

def test_model(model, args, eval_env, eval_episodes=None):
    print('Testing...')
    if eval_episodes is None:
        eval_episodes = args.n_eval_episodes * 5
    print(f'Eval episodes: {eval_episodes}')
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes, deterministic=True)
    print(f'Test res in {eval_episodes} episodes: {mean}/+-{std}')

    return mean, std
