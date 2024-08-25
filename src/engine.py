import copy

import torch
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env

# import wandb
# from wandb.integration.sb3 import WandbCallback
from video_recorder import VideoRecorderCallback


import os

from pruning import PruningCallback, tmp_remove_parametrization
from quantization import QuantizationCallback, get_ptq_model
from tools import seed_everything, get_params, erk_sparsity
import numpy as np

from ppo_tools import ppo_setup, ppo_postprocess
from sac_tools import sac_setup, sac_postprocess
from dqn_tools import dqn_setup, dqn_postprocess
from tools import form_log_name, ModelSaveCallback, test_model, smart_loader

from stable_baselines3.common.env_util import make_vec_env

import time


def engine_train(model_directory, args):
    seed_everything(args.seed)
    log_name = form_log_name(args)
    # AsyncEval??
    # SubprocVecEnv
    if args.alg != 'dqn':
        '''train_env = SubprocVecEnv([
            lambda: gym.make(
                args.env, max_episode_steps=args.max_steps if args.max_steps else None
            ) for _ in range(args.n_actors)
        ])
        eval_env = SubprocVecEnv([
            lambda: gym.make(
                args.env, max_episode_steps=args.max_steps if args.max_steps else None
            ) for _ in range(args.n_test_actors)
        ])'''

        train_env = make_vec_env(
            lambda: gym.make(args.env, max_episode_steps=args.max_steps if args.max_steps else None),
            n_envs=args.n_actors,
            seed=args.seed,
            # vec_env_cls=SubprocVecEnv
        )

        eval_env = make_vec_env(
            lambda: gym.make(args.env, max_episode_steps=args.max_steps if args.max_steps else None),
            n_envs=args.n_test_actors,
            seed=args.seed,
            # vec_env_cls=SubprocVecEnv
        )
    else:
        # vec_env_cls = SubprocVecEnv
        # В AtariWrapper уже frame_skip = 4 => *NoFrameskip-v4"
        train_env = make_atari_env(
            args.env,
            n_envs=args.n_actors,
            seed=args.seed,
            vec_env_cls=SubprocVecEnv if args.n_actors > 1 else None  # DummyVecEnv
            # env_kwargs={'max_episode_steps': args.max_steps if args.max_steps else None})
        )


        # Оказывается, что в Atari и пропускается каждый 4-ый кадр
        # И потом они по 4 стекаются
        # Stack 4 frames
        train_env = VecFrameStack(train_env, n_stack=4)

        # vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

        eval_env = make_atari_env(
            args.env,
            n_envs=args.n_test_actors,
            seed=args.seed,
            wrapper_kwargs={'clip_reward': False},
            vec_env_cls=SubprocVecEnv if args.n_test_actors > 1 else None  # DummyVecEnv
            # env_kwargs={'max_episode_steps': args.max_steps if args.max_steps else None},
        )


        eval_env = VecFrameStack(eval_env, n_stack=4)

    # TBD remove
    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=50, min_evals=200, verbose=1)

    # deterministic = False делает семплирование из распределения
    # eval_freq в steps
    '''callbacks_list = [EvalCallback(eval_env,
                                   best_model_save_path=model_directory,
                                   n_eval_episodes=args.n_eval_episodes,
                                   eval_freq=args.eval_freq,
                                   # callback_after_eval=stop_train_callback,
                                   deterministic=True,  # if args.alg != 'dqn' else False,
                                   render=False,
                                   verbose=1)]'''


    # If a vector env is passed in, this divides the episodes to evaluate onto the different elements of the vector env.
    # max_episode_callback = StopTrainingOnMaxEpisodes(max_episodes=args.episodes, verbose=1)
    if args.alg == 'ppo':
        setup = ppo_setup
        postprocess = ppo_postprocess
    elif args.alg == 'sac':
        setup = sac_setup
        postprocess = sac_postprocess
    elif args.alg == 'dqn':
        setup = dqn_setup
        postprocess = dqn_postprocess
    else:
        print('This algorithm doesn\'t supported')
        raise Exception()

    pruning_sparsity_schedule = None
    common_sparsities_schedule = None

    if args.prune:
        # In stablebaselines are counted as one step for one step for all actors
        real_steps = args.total_steps // args.n_actors
        print('Real_steps:', real_steps)
        # pruning_freq = int(real_steps * (args.pruning_end - args.pruning_start) / (args.pruning_iterations - 1))
        pruning_freq = int(real_steps * (args.pruning_end - args.pruning_start) / args.pruning_iterations)
        '''pruning_schedule = list(
            range(int(real_steps * args.pruning_start), int(real_steps * args.pruning_end), pruning_freq)
        )'''
        print('Pruning frequency:', pruning_freq)

        pruning_schedule = list([
            int(real_steps * args.pruning_start) + pruning_freq * step for step in range(args.pruning_iterations)
        ])

        print('Observation space', train_env.observation_space)
        print('Action space:', train_env.action_space)

        sparsity_schedule = []
        common_sparsities = []

        if args.alg == 'dqn':
            # Maybe better to replace
            dims = []

            # this is temporary only for counting dimensions
            tmp_model, original = setup(args, train_env)
            print("Original:", original)

            # because in Atari we use CNN we need to collect dims in this way
            for submodel in list([tmp_model.q_net.features_extractor.cnn, tmp_model.q_net.features_extractor.linear, tmp_model.q_net.q_net]):
                for (name, module) in submodel.named_modules():
                    if type(module) is nn.Linear or type(module) is nn.Conv2d:
                        dims.append(get_params(module))

            dims.append(train_env.action_space.n)
            print('Actor CnnNature dimensions', dims)
        else:
            dims = [train_env.observation_space.shape[0], ] + \
                    [args.actor_layer_dim for _ in range(args.actor_layers)] + \
                    [train_env.action_space.shape[0]]

            print('Actor MLP dimensions', dims)

        for i in range(args.pruning_iterations + 1):
            sparsity_point = args.target_sparsity - args.target_sparsity * (1 - i / args.pruning_iterations)**3
            common_sparsities.append(sparsity_point)
            # Schedules for each particular module
            if args.er:
                module_sparsities = erk_sparsity(dims, sparsity_point)
            else:
                module_sparsities = [sparsity_point for _ in range(len(dims) - 1)]

            print(f'Sparsity {sparsity_point:.3f}. module sparsities:', module_sparsities)
            sparsity_schedule.append(module_sparsities)

        pruning_sparsity_schedule = {pr: sp for pr, sp in zip(pruning_schedule, sparsity_schedule[1:])}
        common_sparsities_schedule = {pr: sp for pr, sp in zip(pruning_schedule, common_sparsities[1:])}

        print(f'Pruning frequency {pruning_freq}')
        print(f'Pruning schedule {pruning_schedule}')
        # print(f'Sparsity schedule {sparsity_schedule}')
        # print(f'=> {pruning_sparsity_schedule}')

    print('Log name:', log_name)

    # Here model is created or loaded
    model, original = setup(args, train_env)
    print("Original:", original)

    callbacks_list = [ModelSaveCallback(
        args,
        postprocess,
        original,
        eval_env,
        best_model_save_path=model_directory,
        n_eval_episodes=args.n_eval_episodes,
        eval_freq=args.eval_freq,
        # callback_after_eval=stop_train_callback,
        deterministic=True,  # if args.alg != 'dqn' else False,
        render=False,
        verbose=1
    )]


    if args.prune:
        callbacks_list.append(PruningCallback(args.alg, pruning_sparsity_schedule, common_sparsities_schedule, args.res))

    if args.quantize_weights or args.quantize_activations:
         callbacks_list.append(QuantizationCallback(args, model_directory))
         # args.alg, args.quantize_weights, args.quantize_activations, args.quantize_per_channels, args.not_quantize_last))

    if args.video_record:
        callbacks_list.append(VideoRecorderCallback(
            args.env, args.video_record_freq, f'{model_directory}/video'
        ))

    callbacks = CallbackList(callbacks_list)

    # Add DDPG, A2C, SAC
    # Set total_timesteps to maximum. If args.episodes complete early then max_episode_callback will stop

    print(f'START TIME: {time.strftime("%Y-%m-%d %H:%M:%S")}')

    if args.no_train is not True:
        total_timesteps = int(args.total_steps * (1 + args.additional_steps))
        print(f'Total steps with additional {args.additional_steps} = {total_timesteps}')
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
            callback=callbacks,
            tb_log_name=log_name
        )

        print('Postprocssing...')
        model = postprocess(args, model, original)

        print(f'Saving model to {model_directory}')
        model.save(os.path.join(model_directory, 'final_model.pt'))
    if args.ptq:
        tmp_args = copy.deepcopy(args)
        tmp_args.tensorboard = False
        # tmp_args.device='cpu'
        model_tmp, original = setup(tmp_args, train_env, device_p='cpu')
        # model_tmp, original = setup(tmp_args, train_env)
        # Вроде реально загружает
        if args.alg == 'dqn':
            if args.res is False:
                print('tmp:', model_tmp.q_net.features_extractor.cnn[0].weight.sum())
        elif args.alg == 'sac':
            print('tmp:', model_tmp.actor.latent_pi[0].weight.sum())

        best_nq_model = smart_loader(
            model_tmp,
            model_directory,
            'best_nq_model',
            args.alg,
            args.quantize_activations,
            apply_masks_pemanent=True if args.prune else False
        )

        if args.alg == 'dqn':
            if args.res is False:
                print('best_nq:', best_nq_model.q_net.features_extractor.cnn[0].weight.sum())
        elif args.alg == 'sac':
            print('best_nq:', best_nq_model.actor.latent_pi[0].weight.sum())

        print('Pre PTQ testing...')

        mean, std = test_model(best_nq_model, args, eval_env) #, eval_episodes=5)

        model.logger.record("preptq/mean", np.mean(mean))
        model.logger.record("preptq/std", np.mean(std))

        print('PTQ...')

        ptq_model = get_ptq_model(args, best_nq_model)
        print('Calibrating...')
        test_model(ptq_model, args, eval_env) # , eval_episodes=5)

        if args.alg == 'dqn':
            best_nq_model.q_net.features_extractor.cnn.to('cpu')
            torch.ao.quantization.convert(best_nq_model.q_net.features_extractor.cnn, inplace=True)
            best_nq_model.q_net.features_extractor.linear.to('cpu')
            torch.ao.quantization.convert(best_nq_model.q_net.features_extractor.linear, inplace=True)
            best_nq_model.q_net.q_net.to('cpu')
            torch.ao.quantization.convert(best_nq_model.q_net.q_net, inplace=True)
        elif args.alg == 'sac':
            # best_nq_model.actor.features_extractor.to('cpu')
            # torch.ao.quantization.convert(best_nq_model.actor.features_extractor, inplace=True)
            best_nq_model.actor.latent_pi.to('cpu')
            torch.ao.quantization.convert(best_nq_model.actor.latent_pi, inplace=True)
            best_nq_model.actor.mu.to('cpu')
            torch.ao.quantization.convert(best_nq_model.actor.mu, inplace=True)
            best_nq_model.actor.log_std.to('cpu')
            torch.ao.quantization.convert(best_nq_model.actor.log_std, inplace=True)

        print('PTQ process finished')

        if args.alg == 'dqn' and args.res is False:
            print(ptq_model.q_net.features_extractor.linear.model_fp32[0])
            print(ptq_model.q_net.features_extractor.linear.model_fp32[0].weight)
            print(ptq_model.q_net.features_extractor.linear.model_fp32[0].weight())
            print(ptq_model.q_net.features_extractor.linear.model_fp32[0].weight().dequantize())
            print(ptq_model.q_net.features_extractor.linear.model_fp32[0].weight().dequantize().shape)
            print(ptq_model.q_net.features_extractor.linear.model_fp32[0].weight().dequantize().unique().shape)

            # weight_, bias_ = ptq_model.q_net.features_extractor.linear.model_fp32[0]._packed_params.unpack()
            # print(weight_.dequantize())

        # через original загружать?

        # print('Loading ot gpu...')
        # best_nq_model.q_net.features_extractor.cnn.to(f'cuda:{args.gpu}')
        # best_nq_model.q_net.features_extractor.linear.to(f'cuda:{args.gpu}')
        # best_nq_model.q_net.q_net.to(f'cuda:{args.gpu}')
        # tmp_remove_parametrization(args.alg, best_nq_model)

        if args.alg == 'dqn' and args.res is False:
            print(ptq_model.q_net.features_extractor.linear.model_fp32[0])
            # print(ptq_model.q_net.features_extractor.linear.model_fp32[0].weight.unique().shape)

        print('Testing PTQ model...')
        mean, std = test_model(ptq_model, args, train_env) #, eval_episodes=5)

        model.logger.record("ptq/mean", np.mean(mean))
        model.logger.record("ptq/std", np.mean(std))
        model.logger.dump(model.num_timesteps)


    if args.test:
        test_model(model, args, eval_env)

    print(f'END TIME: {time.strftime("%Y-%m-%d %H:%M:%S")}')




