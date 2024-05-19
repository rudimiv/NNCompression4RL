#!/

import torch
import torch.nn as nn
from stable_baselines3.sac import SAC

import copy

def convert_quantized_without_activations(trained_model, number_of_layers, original_latent_pi, original_mu, original_log_std):
    latent_pi = trained_model.policy.actor.latent_pi.to('cpu')
    mu = trained_model.policy.actor.mu.to('cpu')
    log_std = trained_model.policy.actor.log_std.to('cpu')

    new_latent_pi = copy.deepcopy(original_latent_pi.to('cpu'))
    new_mu = copy.deepcopy(original_mu.to('cpu'))
    new_log_std = copy.deepcopy(original_log_std.to('cpu'))

    for l in range(number_of_layers):
        weight = copy.deepcopy(latent_pi.model_fp32[l * 2].weight_fake_quant(latent_pi.model_fp32[l * 2].weight).detach())
        bias = copy.deepcopy(latent_pi.model_fp32[l * 2].weight_fake_quant(latent_pi.model_fp32[l * 2].bias).detach())
        new_latent_pi[l * 2].weight = nn.Parameter(weight)
        new_latent_pi[l * 2].bias = nn.Parameter(bias)

    weight = copy.deepcopy(mu.model_fp32.weight_fake_quant(mu.model_fp32.weight).detach())
    bias = copy.deepcopy(mu.model_fp32.weight_fake_quant(mu.model_fp32.bias).detach())
    new_mu.weight = nn.Parameter(weight)
    new_mu.bias = nn.Parameter(bias)

    weight = copy.deepcopy(log_std.model_fp32.weight_fake_quant(log_std.model_fp32.weight).detach())
    bias = copy.deepcopy(log_std.model_fp32.weight_fake_quant(log_std.model_fp32.bias).detach())
    new_log_std.weight = nn.Parameter(weight)
    new_log_std.bias = nn.Parameter(bias)

    trained_model.policy.actor.latent_pi = new_latent_pi
    trained_model.policy.actor.mu = new_mu
    trained_model.policy.actor.log_std = new_log_std


    print(trained_model.policy.actor.latent_pi)

    # https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#quantization-aware-training

    print('Converting...')
    trained_model.policy.actor.latent_pi.apply(torch.ao.quantization.disable_observer)
    trained_model.policy.actor.mu.apply(torch.ao.quantization.disable_observer)
    trained_model.policy.actor.log_std.apply(torch.ao.quantization.disable_observer)
    trained_model.policy.actor.latent_pi = torch.quantization.convert(trained_model.policy.actor.latent_pi, inplace=False)
    trained_model.policy.actor.mu = torch.quantization.convert(trained_model.policy.actor.mu, inplace=False)
    trained_model.policy.actor.log_std = torch.quantization.convert(trained_model.policy.actor.log_std, inplace=False)

    return trained_model

def sac_setup(args, train_env, device_p=False):
    verbose_level = 1 if args.verbose else 0


    print('Input Channels:', train_env.reset().shape)

    device = args.gpu if device_p is False else device_p

    print('SAC device:', device)

    if args.load:
        print(f'Load model from {args.load}')
        model = SAC.load(args.load, env=train_env, print_system_info=True, kwargs={'device': device})

        print(f'Model loaded')
    else:
        policy_kwargs = dict(
            activation_fn=nn.ReLU,# nn.Tanh if args.tanh else nn.ReLU,
            net_arch=dict(
                pi=[args.actor_layer_dim for _ in range(args.actor_layers)],
                qf=[args.critic_layer_dim for _ in range(args.critic_layers)]
            ),
            optimizer_kwargs=dict(weight_decay=1e-4)  # important
        )

        print(f'Activation function: {policy_kwargs["activation_fn"]}')
        print('Net arch:', policy_kwargs['net_arch'])

        model = SAC(
            'MlpPolicy', train_env,
            learning_rate=3e-4,
            policy_kwargs=policy_kwargs,
            verbose=verbose_level,
            tensorboard_log='./tensorboard_log/' if args.tensorboard else None,
            seed=args.seed,
            device=device, # f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu',
            learning_starts=10000,  # according to paper TBD Ant
        )

    original_latent_pi = copy.deepcopy(model.policy.actor.latent_pi).to('cpu')
    original_mu = copy.deepcopy(model.policy.actor.mu).to('cpu')
    original_log_std = copy.deepcopy(model.policy.actor.log_std).to('cpu')

    print(model.policy.actor)

    return model, [original_latent_pi, original_mu, original_log_std]


def sac_postprocess(args, model, original):
    if args.quantize_weights or args.quantize_activations:
        if args.quantize_activations:
            raise Exception()
            # jit_quantized_model, jit_quantized_action_model = convert_quantized_with_activations(model)
            # torch.jit.save(jit_quantized_model, os.path.join(model_directory, 'final_model_quantized_mlp.pt'))
            # torch.jit.save(jit_quantized_action_model, os.path.join(model_directory, 'final_model_quantized_action.pt'))
        else:
            model = convert_quantized_without_activations(model, args.actor_layers, original[0], original[1], original[2])

    return model