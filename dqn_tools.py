import stable_baselines3.dqn.policies
import torch
import torch.nn as nn
from stable_baselines3.dqn import DQN

import copy
import gymnasium as gym


def convert_quantized_without_activations(
        trained_model,
        original_cnn,
        original_linear,
        original_q_net,
        not_quantize_last=False,
        quantize_per_channels=False
):
    print('DQN convert_quantized_without_activations...')

    cnn = trained_model.q_net.features_extractor.cnn.to('cpu')
    linear = trained_model.q_net.features_extractor.linear.to('cpu')
    q_net = trained_model.q_net.q_net.to('cpu')

    new_cnn = copy.deepcopy(original_cnn.to('cpu'))
    new_linear = copy.deepcopy(original_linear.to('cpu'))
    new_q_net = copy.deepcopy(original_q_net.to('cpu'))

    if isinstance(new_cnn[0], ExtractorResBlock): # чтобы понять что  ResNet
        print('ResNetFeatureExtractor replacing...')

        for i in range(3):
            weight = copy.deepcopy(
                cnn.model_fp32[i].conv[0].weight_fake_quant(cnn.model_fp32[i].conv[0].weight).detach()
            )

            # In Conv2d bias isn't quantized
            # bias = copy.deepcopy(cnn.model_fp32[i].conv[0].weight_fake_quant(cnn.model_fp32[i].conv[0].bias).detach())

            bias = cnn.model_fp32[i].conv[0].bias.detach()

            new_cnn[i].conv[0].weight = nn.Parameter(weight)
            new_cnn[i].conv[0].bias = nn.Parameter(bias)


            for res_block in [2, 3]:
                for conv_in_res_block in [1, 3]:
                    weight = copy.deepcopy(
                        cnn.model_fp32[i].conv[res_block].res_block[conv_in_res_block].weight_fake_quant(
                            cnn.model_fp32[i].conv[res_block].res_block[conv_in_res_block].weight).detach()
                    )
                    # bias = copy.deepcopy(
                    #     cnn.model_fp32[i].conv[res_block].res_block[conv_in_res_block].weight_fake_quant(
                    #         cnn.model_fp32[i].conv[res_block].res_block[conv_in_res_block].bias).detach()
                    # )

                    bias = cnn.model_fp32[i].conv[res_block].res_block[conv_in_res_block].bias.detach()
                    orig_unique = torch.unique(new_cnn[i].conv[res_block].res_block[conv_in_res_block].weight).numel()

                    new_cnn[i].conv[res_block].res_block[conv_in_res_block].weight = nn.Parameter(weight)
                    new_cnn[i].conv[res_block].res_block[conv_in_res_block].bias = nn.Parameter(bias)

                    new_unique = torch.unique(new_cnn[i].conv[res_block].res_block[conv_in_res_block].weight).numel()

                    print(f'Uniqueness before/after quantization: {orig_unique}/{new_unique}')
    else:
        print('NatureCNN replacing...')
        for l, module in enumerate(cnn.model_fp32):
            if isinstance(module, torch.nn.Conv2d):
                print(module)
                weight = copy.deepcopy(module.weight_fake_quant(module.weight).detach())
                '''print(module.bias.shape)
                print(weight.shape)
                for w in weight:
                    print('Unique:', w.unique().shape)'''
                # print('>>>>>', module)
                # print('device:', module.weight.device, module.bias.device)
                # In Conv2d bias isn't quantized
                # https://discuss.pytorch.org/t/how-to-quantize-the-bias-of-a-convolution-in-qat-quantization-aware-training-mode/191755
                # print(module.bias)
                # bias = copy.deepcopy(module.weight_fake_quant(module.bias).detach())
                bias = module.bias.detach()
                # print('Bias device:', bias.device, weight.device)

                new_cnn[l].weight = nn.Parameter(weight)
                new_cnn[l].bias = nn.Parameter(bias)

    for l, module in enumerate(linear.model_fp32):
        if isinstance(module, torch.nn.Linear):
            weight = copy.deepcopy(module.weight_fake_quant(module.weight).detach())

            print('module.bias:', module.bias)
            print('bias:', bias)

            if quantize_per_channels:
                bias = module.bias.detach()
            else:
                bias = copy.deepcopy(module.weight_fake_quant(module.bias).detach())

            new_linear[l].weight = nn.Parameter(weight)
            new_linear[l].bias = nn.Parameter(bias)

    if not_quantize_last is False:
        for l, module in enumerate(q_net.model_fp32):
            if isinstance(module, torch.nn.Linear):
                weight = copy.deepcopy(module.weight_fake_quant(module.weight).detach())

                if quantize_per_channels:
                    bias = module.bias.detach()
                else:
                    bias = copy.deepcopy(module.weight_fake_quant(module.bias).detach())

                new_q_net[l].weight = nn.Parameter(weight)
                new_q_net[l].bias = nn.Parameter(bias)
    else:
        print('Don\'t replace last layer')

    trained_model.q_net.features_extractor.cnn = new_cnn
    trained_model.q_net.features_extractor.linear = new_linear

    if not_quantize_last is False:
        trained_model.q_net.q_net = new_q_net

    print('Converting...')
    trained_model.q_net.features_extractor.cnn.apply(torch.ao.quantization.disable_observer)
    trained_model.q_net.features_extractor.linear.apply(torch.ao.quantization.disable_observer)
    trained_model.q_net.q_net.apply(torch.ao.quantization.disable_observer)
    trained_model.q_net.features_extractor.cnn = torch.quantization.convert(trained_model.q_net.features_extractor.cnn, inplace=False)
    trained_model.q_net.features_extractor.linear = torch.quantization.convert(trained_model.q_net.features_extractor.linear, inplace=False)
    trained_model.q_net.q_net = torch.quantization.convert(trained_model.q_net.q_net, inplace=False)

    return trained_model


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return x + self.res_block(x)

class ExtractorResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels)
        )

    def forward(self, x):
        return self.conv(x)

class ResNetFeatureExtractor(stable_baselines3.dqn.policies.BaseFeaturesExtractor):
    """
    Agent with ResNet, but without LSTM and additional inputs.
    Inspired by Impala
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 512,):
        super().__init__(observation_space, features_dim)

        print(observation_space)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            ExtractorResBlock(n_input_channels, 32),
            ExtractorResBlock(32, 64),
            ExtractorResBlock(64, 64),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

            print('N_flatten:', n_flatten)
            print(self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape)

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, x):
        return self.linear(self.cnn(x))


# for testing size
'''class ResNet(nn.Module):
    def __init__(self, n_input_channels, n_flatten, features_dim: int = 512,):
        super().__init__()

        self.cnn = nn.Sequential(
            ExtractorResBlock(n_input_channels, 32),
            ExtractorResBlock(32, 64),
            ExtractorResBlock(64, 64),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, x):
        return self.linear(self.cnn(x))'''


'''
================================================================
Total params: 1,995,168
Trainable params: 1,995,168
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.11
Forward/backward pass size (MB): 9.40
Params size (MB): 7.61
Estimated Total Size (MB): 17.12
----------------------------------------------------------------
'''


def dqn_setup(args, train_env, device_p=False):
    verbose_level = 1 if args.verbose else 0

    print(f'Activation function: ReLU()')

    print('Input Channels:', train_env.reset().shape)

    # train_freq = 1
    # learning_starts = 20000
    # target_update_interval = 8000
    train_freq = 4  # according to rl baselines
    learning_starts = int(20_000)  # int(50000 / 4)
    target_update_interval = int(8000)  # int(10000/4)

    exploration_fraction = args.exploration_fraction # 2.5 * 10^4 = 0.0025
    exploration_final_eps = 0.01
    gradient_steps = 1
    batch_size = 32
    device = args.gpu if device_p is False else device_p

    print('DQN device:', device)

    print('DQN Parameters:')
    print(f'exploration_fraction: {exploration_fraction}')
    print(f'exploration_final_eps: {exploration_final_eps}')
    print(f'target_update_interval: {target_update_interval}')
    print(f'learning_starts: {learning_starts}')
    print(f'train_freq: {train_freq}')
    print(f'gradient_steps: {gradient_steps}')
    print(f'batch_size: {batch_size}')
    print(f'device: {device}')

    print(f'LR: {args.lr}')

    print(f'ARGS load:{args.load}')

    if args.load:
        print(f'Load model from {args.load}')
        model = DQN.load(args.load, env=train_env, print_system_info=True, kwargs={'device': args.gpu})

        print(f'Model loaded')
    else:
        if args.res is True:
            print('Net arch: ResCNN')
            # for ResNet
            model = DQN(
                'CnnPolicy', train_env,
                learning_rate=1e-4,
                verbose=verbose_level,
                tensorboard_log='./tensorboard_log/' if args.tensorboard else None,
                seed=args.seed,
                device=device,
                buffer_size=args.buffer,  # иначе не влазит
                learning_starts=learning_starts,
                target_update_interval=target_update_interval,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                exploration_final_eps=exploration_final_eps,
                batch_size=batch_size,
                exploration_fraction=exploration_fraction,
                policy_kwargs={'features_extractor_class': ResNetFeatureExtractor,
                              'optimizer_kwargs': {'eps': 3.125e-4, 'weight_decay': 1e-5},
                              'net_arch': []},  # чтобы лишних слоев не создалось
                # eval epsilon=0.001 # не нашел
                # optimize_memory_usage=True
            )
        else:
            print('Net arch: NatureCNN')
            # for CNN
            model = DQN(
                'CnnPolicy', train_env,
                # learning_rate=2.5e-4,
                learning_rate=args.lr,
                verbose=verbose_level,
                tensorboard_log='./tensorboard_log/' if args.tensorboard else None,
                seed=args.seed,
                device=device,
                buffer_size=args.buffer,  # иначе не влазит
                learning_starts=learning_starts,
                target_update_interval=target_update_interval,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                exploration_final_eps=exploration_final_eps,
                batch_size=batch_size,
                exploration_fraction=exploration_fraction,
                policy_kwargs={'optimizer_kwargs': {'eps': 1e-8, 'weight_decay': 0}},
                # eval epsilon=0.001 # не нашел
                # optimize_memory_usage=True
            )

    print('Policy and Optimizer params: >>>>>>>>>>>>>>>>>>>')
    print(model.policy)
    print(model.policy.optimizer)
    # ???? model.policy.q_net

    print(model.q_net.device)
    # maybe cnn, and linear add
    cnn = copy.deepcopy(model.q_net.features_extractor.cnn).to('cpu')
    linear = copy.deepcopy(model.q_net.features_extractor.linear).to('cpu')
    q_net = copy.deepcopy(model.q_net.q_net).to('cpu')

    print(model.q_net)
    print(model.q_net.device)

    return model, [cnn, linear, q_net]


def dqn_postprocess(args, model, original):
    if args.quantize_weights or args.quantize_activations:
        if args.quantize_activations:

            if args.alg == 'sac':
                print('Model.actor.latent_pi:', model.policy.actor.latent_pi)
                print('Model.actor.mu:', model.policy.actor.mu)
                print('Model.actor.log_std:', model.policy.actor.log_std)
            elif args.alg == 'dqn':
                print('Model.fe.CNN:', model.q_net.features_extractor.cnn)
                print('Model.fe.Linear:', model.q_net.features_extractor.linear)
                print('Model.fe.Linear:', model.q_net.q_net)

            raise Exception()
            # jit_quantized_model, jit_quantized_action_model = convert_quantized_with_activations(model)
            # torch.jit.save(jit_quantized_model, os.path.join(model_directory, 'final_model_quantized_mlp.pt'))
            # torch.jit.save(jit_quantized_action_model, os.path.join(model_directory, 'final_model_quantized_action.pt'))
        else:
            model = convert_quantized_without_activations(model, original[0], original[1], original[2],
                                                          args.not_quantize_last, args.quantize_per_channels)

    return model
