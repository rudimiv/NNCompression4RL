from builtins import int

import torch
import torch.nn as nn
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from torch.ao.quantization import (
      get_default_qconfig_mapping,
      get_default_qat_qconfig_mapping,
      QConfigMapping,
    FakeQuantize,
    QConfig,
    MovingAverageMinMaxObserver,
    default_weight_fake_quant, default_per_channel_weight_fake_quant,
    default_activation_only_qconfig,
    default_fake_quant,
    default_placeholder_observer,
    default_weight_observer,
    default_per_channel_weight_observer,
    MinMaxObserver,
    PerChannelMinMaxObserver,
    HistogramObserver
)

from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver

from pruning import tmp_remove_parametrization, set_masks, is_pruned
import tools


class QuantizedActor(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedActor, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        input = x.clone()

        x = self.quant(x)

        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x


def is_quantized(alg, model):
    if alg == 'ppo':
        test_module = model.policy.mlp_extractor.policy_net
    elif alg == 'sac':
        test_module = model.policy.actor.latent_pi
    elif alg == 'dqn':
        test_module = model.q_net.features_extractor.cnn
    else:
        test_module = None
        print(f'Warning: unknown algorithm {alg}')
        already_quantized = False

    if test_module is not None:
        if isinstance(test_module, QuantizedActor):
            already_quantized = True
        else:
            already_quantized = False

    print(f'Check if model is already quantized: {already_quantized}')

    return already_quantized


def prepare_module_quantization(module, qconfig, quant_type):
    module.eval()
    module = QuantizedActor(module)
    module.qconfig = torch.ao.quantization.get_default_qconfig('x86') # qconfig

    if quant_type == 'qat':
        torch.quantization.prepare_qat(module, inplace=True)
        module.train()
    elif quant_type == 'ptq':
        torch.quantization.prepare(module, inplace=True)

    return module

def get_ptq_model(args, model):
    quant_activations = args.quantize_activations
    quant_activations = True
    per_channel_quant = PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric # torch.per_channel_affine if args.asymmetric else torch.per_channel_symmetric
    )

    per_tensor_quant = MinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric # torch.per_tensor_affine if args.asymmetric else torch.per_tensor_symmetric
    )

    if not args.quantize_per_channels:
        qconfig_ptq = QConfig(
            activation=HistogramObserver.with_args(
                reduce_range=True
            ) if quant_activations else default_placeholder_observer,
            weight=default_per_channel_weight_observer #per_tensor_quant
        )
    else:
        qconfig_ptq = QConfig(
            activation=HistogramObserver.with_args(
                reduce_range=True
            ) if quant_activations else default_placeholder_observer,
            weight=default_per_channel_weight_observer # per_channel_quant
        )

        # torch.ao.quantization.get_default_qconfig('x86')

    qconfig_cnn_ptq = QConfig(
        activation=HistogramObserver.with_args(
            reduce_range=True
        ) if quant_activations else default_placeholder_observer,
        weight=default_per_channel_weight_observer # per_channel_quant
    )

    if args.alg == 'sac':
        model.policy.actor.latent_pi = prepare_module_quantization(
            model.policy.actor.latent_pi, qconfig_ptq, 'ptq'
        )
        # torch.ao.quantization.convert(model.policy.latent_pi, inplace=True)

        model.policy.actor.mu = prepare_module_quantization(
            model.policy.actor.mu, qconfig_ptq, 'ptq'
        )
        # torch.ao.quantization.convert(model.policy.actor.mu, inplace=True)

        model.policy.actor.log_std = prepare_module_quantization(
            model.policy.actor.log_std, qconfig_ptq, 'ptq'
        )
        # torch.ao.quantization.convert(model.policy.actor.log_std, inplace=True)
    elif args.alg == 'dqn':
        model.q_net.features_extractor.cnn = prepare_module_quantization(
            model.q_net.features_extractor.cnn, qconfig_cnn_ptq, 'ptq'
        )
        print(model.q_net.features_extractor.cnn.qconfig)
        # model.q_net.features_extractor.cnn.to('cpu')
        # torch.ao.quantization.convert(model.q_net.features_extractor.cnn, inplace=True)

        # print('CNN q_net.features_extractor.cnn config:', model.q_net.features_extractor.cnn.qconfig)
        print('DEBUG: CNN quantized:', model.q_net.features_extractor.cnn)

        model.q_net.features_extractor.linear = prepare_module_quantization(
            model.q_net.features_extractor.linear, qconfig_ptq, 'ptq'
        )
        # torch.ao.quantization.convert(model.q_net.features_extractor.linear, inplace=True)

        print('MLP q_net.features_extractor.linear config:', model.q_net.features_extractor.linear.qconfig)

        if args.not_quantize_last is False:
            model.q_net.q_net = prepare_module_quantization(
                model.q_net.q_net, qconfig_ptq, 'ptq'
            )
            # torch.ao.quantization.convert(model.q_net.q_net, inplace=True)
            print('MLP q_net.q_net config:', model.q_net.q_net.qconfig)
        else:
            print('NOT QUANTIZE LAST LAYER')

    else:
        print('This algorithm isn\'t supported')
        raise Exception()

    return model


# переделать на args
class QuantizationCallback(BaseCallback):
    # for quantization details see https://pytorch.org/docs/stable/quantization-support.html
    def __init__(self, args, best_model_save_path, verbose: int = 1):
        # alg,
        # quantize_weights: bool,
        # quantize_activations: bool,
        # quantize_per_channel: bool=False,
        # not_quantize_last: bool=False,
        # verbose: int = 1,):
        super().__init__(verbose)

        self.alg = args.alg
        self.quant_on = False
        self.quant_end = False
        self.load_best_nq = args.load_best_nq
        self.best_model_save_path = best_model_save_path

        # Moment for switching on the quantization
        self.quantize_start_step = int(args.quantize_start * args.total_steps)
        print(f'Quantization will be started at step: {self.quantize_start_step}')

        if args.quantize_end > 0.0001:
            self.quantize_end_step = int(args.quantize_end * args.total_steps)
        else:
            self.quantize_end_step = np.inf
        print(f'Quantization will be finished at step: {self.quantize_end_step}')

        self.quantize_activations = args.quantize_activations

        self.res = args.res

        per_channel_quant = FakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_affine if args.asymmetric else torch.per_channel_symmetric,  # non symmetric
            reduce_range=False,
            ch_axis=0
        )

        per_tensor_quant = FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine if args.asymmetric else torch.per_tensor_symmetric,
            reduce_range=False
        )



        # torch.quantization.default_weight_only_qconfig
        if not args.quantize_per_channels:
            print('Quantize MLP per Layer')
            self.qconfig_qat = QConfig(
                # activation=FakeQuantize.with_args(
                #     observer=MovingAverageMinMaxObserver, reduce_range=True
                # )
                activation=default_fake_quant if args.quantize_activations else default_placeholder_observer, # nn.Identity,
                weight=per_tensor_quant if args.quantize_weights else nn.Identity
                # weight=default_weight_fake_quant if args.quantize_weights else nn.Identity
            )
            # weight=default_float_qparams_observer_4bit
            # мб добавить отдельный qconfig для последних слоев
            # еще default_fused_wt_fake_quant
        else:
            print('Quantize MLP per Channel')
            # almost torch.quantization.get_default_qat_qconfig('fbgemm')

            self.qconfig_qat = QConfig(
                activation=default_fake_quant if args.quantize_activations else default_placeholder_observer, # nn.Identity,
                weight=per_channel_quant if args.quantize_weights else nn.Identity
                # weight=default_per_channel_weight_fake_quant if args.quantize_weights else nn.Identity
            )


            # activation=HistogramObserver.with_args(reduce_range=False)

        print('Quantize CNN per Channel')
        self.qconfig_cnn_qat = QConfig(
            # activation=FakeQuantize.with_args(
            #     observer=MovingAverageMinMaxObserver, reduce_range=True
            # )
            activation=default_fake_quant if args.quantize_activations else default_placeholder_observer,  # nn.Identity,
            weight=per_channel_quant if args.quantize_weights else nn.Identity
            # weight=default_per_channel_weight_fake_quant if args.quantize_weights else nn.Identity
        )

        self.n_rollouts = 0

        self.not_quantize_last = args.not_quantize_last

    def _on_rollout_start(self) -> None:
        # rollout по факту каждый шаг происходит
        if self.quant_on is True and self.n_calls % 2000 == 0:
            print('Quantization on rollout start callback', self.n_calls, self.n_rollouts)

        self.n_rollouts += 1

    def _on_step(self) -> bool:
        if self.quant_on is False and self.n_calls >= self.quantize_start_step:
            if self.load_best_nq:
                print('Loading best pruned model...')
                self.model = tools.smart_loader(
                    self.model, self.best_model_save_path, 'best_nq_model',
                    self.alg, self.quantize_activations
                )

                # tools.test_model(self.model, args, eval_env)

            self._switch_on_quantization()
            self.quant_on = True

        if self.quant_end is False and self.n_calls >= self.quantize_end_step:
            print('>' * 50)
            print('Switching off quantization')
            self._switch_off_quantization()
            print('>' * 50)
            self.quant_end = True

        return True

    def _on_training_start(self) -> None:
        pass

    def _switch_off_quantization(self) -> None:
        if self.alg == 'sac':
            self.model.policy.actor.latent_pi.apply(torch.ao.quantization.disable_observer)
            self.model.policy.actor.mu.apply(torch.ao.quantization.disable_observer)
            self.model.policy.actor.log_std.apply(torch.ao.quantization.disable_observer)
        elif self.alg == 'dqn':
            self.model.q_net.features_extractor.cnn.apply(torch.ao.quantization.disable_observer)
            self.model.q_net.features_extractor.linear.apply(torch.ao.quantization.disable_observer)
            self.model.q_net.q_net.apply(torch.ao.quantization.disable_observer)
        else:
            print('Incorrext alg')
    def _switch_on_quantization(self) -> None:
        print('Quantization on start (switch on)')

        '''if self.alg == 'ppo':
            already_pruned = torch.nn.utils.prune.is_pruned(self.model.policy.mlp_extractor.policy_net)
        elif self.alg == 'sac':
            already_pruned = torch.nn.utils.prune.is_pruned(self.model.policy.actor.latent_pi)
        elif self.alg == 'dqn':
            already_pruned = torch.nn.utils.prune.is_pruned(self.model.q_net.features_extractor.cnn)
        else:
            print(f'Warning: unknown algorithm {self.alg}')
            already_pruned = False

        print(f'Check if model is already pruned: {already_pruned}')'''

        already_pruned = is_pruned(self.alg, self.model)

        if already_pruned:
            masks = tmp_remove_parametrization(self.alg, self.model)

        if self.alg == 'ppo':
            self.model.policy.mlp_extractor.policy_net.eval()
            self.model.policy.mlp_extractor.policy_net = QuantizedActor(self.model.policy.mlp_extractor.policy_net)
            self.model.policy.mlp_extractor.policy_net.qconfig = self.qconfig_qat # torch.quantization.get_default_qat_qconfig('fbgemm')
            print('MLP extractor config:', self.model.policy.mlp_extractor.policy_net.qconfig)
            torch.quantization.prepare_qat(self.model.policy.mlp_extractor.policy_net, inplace=True)
            self.model.policy.mlp_extractor.policy_net.train()

            self.model.policy.action_net.eval()
            self.model.policy.action_net = QuantizedActor(self.model.policy.action_net)
            self.model.policy.action_net.qconfig = self.qconfig_qat # torch.quantization.get_default_qat_qconfig('fbgemm')
            print('MLP extractor config:', self.model.policy.action_net.qconfig)
            torch.quantization.prepare_qat(self.model.policy.action_net, inplace=True)
            self.model.policy.action_net.train()
        elif self.alg == 'sac':
            self.model.policy.actor.latent_pi.eval()
            self.model.policy.actor.latent_pi = QuantizedActor(self.model.policy.actor.latent_pi)
            self.model.policy.actor.latent_pi.qconfig = self.qconfig_qat # torch.quantization.get_default_qat_qconfig('fbgemm')

            if self.quantize_activations:
                print('Model.actor.latent_pi:', self.model.policy.actor.latent_pi)

                self.model.policy.actor.latent_pi.model_fp32 = torch.ao.quantization.fuse_modules(
                    self.model.policy.actor.latent_pi.model_fp32,
                    [['0', '1'], ['2', '3']]
                )

            print('MLP extractor.latent_pi config:', self.model.policy.actor.latent_pi.qconfig)
            torch.quantization.prepare_qat(self.model.policy.actor.latent_pi, inplace=True)
            self.model.policy.actor.latent_pi.train()

            self.model.policy.actor.mu.eval()
            self.model.policy.actor.mu = QuantizedActor(self.model.policy.actor.mu)
            self.model.policy.actor.mu.qconfig = self.qconfig_qat # torch.quantization.get_default_qat_qconfig('fbgemm')

            if self.quantize_activations:
                print('Model.actor.mu:', self.model.policy.actor.mu)
                # Тут просто Linear нечего квантизовать в активациях

            print('MLP extractor.mu config:', self.model.policy.actor.mu.qconfig)
            torch.quantization.prepare_qat(self.model.policy.actor.mu, inplace=True)
            self.model.policy.actor.mu.train()

            self.model.policy.actor.log_std.eval()
            self.model.policy.actor.log_std = QuantizedActor(self.model.policy.actor.log_std)
            self.model.policy.actor.log_std.qconfig = self.qconfig_qat # torch.quantization.get_default_qat_qconfig('fbgemm')

            if self.quantize_activations:
                print('Model.actor.log_std:', self.model.policy.actor.log_std)
                # Тут просто Linear нечего квантизовать в активациях

            print('MLP extractor.log_std config:', self.model.policy.actor.log_std.qconfig)
            torch.quantization.prepare_qat(self.model.policy.actor.log_std, inplace=True)
            self.model.policy.actor.log_std.train()
        elif self.alg == 'dqn':
            # torch.ao.quantization.fuse_modules(model_fp32, [['conv', 'relu']])
            # torch.ao.quantization.get_default_qat_qconfig
            self.model.q_net.features_extractor.cnn.eval()
            self.model.q_net.features_extractor.cnn = QuantizedActor(self.model.q_net.features_extractor.cnn)

            if self.quantize_activations:
                print('Model.fe.CNN:', self.model.q_net.features_extractor.cnn)

                if self.res is not True:
                    self.model.q_net.features_extractor.cnn.model_fp32 = torch.ao.quantization.fuse_modules(
                        self.model.q_net.features_extractor.cnn.model_fp32,
                        [['0', '1'], ['2', '3'], ['4', '5']]
                    )
                else:
                    print('ResNet not implemented yet')

                    raise NotImplementedError()

            self.model.q_net.features_extractor.cnn.qconfig = self.qconfig_cnn_qat
            # self.model.q_net.features_extractor.cnn.model_fp32.qconfig = self.qconfig_cnn_qat  # torch.quantization.get_default_qat_qconfig('fbgemm')
            print('CNN q_net.features_extractor.cnn config:', self.model.q_net.features_extractor.cnn.qconfig)
            torch.quantization.prepare_qat(self.model.q_net.features_extractor.cnn, inplace=True)
            self.model.q_net.features_extractor.cnn.train()

            print('DEBUG: CNN quantized:', self.model.q_net.features_extractor.cnn)

            self.model.q_net.features_extractor.linear.eval()
            self.model.q_net.features_extractor.linear = QuantizedActor(self.model.q_net.features_extractor.linear)

            if self.quantize_activations:
                print('Model.fe.Linear:', self.model.q_net.features_extractor.linear)
                self.model.q_net.features_extractor.linear = torch.ao.quantization.fuse_modules(
                    self.model.q_net.features_extractor.linear, ['model_fp32.0', 'model_fp32.1']
                 )

            self.model.q_net.features_extractor.linear.qconfig = self.qconfig_qat  # torch.quantization.get_default_qat_qconfig('fbgemm')
            print('MLP q_net.features_extractor.linear config:', self.model.q_net.features_extractor.linear.qconfig)
            torch.quantization.prepare_qat(self.model.q_net.features_extractor.linear, inplace=True)
            self.model.q_net.features_extractor.linear.train()

            if self.not_quantize_last is False:
                self.model.q_net.q_net.eval()
                self.model.q_net.q_net = QuantizedActor(self.model.q_net.q_net)

                if self.quantize_activations:
                    print('Model.fe.Linear:', self.model.q_net.q_net)
                    # Тут нечего фьюзить так-как только Linear
                    # self.model.q_net.q_net = torch.ao.quantization.fuse_modules(
                    #     self.model.q_net.q_net, [['conv', 'relu'], ['linear', 'relu']]
                    # )

                self.model.q_net.q_net.qconfig = self.qconfig_qat  # torch.quantization.get_default_qat_qconfig('fbgemm')
                print('MLP q_net.q_net config:', self.model.q_net.q_net.qconfig)
                torch.quantization.prepare_qat(self.model.q_net.q_net, inplace=True)
                self.model.q_net.q_net.train()
            else:
                print('NOT QUANTIZE LAST LAYER')
        else:
            print('This algorithm isn\'t supported')
            raise Exception()

        if already_pruned:
            set_masks(self.alg, self.model, masks)

    def _on_training_end(self) -> None:
        print(f'Quantization on end. Total calls: {self.n_calls}')