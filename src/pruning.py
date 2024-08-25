import torch
import copy

import torch.nn.utils.prune as prune
from stable_baselines3.common.callbacks import BaseCallback

from typing import List

def prune_mlp(model, sparsities):
    print('Prune MLP:', sparsities)
    # print('model.named_modules()', list(model.named_modules()))
    # Here we make local pruning
    j = 0

    for i, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            if hasattr(module, 'weight_mask'):
                zeros = torch.sum(module.weight_mask == 0)
            else:
                # zeros = torch.sum(torch.abs(module.weight) < 1e-10).cpu().item()
                zeros = 0

            total = module.weight.numel()
            current_sparsity = zeros / total

            sparsity_setting = (sparsities[j] - current_sparsity) / (1 - current_sparsity)
            sparsity_setting = sparsity_setting.item()

            print(f'Prune {name} from {current_sparsity:.5f} to {sparsities[j]:.5f} by add {sparsity_setting:.5f}...')
            # amount is estimated from the rest non-zero elements
            print('Type:', type(sparsity_setting))

            if sparsity_setting < 0.0:
                print(f'Warning: attempt to prune from {current_sparsity} to {sparsities[j]} by add {sparsity_setting}')
                sparsity_setting = 0.0
                j += 1
                continue

            prune.l1_unstructured(module, name='weight', amount=sparsity_setting)
            # for keeping Polyak sparse as well
            # module.weight_orig = torch.nn.Parameter(module.weight_orig * module.weight_mask)
            # module.weight_orig.mul_(module.weight_mask)
            module.weight_orig.data = module.weight_orig * module.weight_mask
            # for keeping order of registered parameters (for Polyak shift tau)
            prune.l1_unstructured(module, name='bias', amount=0.0)
            # module.bias_orig = torch.nn.Parameter(module.bias_orig * module.bias_mask)
            # module.bias_orig.mul_(module.bias_mask)
            module.bias_orig.data = module.bias_orig * module.bias_mask

            j += 1



def prune_mlp_remove_parametrization(model):
    masks = {}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            masks.update({name: [
                copy.deepcopy(module.weight_mask),
                copy.deepcopy(module.bias_mask)
            ]})

            # make the pruning permanent
            prune.remove(module, name='weight')
            prune.remove(module, name='bias')

    return masks


def prune_cnn(model, sparsities):
    # TBD
    print('Prune CNN:', sparsities)
    # print('model.named_modules()', list(model.named_modules()))
    # Here we make local pruning
    j = 0

    # Maybe to change to sequential?
    for i, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, torch.nn.Conv2d):
            if hasattr(module, 'weight_mask'):
                zeros = torch.sum(module.weight_mask == 0)
            else:
                zeros = 0  # torch.sum(torch.abs(module.weight) < 1e-10).cpu().item()

            total = module.weight.numel()

            # так-как иногда число уже отпрюненных может превышать цель, то мы скипаем этот шаг
            current_sparsity = zeros / total

            # наверное можно было бы опираться и на предущие разреженности
            sparsity_setting = (sparsities[j] - current_sparsity) / (1 - current_sparsity)
            sparsity_setting = sparsity_setting.item()

            print(f'Prune {name} from {current_sparsity:.5f} to {sparsities[j]:.5f} by add {sparsity_setting:.5f}...')
            # amount is estimated from the rest non-zero elements
            
            if sparsity_setting < 0.0:
                print(f'Warning: attempt to prune from {current_sparsity} to {sparsities[j]} by add {sparsity_setting}')
                sparsity_setting = 0.0
                j += 1
                continue

            prune.l1_unstructured(module, name='weight', amount=sparsity_setting)
            # for keeping Polyak sparse as well
            # module.weight_orig = torch.nn.Parameter(module.weight_orig * module.weight_mask)
            module.weight_orig.data = module.weight_orig * module.weight_mask
            # for keeping order of registered parameters (for Polyak shift tau)
            prune.l1_unstructured(module, name='bias', amount=0.0)
            # module.bias_orig = torch.nn.Parameter(module.bias_orig * module.bias_mask)
            module.bias_orig.data = module.bias_orig * module.bias_mask

            j += 1


def prune_cnn_remove_parametrization(model):
    masks = {}

    # TBD
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            masks.update({name: [
                copy.deepcopy(module.weight_mask),
                copy.deepcopy(module.bias_mask)
            ]})

            # make the pruning permanent
            prune.remove(module, name='weight')
            prune.remove(module, name='bias')

    return masks


def estimate_sparsity(model):
    print('Sparsity estimation...')

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            zeros = torch.sum(torch.abs(module.weight) < 1e-10)
            total = module.weight.numel()
            print(f'Sparsity of {name} (linear): {zeros} / {total} = {zeros/total:.3f}')

            print('hasattr(Linear, weight_mask):', hasattr(module, "weight_mask"))

            # for processing the case of very small initial pruning, when default zeros exceeds the desired sparsity
            if hasattr(module, "weight_mask"):
                zeros_mask = torch.sum(module.weight_mask == 0)
                total_mask = module.weight_mask.numel()
                print(f'Weight_mask of {name}: {zeros_mask} / {total_mask} = {zeros_mask/total_mask:.3f}')

        elif isinstance(module, torch.nn.Conv2d):
            zeros = torch.sum(torch.abs(module.weight) < 1e-10)
            total = module.weight.numel()
            print(f'Sparsity of {name} (conv2d): {zeros} / {total} = {zeros / total:.3f}')

            print('hasattr(Conv2d, weight_mask):', hasattr(module, "weight_mask"))

            if hasattr(module, "weight_mask"):
                zeros_mask = torch.sum(module.weight_mask == 0)
                total_mask = module.weight_mask.numel()
                print(f'Weight_mask of {name}: {zeros_mask} / {total_mask} = {zeros_mask / total_mask:.3f}')


class PruningCallback(BaseCallback):
    # for pruning details see https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
    def __init__(self, alg, pruning_sparsity_schedule, common_sparsities_schedule, res, verbose: int = 1):
        super().__init__(verbose)

        self.alg = alg
        self.res = res

        self.pruning_schedule = pruning_sparsity_schedule
        self.common_sparsities_schedule = common_sparsities_schedule
        self.last_sparsity = 0.0

        # print(self.pruning_schedule.keys())
        self.pruning_end_step = max(self.pruning_schedule.keys())

        self.finished = False

        print(f'Pruning end step: {self.pruning_end_step}')


    def estimate_all_nns(self):
        if self.alg == 'ppo':
            estimate_sparsity(self.model.policy.mlp_extractor.policy_net)
            estimate_sparsity(self.model.policy.action_net)
        elif self.alg == 'sac':
            estimate_sparsity(self.model.policy.actor.latent_pi)
            estimate_sparsity(self.model.policy.actor.mu)
            estimate_sparsity(self.model.policy.actor.log_std)
        elif self.alg == 'dqn':
            estimate_sparsity(self.model.q_net.features_extractor.cnn)
            estimate_sparsity(self.model.q_net.features_extractor.linear)
            estimate_sparsity(self.model.q_net.q_net)

    def _on_step(self) -> bool:
        self.logger.record("Common_sparsity:", self.last_sparsity)

        if self.n_calls > 0 and self.n_calls in self.pruning_schedule.keys():
            print('='*80)
            print(f'Pruning on step callback: step: {self.n_calls}, sparsity: '
                  f'{self.common_sparsities_schedule[self.n_calls]} {self.pruning_schedule[self.n_calls]}')

            self.last_sparsity = self.common_sparsities_schedule[self.n_calls]
            print(f'New common sparsity: {self.common_sparsities_schedule[self.n_calls]}')
            # self.logger.record("Common_sparsity:", self.pruning_schedule[self.n_calls] )

            if self.alg == 'ppo':
                prune_mlp(self.model.policy.mlp_extractor.policy_net, self.pruning_schedule[self.n_calls][:-1])
                prune_mlp(self.model.policy.action_net, self.pruning_schedule[self.n_calls][-1:])

                estimate_sparsity(self.model.policy.mlp_extractor.policy_net)
                estimate_sparsity(self.model.policy.action_net)
            elif self.alg == 'sac':
                prune_mlp(self.model.policy.actor.latent_pi, self.pruning_schedule[self.n_calls][:-1])
                prune_mlp(self.model.policy.actor.mu, self.pruning_schedule[self.n_calls][-1:])
                # TBD Are mu and log_std the same nets? Seems yes by logs
                prune_mlp(self.model.policy.actor.log_std, self.pruning_schedule[self.n_calls][-1:])

                estimate_sparsity(self.model.policy.actor.latent_pi)
                estimate_sparsity(self.model.policy.actor.mu)
                estimate_sparsity(self.model.policy.actor.log_std)
            elif self.alg == 'dqn':
                # TBD
                if self.res:
                    prune_cnn(self.model.q_net.features_extractor.cnn, self.pruning_schedule[self.n_calls][:-2])
                    prune_mlp(self.model.q_net.features_extractor.linear, self.pruning_schedule[self.n_calls][-2:-1])
                    prune_mlp(self.model.q_net.q_net, self.pruning_schedule[self.n_calls][-1:])
                else:
                    prune_cnn(self.model.q_net.features_extractor.cnn, self.pruning_schedule[self.n_calls][0:3])
                    prune_mlp(self.model.q_net.features_extractor.linear, self.pruning_schedule[self.n_calls][3:4])
                    prune_mlp(self.model.q_net.q_net, self.pruning_schedule[self.n_calls][4:])

                estimate_sparsity(self.model.q_net.features_extractor.cnn)
                estimate_sparsity(self.model.q_net.features_extractor.linear)
                estimate_sparsity(self.model.q_net.q_net)
            else:
                print('This algorithm isn\'t supported')
                raise Exception()

        if self.finished is False and self.n_calls > self.pruning_end_step:
            self.finished = True
            self.estimate_all_nns()

        return True

    def _on_training_end(self) -> None:
        print('Training on end')
        self.remove_parametrization()
        self.estimate_all_nns()


    def remove_parametrization(self) -> None:
        print(f'Pruning on end. Total calls: {self.n_calls}')

        if self.alg == 'ppo':
            prune_mlp_remove_parametrization(self.model.policy.mlp_extractor.policy_net)
            prune_mlp_remove_parametrization(self.model.policy.action_net)
        elif self.alg == 'sac':
            prune_mlp_remove_parametrization(self.model.policy.actor.latent_pi)
            prune_mlp_remove_parametrization(self.model.policy.actor.mu)
            prune_mlp_remove_parametrization(self.model.policy.actor.log_std)
        elif self.alg == 'dqn':
            # TBD
            prune_cnn_remove_parametrization(self.model.q_net.features_extractor.cnn)
            prune_mlp_remove_parametrization(self.model.q_net.features_extractor.linear)
            prune_mlp_remove_parametrization(self.model.q_net.q_net)
        else:
            print('This algorithm isn\'t supported')
            raise Exception()


def tmp_remove_parametrization(alg, model) -> List:
    print(f'Temporary removing mask parametrization')
    masks = []

    if alg == 'ppo':
        masks.append(prune_mlp_remove_parametrization(model.policy.mlp_extractor.policy_net))
        masks.append(prune_mlp_remove_parametrization(model.policy.action_net))
    elif alg == 'sac':
        masks.append(prune_mlp_remove_parametrization(model.policy.actor.latent_pi))
        masks.append(prune_mlp_remove_parametrization(model.policy.actor.mu))
        masks.append(prune_mlp_remove_parametrization(model.policy.actor.log_std))
    elif alg == 'dqn':
        # TBD
        masks.append(prune_cnn_remove_parametrization(model.q_net.features_extractor.cnn))
        masks.append(prune_mlp_remove_parametrization(model.q_net.features_extractor.linear))
        masks.append(prune_mlp_remove_parametrization(model.q_net.q_net))
    else:
        print('This algorithm isn\'t supported')
        raise Exception()

    print(masks[0].keys())

    return masks


def set_mask(model, masks):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            print(f'Set mask for module {name}')  # {module.weight.device} {masks[name][0].device}')
            # make the pruning permanent
            prune.custom_from_mask(module, name='weight', mask=masks[name][0])
            prune.custom_from_mask(module, name='bias', mask=masks[name][1])


def set_masks(alg, model, masks, quant=True) -> None:
    print(f'Restore mask parametrization')

    if alg == 'ppo':
        if quant:
            set_mask(model.policy.mlp_extractor.policy_net.model_fp32, masks[0])
            set_mask(model.policy.action_net.model_fp32, masks[1])
        else:
            set_mask(model.policy.mlp_extractor.policy_net, masks[0])
            set_mask(model.policy.action_net, masks[1])
    elif alg == 'sac':
        if quant:
            set_mask(model.policy.actor.latent_pi.model_fp32, masks[0])
            set_mask(model.policy.actor.mu.model_fp32, masks[1])
            set_mask(model.policy.actor.log_std.model_fp32, masks[2])
        else:
            set_mask(model.policy.actor.latent_pi, masks[0])
            set_mask(model.policy.actor.mu, masks[1])
            set_mask(model.policy.actor.log_std, masks[2])
    elif alg == 'dqn':
        if quant:
            set_mask(model.q_net.features_extractor.cnn.model_fp32, masks[0])
            set_mask(model.q_net.features_extractor.linear.model_fp32, masks[1])
            set_mask(model.q_net.q_net.model_fp32, masks[2])
        else:
            set_mask(model.q_net.features_extractor.cnn, masks[0])
            set_mask(model.q_net.features_extractor.linear, masks[1])
            set_mask(model.q_net.q_net, masks[2])
    else:
        print('This algorithm isn\'t supported')
        raise Exception()


def is_pruned(alg, model):
    if alg == 'ppo':
        already_pruned = torch.nn.utils.prune.is_pruned(model.policy.mlp_extractor.policy_net)
    elif alg == 'sac':
        already_pruned = torch.nn.utils.prune.is_pruned(model.policy.actor.latent_pi)
    elif alg == 'dqn':
        already_pruned = torch.nn.utils.prune.is_pruned(model.q_net.features_extractor.cnn)
    else:
        print(f'Warning: unknown algorithm {alg}')
        already_pruned = False

    print(f'Check if model is already pruned: {already_pruned}')

    return already_pruned
