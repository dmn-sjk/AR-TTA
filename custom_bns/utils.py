import torch
from torch import nn


def replace_bn(model: nn.Module, BN_module: nn.Module, n_repalced=0, number_to_replace=None, **abn_kwargs):
    copy_keys = ['eps', 'momentum', 'affine']

    for mod_name, target_mod in model.named_children():
        # print(mod_name)
        if number_to_replace is not None and n_repalced == number_to_replace:
            # print(n_repalced)
            return n_repalced
        
        if isinstance(target_mod, nn.BatchNorm2d) or isinstance(target_mod, nn.SyncBatchNorm):
            # print(target_mod)
            # print(f" Insert {BN_module.__name__} to ", mod_name)
            n_repalced += 1
            
            new_mod = BN_module(
                target_mod.num_features,
                **{k: getattr(target_mod, k) for k in copy_keys},
                **abn_kwargs,
            )
            new_mod.load_state_dict(target_mod.state_dict())
            new_mod.track_running_stats = False
            setattr(model, mod_name, new_mod)
        else:
            n_repalced = replace_bn(
                target_mod, BN_module, n_repalced=n_repalced, number_to_replace=number_to_replace, **abn_kwargs)
    return n_repalced

def count_bn(model: nn.Module, BN_module):
    cnt = 0
    for n, m in model.named_modules():
        if isinstance(m, BN_module):
            cnt += 1
    return cnt


if __name__ == "__main__":
    def compare_bn(bn1, bn2):
        err = False
        if not torch.allclose(bn1.running_mean, bn2.running_mean):
            print('Diff in running_mean: {} vs {}'.format(
                bn1.running_mean, bn2.running_mean))
            err = True

        if not torch.allclose(bn1.running_var, bn2.running_var):
            print('Diff in running_var: {} vs {}'.format(
                bn1.running_var, bn2.running_var))
            err = True

        if bn1.affine and bn2.affine:
            if not torch.allclose(bn1.weight, bn2.weight):
                print('Diff in weight: {} vs {}'.format(
                    bn1.weight, bn2.weight))
                err = True

            if not torch.allclose(bn1.bias, bn2.bias):
                print('Diff in bias: {} vs {}'.format(
                    bn1.bias, bn2.bias))
                err = True

        if not err:
            print('All parameters are equal!')

    my_bn = DynamicBN(3, affine=True)
    bn = nn.BatchNorm2d(3, affine=True)

    compare_bn(my_bn, bn)  # weight and bias should be different
    # Load weight and bias
    my_bn.load_state_dict(bn.state_dict())
    compare_bn(my_bn, bn)

    # Run train
    for _ in range(10):
        scale = torch.randint(1, 10, (1,)).float()
        bias = torch.randint(-10, 10, (1,)).float()
        x = torch.randn(10, 3, 100, 100) * scale + bias
        out1 = my_bn(x)
        out2 = bn(x)
        compare_bn(my_bn, bn)

        # torch.allclose(out1, out2)
        # print(f"All close: {torch.allclose(out1, out2)}")
        print('Max diff: ', (out1 - out2).abs().max())

    # Run eval
    my_bn.eval()
    bn.eval()
    my_bn.turn_dynamic_bn_on()
    for _ in range(10):
        scale = torch.randint(1, 10, (1,)).float()
        bias = torch.randint(-10, 10, (1,)).float()
        x = torch.randn(10, 3, 100, 100) * scale + bias
        out1 = my_bn(x)
        out2 = bn(x)
        compare_bn(my_bn, bn)

        # print(f"All close: {torch.allclose(out1, out2)}")
        print('Max diff: ', (out1 - out2).abs().max())