from torch import nn
import torch

from utils import InstanceAwareBatchNorm2d, InstanceAwareBatchNorm1d, PBRS


class NOTE(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, memory_size=64, num_classes=10):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.adapt = True
        self.mem = PBRS(capacity=memory_size, num_classes=num_classes)

    def forward(self, x):

        if self.adapt:
            for _ in range(self.steps):
                outputs = forward_and_adapt(x, self.model, self.optimizer, self.mem)
        else:
            outputs = self.model(x)

        return outputs
    

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, memory):
    with torch.no_grad():

        model.eval()

        # if conf.args.memory_type in ['FIFO', 'Reservoir']:
        #     self.mem.add_instance(current_sample)

        # elif conf.args.memory_type in ['PBRS']:

        logits = model(x)
        pseudo_cls = logits.argmax(1)
        for i in range(logits.shape[0]):
            img = x[i]
            cls = pseudo_cls[i].item()
            memory.add_instance(img, cls)

    # setup models
    model.train()

    # if len(x) == 1:  # avoid BN error
    #     model.eval()

    memory_samples = memory.get_memory()
    memory_samples = torch.stack(memory_samples)

    outputs = model(memory_samples) # update bn stats

    loss = hloss(outputs)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return logits

@torch.jit.script
def hloss(x: torch.Tensor) -> torch.Tensor:
    # as in NOTE code, I don't think the mean of every value of multiplication of softmax and logsoftmax values is correct
    # it lacks the sum in class dimension
    softmax = nn.functional.softmax(x, dim=1)
    entropy = -softmax * torch.log(softmax+1e-6)
    return entropy.mean() 

def configure_model(model, bn_momentum):
    """Configure model for use with note."""
    # assume this note param to be True, it was by default
    use_learned_stats=True
    
    model.train()

    for param in model.parameters():  # initially turn off requires_grad for all
        param.requires_grad = False

    # configure norm for tent updates: enable grad + force batch statisics
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
        # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

            if use_learned_stats:
                module.track_running_stats = True
                module.momentum = bn_momentum
            else:
                # With below, this module always uses the test batch statistics (no momentum)
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

            module.weight.requires_grad_(True)
            module.bias.requires_grad_(True)

        elif isinstance(module, nn.InstanceNorm1d) or isinstance(module, nn.InstanceNorm2d): #ablation study
            module.weight.requires_grad_(True)
            module.bias.requires_grad_(True)

        if isinstance(module, InstanceAwareBatchNorm2d) or isinstance(module, InstanceAwareBatchNorm1d):
            for param in module.parameters():
                param.requires_grad = True

    return model
