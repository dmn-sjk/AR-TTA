from copy import deepcopy
import torch
import torch.nn as nn

from utils.cotta_transforms import get_tta_transforms
from utils.math import update_ema_variables, softmax_entropy
from .tta_method import TTAMethod



class CoTTA(TTAMethod, method_name='cotta'):
    """CoTTA adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        
        self.model_state = deepcopy(self.model.state_dict())

        self.model_ema = deepcopy(self.model)
        self.model_ema.eval()
        self.model_ema.requires_grad_(False)
        for param in self.model_ema.parameters():
            param.detach_()
        
        self.model_anchor = deepcopy(self.model)
        self.model_anchor.eval()
        self.model_anchor.requires_grad_(False)
        for param in self.model_anchor.parameters():
            param.detach_()
        
        self.transform = get_tta_transforms(img_size=self.cfg['img_size'])    
        self.mt = cfg['mt']
        self.rst = cfg['rst']
        self.ap = cfg['ap']

    @torch.enable_grad()
    def forward(self, x):
        outputs = self.model(x)
        # Teacher Prediction
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        standard_ema = self.model_ema(x)
        
        # Threshold choice discussed in supplementary
        if anchor_prob.mean(0)<self.ap:
            # Augmentation-averaged Prediction
            N = 32
            outputs_emas = []
            for i in range(N):
                outputs_  = self.model_ema(self.transform(x)).detach()
                outputs_emas.append(outputs_)

            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema
            
        # Student update
        loss = (softmax_entropy(outputs, outputs_ema)).mean(0)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Teacher update
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.mt)
        # Stochastic restore
        if True:
            for nm, m  in self.model.named_modules():
                if 'model.' in nm and 'model.' not in list(self.model_state.keys())[10]:
                    nm = nm.replace('model.', '')
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<self.rst).float().cuda() 
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
        return outputs_ema


    def collect_params(self):
        """Collect all trainable parameters.
        Walk the model's modules and collect all parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if True:#isinstance(m, nn.BatchNorm2d): collect all 
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        self.model.train()
        # disable grad, to (re-)enable only what we update
        self.model.requires_grad_(False)
        # enable all trainable
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            else:
                m.requires_grad_(True)
