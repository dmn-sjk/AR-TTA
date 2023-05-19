from torch import nn


class IntermediateFeaturesGetter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._features = {}
        
    def forward(self, x):
        return self.model(x)

    def register_features(self, layer_name):
        if not hasattr(self.model, layer_name):
            raise ValueError(f"There is no layer {layer_name} in the model")

        def hook(model, input, output):
            self._features[layer_name] = output.detach()

        getattr(self.model, layer_name).register_forward_hook(hook)

    def get_features(self, layer_name):
        if layer_name not in self._features.keys():
            raise ValueError(f"There is no features from layer {layer_name} yet")
            
        return self._features[layer_name]


### TESTING ###
if __name__ == "__main__":
    from torch.nn import functional as F

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.cl1 = nn.Linear(25, 60)
            self.cl2 = nn.Linear(60, 16)
            self.fc1 = nn.Linear(16, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
            
        def forward(self, x):
            x = F.relu(self.cl1(x))
            x = F.relu(self.cl2(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.log_softmax(self.fc3(x), dim=1)
            return x
    
    import torch
    import torchvision
        
    # model = MyModel()
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    
    model = IntermediateFeaturesGetter(model)
    model.register_features('avgpool')
    
    x = torch.rand(size=(10,3, 224,224))
    
    # x = torch.full(size=(1000,3), fill_value=0, dtype=torch.float32)
    # y = torch.full(size=(1000,3), fill_value=0, dtype=torch.float32)
    # y[0] = 1.
    # y[100] = 1001.
    # print(nn.functional.mse_loss(x, y))
    # print(((torch.sum((y - x)**2)**(1/2))**2) / 3000)
    output = model(x)
    print(model.get_features('avgpool').shape)
    
    # from robustbench.utils import load_model
    
    # model = load_model('Standard', './models_checkpoints'
    #                         'cifar10', "corruptions")
    
    # device = torch.device("cuda:0")
    # print(model.to(device))