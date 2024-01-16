import torch
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from custom_bns.dynamic_bn import DynamicBN


features_return_nodes = {
    'resnet50': {
        'fc': 'out',
        'layer4.2.add': 'out_encoder',
        'flatten': 'out_encoder_flatten',
        'layer1.0.bn2': 'layer1.0.bn2',
        'layer2.2.bn2': 'layer2.2.bn2'
    },
    'wideresnet28': {
        'fc': 'out',
        'bn1': 'out_encoder',
        'relu': 'out_encoder_relu',
        'view': 'out_encoder_flatten',
        'block1.layer.0.bn2': 'block1.layer.0.bn2',
        'block2.layer.2.bn2': 'block2.layer.2.bn2'
    }
}

class FeaturesGetterWrapper(torch.nn.Module):
    def __init__(self, model, return_nodes=None):
        super(FeaturesGetterWrapper, self).__init__()
        
        self.model = create_feature_extractor(
            model, return_nodes=return_nodes, 
            tracer_kwargs={'leaf_modules': [torch.nn.BatchNorm2d]})

    def forward(self, x):
        return self.model(x)
    
    @staticmethod
    def get_node_names(model):
        train_nodes, eval_nodes = get_graph_node_names(model,
                                                       tracer_kwargs={'leaf_modules': [torch.nn.BatchNorm2d]})
        # print(train_nodes)
        # import sys
        # sys.exit()
        
        return train_nodes