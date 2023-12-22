import torch
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

# _features_return_nodes = {
#     'deeplabv2': {
#         '1.layer1.2.relu_2': 'layer1',
#         '1.layer2.3.relu_2': 'layer2',
#         '1.layer3.22.relu_2': 'layer3',
#         '1.layer4.2.relu_2': 'layer4',
#         '1.interpolate': 'logits'
#     },
#     'deeplabv3': {
#         '1.backbone.layer1.2.relu_2': 'layer1',
#         '1.backbone.layer2.3.relu_2': 'layer2',
#         '1.backbone.layer3.5.relu_2': 'layer3',
#         # '1.backbone.layer4.2.relu_2': 'layer4',
#         '1.classifier.3': 'layer4',
#         '1.interpolate': 'logits'
#     }
# }

class FeaturesGetterWrapper(torch.nn.Module):
    def __init__(self, model, return_nodes=None):
        # train_nodes, eval_nodes = get_graph_node_names(model)
        # print(train_nodes)
        # import sys
        # sys.exit()
        
        super(FeaturesGetterWrapper, self).__init__()
        
        self.model = create_feature_extractor(
            model, return_nodes=return_nodes)

    def forward(self, x):
        return self.model(x)