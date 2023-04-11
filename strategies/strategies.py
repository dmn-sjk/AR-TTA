from avalanche.training import Naive

def get_strategy(args):
    return None

    if args.model == 'resnet18':
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown model name: {args.model}")

    model.fc = Linear(model.fc.in_features, len(clad.SODA_CATEGORIES), bias=True)

    if args.pretrained_model_path is not None:
        model.load_state_dict(torch.load(args.pretrained_model_path))
        
    criterion = torch.nn.CrossEntropyLoss()
    batch_size = 10

    tented_model = None

    if args.method == "finetune":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        strategy = Naive(
            model, optimizer, criterion, train_mb_size=batch_size, train_epochs=1, eval_mb_size=128,
            device=device, evaluator=eval_plugin, plugins=plugins, eval_every=-1)
    elif args.method == "frozen":
        strategy = FrozenModel(
            model, train_mb_size=batch_size, eval_mb_size=128,
            device=device, evaluator=eval_plugin, plugins=plugins, eval_every=-1)
    elif args.method == "tent":
        # model, params = get_tented_model_and_params(model)
        # optimizer = torch.optim.SGD(params, lr=1e-3)
        #
        # from tent import softmax_entropy
        #
        # def softmax_entropy_loss(x: torch.Tensor, _):
        #     return softmax_entropy(x).mean(0)
        #
        # criterion = softmax_entropy_loss
        #
        # plugins.append(TentPlugin())
        #
        # strategy = Naive(
        #     model, optimizer, criterion, train_mb_size=batch_size, train_epochs=1, eval_mb_size=128,
        #     device=device, evaluator=eval_plugin, plugins=plugins, eval_every=-1)

        # plugins.append(TentPlugin(lr=1e-3))

        import tent
        model = tent.configure_model(model)
        params, param_names = tent.collect_params(model)
        optimizer = torch.optim.SGD(params, lr=1e-3)
        tented_model = tent.Tent(model, optimizer)

        strategy = FrozenModel(
            tented_model, train_mb_size=batch_size, eval_mb_size=32,
            device=device, evaluator=eval_plugin, plugins=plugins, eval_every=-1)
    else:
        raise ValueError("Unknown method")