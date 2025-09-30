from torchvision.transforms import transforms



def get_transforms(cfg, train: bool = False):

    if train:
        return transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(
                kernel_size=5, sigma=[0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
        ])

    else:
        return transforms.Compose([
        ])
