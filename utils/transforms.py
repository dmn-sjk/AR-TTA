from torchvision.transforms import transforms


# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

# mean = [0.5, 0.5, 0.5]
# std = [0.5, 0.5, 0.5]

# mean_clad = (0.3540, 0.3393, 0.3285)
# std_clad = (0.1615, 0.1588, 0.1592)

def get_transforms(cfg, train: bool = False):
    # img_resize = (224, 224)

    if train:
        # transforms_train = transforms.Compose([
        #     transforms.ToTensor(),
        #     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # vals already in range (0 - 1)
        #     transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5),
        #                            hue=(-0.2, 0.2)),
        #     transforms.GaussianBlur(5),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomAffine(degrees=10, translate=(0.1, 0.4), scale=(0.9, 1.0)),
        # ])

        return transforms.Compose([
            # transforms.RandomResizedCrop(cfg['img_size'], scale=(0.6, 1.0)),
            # transforms.Normalize(mean_clad, std_clad),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(
                kernel_size=5, sigma=[0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
        ])

    else:
        return transforms.Compose([
            # transforms.Normalize(mean, std)
            # transforms.Normalize(mean_clad, std_clad),
            # transforms.Resize(size=img_resize),
            # transforms.ToTensor(),
        ])
