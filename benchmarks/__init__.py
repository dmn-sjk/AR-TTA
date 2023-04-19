from .cladc import get_cladc_benchmark 
from .cifar10c import get_cifar10c_benchmark
from .shift import get_shift_benchmark


def get_benchmark(cfg):
    if cfg['dataset'] == "cifar10c":
        return get_cifar10c_benchmark(cfg)
    elif cfg['dataset'] == "clad":
        return get_cladc_benchmark(cfg)
    elif cfg['dataset'] == "shift":
        return get_shift_benchmark(cfg)
    else:
        raise ValueError("Unknown benchmark name")


### ----TESTING----
if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt
    # from continual_adaptation.constants.cifar import CORRUPTIONS
    from constants.cifar import CORRUPTIONS
    from utils.config_parser import ConfigParser
            
    def display_data(imgs, targets):
        nrows, ncols = 3, 3
        fig, axs = plt.subplots(figsize=(15, 15), nrows=nrows, ncols=ncols)
        image_i = 0

        for row in range(nrows):
            for col in range(ncols):
                max_val = imgs[image_i].max().item()
                min_val = imgs[image_i].min().item()
                assert max_val <= 1.0
                assert min_val >= 0.0

                axs[row, col].imshow(imgs[image_i].permute(1, 2, 0))
                axs[row, col].title.set_text(str(targets[image_i]))
                image_i += 1

        fig.tight_layout(pad=2.0)
        plt.show()

    cfg = ConfigParser(mode="tta").get_config()
    
    benchmark = get_benchmark(cfg)

    for i, experience in enumerate(benchmark.train_stream):
        # ...as well as the task_label
        current_training_set = experience.dataset
        print('Task {}'.format(experience.task_label))
        print('This task contains', len(current_training_set), 'training examples')

        # TODO: [0] because CLAD has only one test stream, the question is what about e.g., shift?
        current_test_set = benchmark.test_stream[0].dataset
        print('This task contains', len(current_test_set), 'test examples')
        
        # print(current_training_set[0])
        
        if i == 0:
            idxs_train = random.sample(range(0, len(current_training_set)), 9)
            idxs_test = random.sample(range(0, len(current_test_set)), 9)
            
            train_batch = []
            for idx in idxs_train:
                train_batch.append(current_training_set[idx])
            test_batch = []
            for idx in idxs_test:
                test_batch.append(current_test_set[idx])
                
            display_data(imgs=[sample[0] for sample in train_batch],
                         targets=[sample[1] for sample in train_batch])
            display_data(imgs=[sample[0] for sample in test_batch],
                         targets=[sample[1] for sample in test_batch])
            