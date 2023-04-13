import torch
import os
from utils.utils import set_seed
from utils.config_parser import ConfigParser
from benchmarks import get_benchmark
from strategies import get_strategy
import json


def main():
    cfg = ConfigParser(mode="tta").get_config()

    cfg['device'] = torch.device(f"cuda:{cfg['cuda']}"
                                 if torch.cuda.is_available() and cfg['cuda'] >= 0
                                 else "cpu")

    if cfg['seed'] is not None:
        set_seed(cfg['seed'])

    benchmark = get_benchmark(cfg)
    strategy = get_strategy(cfg)

    results_dict = {"train_sequences": {"running_acc": [], "batch_acc": []},
                    "val_sequences": {"acc": [], "refined_acc": []}, # refined for AdaCon, not used here
                    "test_sequences": {"acc": [], "refined_acc": []}}

    for i, experience in enumerate(benchmark.train_stream):
        # strategy.train(experience, eval_streams=[cladc.streams['val_sets']], shuffle=False,
        #                num_workers=cfg['num_workers)

        if i == 0:
            print("Initial eval...")
            results = strategy.eval(benchmark.test_stream, num_workers=cfg['num_workers'])
            print(results)

            if cfg['save_results']:
                metrics = strategy.evaluator.get_all_metrics()
                results_dict["test_sequences"]["acc"]. \
                    append(metrics[f"Top1_Acc_Stream/eval_phase/test_stream/Task000"][1])

        strategy.train(experience, eval_streams=[], shuffle=False,
                       num_workers=cfg['num_workers'])

        results = strategy.eval(benchmark.test_stream, num_workers=cfg['num_workers'])
        print(results)

        if cfg['save_results']:
            metrics = strategy.evaluator.get_all_metrics()

            task_nr = str(i)
            while len(task_nr) < 3:
                task_nr = '0' + task_nr

            results_dict["train_sequences"]["running_acc"]. \
                append(metrics[f"Top1_RunningAcc_Epoch/train_phase/train_stream/Task{task_nr}"][1])
            results_dict["train_sequences"]["batch_acc"]. \
                append(metrics[f"Top1_Acc_MB/train_phase/train_stream/Task{task_nr}"][1])

            results_dict["test_sequences"]["acc"]. \
                append(metrics[f"Top1_Acc_Stream/eval_phase/test_stream/Task000"][1])

    if cfg['save_results']:
        results_path = os.path.join("experiments", f"{cfg['run_name']}_{cfg['method']}")
        if not os.path.isdir(results_path):
            os.makedirs(results_path)
        with open(os.path.join(results_path, f"{cfg['run_name']}_{cfg['method']}_results.json"), 'w') \
                as file:
            json.dump(results_dict, file)

    

    # for i, experience in enumerate(benchmark.train_stream):
    #     if i == 0:
    #         print("Initial eval...")
    #         strategy.eval(benchmark.test_stream, num_workers=cfg['num_workers'])
    #         strategy.eval(benchmark.streams['val_sets'], num_workers=cfg['num_workers'])

    #     strategy.train(experience, eval_streams=[], shuffle=False,
    #                    num_workers=cfg['num_workers'])

    #     strategy.eval(benchmark.test_stream, num_workers=cfg['num_workers'])
    #     strategy.eval(benchmark.streams['val_sets'], num_workers=cfg['num_workers'])


if __name__ == '__main__':
    main()
