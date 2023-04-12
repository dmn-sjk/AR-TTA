

results_dict = {"train_sequences": {"running_acc": [], "batch_acc": []},
                    "val_sequences": {"acc": [], "refined_acc": []}, # refined for AdaCon, not used here
                    "test_sequences": {"acc": [], "refined_acc": []}}



if args.save_results:
    metrics = eval_plugin.get_all_metrics()
    results_dict["val_sequences"]["acc"]. \
        append(metrics[f"Top1_Acc_Stream/eval_phase/val_sets_stream/Task000"][1])
    results_dict["test_sequences"]["acc"]. \
        append(metrics[f"Top1_Acc_Stream/eval_phase/test_stream/Task000"][1])
        
        
    metrics = eval_plugin.get_all_metrics()
    results_dict["train_sequences"]["running_acc"]. \
        append(metrics[f"Top1_RunningAcc_Epoch/train_phase/train_stream/Task00{i + 1}"][1])
    results_dict["train_sequences"]["batch_acc"]. \
        append(metrics[f"Top1_Acc_MB/train_phase/train_stream/Task00{i + 1}"][1])

    results_dict["val_sequences"]["acc"]. \
        append(metrics[f"Top1_Acc_Stream/eval_phase/val_sets_stream/Task000"][1])
    results_dict["test_sequences"]["acc"]. \
        append(metrics[f"Top1_Acc_Stream/eval_phase/test_stream/Task000"][1])
                

if args.save_results:
    results_path = os.path.join("experiments", f"{args.run_name}_{args.method}")
    if not os.path.isdir(results_path):
        os.mkdir(results_path, exist_ok = True)
    with open(os.path.join(results_path, f"{args.run_name}_{args.method}_results.json"), 'w') \
            as file:
        json.dump(results_dict, file)