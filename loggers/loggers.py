from avalanche.logging import TextLogger, InteractiveLogger, WandBLogger

# TODO: write custom logger for avalanche to save to json

def get_loggers(args):
    return None

    loggers = [InteractiveLogger()]
    if args.text_logger:
        loggers.append(TextLogger(open(f"./experiments/{args.run_name}/{args.run_name}.log", 'w')))
    if args.wandb:
        if args.watch_model:
            wandb_logger = ImprovedWandBLogger(model=model, project_name=args.project_name, run_name=args.run_name)
        else:
            wandb_logger = WandBLogger(project_name=args.project_name, run_name=args.run_name)
        loggers.append(wandb_logger)