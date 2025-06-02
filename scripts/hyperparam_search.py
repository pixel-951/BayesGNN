import optuna
import wandb
import argparse
import json
import copy
import tqdm

from bayes.training.trainer import ModelTrainer



def search(base_config, searchspace, n_trials):
    def objective(trial):
        sweep_config = {}
        
        for param, config in searchspace.items():
            param_type = config["type"]
            
            if param_type == "float":
                sweep_config[param] = trial.suggest_float(
                    param,
                    low=config["low"],
                    high=config["high"],
                    log=config.get("log", False)
                )
            elif param_type == "int":
                sweep_config[param] = trial.suggest_int(
                    param,
                    low=config["low"],
                    high=config["high"], 
                    log=config.get("log", False)
                )
            elif param_type == "categorical":
                sweep_config[param] = trial.suggest_categorical(
                    param,
                    choices=config["values"]
                )
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

        #base_copy = copy.deepcopy(base_config)
        #merged_congfig = base_copy.update(sweep_config)
        merged_config = {**base_config, **sweep_config}
        
        wandb.init(
            project="optuna_sweep",
            config=merged_config,
            reinit=True
        )

        trainer = ModelTrainer(merged_config)
        trainer.train()
        # or use loss instead?
        val_acc = trainer.best_val_acc
        
        wandb.log({"val/accuracy": val_acc})
        wandb.finish()
        
        return val_acc
    
    progress = tqdm(total=n_trials, desc="Optuna Trials")
    
    def callback(study, trial):
        progress.update(1)
        progress.set_postfix({"Best Val Acc": study.best_value})

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify sweep arguments.")
    parser.add_argument(
        "--config",
        type=str)
    
    parser.add_argument(
        "--n_trials",
        type=int, 
        default=50)
    
    args = parser.parse_args()

    try:
        with open(f"../configs/hyperparam/{args.config.lower()}.json", "r") as file:
            config = json.load(file)
            search(config["base_config"], config["searchspace"], args.n_trials)
            

    except FileNotFoundError:
        print(f"Error: File {args.config.lower()}.json not found.")