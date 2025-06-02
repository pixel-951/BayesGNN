import os
import json
import argparse


from bayes.training.trainer import ModelTrainer


def get_available_configs(directory):
    """Parameters:
        - directory (str): Path to the directory containing config files.
    Returns:
        - config_files (list): List of config file names.
    Processing Logic:
        - Get all files in directory.
        - Filter for only JSON files.
        - Extract file names without extension."""
    config_files = [
        filename.split(".")[0]
        for filename in os.listdir(directory)
        if filename.endswith(".json")
    ]
    return config_files


def setup(config):

    tensorboard_dir = os.path.abspath(os.path.join(os.getcwd(), "../tensorboard/optimizers/"))
    config["tensorboard_dir"] = tensorboard_dir
    trainer = ModelTrainer(config)
    trainer.train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process config files in executing.")
    parser.add_argument(
        "--config",
        type=str,
        default="model_params",
        help="Name of the configuration to load as a dictionary. "
        "Available options:\n" + "\n".join(get_available_configs("../configs/")),
    )
    parser.add_argument(
        "--num_seeds", type=int, default=5, help="The number of seeds to be executed"
    )

    args = parser.parse_args()
    if args.config != "model_params":
        try:
            with open(f"../configs/training/{args.config.lower()}.json", "r") as file:
                config = json.load(file)
                setup(config)
                

        except FileNotFoundError:
            print(f"Error: File {args.config.lower()}.json not found.")
