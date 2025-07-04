import argparse
import yaml
from loguru import logger
import mlflow

__all__ = ["parse_args", "save_config", "setup_mlflow"]


def parse_args() -> argparse.Namespace:
    # TODO: we can only save the config but not load it yet
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Configuration for the project.")

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (default: cuda).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10).",
    )
    args = parser.parse_args()
    return args


@logger.catch
def save_config(args: argparse.Namespace) -> None:
    """Save the configuration to a file."""

    config_file_path = args.config
    try:
        with open(config_file_path, "w") as f:
            yaml.dump(vars(args), f, default_flow_style=False)
        logger.info(f"Configuration saved to {config_file_path}")
    except Exception as e:
        logger.error(f"Error saving configuration to {config_file_path}: {e}")


@logger.catch
def setup_mlflow(args: argparse.Namespace) -> None:
    """Set up MLflow tracking for recording training metrics.

    This function configures MLflow for tracking experiments based on the provided arguments.
    If specified in the arguments, it will also start a run and log the configuration parameters.

    Args:
        args: Command line arguments containing MLflow configuration
    """

    # Set tracking URI if specified
    tracking_uri = getattr(args, "mlflow_tracking_uri", "http://127.0.0.1:8080")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Set experiment name
    experiment_name = getattr(args, "mlflow_experiment_name", "default")
    mlflow.set_experiment(experiment_name)

    logger.info(f"MLflow tracking set up with experiment '{experiment_name}'")
