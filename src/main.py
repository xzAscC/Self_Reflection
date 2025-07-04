from logger import loguru_setup
from config import parse_args, save_config, setup_mlflow
from utils import set_random_seed
from trainer import Trainer
import yaml


def main():
    # 1. basic setup
    logger = loguru_setup()
    logger.info("Begin running")

    args = parse_args()
    logger.info(f"Arguments parsed: {args}")
    save_config(args)
    verbose = args.verbose
    if verbose:
        logger.info("Verbose mode is enabled")
    else:
        logger.info("Verbose mode is disabled")

    # setup_mlflow(args)
    logger.info("MLflow setup completed")
    set_random_seed(args.seed, args.deterministic)
    device = args.device

    # 2. create dataloader, model, optimizer, lr_scheduler and loss function
    train_loader, test_loader = None, None  # Placeholder for dataloaders
    model = None  # Placeholder for model
    optimizer = None  # Placeholder for optimizer
    lr_scheduler = None  # Placeholder for learning rate scheduler
    loss_fn = None  # Placeholder for loss function

    # 3. create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        lr_scheduler=lr_scheduler,
        loss_fn=loss_fn,
        device=device,
        verbose=verbose,
        epochs=args.epochs,
    )
    trainer.register_hooks([])
    trainer.train()


if __name__ == "__main__":
    main()
