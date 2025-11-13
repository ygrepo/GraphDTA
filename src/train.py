import numpy as np
import pandas as pd
from pathlib import Path
import sys, os
from random import shuffle
import torch
import torch.nn as nn
import torch.utils.data  # Add this import
import argparse

from torch_geometric.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import (
    setup_logging,
    get_logger,
    TestbedDataset,
    rmse,
    mse,
    pearson,
    spearman,
    ci,  # Import metrics here instead
)

from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet

logger = get_logger(__name__)


# training function at each epoch
def train(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss_fn: nn.Module,
    log_interval: int,
):
    """
    Runs a single training epoch.
    """
    logger.info(f"Training on {len(train_loader.dataset)} samples...")
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            logger.info(
                f"Train epoch: {epoch} [{batch_idx * len(data.x)}/{len(train_loader.dataset)} ({ 100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )


def predict(model: nn.Module, device: torch.device, loader: DataLoader):
    """
    Evaluates the model on a given data loader.
    """
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    logger.info(f"Make prediction for {len(loader.dataset)} samples...")
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def main_loop(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    num_epochs: int,
    log_interval: int,
    model_file_name: Path,
    result_file_name: Path,
    model_st: str,
    dataset: str,
    metrics: dict,
):
    """
    The main training and validation loop.
    """
    logger.info(
        f"Starting training for {model_st} on {dataset} for {num_epochs} epochs."
    )

    # Unpack metrics functions
    mse_fn, rmse_fn, pearson_fn, spearman_fn, ci_fn = (
        metrics["mse"],
        metrics["rmse"],
        metrics["pearson"],
        metrics["spearman"],
        metrics["ci"],
    )

    best_mse = 1000
    best_test_mse = 1000
    best_test_ci = 0
    best_epoch = -1

    for epoch in range(num_epochs):
        # Pass log_interval and loss_fn to the train function
        train(model, device, train_loader, optimizer, epoch + 1, loss_fn, log_interval)

        logger.info("predicting for valid data")
        G, P = predict(model, device, valid_loader)
        val = mse_fn(G, P)

        if val < best_mse:
            best_mse = val
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_file_name)
            logger.info("predicting for test data")
            G, P = predict(model, device, test_loader)
            ret = [
                rmse_fn(G, P),
                mse_fn(G, P),
                pearson_fn(G, P),
                spearman_fn(G, P),
                ci_fn(G, P),
            ]

            with open(result_file_name, "w") as f:
                f.write(",".join(map(str, ret)))

            best_test_mse = ret[1]
            best_test_ci = ret[-1]
            logger.info(
                f"Improvement at epoch {best_epoch}: best_test_mse={best_test_mse:.4f}, "
                f"best_test_ci={best_test_ci:.4f} for {model_st} on {dataset}"
            )
        else:
            logger.info(
                f"No improvement since epoch {best_epoch}; best_test_mse={best_test_mse:.4f}, "
                f"best_test_ci={best_test_ci:.4f} for {model_st} on {dataset}"
            )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GNN models for drug-target affinity prediction."
    )
    parser.add_argument(
        "--model",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Model to use: 0=GINConvNet, 1=GATNet, 2=GAT_GCN, 3=GCNNet.",
    )
    parser.add_argument(
        "--model_dir", type=Path, default="output/models", help="Model directory."
    )
    parser.add_argument(
        "--result_dir", type=Path, default="output/results", help="Result directory."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=512, help="Training batch size."
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=512, help="Validation/Test batch size."
    )
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate.")
    parser.add_argument(
        "--num_epochs", type=int, default=1000, help="Number of training epochs."
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="Log interval (in batches) during training.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/data",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="All_binding_db_genes",
        help="Name of the dataset",
    )
    parser.add_argument("--log_fn", type=str, default="logs/create_data.log")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()
    return args


def main():
    """
    Main function to parse arguments, set up training, and run the main loop.
    """

    args = parse_args()
    setup_logging(Path(args.log_fn), args.log_level)

    try:
        # --- Configuration from Arguments ---
        if args.model == 0:
            modeling = GINConvNet
        elif args.model == 1:
            modeling = GATNet
        elif args.model == 2:
            modeling = GAT_GCN
        elif args.model == 3:
            modeling = GCNNet
        model_st = modeling.__name__

        # These are now local variables, not global constants
        train_batch_size = args.train_batch_size
        test_batch_size = args.test_batch_size
        lr = args.lr
        log_interval = args.log_interval
        num_epochs = args.num_epochs
        output_dir = Path(args.output_dir)
        dataset_name = args.dataset_name

        logger.info(f"Learning rate: {lr}")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Log interval: {log_interval}")
        # Pack metrics for passing to main_loop
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "pearson": pearson,
            "spearman": spearman,
            "ci": ci,
        }

        # Main program: iterate over different datasets
        logger.info(f"\nrunning on {model_st}_{dataset}")
        processed_data_file_train = output_dir / f"{dataset_name}_train.pt"
        processed_data_file_test = output_dir / f"{dataset_name}_test.pt"

        if not os.path.isfile(processed_data_file_train) or not os.path.isfile(
            processed_data_file_test
        ):
            logger.info("Please run create_data.py to prepare data in PyTorch format!")
            return  # Exit the function if data is not prepared

        dataset = args.dataset_name
        train_data_full = TestbedDataset(
            root=str(output_dir), dataset=dataset + "_train"
        )
        test_data = TestbedDataset(root=str(output_dir), dataset=dataset + "_test")

        # 80/20 train/validation split
        train_size = int(0.8 * len(train_data_full))
        valid_size = len(train_data_full) - train_size
        train_data, valid_data = torch.utils.data.random_split(
            train_data_full, [train_size, valid_size]
        )

        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

        # training the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model_dir = args.model_dir.resolve()
        model_file_name = model_dir / f"model_{model_st}_{dataset}.model"
        result_file_name = (
            args.result_dir.resolve() / f"result_{model_st}_{dataset}.csv"
        )

        # Call the new main_loop function
        main_loop(
            model,
            device,
            train_loader,
            valid_loader,
            test_loader,
            optimizer,
            loss_fn,
            num_epochs,
            log_interval,
            model_file_name,
            result_file_name,
            model_st,
            dataset,
            metrics,
        )
    except Exception as e:
        logger.exception("Script failed: %s", e)
        sys.exit(1)


# Standard Python entry point
if __name__ == "__main__":
    main()
