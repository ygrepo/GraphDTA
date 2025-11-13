import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import os
from rdkit import Chem
import networkx as nx
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.utils import (
    setup_logging,
    get_logger,
    TestbedDataset,
    read_csv_parquet_torch,
)

logger = get_logger(__name__)


# --- Helper Functions (Unchanged from your script) ---


def atom_features(atom: Chem.Atom) -> np.ndarray:
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                "C",
                "N",
                "O",
                "S",
                "F",
                "Si",
                "P",
                "Cl",
                "Br",
                "Mg",
                "Na",
                "Ca",
                "Fe",
                "As",
                "Al",
                "I",
                "B",
                "V",
                "K",
                "Tl",
                "Yb",
                "Sb",
                "Sn",
                "Ag",
                "Pd",
                "Co",
                "Se",
                "Ti",
                "Zn",
                "H",
                "Li",
                "Ge",
                "Cu",
                "Au",
                "Ni",
                "Cd",
                "In",
                "Mn",
                "Zr",
                "Cr",
                "Pt",
                "Hg",
                "Pb",
                "Unknown",
            ],
        )
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(
            atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
        + one_of_k_encoding_unk(
            atom.GetValence(getExplicit=False),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )
        + [atom.GetIsAromatic()]
    )


def one_of_k_encoding(x: int, allowable_set: list) -> list:
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x: int, allowable_set: list) -> list:
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile: str) -> tuple[int, list, list]:
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smile}")

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index


# --- Sequence Encoding (Unchanged) ---
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000  # You may want to adjust this based on your 'Target' lengths


def seq_cat(prot: str) -> np.ndarray:
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        if ch in seq_dict:
            x[i] = seq_dict[ch]
    return x


def process_data(df: pd.DataFrame, dataset_name: str, output_dir: Path):

    # We'll map your columns to the names the script expects
    # Rename columns for compatibility
    df_processed = df[["Drug", "Target", "True_Affinity"]]
    df_processed.rename(
        columns={
            "Drug": "compound_iso_smiles",
            "Target": "target_sequence",
            "True_Affinity": "affinity",
        },
        inplace=True,
    )

    # Drop any rows with missing essential data
    df_processed.dropna(
        subset=["compound_iso_smiles", "target_sequence", "affinity"], inplace=True
    )

    logger.info(
        f"Successfully loaded and mapped {len(df_processed)} data points from 'df'."
    )

    # Build smile_graph dictionary from YOUR data
    logger.info("Building graph representations for all unique SMILES...")
    compound_iso_smiles = set(df_processed["compound_iso_smiles"])
    smile_graph = {}
    failed_smiles = []

    for smile in compound_iso_smiles:
        try:
            g = smile_to_graph(smile)
            smile_graph[smile] = g
        except Exception as e:
            logger.warning(f"Warning: Could not process SMILES: {smile}. Error: {e}")
            failed_smiles.append(smile)

    logger.info(f"Successfully processed {len(smile_graph)} unique SMILES.")
    if failed_smiles:
        logger.info(
            f"Failed to process {len(failed_smiles)} SMILES. They will be removed."
        )
        # Remove rows with unprocessable SMILES
        df_processed = df_processed[
            ~df_processed["compound_iso_smiles"].isin(failed_smiles)
        ]

    # Split YOUR data into train and test
    train_df, test_df = train_test_split(df_processed, test_size=0.2, random_state=42)

    # Process and save the datasets (using 'my_dataset' as the name)
    datasets_to_process = {"train": train_df, "test": test_df}

    for split_name, df_split in datasets_to_process.items():
        logger.info(f"Processing {split_name} split...")

        # Extract data from the DataFrame split
        drugs = list(df_split["compound_iso_smiles"])
        prots = list(df_split["target_sequence"])
        affinities = list(df_split["affinity"])

        # Convert sequences to numerical representation
        XT = [seq_cat(t) for t in prots]

        # Convert to numpy arrays
        drugs_np, prots_np, affinities_np = (
            np.asarray(drugs),
            np.asarray(XT),
            np.asarray(affinities),
        )

        # Create the PyTorch Geometric dataset
        TestbedDataset(
            root=str(output_dir),
            dataset=dataset_name + "_" + split_name,
            xd=drugs_np,
            xt=prots_np,
            y=affinities_np,
            smile_graph=smile_graph,
        )

    logger.info("\nData processing complete.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create processed data for GraphDTA model training."
    )
    parser.add_argument(
        "--data_fn",
        type=Path,
        default=Path("output/data/20251031_all_binding_db_genes.parquet"),
        required=True,
        help="Path to the combined predictions parquet/CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/data",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=0,
        help="Minimum number of samples per target to include in the dataset",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="All_binding_db_genes",
        help="Name of the dataset",
    )
    parser.add_argument("--log_fn", type=str, default="logs/create_data.log")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    setup_logging(Path(args.log_fn), args.log_level)
    try:
        # Log configuration
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Logging to: {args.log_fn}")
        logger.info(f"Data file: {args.data_fn}")
        data_fn = args.data_fn.resolve()
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Dataset name: {args.dataset_name}")

        # Load data
        df = read_csv_parquet_torch(data_fn)
        logger.info(f"Loaded {len(df)} samples")
        if args.n > 0:
            df = df.head(args.n)
            logger.info(f"Filtered to {len(df)} samples with at least {args.n} samples")
        df["Mutant"] = df["Mutant"].astype(str)
        logger.info(f"Shape: {df.shape}")
        df.query("Mutant == 'Mutant'", inplace=True)
        logger.info(f"Shape: {df.shape}")
        process_data(df, args.dataset_name, Path(args.output_dir))

    except Exception as e:
        logger.exception("Script failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
