import os
import sys
from pathlib import Path

import numpy as np
from math import sqrt
from scipy import stats
import logging
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import InMemoryDataset, Data
import pandas as pd
import torch
import logging

logger = logging.getLogger(__name__)


# ---- One base for everything ----
BASE_LOGGER = "drugdiscovery"
_BASE = logging.getLogger(BASE_LOGGER)  # the only logger we configure here


def setup_logging(log_path: str | Path | None, level: str = "INFO") -> logging.Logger:
    """Configure the base logger once (file + console)."""
    if getattr(_BASE, "_configured", False):
        return _BASE

    _BASE.handlers.clear()
    _BASE.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Optional file handler
    if log_path:
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setFormatter(fmt)
        _BASE.addHandler(fh)

    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    _BASE.addHandler(sh)

    # Do not bubble to the *root* logger
    _BASE.propagate = False
    _BASE._configured = True
    return _BASE


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a child logger that inherits the base handlers (no child handlers)."""
    full_name = BASE_LOGGER if not name else f"{BASE_LOGGER}.{name}"
    logger = logging.getLogger(full_name)
    # Ensure children don't keep their own handlers (which would double-log)
    if logger is not _BASE and logger.handlers:
        logger.handlers.clear()
    logger.propagate = True  # bubble to BASE only
    return logger


# Convenience logger for this module
logger = get_logger(__name__)


class TestbedDataset(InMemoryDataset):
    def __init__(
        self,
        root="/tmp",
        dataset="davis",
        xd=None,
        xt=None,
        y=None,
        transform=None,
        pre_transform=None,
        smile_graph=None,
    ):

        self.dataset = dataset
        self.root = root

        # We MUST call super() *after* defining custom properties
        # that it depends on (like 'dataset' for processed_file_names)
        # BUT, we must NOT call it until we are ready to process.
        # This is tricky.

        # We need to store these for self.process() to access
        self._xd = xd
        self._xt = xt
        self._y = y
        self._smile_graph = smile_graph

        super(TestbedDataset, self).__init__(root, transform, pre_transform)

        # The super() call above will have already
        # 1. Checked for processed_paths[0]
        # 2. If not found, called self.process() (which uses our _xd, _xt args)
        # 3. If found, it does nothing.
        # EITHER WAY, the file now exists. We just need to load it.

        logger.info(f"Loading processed data from: {self.processed_paths[0]}")
        # Tell PyTorch to load everything, not just weights
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

        # Clean up temp args
        del self._xd, self._xt, self._y, self._smile_graph

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # This property is used by super() to find the file
        return [self.dataset + ".pt"]

    def download(self):
        pass

    def process(self):
        # This method is called by super() IF the file in processed_paths[0] is not found.
        # It takes NO arguments, so it must use member variables.

        logger.info("Pre-processed data not found, doing pre-processing...")

        # Use the stored attributes
        xd, xt, y, smile_graph = self._xd, self._xt, self._y, self._smile_graph

        if xd is None:
            raise RuntimeError(
                "Raw data (xd, xt, y) was not provided to the constructor."
            )

        assert len(xd) == len(xt) and len(xt) == len(
            y
        ), "The three lists must be the same length!"

        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            if (i + 1) % 100 == 0:
                logger.info(f"Converting SMILES to graph: {i+1}/{data_len}")

            smiles = xd[i]
            target = xt[i]  # This is a np.array
            labels = y[i]

            c_size, features, edge_index = smile_graph[smiles]

            GCNData = Data(
                x=torch.Tensor(np.array(features)),  # <--- THE FIX
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                y=torch.FloatTensor([labels]),
            )

            # Make sure target is a 1D tensor, not a 0-dim array
            GCNData.target = torch.LongTensor(target)
            GCNData.__setitem__("c_size", torch.LongTensor([c_size]))
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        logger.info("Graph construction done. Saving to file.")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


def save_csv_parquet_torch(df: pd.DataFrame, fn: Path) -> None:
    if fn.suffix == ".parquet":
        logger.info(f"Saving to parquet: {fn}")
        df.to_parquet(fn)
        return
    if fn.suffix == ".csv":
        logger.info(f"Saving to csv: {fn}")
        df.to_csv(fn, index=False)
        return

    if fn.suffix == ".pt":
        logger.info(f"Saving to torch: {fn}")
        torch.save(df, fn)
        return

    raise ValueError(f"Unsupported file format: {fn.suffix}")


def read_csv_parquet_torch(fn: Path) -> pd.DataFrame:
    if fn.suffix == ".parquet":
        return pd.read_parquet(fn)
    if fn.suffix == ".csv":
        return pd.read_csv(fn)
    if fn.suffix == ".pt":
        return torch.load(fn)
    raise ValueError(f"Unsupported file format: {fn.suffix}")
