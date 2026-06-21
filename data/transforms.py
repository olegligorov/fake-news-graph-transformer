"""Positional encoding transforms applied once during dataset processing.

Both PEs are appended to x after structural features are computed.
AddLaplacianEigenvectorPE and AddRandomWalkPE write to separate attrs;
this module concatenates them into x and removes the interim attrs.
"""

import warnings

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE

_lap_pe = AddLaplacianEigenvectorPE(k=8, attr_name="laplacian_pe", is_undirected=True)
_rw_pe = AddRandomWalkPE(walk_length=16, attr_name="rw_pe")


def add_positional_encodings(data: Data) -> Data:
    """Compute Laplacian PE (k=8) and RWPE (length=16) and append to data.x.

    If the graph is too small for k=8 eigenvectors the PE is zero-padded.
    """
    n = data.num_nodes

    # Laplacian PE
    try:
        data = _lap_pe(data)
        lap = data.laplacian_pe  # [N, 8]
        del data.laplacian_pe
    except Exception as exc:
        warnings.warn(f"LapPE failed ({type(exc).__name__}: {exc}), using zeros. Install scipy.", stacklevel=2)
        lap = torch.zeros(n, 8)
        if hasattr(data, "laplacian_pe"):
            del data.laplacian_pe

    # Random Walk PE
    try:
        data = _rw_pe(data)
        rw = data.rw_pe  # [N, 16]
        del data.rw_pe
    except Exception as exc:
        warnings.warn(f"RWPE failed ({type(exc).__name__}: {exc}), using zeros.", stacklevel=2)
        rw = torch.zeros(n, 16)
        if hasattr(data, "rw_pe"):
            del data.rw_pe

    data.x = torch.cat([data.x, lap, rw], dim=1)  # [N, 6+8+16=30]
    return data
