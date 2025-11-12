import numpy as np
import torch
import warnings
from scipy.sparse.linalg import eigs
from typing import Union, Dict, List


def gsp_design_meyer(G, Nf=6, param=None):
    param = param or {}
    verbose = param.get("verbose", True)

    if isinstance(G, dict) and "lmax" in G:
        lmax = G["lmax"]
    elif isinstance(G, (float, int)):
        lmax = G
    elif isinstance(G, torch.Tensor):
        if G.numel() == 1:
            lmax = G.item()
        else:
            lmax = G[-1].item()  # 取最后一个元素
    else:
        raise ValueError("Invalid G type")

    if "t" not in param:
        t = (4 / (3 * lmax)) * 2.0 ** torch.arange(Nf - 2, -1, -1)
    else:
        t = param["t"]
        if verbose and len(t) != Nf - 1:
            warnings.warn("length of t not equal to #filters", UserWarning)

    g = [lambda x, t_j=t[0]: kernel_meyer(t_j * x, "sf")]
    for j in range(Nf - 1):
        g.append(lambda x, t_j=t[j]: kernel_meyer(t_j * x, "wavelet"))

    return g, t


def kernel_meyer(x, kernel_type):
    l1 = 2 / 3
    l2 = 4 / 3  # 2*l1
    l3 = 8 / 3  # 4*l1

    # Define auxiliary function v(x)
    def v(x):
        return x**4 * (35 - 84 * x + 70 * x**2 - 20 * x**3)

    # Initialize output array
    r = torch.zeros_like(x)

    # Compute based on kernel type
    if kernel_type == "sf":
        # Scaling function branch
        mask1 = x < l1
        mask2 = (x >= l1) & (x < l2)

        r[mask1] = 1.0
        r[mask2] = torch.cos((torch.pi / 2) * v(torch.abs(x[mask2]) / l1 - 1))

    elif kernel_type == "wavelet":
        # Wavelet function branch
        mask1 = (x >= l1) & (x < l2)
        mask2 = (x >= l2) & (x < l3)

        r[mask1] = torch.sin((torch.pi / 2) * v(torch.abs(x[mask1]) / l1 - 1))
        r[mask2] = torch.cos((torch.pi / 2) * v(torch.abs(x[mask2]) / l2 - 1))

    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    return r


def gsp_estimate_lmax(G: Union[Dict, List[Dict]]) -> Union[Dict, List[Dict]]:
    if isinstance(G, list):
        return [gsp_estimate_lmax(g) for g in G]

    # Make sure we have required fields
    if "L" not in G or "N" not in G or "d" not in G:
        raise ValueError(
            "Graph must contain L (Laplacian), N (node count) and d (degree) fields"
        )

    try:
        # Compute largest eigenvalue with relaxed tolerance
        opts = {
            "tol": 5e-3,
            "k": min(G["N"], 10),  # Number of eigenvalues requested
            "which": "LM",  # Largest magnitude
            "maxiter": 300,  # Max iterations
        }

        # Note: eigs returns complex array even for real matrices
        eigenvalues = eigs(G["L"], **opts)
        lmax = eigenvalues[0]
        G["lmax"] = abs(lmax) * 1.01  # Add 1% buffer
    except Exception as e:
        warnings.warn(
            f"Cannot use default method: {str(e)}. Falling back to 2*max degree."
        )
        G["lmax"] = 2 * max(G["d"])

    # Handle multi-graph case if present
    # if 'Gm' in G:
    #    G = gsp_estimate_oose_lmax(G)  # Assuming this is another function to be implemented

    return G
