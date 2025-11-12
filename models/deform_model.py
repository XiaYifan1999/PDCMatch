import torch
import math
import torch.nn.functional as F

from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.sinkhorn_util import *
from utils.fmap_util import nn_query, fmap2pointmap
from utils.geometry_util import get_all_operators


def cache_operators(data, cache_dir=None):
    data_x, data_y = data['first'], data['second']
    if 'operators' not in data_x.keys():
        cache_dir = cache_dir or data_x.get('cache_dir', None)
        _, mass, L, evals, evecs, gradX, gradY = get_all_operators(data_x['verts'].cpu(), data_x['faces'].cpu(), k=128,
                                                                    cache_dir=cache_dir)
        
        data_x['operators'] = {'mass': mass, 'L': L, 'evals': evals, 'evecs': evecs, 'gradX': gradX, 'gradY': gradY}
    if 'operators' not in data_y.keys():
        cache_dir = cache_dir or data_y.get('cache_dir', None)
        _, mass, L, evals, evecs, gradX, gradY = get_all_operators(data_y['verts'].cpu(), data_y['faces'].cpu(), k=128,
                                                                    cache_dir=cache_dir)
        data_y['operators'] = {'mass': mass, 'L': L, 'evals': evals, 'evecs': evecs, 'gradX': gradX, 'gradY': gradY}
        
@MODEL_REGISTRY.register()
class DEFNetModel(BaseModel):
    def __init__(self, opt):
        self.with_refine = opt.get("refine", -1)
        self.partial = opt.get("partial", False)
        self.non_isometric = opt.get("non-isometric", False)
        self.middle_iters = opt.get("middle_iters", 2000)
        if self.with_refine > 0:
            opt["is_train"] = True
        super(DEFNetModel, self).__init__(opt)

    def feed_data(self, data):
        # get data pair
        cache_dir = self.opt['networks']['feature_extractor'].get('cache_dir', None)
        cache_operators(data, cache_dir=cache_dir)
        data_x, data_y = to_device(data["first"], self.device), to_device(
            data["second"], self.device
        )

        # feature extractor for mesh
        feat_x = self.networks["feature_extractor"](
            data_x["verts"], data_x["faces"]
        )  # [B, Nx, C]
        feat_y = self.networks["feature_extractor"](
            data_y["verts"], data_y["faces"]
        )  # [B, Ny, C]

        # get spectral operators
        evals_x = data_x["evals"]
        evals_y = data_y["evals"]
        evecs_x = data_x["evecs"]
        evecs_y = data_y["evecs"]
        evecs_trans_x = data_x["evecs_trans"]  # [B, K, Nx]
        evecs_trans_y = data_y["evecs_trans"]  # [B, K, Ny]

        Cxy, Cyx = self.networks["fmap_net"](
            feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y
        )

        self.loss_metrics = self.losses["surfmnet_loss"](Cxy, Cyx, evals_x, evals_y)
        Pxy, Pyx = self.compute_permutation_matrix(feat_x, feat_y, bidirectional=True)

        # compute C
        Cxy_est = torch.bmm(evecs_trans_y, torch.bmm(Pyx, evecs_x))

        self.loss_metrics["l_align"] = self.losses["align_loss"](Cxy, Cxy_est)
        if not self.partial:
            Cyx_est = torch.bmm(evecs_trans_x, torch.bmm(Pxy, evecs_y))
            self.loss_metrics["l_align"] += self.losses["align_loss"](Cyx, Cyx_est)

        if "dirichlet_loss" in self.losses:
            Lx, Ly = data_x['operators']['L'], data_y['operators']['L']
            verts_x, verts_y = data_x["verts"], data_y["verts"]
            self.loss_metrics["l_d"] = self.losses["dirichlet_loss"](
                torch.bmm(Pxy, verts_y), Lx
            ) + self.losses["dirichlet_loss"](torch.bmm(Pyx, verts_x), Ly)

        # deform loss
        data_X, data_Y = self.norm2(data_x["verts"], torch.bmm(Pxy, data_y["verts"]))
        conf = self.GCPD_initial_settings()
        W, P = self.GCPD_initial(data_X, data_Y, evecs_x, evals_x, conf)
        W = W.unsqueeze(0)
        flow_x = torch.bmm(evecs_x, W)
        deform_x = flow_x + data_X
        if self.curr_iter > self.middle_iters:
            self.loss_metrics["l_deform"] = self.losses["deform_loss"](
                deform_x, data_Y, P
            )
        else:
            self.loss_metrics["l_deform"] = 0

    def validate_single(self, data, timer):
        # get data pair
        data_x, data_y = to_device(data["first"], self.device), to_device(
            data["second"], self.device
        )

        # get previous network state dict
        if self.with_refine > 0:
            state_dict = {"networks": self._get_networks_state_dict()}

        # start record
        timer.start()

        # test-time refinement
        if self.with_refine > 0:
            self.refine(data)

        # feature extractor
        feat_x = self.networks["feature_extractor"](
            data_x["verts"], data_x.get("faces")
        )
        feat_y = self.networks["feature_extractor"](
            data_y["verts"], data_y.get("faces")
        )

        # get spectral operators
        evecs_x = data_x["evecs"].squeeze()
        evecs_y = data_y["evecs"].squeeze()
        evecs_trans_x = data_x["evecs_trans"].squeeze()
        evecs_trans_y = data_y["evecs_trans"].squeeze()

        if self.non_isometric:
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)

            # nearest neighbour query
            p2p = nn_query(feat_x, feat_y).squeeze()

            # compute Pyx from functional map
            Cxy = evecs_trans_y @ evecs_x[p2p]
            Pyx = evecs_y @ Cxy @ evecs_trans_x
        else:
            # compute Pxy
            Pyx = self.compute_permutation_matrix(
                feat_y, feat_x, bidirectional=False
            ).squeeze()
            Cxy = evecs_trans_y @ (Pyx @ evecs_x)

            # convert functional map to point-to-point map
            p2p = fmap2pointmap(Cxy, evecs_x, evecs_y)

            # compute Pyx from functional map
            Pyx = evecs_y @ Cxy @ evecs_trans_x

        # finish record
        timer.record()

        # resume previous network state dict
        if self.with_refine > 0:
            self.resume_model(state_dict, net_only=True, verbose=False)
        return p2p, Pyx, Cxy

    def compute_permutation_matrix(
        self, feat_x, feat_y, bidirectional=False, normalize=True
    ):
        if normalize:
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)
        similarity = torch.bmm(feat_x, feat_y.transpose(1, 2))

        # sinkhorn normalization
        Pxy = self.networks["permutation"](similarity)

        if bidirectional:
            Pyx = self.networks["permutation"](similarity.transpose(1, 2))
            return Pxy, Pyx
        else:
            return Pxy

    def refine(self, data):
        self.networks["permutation"].hard = False
        self.networks["fmap_net"].bidirectional = True

        with torch.set_grad_enabled(True):
            for _ in range(self.with_refine):
                self.feed_data(data)
                self.optimize_parameters()

        self.networks["permutation"].hard = True
        self.networks["fmap_net"].bidirectional = False

    @torch.no_grad()
    def validation(self, dataloader, tb_logger, update=True):
        # change permutation prediction status
        if "permutation" in self.networks:
            self.networks["permutation"].hard = True
        if "fmap_net" in self.networks:
            self.networks["fmap_net"].bidirectional = False
        super(DEFNetModel, self).validation(dataloader, tb_logger, update)
        if "permutation" in self.networks:
            self.networks["permutation"].hard = False
        if "fmap_net" in self.networks:
            self.networks["fmap_net"].bidirectional = True

    # def gcpd_initial(self, X, Y, evecs_x,evals_x,conf)

    def GCPD_initial_settings(self, conf=None, device=None):
        """
        Initialize the conf dictionary with default values for GCPD settings in PyTorch.

        Parameters:
        conf (dict): A dictionary with user-defined configuration values.
                    Missing keys will be set to default values.

        Returns:
        dict: The updated conf dictionary with all necessary keys as PyTorch tensors.
        """
        if conf is None:
            conf = {}
        elif not isinstance(conf, dict):
            raise TypeError(
                f"Expected conf to be a dictionary or None, but got {type(conf)}"
            )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Default settings with PyTorch tensors
        defaults = {
            "MaxIter": torch.tensor(20, dtype=torch.int32, device=device),  # default: 20
            "gamma": torch.tensor(0.9, dtype=torch.float32, device=device),
            "beta": torch.tensor(0.0, dtype=torch.float32, device=device),
            "lambda": torch.tensor(10.0, dtype=torch.float32, device=device),
            "theta": torch.tensor(0.75, dtype=torch.float32, device=device),
            "a": torch.tensor(0.05, dtype=torch.float32, device=device),
            "ecr": torch.tensor(1e-5, dtype=torch.float32, device=device),
            "minP": torch.tensor(1e-5, dtype=torch.float32, device=device),
        }

        # Update conf with defaults for missing keys
        for key, value in defaults.items():
            if key not in conf:
                conf[key] = value

        return conf

    def GCPD_initial(self, X, Y, U, e, conf):
        """
        GCPD initialization process in PyTorch.

        Parameters:
        X: torch.Tensor - Source point cloud, shape (N, D)
        Y: torch.Tensor - Target point cloud, shape (N, D)
        U: torch.Tensor - Basis functions, shape (N, k)
        e: torch.Tensor - Diagonal entries for K, shape (N,)
        conf: dict - Configuration parameters
        device: str - Device to run the computations (default: "cuda")

        Returns:
        T: torch.Tensor - Transformed target points
        C: torch.Tensor - Coefficients
        index: torch.Tensor - Indices of significant points
        """
        # Initialization
        gamma = conf["gamma"]
        lambda_ = conf["lambda"]
        theta = conf["theta"]
        a = conf["a"]
        MaxIter = conf["MaxIter"]
        ecr = conf["ecr"]
        minP = conf["minP"]

        device = X.device

        Y = torch.squeeze(Y, dim=0)
        X = torch.squeeze(X, dim=0)
        U = torch.squeeze(U, dim=0)
        e = torch.squeeze(e, dim=0)

        K = torch.diag(e).to(device)
        N, D = X.shape
        k = U.shape[1]

        W = torch.zeros((k, D), device=device)
        P = torch.eye(N, device=device)
        T = X.clone()
        iter_ = 1
        tecr = 1
        E = 1
        sigma2 = torch.sum((Y - X) ** 2) / (N * D)

        while iter_ < MaxIter and abs(tecr) > ecr and sigma2 > 1e-8:
            # E-step
            E_old = E
            P1, E = self.get_P(Y, T, sigma2, gamma, a, device)
            P1 = torch.clamp(P1, min=minP)  # Ensure P1 >= minP
            P = torch.diag(P1)

            temp1 = torch.matmul(K, W)
            temp2 = torch.matmul(W.T, temp1)
            E += lambda_ / 2 * torch.trace(temp2)
            tecr = (E - E_old) / E

            # M-step
            W = torch.linalg.solve(
                U.T @ P @ U + lambda_ * sigma2 * K, U.T @ P @ (Y - X)
            )
            T = X + U @ W
            V = Y - T
            sigma2 = torch.trace(V.T @ P @ V) / (D * torch.trace(P))

            numcorr = torch.sum(P.diagonal() > theta).item()
            gamma = torch.tensor(numcorr / N, device=device)
            gamma = torch.clamp(
                gamma, min=0.05, max=0.95
            )  # Clamp gamma to [0.05, 0.95]

            iter_ += 1
        P = P.diagonal()
        return W, P

    def get_P(self, X, Tx, sigma2, gamma, a, device="cuda"):
        """
        Estimate posterior probability and energy in PyTorch.

        Parameters:
        X: torch.Tensor - Source points, shape (N, D)
        Tx: torch.Tensor - Transformed target points, shape (N, D)
        sigma2: float - Variance
        gamma: float - Gamma parameter
        a: float - Regularization parameter
        device: str - Device to run computations (default: "cuda")

        Returns:
        P: torch.Tensor - Posterior probabilities, shape (N,)
        E: float - Energy value
        """
        D = X.shape[1]
        diff = X - Tx
        temp1 = torch.exp(-torch.sum(diff**2, dim=1) / (2 * sigma2))
        temp2 = (2 * math.pi * sigma2) ** (D / 2) * (1 - gamma) / (gamma * a)
        P = temp1 / (temp1 + temp2)

        E = (
            P @ torch.sum(diff**2, dim=1) / (2 * sigma2)
            + torch.sum(P) * torch.log(sigma2) * D / 2
            - torch.log(gamma) * torch.sum(P)
            - torch.log(1 - gamma) * torch.sum(1 - P)
        )

        return P, E

    def norm2(self, x, y):
        """
        Normalizes the data so that it has zero means and unit covariance.

        Args:
            x (torch.Tensor): Input tensor x of shape (b, n, d).
            y (torch.Tensor): Input tensor y of shape (b, m, d).

        Returns:
            X (torch.Tensor): Normalized x.
            Y (torch.Tensor): Normalized y.
        """
        x = x.squeeze(0)  # shape: (N, d)
        y = y.squeeze(0)  # shape: (M, d)

        # Compute the mean of x and y
        xm = torch.mean(x, dim=0)
        ym = torch.mean(y, dim=0)

        # Center the data by subtracting the mean
        xx = x - xm
        yy = y - ym

        # Compute the scale factor (L2 norm)
        xscale = torch.sqrt(torch.sum(xx**2, dim=1).mean())
        yscale = torch.sqrt(torch.sum(yy**2, dim=1).mean())

        # Normalize the data by dividing by the scale
        X = xx / xscale
        X = X.unsqueeze(0)
        Y = yy / yscale
        Y = Y.unsqueeze(0)

        return X, Y
