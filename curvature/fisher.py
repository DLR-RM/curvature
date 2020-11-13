"""Various Fisher information matrix approximations."""

from abc import ABC, abstractmethod
from typing import Union, List, Any

import torch
from torch import Tensor
from torch.nn import Module, Sequential
import torch.nn.functional as F
from tqdm import tqdm

from .utils import get_eigenvectors, kron


class Curvature(ABC):
    """Base class for all curvature approximations.

    All curvature approximations are computed layer-wise (i.e. layer-wise independence is assumed s.t. no
    covariances between layers are computed, aka block-wise approximation) and stored in `state`.

    The curvature of the loss function is the matrix of 2nd-order derivatives of the loss w.r.t. the networks weights
    (i.e. the expected Hessian). It can be approximated by the expected Fisher information matrix and, under exponential
    family loss functions (like mean-squared error and cross-entropy loss) and piecewise linear activation functions
    (i.e. ReLU), becomes identical to the Fisher.

    Note:
        The aforementioned identity does not hold for the empirical Fisher, where the expectation is computed w.r.t.
        the data distribution instead of the models' output distribution. Also, because the latter is usually unknown,
        it is approximated through Monte Carlo integration using samples from a categorical distribution, initialized by
        the models' output.

    Source: `Optimizing Neural Networks with Kronecker-factored Approximate Curvature
    <https://arxiv.org/abs/1503.05671>`_
    """

    def __init__(self,
                 model: Union[Module, Sequential]):
        """Curvature class initializer.

        Args:
            model: Any (pre-trained) PyTorch model including all `torchvision` models.
        """
        self.model = model
        self.state = dict()

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any):
        """Abstract method to be implemented by each derived class individually."""
        pass


class DIAG(Curvature):
    r"""The diagonal Fisher information matrix approximation.

    It is defined as :math:`F_{DIAG}=\mathrm{diag}(F)` with `F` being the Fisher defined in the `FISHER` class.
    Code inspired by https://github.com/wjmaddox/swa_gaussian/blob/master/swag/posteriors/diag_laplace.py.

    Source: `A Scalable Laplace Approximation for Neural Networks <https://openreview.net/pdf?id=Skdvd2xAZ>`_
    """

    def update(self,
               batch_size: int):
        """Computes the diagonal curvature for each `Conv2d` or `Linear` layer, skipping all others.

        Args:
            batch_size: The size of the current batch.
        """
        for module in self.model.modules():
            if module.__class__.__name__ in ['Linear', 'Conv2d']:
                grads = module.weight.grad.contiguous().view(module.weight.grad.shape[0], -1)
                if module.bias is not None:
                    grads = torch.cat([grads, module.bias.grad.unsqueeze(dim=1)], dim=1)
                grads = grads ** 2 * batch_size
                if module in self.state:
                    self.state[module] += grads
                else:
                    self.state[module] = grads


class FISHER(Curvature):
    r"""The Fisher information matrix.

    It can be defined as the expectation of the outer product of the gradient of the networks loss E w.r.t. its
    weights W: :math:`F=\mathbb{E}\left[\nabla_W E(W)\nabla_W E(W)^T\right]`

    Source: `A Scalable Laplace Approximation for Neural Networks <https://openreview.net/pdf?id=Skdvd2xAZ>`_
    """
    def update(self,
               batch_size: int):
        """Computes the curvature for each `Conv2d` or `Linear` layer, skipping all others.

        Args:
            batch_size: The size of the current batch.
        """
        for module in self.model.modules():
            if module.__class__.__name__ in ['Linear', 'Conv2d']:
                grads = module.weight.grad.contiguous().view(-1)
                if module.bias is not None:
                    grads = torch.cat([grads, module.bias.grad])
                grads = torch.ger(grads, grads) * batch_size
                if module in self.state:
                    self.state[module] += grads
                else:
                    self.state[module] = grads


class KFAC(Curvature):
    r"""The Kronecker-factored Fisher information matrix approximation.

    For a single datum, the Fisher can be Kronecker-factorized into two much smaller matrices `Q` and `H`, aka
    `Kronecker factors`, s.t. :math:`F=Q\otimes H` where :math:`Q=zz^T` and :math:`H=\nabla_a^2 E(W)` with `z` being the
    output vector of the previous layer, `a` the `pre-activation` of the current layer (i.e. the output of the previous
    layer before being passed through the non-linearity) and `E(W)` the loss. For the expected Fisher,
    :math:`\mathbb{E}[Q\otimes H]\approx\mathbb{E}[Q]\otimes\mathbb{E}[H]` is assumed, which might not necessarily be
    the case.

    Code adapted from https://github.com/Thrandis/EKFAC-pytorch/kfac.py.

    Linear: `Optimizing Neural Networks with Kronecker-factored Approximate Curvature
    <https://arxiv.org/abs/1503.05671>`_

    Convolutional: `A Kronecker-factored approximate Fisher matrix for convolutional layers
    <https://arxiv.org/abs/1602.01407>`_
    """
    def __init__(self,
                 model: Union[Module, Sequential]):
        """KFAC class initializer.

        For the recursive computation of `H`, outputs and inputs for each layer are recorded in `record`. Forward and
        backward hook handles are stored in `hooks` for subsequent removal.

        Args:
            model: Any (pre-trained) PyTorch model including all `torchvision` models.
        """
        super().__init__(model)
        self.hooks = list()
        self.record = dict()

        for module in model.modules():
            if module.__class__.__name__ in ['Linear', 'Conv2d']:
                self.record[module] = [None, None]
                self.hooks.append(module.register_forward_pre_hook(self._save_input))
                self.hooks.append(module.register_backward_hook(self._save_output))

    def _save_input(self, module, input):
        self.record[module][0] = input[0]

    def _save_output(self, module, grad_input, grad_output):
        self.record[module][1] = grad_output[0] * grad_output[0].size(0)

    def update(self,
               batch_size: int):
        """Computes the 1st and 2nd Kronecker factor `Q` and `H` for each `Conv2d` or `Linear` layer, skipping all
        others.

        Todo: Check code against papers.

        Args:
            batch_size: The size of the current batch.
        """
        for module in self.model.modules():
            module_class = module.__class__.__name__
            if module_class in ['Linear', 'Conv2d']:
                forward, backward = self.record[module]

                # 1st factor: Q
                if module_class == 'Conv2d':
                    forward = F.unfold(forward, module.kernel_size, padding=module.padding, stride=module.stride)
                    forward = forward.data.permute(1, 0, 2).contiguous().view(forward.shape[1], -1)
                else:
                    forward = forward.data.t()
                if module.bias is not None:
                    ones = torch.ones_like(forward[:1])
                    forward = torch.cat([forward, ones], dim=0)
                first_factor = torch.mm(forward, forward.t()) / float(forward.shape[1])

                # 2nd factor: H
                if module_class == 'Conv2d':
                    backward = backward.data.permute(1, 0, 2, 3).contiguous().view(backward.shape[1], -1)
                else:
                    backward = backward.data.t()
                second_factor = torch.mm(backward, backward.t()) / float(backward.shape[1])

                # Expectation
                if module in self.state:
                    self.state[module][0] += first_factor
                    self.state[module][1] += second_factor
                else:
                    self.state[module] = [first_factor, second_factor]


class EFB(Curvature):
    """The eigenvalue corrected Kronecker-factored Fisher information matrix.

    Todo: Add source/equations.
    """
    def __init__(self,
                 model: Union[Module, Sequential],
                 factors: List[Tensor]):
        """EFB class initializer.

        Args:
            model: Any (pre-trained) PyTorch model including all `torchvision` models.
            factors: The Kronecker factors Q and H, computed using the `KFAC` class.
        """
        super().__init__(model)
        self.eigvecs = get_eigenvectors(factors)
        self.diags = dict()

    def update(self,
               batch_size: int):
        """Computes the eigenvalue corrected diagonal of the Fisher information matrix.

        Args:
            batch_size: The size of the current batch.
        """
        layer = 0
        for module in self.model.modules():
            if module.__class__.__name__ in ['Linear', 'Conv2d']:
                grads = module.weight.grad.contiguous().view(module.weight.grad.shape[0], -1)
                if module.bias is not None:
                    grads = torch.cat([grads, module.bias.grad.unsqueeze(dim=1)], dim=1)
                lambdas = (self.eigvecs[layer][1].t() @ grads @ self.eigvecs[layer][0]) ** 2

                if module in self.state:
                    self.state[module] += lambdas
                    self.diags[module] += grads ** 2 * batch_size
                else:
                    self.state[module] = lambdas
                    self.diags[module] = grads ** 2 * batch_size
                layer += 1


class INF:
    """Computes the diagonal correction term and low-rank approximations of KFAC factor eigenvectors and EFB diagonals.

    Todo: Add more info from paper.
    """
    def __init__(self,
                 factors: List[Tensor],
                 lambdas: List[Tensor],
                 diags: List[Tensor]):
        self.eigvecs = get_eigenvectors(factors)
        self.lambdas = lambdas
        self.diags = diags
        self.state = list()

    def accumulate(self,
                   rank: int = 100):
        """Accumulates the diagonal values used for the diagonal correction term.

        Todo: Add more info from paper.
        Args:
            rank: The rank of the low-rank approximations.
        """
        for eigvecs, lambdas, diags in tqdm(zip(self.eigvecs, self.lambdas, self.diags), total=len(self.eigvecs)):
            xxt_eigvecs, ggt_eigvecs = eigvecs
            lambda_vec = lambdas.t().contiguous().view(-1)
            diag_vec = diags.t().contiguous().view(-1)

            lr_xxt_eigvecs, lr_ggt_eigvecs, lr_lambda = self._dim_reduction(xxt_eigvecs, ggt_eigvecs, lambda_vec, rank)
            sif_diag = self._diagonal_accumulator(lr_xxt_eigvecs, lr_ggt_eigvecs, lr_lambda)

            self.state.append([lr_xxt_eigvecs, lr_ggt_eigvecs, lr_lambda, diag_vec - sif_diag])

    @staticmethod
    def _dim_reduction(xxt_eigvecs: List[Tensor],
                       ggt_eigvecs: List[Tensor],
                       lambda_vec: Tensor,
                       rank: int):
        """

        Args:
            xxt_eigvecs:
            ggt_eigvecs:
            lambda_vec:
            rank:

        Returns:

        """
        if rank >= lambda_vec.shape[0]:
            return xxt_eigvecs, ggt_eigvecs, lambda_vec
        else:
            m = ggt_eigvecs.shape[1]
            idx_total_0 = torch.argsort(-torch.abs(lambda_vec))
            idx_total = idx_total_0 + 1
            idx_top_l = idx_total[0:rank]
            idx_left = list()
            idx_right = list()
            for z in range(rank):
                i = int((idx_top_l[z] - 1.) / m + 1.)
                j = idx_top_l[z] - (m * (i - 1))
                idx_left.append(i)
                idx_right.append(j)

            idx_top_lm = list()
            idx_left = torch.unique(torch.tensor(idx_left))
            idx_right = torch.unique(torch.tensor(idx_right))
            len_l = len(idx_left)
            len_r = len(idx_right)

            for i in range(0, len_l):
                for j in range(0, len_r):
                    idx_top_lm.append(m * (idx_left[i] - 1) + idx_right[j])

            lr_lambda = lambda_vec[[idx - 1 for idx in idx_top_lm]]
            lr_cov_inner = xxt_eigvecs[:, [idx - 1 for idx in idx_left]]
            lr_cov_outer = ggt_eigvecs[:, [idx - 1 for idx in idx_right]]

            return lr_cov_inner, lr_cov_outer, lr_lambda

    @staticmethod
    def _diagonal_accumulator(xxt_eigvecs: List[Tensor],
                              ggt_eigvecs: List[Tensor],
                              lambda_vec: List[Tensor]):
        """

        Args:
            xxt_eigvecs:
            ggt_eigvecs:
            lambda_vec:

        Returns:

        """
        n = xxt_eigvecs.shape[0]
        m = ggt_eigvecs.shape[0]
        diag_vec = torch.zeros(n * m).to(lambda_vec.device)
        k = 0

        for i in range(n):
            diag_kron = kron(xxt_eigvecs[i, :].unsqueeze(0), ggt_eigvecs) ** 2
            diag_vec[k:k + m] = diag_kron @ lambda_vec
            k += m
        return diag_vec
