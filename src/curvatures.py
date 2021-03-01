"""Various Fisher information matrix approximations."""

from abc import ABC, abstractmethod
from typing import Union, List, Any, Dict
import copy

from numpy.linalg import inv, cholesky
import torch
from torch import Tensor
from torch.nn import Module, Sequential
import torch.nn.functional as F
from tqdm import tqdm

from src.utils import get_eigenvectors, kron


class Curvature(ABC):
    """Base class for all src approximations.

    All src approximations are computed layer-wise (i.e. layer-wise independence is assumed s.t. no
    covariances between layers are computed, aka block-wise approximation) and stored in `state`.

    The src of the loss function is the matrix of 2nd-order derivatives of the loss w.r.t. the networks weights
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
                 model: Union[Module, Sequential],
                 layer_types: Union[List[str], str] = None):
        """Curvature class initializer.

        Args:
            model: Any (pre-trained) PyTorch model including all `torchvision` models.
            layer_types: Types of layers for which to compute src information. Supported are `Linear`, `Conv2d`
                         and `MultiheadAttention`. If `None`, all supported types are considered. Default: None.
        """
        self.model = model
        self.model_state = copy.deepcopy(model.state_dict())
        self.layer_types = list()
        if isinstance(layer_types, str):
            self.layer_types.append(layer_types)
        elif isinstance(layer_types, list):
            if layer_types:
                self.layer_types.extend(layer_types)
            else:
                self.layer_types.extend(['Linear', 'Conv2d', 'MultiheadAttention'])
        elif layer_types is None:
            self.layer_types.extend(['Linear', 'Conv2d', 'MultiheadAttention'])
        else:
            raise TypeError
        for _type in self.layer_types:
            assert _type in ['Linear', 'Conv2d', 'MultiheadAttention']
        self.state = dict()
        self.inv_state = dict()

    @staticmethod
    def _replace(sample: Tensor,
                 weight: Tensor,
                 bias: Tensor = None):
        """Modifies current model parameters by adding/subtracting quantity given in `sample`.

        Args:
            sample: Sampled offset from the mean dictated by the inverse src (variance).
            weight: The weights of one model layer.
            bias: The bias of one model layer. Optional.
        """
        if bias is not None:
            bias_sample = sample[:, -1].contiguous().view(*bias.shape)
            bias.data.add_(bias_sample)
            sample = sample[:, :-1]
        weight.data.add_(sample.contiguous().view(*weight.shape))

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any):
        """Abstract method to be implemented by each derived class individually."""
        raise NotImplementedError

    @abstractmethod
    def invert(self,
               add: Union[float, list, tuple] = 0.,
               multiply: Union[float, list, tuple] = 1.):
        """Abstract method to be implemented by each derived class individually. Inverts state.

        Args:
            add: This quantity times the identity is added to each src factor.
            multiply: Each factor is multiplied by this quantity.

        Returns:
            A dict of inverted factors and potentially other quantities required for sampling.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self,
               layer: Module) -> Tensor:
        """Abstract method to be implemented by each derived class individually. Samples from inverted state.

        Args:
            layer: A layer instance from the current model.

        Returns:
            A tensor with newly sampled weights for the given layer.
        """
        raise NotImplementedError

    def sample_and_replace(self):
        """Samples new model parameters and replaces old ones for selected layers, skipping all others."""
        self.model.load_state_dict(self.model_state)
        for layer in self.model.modules():
            if layer.__class__.__name__ in self.layer_types:
                if layer.__class__.__name__ in ['Linear', 'Conv2d']:
                    _sample = self.sample(layer)
                    self._replace(_sample, layer.weight, layer.bias)
                elif layer.__class__.__name__ == 'MultiheadAttention':
                    for weight, bias, layer in [(layer.in_proj_weight, layer.in_proj_bias, 'attn_in'),
                                                (layer.out_proj.weight, layer.out_proj.bias, 'attn_out')]:
                        _sample = self.sample(layer)
                        self._replace(_sample, weight, bias)


class Diagonal(Curvature):
    r"""The diagonal Fisher information or Generalized Gauss Newton matrix approximation.

    It is defined as :math:`F_{DIAG}=\mathrm{diag}(F)` with `F` being the Fisher defined in the `FISHER` class.
    Code inspired by https://github.com/wjmaddox/swa_gaussian/blob/master/swag/posteriors/diag_laplace.py.

    Source: `A Scalable Laplace Approximation for Neural Networks <https://openreview.net/pdf?id=Skdvd2xAZ>`_
    """

    def update(self,
               batch_size: int):
        """Computes the diagonal src for selected layer types, skipping all others.

        Args:
            batch_size: The size of the current batch.
        """
        for layer in self.model.modules():
            if layer.__class__.__name__ in self.layer_types:
                if layer.__class__.__name__ in ['Linear', 'Conv2d']:
                    grads = layer.weight.grad.contiguous().view(layer.weight.grad.shape[0], -1)
                    if layer.bias is not None:
                        grads = torch.cat([grads, layer.bias.grad.unsqueeze(dim=1)], dim=1)
                    grads = grads ** 2 * batch_size
                    if layer in self.state:
                        self.state[layer] += grads
                    else:
                        self.state[layer] = grads
                elif layer.__class__.__name__ == 'MultiheadAttention':
                    grads = layer.in_proj_weight.grad.contiguous().view(layer.in_proj_weight.grad.shape[0], -1)
                    grads = torch.cat([grads, layer.in_proj_bias.grad.unsqueeze(dim=1)], dim=1)
                    grads = grads ** 2 * batch_size
                    if 'attn_in' in self.state:
                        self.state['attn_in'] += grads
                    else:
                        self.state['attn_in'] = grads

                    grads = layer.out_proj.weight.grad.contiguous().view(layer.out_proj.weight.grad.shape[0], -1)
                    grads = torch.cat([grads, layer.out_proj.bias.grad.unsqueeze(dim=1)], dim=1)
                    grads = grads ** 2 * batch_size
                    if 'attn_out' in self.state:
                        self.state['attn_out'] += grads
                    else:
                        self.state['attn_out'] = grads

    def invert(self,
               add: Union[float, list, tuple] = 0.,
               multiply: Union[float, list, tuple] = 1.):
        assert self.state, "State dict is empty. Did you call 'update' prior to this?"
        if self.inv_state:
            Warning("State has already been inverted. Is this expected?")
        for index, (layer, value) in enumerate(self.state.items()):
            if isinstance(add, (list, tuple)) and isinstance(multiply, (list, tuple)):
                assert len(add) == len(multiply) == len(self.state)
                n, s = add[index], multiply[index]
            else:
                n, s = add, multiply
            self.inv_state[layer] = torch.reciprocal(s * value + n).sqrt()

    def sample(self,
               layer: Union[Module, str]):
        assert self.inv_state, "Inverse state dict is empty. Did you call 'invert' prior to this?"
        return self.inv_state[layer].new(self.inv_state[layer].size()).normal_() * self.inv_state[layer]


class BlockDiagonal(Curvature):
    r"""The block-diagonal Fisher information or Generalized Gauss Newton matrix approximation.

    It can be defined as the expectation of the outer product of the gradient of the networks loss E w.r.t. its
    weights W: :math:`F=\mathbb{E}\left[\nabla_W E(W)\nabla_W E(W)^T\right]`

    Source: `A Scalable Laplace Approximation for Neural Networks <https://openreview.net/pdf?id=Skdvd2xAZ>`_
    """
    def update(self,
               batch_size: int):
        """Computes the block-diagonal (per-layer) src selected layer types, skipping all others.

        Args:
            batch_size: The size of the current batch.
        """
        for layer in self.model.modules():
            if layer.__class__.__name__ in self.layer_types:
                if layer.__class__.__name__ in ['Linear', 'Conv2d']:
                    grads = layer.weight.grad.contiguous().view(-1)
                    if layer.bias is not None:
                        grads = torch.cat([grads, layer.bias.grad])
                    grads = torch.ger(grads, grads) * batch_size
                    if layer in self.state:
                        self.state[layer] += grads
                    else:
                        self.state[layer] = grads
                elif layer.__class__.__name__ == 'MultiheadAttention':
                    grads = layer.in_proj_weight.grad.contiguous().view(layer.in_proj_weight.grad.shape[0], -1)
                    if layer.in_proj_bias is not None:
                        grads = torch.cat([grads, layer.in_proj_bias.grad])
                    grads = torch.ger(grads, grads) * batch_size
                    if 'attn_in' in self.state:
                        self.state['attn_in'] += grads
                    else:
                        self.state['attn_in'] = grads

                    grads = layer.out_proj.weight.grad.contiguous().view(layer.out_proj.weight.grad.shape[0], -1)
                    if layer.out_proj.bias is not None:
                        grads = torch.cat([grads, layer.out_proj.bias.grad])
                    grads = torch.ger(grads, grads) * batch_size
                    if 'attn_out' in self.state:
                        self.state['attn_out'] += grads
                    else:
                        self.state['attn_out'] = grads

    def invert(self,
               add: Union[float, list, tuple] = 0.,
               multiply: Union[float, list, tuple] = 1.):
        assert self.state, "State dict is empty. Did you call 'update' prior to this?"
        if self.inv_state:
            Warning("State has already been inverted. Is this expected?")
        for index, (layer, value) in enumerate(self.state.items()):
            if not isinstance(add, float) and not isinstance(multiply, float):
                assert len(add) == len(multiply) == len(self.state)
                n, s = add[index], multiply[index]
            else:
                n, s = add, multiply
            reg = torch.diag(value.new(value.shape[0]).fill_(n))
            self.inv_state[layer] = (s * value + reg).inverse().cholesky()

    def sample(self,
               layer: Module) -> Tensor:
        assert self.inv_state, "Inverse state dict is empty. Did you call 'invert' prior to this?"
        x = self.inv_state[layer].new(self.inv_state[layer].shape[0]).normal_() @ self.inv_state[layer]
        return torch.cat([x[:layer.weight.numel()].contiguous().view(*layer.weight.shape),
                          torch.unsqueeze(x[layer.weight.numel():], dim=1)], dim=1)


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
                 model: Union[Module, Sequential],
                 layer_types: Union[List[str], str] = None):
        """KFAC class initializer.

        For the recursive computation of `H`, outputs and inputs for each layer are recorded in `record`. Forward and
        backward hook handles are stored in `hooks` for subsequent removal.

        Args:
            model: Any (pre-trained) PyTorch model including all `torchvision` models.
        """
        super().__init__(model, layer_types)
        self.hooks = list()
        self.record = dict()

        for layer in model.modules():
            if layer.__class__.__name__ in self.layer_types:
                if layer.__class__.__name__ in ['Linear', 'Conv2d']:
                    self.record[layer] = [None, None]
                    self.hooks.append(layer.register_forward_pre_hook(self._save_input))
                    self.hooks.append(layer.register_backward_hook(self._save_output))
                elif layer.__class__.__name__ == 'MultiheadAttention':
                    raise NotImplementedError

    def _save_input(self, module, input):
        self.record[module][0] = input[0]

    def _save_output(self, module, grad_input, grad_output):
        self.record[module][1] = grad_output[0] * grad_output[0].size(0)

    def update(self,
               batch_size: int):
        """Computes the 1st and 2nd Kronecker factor `Q` and `H` for each selected layer type, skipping all others.

        Todo: Check code against papers.

        Args:
            batch_size: The size of the current batch.
        """
        for layer in self.model.modules():
            module_class = layer.__class__.__name__
            if layer.__class__.__name__ in self.layer_types:
                if module_class in ['Linear', 'Conv2d']:
                    forward, backward = self.record[layer]

                    # 1st factor: Q
                    if module_class == 'Conv2d':
                        forward = F.unfold(forward, layer.kernel_size, padding=layer.padding, stride=layer.stride)
                        forward = forward.data.permute(1, 0, 2).contiguous().view(forward.shape[1], -1)
                    else:
                        forward = forward.data.t()
                    if layer.bias is not None:
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
                    if layer in self.state:
                        self.state[layer][0] += first_factor
                        self.state[layer][1] += second_factor
                    else:
                        self.state[layer] = [first_factor, second_factor]
                elif layer.__class__.__name__ == 'MultiheadAttention':
                    raise NotImplementedError

    def invert(self,
               add: Union[float, list, tuple] = 0.,
               multiply: Union[float, list, tuple] = 1.):
        assert self.state, "State dict is empty. Did you call 'update' prior to this?"
        if self.inv_state:
            Warning("State has already been inverted. Is this expected?")
        for index, (layer, value) in enumerate(self.state.items()):
            if not isinstance(add, (float, int)) and not isinstance(multiply, (float, int)):
                assert len(add) == len(multiply) == len(self.state)
                n, s = add[index], multiply[index]
            else:
                n, s = float(add), float(multiply)
            first, second = value

            diag_frst = torch.diag(first.new(first.shape[0]).fill_(n ** 0.5))
            diag_scnd = torch.diag(second.new(second.shape[0]).fill_(n ** 0.5))

            reg_frst = s ** 0.5 * first + diag_frst
            reg_scnd = s ** 0.5 * second + diag_scnd

            reg_frst = (reg_frst + reg_frst.t()) / 2.0
            reg_scnd = (reg_scnd + reg_scnd.t()) / 2.0

            try:
                chol_ifrst = reg_frst.inverse().cholesky()
                chol_iscnd = reg_scnd.inverse().cholesky()
            except RuntimeError:
                print("PyTorch Cholesky is singular. Using Numpy.")
                chol_ifrst = torch.from_numpy(cholesky(inv(reg_frst.cpu().numpy()))).to(first.device)
                chol_iscnd = torch.from_numpy(cholesky(inv(reg_scnd.cpu().numpy()))).to(second.device)

            self.inv_state[layer] = (chol_ifrst, chol_iscnd)

    def sample(self,
               layer: Module) -> Tensor:
        assert self.inv_state, "Inverse state dict is empty. Did you call 'invert' prior to this?"
        first, second = self.inv_state[layer]
        z = torch.randn(first.size(0), second.size(0), device=first.device, dtype=first.dtype)
        return (first @ z @ second.t()).t()  # Final transpose because PyTorch uses channels first


class EFB(Curvature):
    """The eigenvalue corrected Kronecker-factored Fisher information or Generalized Gauss Newton matrix.

    Todo: Add source/equations.
    """
    def __init__(self,
                 model: Union[Module, Sequential],
                 factors: Dict[Module, Tensor],
                 layer_types: Union[List[str], str] = None):
        """EFB class initializer.

        Args:
            model: Any (pre-trained) PyTorch model including all `torchvision` models.
            factors: The Kronecker factors Q and H, computed using the `KFAC` class.
        """
        super().__init__(model, layer_types)
        self.eigvecs = get_eigenvectors(factors)
        self.diags = dict()

    def update(self,
               batch_size: int):
        """Computes the eigenvalue corrected diagonal of the FiM or GNN.

        Args:
            batch_size: The size of the current batch.
        """
        for layer in self.model.modules():
            if layer.__class__.__name__ in self.layer_types:
                if layer.__class__.__name__ in ['Linear', 'Conv2d']:
                    grads = layer.weight.grad.contiguous().view(layer.weight.grad.shape[0], -1)
                    if layer.bias is not None:
                        grads = torch.cat([grads, layer.bias.grad.unsqueeze(dim=1)], dim=1)
                    lambdas = (self.eigvecs[layer][1].t() @ grads @ self.eigvecs[layer][0]) ** 2

                    if layer in self.state:
                        self.state[layer] += lambdas
                        self.diags[layer] += grads ** 2 * batch_size
                    else:
                        self.state[layer] = lambdas
                        self.diags[layer] = grads ** 2 * batch_size
                elif layer.__class__.__name__ == 'MultiheadAttention':
                    raise NotImplementedError

    def invert(self,
               add: Union[float, list, tuple] = 0.,
               multiply: Union[float, list, tuple] = 1.):
        assert self.state, "State dict is empty. Did you call 'update' prior to this?"
        if self.inv_state:
            Warning("State has already been inverted. Is this expected?")
        for index, (layer, value) in enumerate(self.state.items()):
            if not isinstance(add, float) and not isinstance(multiply, float):
                assert len(add) == len(multiply) == len(self.state)
                n, s = add[index], multiply[index]
            else:
                n, s = add, multiply
            reg_inv_lambda = torch.reciprocal(s * value + n).sqrt()
            self.inv_state[layer] = reg_inv_lambda

    def sample(self,
               layer: Module) -> Tensor:
        assert self.inv_state, "Inverse state dict is empty. Did you call 'invert' prior to this?"
        first, second = self.eigvecs[layer]
        lambdas = self.inv_state[layer]
        z = torch.randn(first.size(0), second.size(0), device=first.device, dtype=first.dtype)
        z *= lambdas.t()
        return (first @ z @ second.t()).t()  # Final transpose because PyTorch uses channels first


class INF(Curvature):
    """Computes the diagonal correction term and low-rank approximations of KFAC factor eigenvectors and EFB diagonals.

    Todo: Add more info from paper.
    """
    def __init__(self,
                 model: Union[Module, Sequential],
                 diags: Dict[Module, Tensor],
                 factors: Dict[Module, Tensor],
                 lambdas: Dict[Module, Tensor],
                 layer_types: Union[List[str], str] = None):
        """INF class initializer.

        Args:
            diags: Diagonal FiM or GNN computed by `Diagonal` class.
            factors: Kronecker-factored FiM or GNN computed by `KFAC` class.
            lambdas: Eigenvalue corrected diagonal FiM or GNN computed by `EFB` class.
        """
        super().__init__(model, layer_types)
        assert diags.keys() == factors.keys() == lambdas.keys()
        self.eigvecs = get_eigenvectors(factors)
        self.lambdas = lambdas
        self.diags = diags

    def update(self,
               rank: int = 100):
        """Accumulates the diagonal values used for the diagonal correction term.

        Todo: Add more info from paper.
        Args:
            rank: The rank of the low-rank approximations.
        """
        values = zip(list(self.diags.keys()),
                     list(self.eigvecs.values()),
                     list(self.lambdas.values()),
                     list(self.diags.values()))
        for layer, eigvecs, lambdas, diags in tqdm(values, total=len(self.diags)):
            xxt_eigvecs, ggt_eigvecs = eigvecs
            lambda_vec = lambdas.t().contiguous().view(-1)
            diag_vec = diags.t().contiguous().view(-1)

            lr_xxt_eigvecs, lr_ggt_eigvecs, lr_lambda = self._dim_reduction(xxt_eigvecs, ggt_eigvecs, lambda_vec, rank)
            sif_diag = self._diagonal_accumulator(lr_xxt_eigvecs, lr_ggt_eigvecs, lr_lambda)

            self.state[layer] = (lr_xxt_eigvecs, lr_ggt_eigvecs, lr_lambda, diag_vec - sif_diag)

    def invert(self,
               add: Union[float, list, tuple] = 0.,
               multiply: Union[float, list, tuple] = 1.):
        assert self.state, "State dict is empty. Did you call 'update' prior to this?"
        if self.inv_state:
            Warning("State has already been inverted. Is this expected?")
        for index, (layer, value) in enumerate(self.state.items()):
            if not isinstance(add, float) and not isinstance(multiply, float):
                assert len(add) == len(multiply) == len(self.state)
                n, s = add[index], multiply[index]
            else:
                n, s = add, multiply
            # Low-rank U_A, low-rank U_G, low-rank Lambda, D
            lr_frst_eigvecs, lr_scnd_eigvecs, lr_lambda, correction = value
            correction[correction < 0] = 0

            reg_lr_lambda = (s * lr_lambda).sqrt()
            reg_inv_correction = torch.reciprocal(s * correction + n).sqrt()

            pre_sample = self.pre_sampler(lr_frst_eigvecs, lr_scnd_eigvecs, reg_lr_lambda, reg_inv_correction)

            self.inv_state[layer] = (lr_frst_eigvecs, lr_scnd_eigvecs, reg_inv_correction, pre_sample)

    def sample(self,
               layer: Module) -> Tensor:
        assert self.inv_state, "Inverse state dict is empty. Did you call 'invert' prior to this?"
        a, b, c, d = self.inv_state[layer]
        return self.sampler(a, b, c, d).reshape(a.shape[0], b.shape[0]).t()

    @staticmethod
    def pre_sampler(frst_eigvecs: torch.Tensor,
                    scnd_eigvecs: torch.Tensor,
                    reg_lambda: torch.Tensor,
                    reg_inv_correction: torch.Tensor) -> torch.Tensor:
        """Pre-sampler for INF sampling. Only needs to be called once.

        Args:
            frst_eigvecs: Eigenvectors of first KFAC factor.
            scnd_eigvecs: Eigenvectors of second KFAC factor.
            reg_lambda: Regularized, eigenvalue corrected diagonal FIM (computed by EFB)
            reg_inv_correction: Regularized inverse of the diagonal correction term of INF.

        Returns:
            A pre-sample used in `sampler` to sample weight sets.
        """
        scale_sqrt = torch.diag(reg_lambda)
        try:
            V_s = reg_inv_correction.contiguous().view(-1, 1) * kron(frst_eigvecs, scnd_eigvecs) @ scale_sqrt
        except RuntimeError:
            print("GPU capacity exhausted. Pre-sampling on CPU.")
            frst_eigvecs = frst_eigvecs.cpu()
            scnd_eigvecs = scnd_eigvecs.cpu()
            reg_inv_correction = reg_inv_correction.cpu()
            scale_sqrt = scale_sqrt.cpu()
            V_s = reg_inv_correction.contiguous().view(-1, 1) * kron(frst_eigvecs, scnd_eigvecs) @ scale_sqrt
        vtv = V_s.t() @ V_s
        vtv = (vtv + vtv.t()) / 2.
        A_c_inv = vtv.cholesky().inverse()
        B_c = (vtv + torch.eye(scale_sqrt.shape[0], device=scale_sqrt.device)).cholesky()
        C = A_c_inv.t() @ (B_c - torch.eye(scale_sqrt.shape[0], device=scale_sqrt.device)) @ A_c_inv
        L_c = (C.inverse() + vtv).inverse()
        P_c = scale_sqrt @ L_c @ scale_sqrt

        return P_c.to(reg_lambda.device)

    @staticmethod
    def sampler(frst_eigvecs: Tensor,
                scnd_eigvecs: Tensor,
                reg_inv_correction: Tensor,
                pre_sample: Tensor) -> Tensor:
        """Samples a new set of weights from the INF weight posterior distribution for the current layer.

        Args:
            frst_eigvecs: Eigenvectors of first KFAC factor.
            scnd_eigvecs: Eigenvectors of second KFAC factor.
            reg_inv_correction: Regularized inverse of the diagonal correction term of INF.
            pre_sample: Pre-sample computed by the pre-sampler.

        Returns:
            A new set of weights for the current layer.
        """
        X = torch.randn(frst_eigvecs.shape[0] * scnd_eigvecs.shape[0], device=frst_eigvecs.device,
                        dtype=frst_eigvecs.dtype)
        Y_l = reg_inv_correction * X
        unvec_Y_l = Y_l.t().reshape((scnd_eigvecs.shape[0], frst_eigvecs.shape[0]))
        Xq = scnd_eigvecs.t() @ unvec_Y_l @ frst_eigvecs
        Qx = pre_sample @ Xq.t().contiguous().view(-1)
        unvec_Qx = Qx.t().reshape((scnd_eigvecs.shape[1], frst_eigvecs.shape[1]))
        X_p_s = scnd_eigvecs @ unvec_Qx @ frst_eigvecs.t()
        Y_r = reg_inv_correction ** 2 * X_p_s.t().contiguous().view(-1)

        return Y_l.t() - Y_r.t()

    @staticmethod
    def _dim_reduction(frst_eigvecs: Tensor,
                       scnd_eigvecs: Tensor,
                       lambda_vec: Tensor,
                       rank: int):
        """

        Args:
            frst_eigvecs:
            scnd_eigvecs:
            lambda_vec:
            rank:

        Returns:

        """
        if rank >= lambda_vec.shape[0]:
            return frst_eigvecs, scnd_eigvecs, lambda_vec
        else:
            m = scnd_eigvecs.shape[1]
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
            lr_cov_inner = frst_eigvecs[:, [idx - 1 for idx in idx_left]]
            lr_cov_outer = scnd_eigvecs[:, [idx - 1 for idx in idx_right]]

            return lr_cov_inner, lr_cov_outer, lr_lambda

    @staticmethod
    def _diagonal_accumulator(xxt_eigvecs: Tensor,
                              ggt_eigvecs: Tensor,
                              lambda_vec: Tensor):
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
