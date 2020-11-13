"""Provides utility functions except for plotting which are in `plot.py`."""

import argparse
import multiprocessing
import os
from typing import Tuple, List, Union
from datetime import datetime
import random
import logging

import numpy as np
from numpy import ndarray as array
import psutil
import torch
from torch import Tensor
from tqdm import tqdm
from scipy.stats import entropy


def get_eigenvalues(factors: List[Tensor],
                    verbose: bool = False) -> Tensor:
    """Computes the eigenvalues of KFAC, EFB or diagonal factors.

    Args:
        factors: A list of KFAC, EFB or diagonal factors.
        verbose: Prints out progress if True.

    Returns:
        The eigenvalues of all KFAC, EFB or diagonal factors.
    """
    eigenvalues = Tensor()
    factors = tqdm(factors, disable=not verbose)
    for layer, factor in enumerate(factors):
        factors.set_description(desc=f"Layer [{layer + 1}/{len(factors)}]")
        if len(factor) == 2:
            xxt_eigvals = torch.symeig(factor[0])[0]
            ggt_eigvals = torch.symeig(factor[1])[0]
            eigenvalues = torch.cat([eigenvalues, torch.ger(xxt_eigvals, ggt_eigvals).contiguous().view(-1)])
        else:
            eigenvalues = torch.cat([eigenvalues, factor.contiguous().view(-1)])
    return eigenvalues


def get_eigenvectors(factors: List[Tensor]) -> List[List[Tensor]]:
    """Computes the eigenvectors of KFAC factors.

    Args:
        factors: A list of KFAC factors.

    Returns:
        A list where each element is a list of first and second KFAC factors eigenvectors.
    """
    eigenvectors = list()
    for (xxt, ggt) in factors:
        sym_xxt, sym_ggt = xxt + xxt.t(), ggt + ggt.t()
        _, xxt_eigvecs = torch.symeig(sym_xxt, eigenvectors=True)
        _, ggt_eigvecs = torch.symeig(sym_ggt, eigenvectors=True)
        eigenvectors.append([xxt_eigvecs, ggt_eigvecs])
    return eigenvectors


def linear_interpolation(min_val: float,
                         max_val: float,
                         data: array) -> array:
    """Performs a linear interpolation of `data` between `min_val` and `max_val`.

    Args:
        min_val: The lower bound of the interpolation.
        max_val: The upper bound of the interpolation.
        data: The data to be interpolated.

    Returns:
        The linearly interpolated data.
    """
    return (max_val - min_val) * (data - np.min(data)) / (np.max(data) - np.min(data)) + min_val


def accuracy(probabilities: array,
             labels: array) -> float:
    """Computes the top 1 accuracy of the predicted class probabilities in percent.

    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.

    Returns:
        The top 1 accuracy in percent.
    """
    return 100.0 * np.mean(np.argmax(probabilities, axis=1) == labels)


def binned_kl_distance(dist1: array,
                       dist2: array,
                       smooth: float = 1e-7,
                       bins: array = np.logspace(-7, 1, num=200)) -> float:
    """Computes the symmetric, discrete Kulback-Leibler divergence (JSD) between two distributions.

    Source: `A Simple Baseline for Bayesian Neural Networks <https://arxiv.org/abs/1902.02476>`_.

    Args:
        dist1: The first distribution.
        dist2: The second distribution.
        smooth: Smoothing factor to prevent numerical instability.
        bins: How to discretize the distributions.

    Returns:
        The JSD.
    """
    dist1_pdf, _ = np.histogram(dist1, bins)
    dist2_pdf, _ = np.histogram(dist2, bins)

    dist1_pdf = dist1_pdf + smooth
    dist2_pdf = dist2_pdf + smooth

    dist1_pdf_normalized = dist1_pdf / dist1_pdf.sum()
    dist2_pdf_normalized = dist2_pdf / dist2_pdf.sum()

    dir1_normalized_entropy = entropy(dist1_pdf_normalized, dist2_pdf_normalized)
    dir2_normalized_entropy = entropy(dist2_pdf_normalized, dist1_pdf_normalized)

    return dir1_normalized_entropy + dir2_normalized_entropy


def confidence(probabilities: array,
               mean: bool = True) -> Union[float, array]:
    """The confidence of a prediction is the maximum of the predicted class probabilities.

    Args:
        probabilities: The predicted class probabilities.
        mean: If True, returns the average confidence over all provided predictions.

    Returns:
        The confidence.
    """
    if mean:
        return np.mean(np.max(probabilities, axis=1))
    return np.max(probabilities, axis=1)


def negative_log_likelihood(probabilities: array,
                            labels: array) -> float:
    """Computes the Negative Log-Likelihood (NLL) of the predicted class probabilities.

    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.

    Returns:
        The NLL.
    """
    return -np.mean(np.log(probabilities[np.arange(probabilities.shape[0]), labels] + 1e-12))


def calibration_curve(probabilities: array,
                      labels: array,
                      bins: int = 20) -> Tuple[float, array, array, array]:
    r"""Computes the Expected Calibration Error (ECE) of the predicted class probabilities.

    With accuracy `acc` and confidence `conf`, it is defined as
    :math:`ECE=\sum_{m=1}^M\frac{\left|B_m\right|}{n}\left|\mathrm{acc}(B_M)-\mathrm{conf}(B_m)\right|`
    where `n` is the number of samples and `B_m` are all samples in bin `m` from `M` with equal number of samples.

    Source: `A Simple Baseline for Bayesian Neural Networks <https://arxiv.org/abs/1902.02476>`_.

    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.
        bins: The number of bins into which the probabilities are discretized.

    Returns:
        The ECE alongside the average confidence, accuracy and proportion of data points in each bin respectively.
    """
    confidences = np.max(probabilities, 1)
    step = (confidences.shape[0] + bins - 1) // bins
    bins = np.sort(confidences)[::step]
    if confidences.shape[0] % step != 1:
        bins = np.concatenate((bins, [np.max(confidences)]))
    # bins = np.linspace(0.1, 1.0, 30)
    predictions = np.argmax(probabilities, 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    accuracies = predictions == labels

    xs = []
    ys = []
    zs = []

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) * (confidences < bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            xs.append(avg_confidence_in_bin)
            ys.append(accuracy_in_bin)
            zs.append(prop_in_bin)
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    return ece, xs, ys, zs


def expected_calibration_error(probabilities: array,
                               labels: array,
                               bins: int = 10) -> Tuple[float, array, array, array]:
    r"""Computes the Expected Calibration Error (ECE) of the predicted class probabilities.

    With accuracy `acc` and confidence `conf`, it is defined as
    :math:`ECE=\sum_{m=1}^M\frac{\left|B_m\right|}{n}\left|\mathrm{acc}(B_M)-\mathrm{conf}(B_m)\right|`
    where `n` is the number of samples and `B_m` are all samples in bin `m` from `M` equally-spaced bins.

    Source: `On Calibration of Modern Neural Networks <https://arxiv.org/pdf/1706.04599.pdf)?>`_.

    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.
        bins: The number of bins into which the probabilities are discretized.

    Returns:
        The ECE alongside the average confidence, accuracy and proportion of data points in each bin respectively.
    """
    conf = confidence(probabilities, mean=False)
    edges = np.linspace(0, 1, bins + 1)
    bin_ace = list()
    bin_accuracy = list()
    bin_confidence = list()
    ece = 0
    for i in range(bins):
        mask = np.logical_and(conf > edges[i], conf <= edges[i + 1])
        if any(mask):
            bin_acc = accuracy(probabilities[mask], labels[mask]) / 100
            bin_conf = conf[mask].mean()
            ace = bin_conf - bin_acc
            ece += mask.mean() * np.abs(ace)

            bin_ace.append(ace)
            bin_accuracy.append(bin_acc)
            bin_confidence.append(bin_conf)
        else:
            bin_ace.append(0)
            bin_accuracy.append(0)
            bin_confidence.append(0)
    return ece, np.array(bin_ace), np.array(bin_accuracy), np.array(bin_confidence)


def predictive_entropy(probabilities: array,
                       mean: bool = False) -> Union[array, float]:
    r"""Computes the predictive entropy of the predicted class probabilities.

    It is defined as :math:`H(y)=-\sum_{c=1}^K y_c\ln y_c` where `y_c` is the predicted class
    probability for class c and `K` is the number of classes.

    Args:
        probabilities: The predicted class probabilities.
        mean: If True, returns the average predictive entropy over all provided predictions.

    Returns:
        The predictive entropy.
    """
    pred_ent = np.apply_along_axis(entropy, axis=1, arr=probabilities)
    if mean:
        return np.mean(pred_ent)
    return pred_ent


def ram() -> float:
    """Returns the total amount of utilized system memory (RAM) in percent.

    Returns:
        RAM usage in percent.
    """
    return psutil.virtual_memory()[2]


def vram() -> float:
    """Determines the amount of video memory (VRAM) utilized by the current process in GB.

    Returns:
        VRAM usage in GB.
    """
    return torch.cuda.memory_allocated() / (1024.0 ** 3)


def kron(a: Tensor,
         b: Tensor) -> Tensor:
    r"""Computes the Kronecker product between the two 2D-matrices (tensors) `a` and `b`.

    `Wikipedia example <https://en.wikipedia.org/wiki/Kronecker_product>`_.

    Args:
        a: A 2D-matrix
        b: A 2D-matrix

    Returns:
        The Kronecker product between `a` and `b`.

    Examples:
        >>> a = torch.tensor([[1, 2], [3, 4]])
        >>> b = torch.tensor([[0, 5], [6, 7]])
        >>> kron(a, b)
        tensor([[ 0,  5,  0, 10],
                [ 6,  7, 12, 14],
                [ 0, 15,  0, 20],
                [18, 21, 24, 28]])
    """
    return torch.einsum("ab,cd->acbd", [a, b]).contiguous().view(a.size(0) * b.size(0), a.size(1) * b.size(1))


def seed_all_rng(seed: Union[int, None] = None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed: The seed value to use. If None, will use a strong random seed.
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)


def setup(required: bool = True,
          seed: Union[int, None] = 42):
    """Initializes values of importance for most modules and parses command line arguments.

    Returns:
        The parsed arguments.
    """

    # Prepare CUDA
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    # Test existence of GPU and its capabilities
    gpu = torch.cuda.is_available()
    if gpu:
        capability = torch.cuda.get_device_capability()
        capable = capability[0] + 0.1 * capability[1] >= 3.5
        device = torch.device('cuda') if capable else torch.device('cpu')
    else:
        device = torch.device('cpu')

    # Obtain number of CPUs of the system
    cpus = multiprocessing.cpu_count()

    # Set base directories
    root_dir = os.getcwd()  # Location where data and weights are stored
    torch_dir = "~/.torch"  # Location of PyTorch
    results_dir = os.getcwd()  # Location where results should be stored

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=device, type=type(device), help="Computation device: GPU or CPU")
    parser.add_argument("--torch_dir", default=torch_dir, type=str, help="Path to torchvision modelzoo location")
    parser.add_argument("--root_dir", default=root_dir, type=str, required=required, help="Path to root dir")
    parser.add_argument("--results_dir", default=results_dir, type=str, required=required, help="Path to results dir")

    parser.add_argument("--mode", default="torch", type=str, help="GPU/PyTorch or CPU computation (default: Torch)")
    parser.add_argument("--parallel", action="store_true", help="Use data parallelism (default: off)")
    parser.add_argument("--ram", default=psutil.virtual_memory()[1] / 1023 ** 3,
                        help="Amount of available system memory on process start.")
    parser.add_argument("--cpus", default=cpus, type=int, help="Number of CPUs (default: Auto)")
    parser.add_argument("--workers", default=cpus - 1 if cpus < 6 else cpus - 2, type=int,
                        help="Data loading workers (default: cpus - 1)")
    parser.add_argument("--prefix", default="", type=str, help="Filename prefix (default: None)")
    parser.add_argument("--suffix", default="", type=str, help="Filename suffix (default: None)")

    parser.add_argument("--model", default=None, type=str, required=required,
                        help="Name of model to use (default: None)")
    parser.add_argument("--data", default="imagenet", type=str, help="Name of dataset (default: imagenet)")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size (default: 32)")
    parser.add_argument("--epochs", default=1, type=int, help="Number of (training) epochs (default: 1)")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate in SGD training (default: 1e-3)")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum in SGD training (default: 0.9)")
    parser.add_argument("--l2", default=0, type=float, help="L2-norm regularization strength (default: 0)")
    parser.add_argument("--optimizer", default="random", type=str,
                        help="Optimizer used for hyperparameter optimization (deafult: random)")

    parser.add_argument("--estimator", default="kfac", type=str, help="Fisher estimator (default: kfac)")
    parser.add_argument("--samples", default=30, type=int, help="Number of posterior weight samples (default: 30)")
    parser.add_argument("--calls", default=50, type=int, help="Number of hyperparameter search calls (default: 50)")
    parser.add_argument("--boundaries", action="store_true",
                        help="Whether the to search the hyperparameter space boundaries (default: off)")
    parser.add_argument("--exp_id", default=-1, type=str, help="Experiment ID (default: -1)")
    parser.add_argument("--layer", action="store_true", help="If layer-wise damping should be used. (default: off)")
    parser.add_argument("--pre_scale", default=1, type=int,
                        help="Size of dataset, multiplied by scaling factor (default: 1)")
    parser.add_argument("--augment", action="store_true", help="Whether to use data augmentation (default: off)")
    parser.add_argument("--norm", default=-1, type=float,
                        help="This times identity is added to Kronecker factors (default: -1)")
    parser.add_argument("--scale", default=-1, type=float,
                        help="Kronecker factors are multiplied by this times pre-scale (default: -1)")
    parser.add_argument("--epsilon", default=0, type=float, help="Step size for FGSM (default: 0)")
    parser.add_argument("--rank", default=100, type=int, help="Rank for information form sparsification (default: 100)")

    parser.add_argument("--plot", action="store_true",
                        help="Whether to plot the evaluation results or not (default: off)")
    parser.add_argument("--no_results", action="store_true",
                        help="Whether to not save the evaluation results (default: off)")
    parser.add_argument("--stats", action="store_true", help="Whether to compute running statistics (default: off)")
    parser.add_argument("--calibration", action="store_true", help="Make calibration plots (default: off)")
    parser.add_argument("--ood", action="store_true", help="Run ood evaluation/make ood plots (default: off)")
    parser.add_argument("--fgsm", action="store_true", help="Run FGSM evaluation/make fgsm plots (default: off)")
    parser.add_argument("--loss1d", action="store_true", help="Evaluate 1D loss data/make 1D loss plots (default: off)")
    parser.add_argument("--loss2d", action="store_true", help="Evaluate 2D loss data/make 2D loss plots (default: off)")
    parser.add_argument("--ecdf", action="store_true", help="Plot inverse ECDF vs. predictive entropy (default: off)")
    parser.add_argument("--entropy", action="store_true", help="Plot predictive entropy histogram (default: off)")
    parser.add_argument("--summary", action="store_true", help="Print a model summary (default: off)")
    parser.add_argument("--eigvals", action="store_true", help="Plot eigenvalue histogram (default: off)")
    parser.add_argument("--hyper", action="store_true", help="Plot hyperparameter optimization results (default: off)")
    parser.add_argument("--networks", action="store_true", help="Plot network calibration comparison")
    parser.add_argument("--landscapes", action="store_true", help="Plot loss and accuracy landscapes")
    parser.add_argument("--verbose", action="store_true", help="Give verbose output during execution. (default: off)")
    parser.add_argument("--seed", default=seed, type=int, help="Random seed (default: 42).")

    args = parser.parse_args()

    os.environ['TORCH_HOME'] = args.torch_dir  # Set `TORCH_HOME` to provided location
    seed_all_rng(args.seed)
    return args


if __name__ == "__main__":
    import doctest
    doctest.testmod()
