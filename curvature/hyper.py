import os
import warnings

import numpy as np
import skopt
import torch
import torchvision

import curvature.datasets as datasets
from .lenet5 import lenet5
from .resnet import resnet18
from .evaluate import eval_bnn
from .sampling import invert_factors
from .utils import (accuracy, setup, expected_calibration_error, predictive_entropy, negative_log_likelihood,
                    get_eigenvectors)
from .visualize import hyperparameters


def grid(func, dimensions):
    cost = list()
    norms, scales = dimensions
    for norm in norms:
        for scale in scales:
            cost.append(func(norm, scale))
    return cost


def main():
    args = setup()

    print("Preparing directories")
    filename = f"{args.prefix}{args.model}_{args.data}{args.suffix}"
    factors_path = os.path.join(args.root_dir, "factors",
                                f"{args.prefix}{args.model}_{args.data}_{args.estimator}{args.suffix}")
    weights_path = os.path.join(args.root_dir, "weights", f"{args.model}_{args.data}.pth")
    if args.exp_id == -1:
        if not args.no_results:
            os.makedirs(os.path.join(args.results_dir, args.model, "data", args.estimator, args.optimizer), exist_ok=True)
        if args.plot:
            os.makedirs(os.path.join(args.results_dir, args.model, "figures", args.estimator, args.optimizer), exist_ok=True)
        results_path = os.path.join(args.results_dir, args.model, "data", args.estimator, args.optimizer, filename)
    else:
        if not args.no_results:
            os.makedirs(os.path.join(args.results_dir, args.model, "data", args.estimator, args.optimizer, args.exp_id), exist_ok=True)
        if args.plot:
            os.makedirs(os.path.join(args.results_dir, args.model, "figures", args.estimator, args.optimizer, args.exp_id), exist_ok=True)
        results_path = os.path.join(args.results_dir, args.model, "data", args.estimator, args.optimizer, args.exp_id, filename)

    print("Loading model")
    if args.model == 'lenet5':
        model = lenet5(pretrained=args.data, device=args.device)
    elif args.model == 'resnet18' and args.data != 'imagenet':
        model = resnet18(pretrained=weights_path, num_classes=43 if args.data == 'gtsrb' else 10, device=args.device)
    else:
        model_class = getattr(torchvision.models, args.model)
        if args.model in ['googlenet', 'inception_v3']:
            model = model_class(pretrained=True, aux_logits=False)
        else:
            model = model_class(pretrained=True)
    model.to(args.device).eval()
    if args.parallel:
        model = torch.nn.parallel.DataParallel(model)

    print("Loading data")
    if args.data == 'mnist':
        val_loader = datasets.mnist(args.torch_data, splits='val')
    elif args.data == 'cifar10':
        val_loader = datasets.cifar10(args.torch_data, splits='val')
    elif args.data == 'gtsrb':
        val_loader = datasets.gtsrb(args.data_dir, batch_size=args.batch_size, splits='val')
    elif args.data == 'imagenet':
        img_size = 224
        if args.model in ['googlenet', 'inception_v3']:
            img_size = 299
        data_dir = os.path.join(args.root_dir, "datasets", "imagenet")
        val_loader = datasets.imagenet(data_dir, img_size, args.batch_size, splits="val")
    else:
        raise ValueError

    print("Loading factors")
    if args.estimator in ["diag", "kfac"]:
        factors = torch.load(factors_path + '.pth')
    elif args.estimator == 'efb':
        kfac_factors = torch.load(factors_path.replace("efb", "kfac") + '.pth')
        lambdas = torch.load(factors_path + '.pth')

        factors = list()
        eigvecs = get_eigenvectors(kfac_factors)

        for eigvec, lambda_ in zip(eigvecs, lambdas):
            factors.append((eigvec[0], eigvec[1], lambda_))
    elif args.estimator == 'inf':
        factors = torch.load(f"{factors_path}{args.rank}.pth")
    torch.backends.cudnn.benchmark = True

    norm_min = -10
    norm_max = 10
    scale_min = -10
    scale_max = 10
    if args.boundaries:
        x0 = [[norm_min, scale_min],
              [norm_max, scale_max],
              [norm_min, scale_max],
              [norm_max, scale_min],
              [norm_min / 2., scale_min],
              [norm_max / 2., scale_max],
              [norm_min, scale_max / 2.],
              [norm_max, scale_min / 2.],
              [norm_min / 2., scale_min / 2.],
              [norm_max / 2., scale_max / 2.],
              [norm_min / 2., scale_max / 2.],
              [norm_max / 2., scale_min / 2.]]
    else:
        x0 = None

    space = list()
    space.append(skopt.space.Real(norm_min, norm_max, name=f"norm", prior='uniform'))
    space.append(skopt.space.Real(scale_min, scale_max, name=f"scale", prior='uniform'))

    try:
        stats = np.load(results_path + f"_hyperopt_stats{'_layer.npy' if args.layer else '.npy'}", allow_pickle=True).item()
        print(f"Found {len(stats['cost'])} Previous evaluations.")
    except FileNotFoundError:
        stats = {"norms": [], "scales": [], "acc": [], "ece": [], "nll": [], "ent": [], "cost": []}

    @skopt.utils.use_named_args(dimensions=space)
    def objective(**params):
        norms = [10 ** params["norm"]] * len(factors)
        scales = [10 ** params["scale"]] * len(factors)
        print("Norm:", norms[0], "Scale:", scales[0])
        try:
            inv_factors = invert_factors(factors, norms, args.pre_scale * scales, args.estimator)
        except (RuntimeError, np.linalg.LinAlgError):
            print(f"Error: Singular matrix")
            return 200

        predictions, labels, _ = eval_bnn(model, val_loader, inv_factors, args.estimator, args.samples, stats=False,
                                          device=args.device, verbose=args.verbose)

        err = 100 - accuracy(predictions, labels)
        ece = 100 * expected_calibration_error(predictions, labels)[0]
        nll = negative_log_likelihood(predictions, labels)
        ent = predictive_entropy(predictions, mean=True)
        stats["norms"].append(norms)
        stats["scales"].append(scales)
        stats["acc"].append(100 - err)
        stats["ece"].append(ece)
        stats["nll"].append(nll)
        stats["ent"].append(ent)
        stats["cost"].append(err + ece)
        print(f"Err.: {err:.2f}% | ECE: {ece:.2f}% | NLL: {nll:.3f} | Ent.: {ent:.3f}")
        np.save(results_path + f"_hyperopt_stats{'_layer.npy' if args.layer else '.npy'}", stats)

        return err + ece

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        if args.optimizer == "gbrt":
            res = skopt.gbrt_minimize(func=objective, dimensions=space, n_calls=args.calls, x0=x0, verbose=True,
                                      n_jobs=args.workers, n_random_starts=0 if x0 else 10, acq_func='EI')

        # EI (neg. expected improvement)
        # LCB (lower confidence bound)
        # PI (neg. prob. of improvement): Usually favours exploitation over exploration
        # gp_hedge (choose probabilistically between all)
        if args.optimizer == "gp":
            res = skopt.gp_minimize(func=objective, dimensions=space, n_calls=args.calls, x0=x0, verbose=True,
                                    n_jobs=args.workers, n_random_starts=0 if x0 else 1, acq_func='gp_hedge')

        # acq_func: EI (neg. expected improvement), LCB (lower confidence bound), PI (neg. prob. of improvement)
        # xi: how much improvement one wants over the previous best values.
        # kappa: Importance of variance of predicted values. High: exploration > exploitation
        # base_estimator: RF (random forest), ET (extra trees)
        elif args.optimizer == "forest":
            res = skopt.forest_minimize(func=objective, dimensions=space, n_calls=args.calls, x0=x0, verbose=True,
                                        n_jobs=args.workers, n_random_starts=0 if x0 else 1, acq_func='EI')

        elif args.optimizer == "random":
            res = skopt.dummy_minimize(func=objective, dimensions=space, n_calls=args.calls, x0=x0, verbose=True)

        elif args.optimizer == "grid":
            space = [np.arange(norm_min, norm_max + 1, 10), np.arange(scale_min, scale_max + 1, 10)]
            res = grid(func=objective, dimensions=space)
        else:
            raise ValueError

        print(f"Minimal cost of {min(stats['cost'])} found at:")
        print("Norm:", stats['norms'][np.argmin(stats['cost'])][0],
              "Scale:", stats['scales'][np.argmin(stats['cost'])][0])

    if not args.no_results:
        print("Saving results")
        del res.specs['args']['func']
        np.save(f"{results_path}_hyperopt_stats.npy", stats)
        skopt.dump(res, f"{results_path}_hyperopt_dump.pkl")

        all_stats = {"norms": [], "scales": [], "acc": [], "ece": [], "nll": [], "ent": [], "cost": []}
        path = os.path.join(args.results_dir, args.model, "data", args.estimator)
        paths = [subdir[0] for subdir in os.walk(path)]
        for p in paths:
            try:
                tmp_stats = np.load(p, allow_pickle=True).item()
                for key, value in tmp_stats.items():
                    all_stats[key].extend(value)
            except FileNotFoundError:
                pass
        np.save(os.path.join(path, f"{filename}_best_params.npy"),
                [all_stats['norms'][np.argmin(all_stats['cost'])],
                 all_stats['scales'][np.argmin(all_stats['cost'])]])

    if args.plot:
        print("Plotting results")
        hyperparameters(args)


if __name__ == "__main__":
    main()
