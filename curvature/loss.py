import os
import sys
import copy
from math import isqrt

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.backends import cudnn
import torchvision
from tqdm import tqdm

from .utils import setup
from .plot import plot_surfaces, plot_loss1d
import curvature.lenet5 as lenet5
import curvature.resnet as resnet
import curvature.datasets as datesets
from .imagenet import imagenet


def tensorlist_to_tensor(weights_or_state):
    """ Concatenate a list of tensors into one tensor.

        Args:
            weights_or_state: A list of weights or a state dict of a PyTorch model.
        Returns:
            concatenated 1D tensor
    """
    return torch.cat([w.unsqueeze(0).view(-1) for w in weights_or_state])


def get_weights(net):
    """Extract parameters from net, and return a list of tensors.

    Args:
        net:

    Returns:

    """
    return [p.data for p in net.parameters()]


def get_random_weights(weights):
    """Produce a random direction that is a list of random Gaussian tensors with the same shape as the network's
    weights, so one direction entry per weight.

    Args:
        weights:

    Returns:

    """
    return [torch.randn(w.size()) for w in weights]


def get_random_state(state):
    """Produce a random direction that is a list of random Gaussian tensors with the same shape as the network's
    state_dict(), so one direction entry per weight, including BN's running_mean/var.

    Args:
        state:

    Returns:

    """
    return [torch.randn(w.size()) for w in state.values()]


def set_state(model, state, directions=None, steps=None):
    """Overwrite the network's state_dict or change it along directions with a step size."""
    if directions is None:
        model.load_state_dict(state)
    else:
        assert steps is not None, 'If direction is provided then the step must be specified as well'
        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            changes = [d0 * steps[0] + d1 * steps[1] for (d0, d1) in zip(dx, dy)]
        else:
            changes = [d * steps[0] for d in directions[0]]

        new_state = copy.deepcopy(state)
        assert len(new_state) == len(changes)
        for (k, v), d in zip(new_state.items(), changes):
            v.add_(d.type(v.type()))

        model.load_state_dict(new_state)


def normalize_layer(direction, weights, norm='filter'):
    """Rescale the direction so that it has similar norm as their corresponding model in different levels.

        Args:
          direction: a variables of the random direction for one layer
          weights: a variable of the original model for one layer
          norm: normalization method, 'filter' | 'layer' | 'weight'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm() / (d.norm() + 1e-10))
    elif norm == 'layer':
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm() / direction.norm())
    elif norm == 'weight':
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        direction.mul_(weights)
    elif norm == 'dfilter':
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == 'dlayer':
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())


def normalize_direction(direction, weights_or_state, norm='filter', ignore='biasbn'):
    """The normalization scales the direction entries according to the entries of weights.

    Args:
        direction:
        weights_or_state:
        norm:
        ignore:

    Returns:

    """
    assert (len(direction) == len(weights_or_state))
    for d, w in zip(direction, weights_or_state if isinstance(weights_or_state, list) else weights_or_state.values()):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0)  # ignore directions for weights with 1 dimension
            else:
                d.copy_(w)  # keep directions for weights/bias that are only 1 per node
        else:
            normalize_layer(d, w, norm)


def create_random_direction(net, dir_type='state', norm='filter', ignore='biasbn'):
    """Setup a random (normalized) direction with the same dimension as the weights or states.

        Args:
          net: the given trained model
          dir_type: 'weights' or 'state', type of directions.
          norm: direction normalization method, including
                'filter" | 'layer' | 'weight' | 'dlayer' | 'dfilter'
          ignore: 'biasbn', ignore biases and BN parameters.
        Returns:
          direction: a random direction with the same dimension as weights or states.
    """
    if dir_type == 'weights':
        weights = get_weights(net)  # a list of parameters.
        direction = get_random_weights(weights)
        normalize_direction(direction, weights, norm, ignore)
    elif dir_type == 'state':
        state = net.state_dict()  # a dict of parameters, including BN's running mean/var.
        direction = get_random_state(state)
        normalize_direction(direction, state, norm, ignore)
    else:
        print(f"Error: Argument 'dir_type' needs to be either 'weights' or 'state' but not {dir_type}.")
        sys.exit(0)
    return direction


def loss1d(args,
           model,
           train_data,
           val_data,
           directions_path,
           results_path,
           min=-1,
           max=1,
           from2d: str = 'none',
           linear=True):
    """

    Args:
        args:
        model:
        train_data:
        val_data:
        directions_path:
        results_path:
        min:
        max:
        from2d:
        linear:

    Returns:

    """
    if linear:
        samples = args.samples if args.samples % 2 != 0 else args.samples - 1
    else:
        samples = args.samples

    print("Looking for an existing direction.")
    try:
        if from2d == 'x':
            direction = torch.load(directions_path + "_xdirection.pt")
            results_path = results_path + "_2dx"
            try:
                results = np.load(results_path + ".npy")
            except IOError:
                results = np.zeros((samples, 5))
                np.save(results_path + ".npy", results)
        elif from2d == 'y':
            direction = torch.load(directions_path + "_ydirection.pt")
            results_path = results_path + "_2dy"
            try:
                results = np.load(results_path + ".npy")
            except IOError:
                results = np.zeros((samples, 5))
                np.save(results_path + ".npy", results)
        else:
            direction = torch.load(directions_path + "_direction.pt")
            results = np.load(results_path + ".npy")
    except IOError:
        print("No existing direction found, creating new one.")
        direction = create_random_direction(model, norm='filter')
        torch.save(direction, directions_path + "_direction.pt")

        results = np.zeros((samples, 5))
        np.save(results_path + ".npy", results)

    # Generate (random) coordinates along random direction
    if linear:
        coordinates = np.linspace(min, max, samples)
    else:
        coordinates = np.random.uniform(min, max, samples)

    # Remove previously evaluated coordinates
    if results.sum() != 0:
        coordinates = coordinates[~np.isin(coordinates, results[:, 0])]

    initial_state = copy.deepcopy(model.state_dict())
    criterion = CrossEntropyLoss().to(args.device)

    with torch.no_grad():
        # Training data
        for point, coord in enumerate(coordinates):
            loss = 0
            correct = 0
            total = 0
            results = np.load(results_path + ".npy")
            set_state(model, initial_state, directions=[direction], steps=[coord])
            data = tqdm(train_data, desc=f"Train point [{point + 1}/{len(coordinates)}]", disable=not args.verbose)

            for images, labels in data:
                logits = model(images.to(args.device, non_blocking=True))
                loss += criterion(logits, labels.to(args.device, non_blocking=True)).item() * images.size(0)
                _, predicted = torch.max(logits.detach().cpu(), 1)
                correct += predicted.eq(labels).sum().item()
                total += images.size(0)

                if args.verbose:
                    data.set_postfix({'coord': coord,
                                      'loss': loss / total,
                                      'acc': 100. * correct / total})

            results[point, :3] = [coord, loss / total, 100. * correct / total]
            np.save(results_path + ".npy", results)

        # Evaluation data
        for point, coord in enumerate(coordinates):
            loss = 0
            correct = 0
            total = 0
            results = np.load(results_path + ".npy")
            set_state(model, initial_state, directions=[direction], steps=[coord])
            data = tqdm(val_data, desc=f"Val point [{point + 1}/{len(coordinates)}]", disable=not args.verbose)

            for images, labels in data:
                logits = model(images.to(args.device, non_blocking=True))
                loss += criterion(logits, labels.to(args.device, non_blocking=True)).item() * images.size(0)
                _, predicted = torch.max(logits.detach().cpu(), 1)
                correct += predicted.eq(labels).sum().item()
                total += images.size(0)

                if args.verbose:
                    data.set_postfix({'coord': coord,
                                      'loss': loss / total,
                                      'acc': 100. * correct / total})

            results[point, 3:] = [loss / total, 100. * correct / total]
            np.save(results_path + ".npy", results)

    plot_loss1d(np.array(results), path=results_path)


def loss2d(args,
           model,
           data,
           directions_path,
           results_path,
           xmin=-1,
           xmax=1,
           ymin=-1,
           ymax=1,
           mode="random"):
    """

    Args:
        args:
        model:
        data:
        directions_path:
        results_path:
        xmin:
        xmax:
        ymin:
        ymax:
        mode:

    Returns:

    """
    print("Looking for existing directions.")
    try:
        xdirection = torch.load(directions_path + "_xdirection.pt")
        ydirection = torch.load(directions_path + "_ydirection.pt")
        try:
            results = list(np.load(results_path + ".npy"))
        except IOError:
            results = list()
            np.save(results_path + ".npy", results)
    except IOError:
        print("No existing directions found, creating new ones.")
        xdirection = create_random_direction(model, norm='filter')
        ydirection = create_random_direction(model, norm='filter')
        torch.save(xdirection, directions_path + "_xdirection.pt")
        torch.save(ydirection, directions_path + "_ydirection.pt")

        results = list()
        np.save(results_path + ".npy", results)

    # Check direction similarity: 1/-1 -> perpendicular, 0 -> orthogonal
    xdir = tensorlist_to_tensor(xdirection)
    ydir = tensorlist_to_tensor(ydirection)
    print("Direction similarity:", (torch.dot(xdir, ydir) / (xdir.norm() * ydir.norm()).item()).item())
    print("Locations evaluated:", len(results))

    # Generate (random) points in xy-plane
    if mode == "random":
        samples = args.samples if args.samples % 2 != 0 else args.samples - 1
        xcoordinates = np.random.uniform(xmin, xmax, samples)
        ycoordinates = np.random.uniform(ymin, ymax, samples)
    elif mode == "grid":
        samples = isqrt(args.samples)
        samples = samples if samples % 2 != 0 else samples - 1
        xcoordinates, ycoordinates = np.mgrid[xmin:xmax:samples * 1j, ymin:ymax:samples * 1j]
        xcoordinates, ycoordinates = np.vstack([xcoordinates.ravel(), ycoordinates.ravel()])

    # Remove previously evaluated points
    if len(results) > 0:
        xcoordinates = xcoordinates[~np.isin(xcoordinates, np.array(results)[:, 0])]
        ycoordinates = ycoordinates[~np.isin(ycoordinates, np.array(results)[:, 1])]
    else:
        xcoordinates[0] = 0.
        ycoordinates[0] = 0.

    # Setup initial state and loss
    initial_state = copy.deepcopy(model.state_dict())
    criterion = CrossEntropyLoss().to(args.device)

    with torch.no_grad():
        for point, (x, y) in tqdm(enumerate(zip(xcoordinates, ycoordinates)),
                                  total=len(xcoordinates),
                                  disable=args.verbose):
            loss = 0
            correct = 0
            total = 0
            results = list(np.load(results_path + ".npy"))
            set_state(model, initial_state, directions=[xdirection, ydirection], steps=[x, y])
            data = tqdm(data, desc=f"Point [{point + 1}/{len(xcoordinates)}]", disable=not args.verbose)

            for images, labels in data:
                logits = model(images.to(args.device, non_blocking=True))
                loss += criterion(logits, labels.to(args.device, non_blocking=True)).item() * images.size(0)
                _, predicted = torch.max(logits.detach().cpu(), 1)
                correct += predicted.eq(labels).sum().item()
                total += images.size(0)

                if args.verbose:
                    data.set_postfix({'x': x,
                                      'y': y,
                                      'loss': loss / total,
                                      'acc': 100. * correct / total})

            results.append([x, y, loss / total, 100. * correct / total])
            np.save(results_path + ".npy", results)
    plot_surfaces(np.array(results), path=results_path)


def main():
    args = setup(seed=None)  # Disable seed to get random loss samples

    print("Preparing directories")
    filename = f"{args.prefix}{args.model}_{args.data}{args.suffix}"
    os.makedirs(os.path.join(args.root_dir, "directions"), exist_ok=True)
    directions_path = os.path.join(args.root_dir, "directions", filename)
    os.makedirs(os.path.join(args.results_dir, "loss1d" if args.loss1d else "loss2d"), exist_ok=True)
    results_path = os.path.join(args.results_dir, "loss1d" if args.loss1d else "loss2d", filename)

    print("Loading model")
    if args.model == 'lenet5':
        model = lenet5.lenet5(pretrained=args.data, device=args.device)
    elif args.model == 'resnet18' and args.data != 'imagenet':
        model = resnet.resnet18(pretrained=os.path.join(args.root_dir, 'weights', f"{args.model}_{args.data}.pt"),
                                num_classes=43 if args.data == 'gtsrb' else 10, device=args.device)
    else:
        model_class = getattr(torchvision.models, args.model)
        if args.model in ['googlenet', 'inception_v3']:
            model = model_class(pretrained=True, aux_logits=False)
        else:
            model = model_class(pretrained=True)
    model.to(args.device).eval()
    if args.parallel:
        model = torch.nn.parallel.DataParallel(model)

    print(f"Loading data")
    data_dir = os.path.join(args.torch_dir, "datasets")
    if args.data == 'cifar10':
        train_data, val_data = datasets.cifar10(data_dir, args.batch_size, args.workers, augment=False)
    elif args.data == 'mnist':
        train_data, val_data = datasets.mnist(data_dir, args.batch_size, args.workers, augment=False)
    elif args.data == 'gtsrb':
        data_dir = os.path.join(args.root_dir, "datasets", "gtsrb")
        train_data, val_data = datasets.gtsrb(data_dir, batch_size=args.batch_size, workers=args.workers)
    elif args.data == 'tiny':
        img_size = 64
        data_dir = os.path.join(args.root_dir, "datasets", "imagenet")
        train_data, val_data = datasets.imagenet(data_dir, img_size, args.batch_size, augment=False,
                                                 workers=args.workers, tiny=True)
    elif args.data == 'imagenet':
        img_size = 224
        if args.model in ['googlenet', 'inception_v3']:
            img_size = 299
        data_dir = os.path.join(args.root_dir, "datasets", "imagenet")
        train_data, val_data = imagenet(data_dir, img_size, args.batch_size, augment=False, shuffle=False)
    else:
        raise ValueError
    cudnn.benchmark = True

    if args.loss1d:
        loss1d(args, model, train_data, val_data, directions_path, results_path)
    elif args.loss2d:
        loss2d(args, model, train_data, directions_path, results_path)
    else:
        print(f"You need to specify either --loss1d or --loss2d.")


if __name__ == "__main__":
    main()
