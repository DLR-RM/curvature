"""Provides data loading functionality through data loaders, wrappers and helper functions."""

import os
import multiprocessing as mp
import ctypes
from typing import Tuple, List, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, Sequential, CrossEntropyLoss
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import (Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop, CenterCrop, Resize,
                                    RandomResizedCrop, RandomAffine, ColorJitter, RandomRotation)
from torchvision.datasets import MNIST, KMNIST, CIFAR10, SVHN, ImageFolder
from tqdm import tqdm
from PIL import Image
from PIL.Image import Image as Img
import scipy.io
import pandas as pd
from sklearn.model_selection import train_test_split


Image.MAX_IMAGE_PIXELS = 688290000
torch.manual_seed(0)


def fgsm(model: Union[Module, Sequential],
         images: Tensor,
         labels: Tensor,
         criterion=CrossEntropyLoss(),
         epsilon: float = 0.1) -> Tensor:
    """The Fast Gradient Sign Method from `Explaining and Harnessing Adversarial Examples
    <https://arxiv.org/abs/1412.6572>`_ by Goodfellow et al.

    The main idea is to change each pixel in the image according to the sign of the gradient of the loss w.r.t. the
    image pixels by a small amount. This implementation is adapted from the `PyTorch tutorial
    <https://pytorch.org/tutorials/beginner/fgsm_tutorial.html>`_.

    Args:
        model: A `torchvision` or custom neural network.
        images: The image data.
        labels: The class labels.
        criterion (optional): Any PyTorch loss criterion.
        epsilon (optional): The step size of the FGSM.

    Returns:
        Tensor: The perturbed images.
    """
    vmin, vmax = images.min().numpy(), images.max().numpy()
    images.requires_grad = True

    logits = model(images)
    loss = criterion(logits, labels)
    model.zero_grad()
    loss.backward()

    data_grad = images.grad.data
    sign_data_grad = data_grad.sign()
    perturbed_image = images + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, vmin, vmax)

    return perturbed_image


class Binarize:
    """Randomly binarizes monochrome images where pixel values determine the probability."""

    def __call__(self,
                 pic: Img):
        """Randomly binarizes a monochrome image where pixel values determine the probability.

        Args:
            pic: A single monochrome image.

        Returns:
            The binarized image.
        """
        return Image.fromarray(np.uint8(np.random.binomial(1, np.array(pic) / 255) * 255))


class Memory(Dataset):
    """Stores a dataset in RAM."""

    def __init__(self,
                 data: Dataset,
                 img_size: int = 224,
                 channels: int = 3):
        """Creates a `Memory` object, storing a dataset in RAM.

        Args:
            data: A PyTorch dataset.
            img_size (optional): The size of the image.
            channels (optional): Number of color channels, i.e. 3 for RGB and 1 for monochrome images.
        """
        self.data = data
        self.images = torch.zeros(len(data), channels, img_size, img_size)
        self.targets = torch.zeros(len(data)).long()
        self.use_cache = False

    def pin_memory(self):
        """Uses the PyTorch mechanism of pinning data to memory."""
        self.images = self.images.pin_memory()
        self.targets = self.targets.pin_memory()
        return self

    def set_use_cache(self,
                      use_cache: bool):
        """Switch activating the use of the stored data.

        Args:
            use_cache: Whether to use the cache.
        """
        self.use_cache = use_cache

    def __getitem__(self,
                    index: int):
        if self.use_cache:
            if index < 1:
                return self.images, self.targets
            else:
                raise StopIteration()
        else:
            self.images[index] = self.data[index][0]
            self.targets[index] = self.data[index][1]
            return self.images[index], self.targets[index]

    def __len__(self):
        if self.use_cache:
            return 1
        else:
            return len(self.data)


class Cashed(Dataset):
    """Similar to the Memory class, but can be used with multiprocessing."""

    def __init__(self,
                 data: Dataset,
                 img_size: int = 224,
                 channels: int = 3):
        """Creates a `Cashed` object, storing a dataset in RAM.

        Args:
            data: A PyTorch dataset.
            img_size: The size of the image.
            channels: Number of color channels, i.e. 3 for RGB and 1 for monochrome images.
        """
        self.data = data
        shared_array_base = mp.Array(ctypes.c_float, len(data) * channels * img_size ** 2)
        shared_array_base_labels = mp.Array(ctypes.c_long, len(data))
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array_labels = np.ctypeslib.as_array(shared_array_base_labels.get_obj())
        shared_array = shared_array.reshape(len(data), channels, img_size, img_size)
        self.shared_array = torch.from_numpy(shared_array)
        self.shared_array_labels = torch.from_numpy(shared_array_labels).long()
        self.use_cache = False

    def pin_memory(self):
        """Uses the PyTorch mechanism of pinning data to memory."""
        self.shared_array = self.shared_array.pin_memory()
        self.shared_array_labels = self.shared_array_labels.pin_memory()
        return self

    def set_use_cache(self,
                      use_cache: bool):
        """Switch activating the use of the stored data.

        Args:
            use_cache (bool): Whether to use the cache.
        """
        self.use_cache = use_cache

    def __getitem__(self,
                    index: int):
        if not self.use_cache:
            self.shared_array[index] = self.data[index][0]
            self.shared_array_labels[index] = self.data[index][1]
        return self.shared_array[index], self.shared_array_labels[index]

    def __len__(self):
        return len(self.data)


LoaderTypes = Union[DataLoader,
                    Memory,
                    List[DataLoader],
                    List[Memory]]


def uci(root: str,
        name: str,
        split: int = 1):
    if name.lower() in ["boston", "housing", "boston housing", "boston_housing"]:
        path = os.path.join(root, "boston_housing.data")
        data = np.loadtxt(path)
        inputs, targets = data[:, :-1], data[:, -1]
    elif name.lower() == "kin8nm":
        path = os.path.join(root, "kin8nm.csv")
        data = np.loadtxt(path, delimiter=',', skiprows=1)
        inputs, targets = data[:, :-1], data[:, -1]
    elif name.lower() in ["naval", "naval propulsion", "naval_propulsion"]:
        path = os.path.join(root, "naval_propulsion.txt")
        data = np.loadtxt(path)
        inputs, targets = data[:, :-2], data[:, -2:]
    elif name.lower() in ["protein", "protein structure", "protein_structure"]:
        path = os.path.join(root, "protein_structure.csv")
        data = np.loadtxt(path, delimiter=',', skiprows=1)
        inputs, targets = data[:, 1:], data[:, 0]
    elif name.lower() in ["wine", "wine quality", "wine quality red", "wine_quality", "wine_quality_red"]:
        path = os.path.join(root, "wine_quality_red.csv")
        data = np.loadtxt(path, delimiter=';', skiprows=1)
        inputs, targets = data[:, :-1], data[:, -1]
    if name.lower() in ["yacht", "yacht hydrodynamics", "yacht_hydrodynamics"]:
        path = os.path.join(root, "yacht_hydrodynamics.data")
        data = np.loadtxt(path)
        inputs, targets = data[:, :-1], data[:, -1]
    elif name.lower() in ["power", "power plant", "combined cycle power plant", "power_plant",
                          "combined_cycle_power_plant"]:
        path = os.path.join(root, "combined_cycle_power_plant.xlsx")
        data = pd.read_excel(path).to_numpy()
        inputs, targets = data[:, :-1], data[:, -1]
    elif name.lower() in ["concrete", "concrete compression", "concrete compression strength", "concrete_compression",
                          "concrete_compression_strength"]:
        path = os.path.join(root, "concrete_compression_strength.xls")
        data = pd.read_excel(path).to_numpy()
        inputs, targets = data[:, :-1], data[:, -1]
    elif name.lower() in ["energy", "efficiency", "energy efficiency", "energy_efficiency"]:
        path = os.path.join(root, "energy_efficiency.xlsx")
        data = pd.read_excel(path).to_numpy()
        inputs, targets = data[:, :-2], data[:, -2:]

    if len(targets.shape) < 2:
        targets = np.expand_dims(targets, axis=1)

    x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=.1, random_state=split)
    return (x_train, y_train), (x_test, y_test)


def sarcos(root: str):
    sarcos_inv = scipy.io.loadmat(os.path.join(root, "sarcos_inv.mat"))
    sarcos_inv_test = scipy.io.loadmat(os.path.join(root, "sarcos_inv_test.mat"))

    x_train = sarcos_inv["sarcos_inv"][:, :21]
    y_train = sarcos_inv["sarcos_inv"][:, 21:]
    x_test = sarcos_inv_test["sarcos_inv_test"][:, :21]
    y_test = sarcos_inv_test["sarcos_inv_test"][:, 21:]

    return (x_train, y_train), (x_test, y_test)


def kuka(root: str, part=1):
    train = np.loadtxt(os.path.join(root, f"kuka_real_dataset{part}", f"kuka{part}_online.txt"))
    test = np.loadtxt(os.path.join(root, f"kuka_real_dataset{part}", f"kuka{part}_offline.txt"))

    x_train = train[:, :21]
    y_train = train[:, 21:]
    x_test = test[:, :21]
    y_test = test[:, 21:]

    return (x_train, y_train), (x_test, y_test)


def mnist(root: str,
          batch_size: int = 32,
          workers: int = 6,
          augment: bool = True,
          splits: Union[str, Tuple[str]] = ('train', 'val')) -> LoaderTypes:
    """Wrapper for loading the `MNIST` dataset.

    Args:
        root: The root directory where the dataset is stored. Usually ~/.torch/datasets.
        batch_size: The batch size.
        workers: The number of CPUs to use for when loading the data from disk.
        augment: Whether to use data augmentation when training.
        splits: Which splits of the data to return. Possible values are `train`, `val` and `test`.

    Returns:
        A list data loaders of the chosen splits.
    """
    val_transform = ToTensor()
    if augment:
        transform = Compose([Binarize(), ToTensor()])
    else:
        transform = val_transform

    loader_list = list()
    if 'train' in splits:
        train_set = MNIST(root, train=True, transform=transform, download=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        loader_list.append(train_loader)
    if 'test' in splits or 'val' in splits:
        val_test_set = MNIST(root, train=False, transform=val_transform, download=True)
        val_set, test_set = torch.utils.data.random_split(val_test_set, [5000, 5000])

        if 'val' in splits:
            val_set = Memory(val_set, img_size=28, channels=1)
            for _ in val_set:
                pass
            val_set.set_use_cache(True)
            val_set.pin_memory()
            loader_list.append(val_set)

        if 'test' in splits:
            test_set = Memory(test_set, img_size=28, channels=1)
            for _ in test_set:
                pass
            test_set.set_use_cache(True)
            test_set.pin_memory()
            loader_list.append(test_set)

    if len(loader_list) == 1:
        return loader_list[0]
    return loader_list


def kmnist(root: str,
           batch_size: int = 32,
           workers: int = 6,
           splits: Union[str, Tuple[str]] = ('train', 'val')) -> LoaderTypes:
    """Wrapper for loading the `KMNIST` dataset.

    Args:
        root: The root directory where the dataset is stored. Usually ~/.torch/datasets.
        batch_size: The batch size.
        workers: The number of CPUs to use for when loading the data from disk.
        splits: Which splits of the data to return. Possible values are `train`, `val` and `test`.

    Returns:
        A list data loaders of the chosen splits.
    """
    loader_list = list()
    if 'train' in splits or 'val' in splits:
        train_val_set = KMNIST(root, train=True, download=True, transform=ToTensor())

        val_set, train_set = torch.utils.data.random_split(train_val_set, [10000, len(train_val_set) - 10000])
        if 'train' in splits:
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
                                      pin_memory=True)
            loader_list.append(train_loader)
        if 'val' in splits:
            val_set = Memory(val_set, img_size=28, channels=1)
            for _ in val_set:
                pass
            val_set.set_use_cache(True)
            val_set.pin_memory()
            loader_list.append(val_set)
    if 'test' in splits:
        test_set = KMNIST(root, train=False, download=True, transform=ToTensor())
        test_set = Memory(test_set, img_size=28, channels=1)
        for _ in test_set:
            pass
        test_set.set_use_cache(True)
        test_set.pin_memory()
        loader_list.append(test_set)

    if len(loader_list) == 1:
        return loader_list[0]
    return loader_list


def cifar10(root: str,
            batch_size: int = 32,
            workers: int = 6,
            augment: bool = True,
            splits: Union[str, Tuple[str]] = ('train', 'val')) -> LoaderTypes:
    """Wrapper for loading the `CIFAR10` dataset.

    Args:
        root: The root directory where the dataset is stored. Usually ~/.torch/datasets.
        batch_size: The batch size.
        workers: The number of CPUs to use for when loading the data from disk.
        augment: Whether to use data augmentation when training.
        splits: Which splits of the data to return. Possible values are `train`, `val` and `test`.

    Returns:
        A list data loaders of the chosen splits.
    """
    normalize = Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    val_transform = Compose([ToTensor(), normalize])
    if augment:
        transform = Compose(
            [RandomCrop(32, padding=4),
             RandomHorizontalFlip(),
             ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
             RandomRotation(degrees=5),
             ToTensor(),
             normalize])
    else:
        transform = val_transform

    loader_list = list()
    if 'train' in splits:
        train_val_set = CIFAR10(root, train=True, transform=transform, download=True)
        train_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True, num_workers=workers,
                                  pin_memory=True)
        loader_list.append(train_loader)
    if 'test' in splits or 'val' in splits:
        val_test_set = CIFAR10(root, train=False, transform=val_transform, download=True)
        val_set, test_set = torch.utils.data.random_split(val_test_set, [5000, 5000])

        if 'val' in splits:
            val_set = Memory(val_set, img_size=32, channels=3)
            for _ in val_set:
                pass
            val_set.set_use_cache(True)
            val_set.pin_memory()
            loader_list.append(val_set)

        if 'test' in splits:
            test_set = Memory(test_set, img_size=32, channels=3)
            for _ in test_set:
                pass
            test_set.set_use_cache(True)
            test_set.pin_memory()
            loader_list.append(test_set)

    if len(loader_list) == 1:
        return loader_list[0]
    return loader_list


def svhn(root: str,
         batch_size: int = 32,
         workers: int = 6,
         splits: Union[str, Tuple[str]] = ('train', 'val')) -> LoaderTypes:
    """Wrapper for loading the `SVHN` dataset.

    Args:
        root: The root directory where the dataset is stored. Usually ~/.torch/datasets.
        batch_size: The batch size.
        workers: The number of CPUs to use for when loading the data from disk.
        splits: Which splits of the data to return. Possible values are `train`, `val` and `test`.

    Returns:
        A list data loaders of the chosen splits.
    """
    transform = Compose([ToTensor(), Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    loader_list = list()
    if 'train' in splits:
        train_set = SVHN(root, split='train', transform=transform, download=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        loader_list.append(train_loader)
    if 'test' in splits or 'val' in splits:
        val_test_set = SVHN(root, split='test', transform=transform, download=True)
        val_set, test_set, rest = torch.utils.data.random_split(val_test_set, [5000, 5000, len(val_test_set) - 10000])

        if 'val' in splits:
            val_set = Memory(val_set, img_size=32, channels=3)
            for _ in val_set:
                pass
            val_set.set_use_cache(True)
            val_set.pin_memory()
            loader_list.append(val_set)

        if 'test' in splits:
            test_set = Memory(test_set, img_size=32, channels=3)
            for _ in test_set:
                pass
            test_set.set_use_cache(True)
            test_set.pin_memory()
            loader_list.append(test_set)

    if len(loader_list) == 1:
        return loader_list[0]
    return loader_list


def art(root: str,
        img_size: int = 224,
        batch_size: int = 32,
        workers: int = 6,
        pin_memory: bool = True,
        use_cache: bool = False,
        pre_cache: bool = False) -> DataLoader:
    """A dataset consisting of works of arts; mostly paintings.
    Source: https://www.kaggle.com/c/painter-by-numbers/data.

    Args:
        root: The root directory where the image data is stored.
        img_size: The size of the image.
        batch_size: The batch size.
        workers: The number of CPUs to use for when loading the data from disk.
        pin_memory: Whether to use the PyTorchs `pin memory` mechanism.
        use_cache: Whether to cache data in a `Cache` object.
        pre_cache: Whether to run caching before the first epoch.

    Returns:
        Data loader of the test split.
    """
    transform = Compose([
        Resize(int(img_size * 8 / 7)),
        CenterCrop(img_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data = ImageFolder(root, transform)
    test_set, rest_set = torch.utils.data.random_split(data, [25000, len(data) - 25000])
    if use_cache:
        test_set = Cashed(test_set, img_size, channels=3)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=workers, pin_memory=pin_memory)
    if use_cache and pre_cache:
        print("Caching")
        for _ in tqdm(test_loader):
            pass
        test_loader.dataset.set_use_cache(True)
        # test_loader.dataset.pin_memory()
    return test_loader


def imagenet(root: str,
             img_size: int = 224,
             batch_size: int = 32,
             augment: bool = True,
             workers: int = 6,
             splits: Union[str, Tuple[str]] = ('train', 'val'),
             tiny: bool = False,
             pin_memory: bool = True,
             use_cache: bool = False,
             pre_cache: bool = False) -> Union[DataLoader, List[DataLoader]]:
    """Data loader for the ImageNet dataset.

    Args:
        root: The root directory where the image data is stored. Must contain a `train` and `val` directory with
          training and validation data respectively. If `tiny` is set to True, it must contain a `tiny` directory.
        img_size: The size of the image.
        batch_size: The batch size.
        augment: Whether to use data augmentation techniques.
        workers: The number of CPUs to use for when loading the data from disk.
        splits: Which splits of the data to return. Possible values are `train` and `val`.
        tiny: Whether to use the `Tiny ImageNet dataset <https://tiny-imagenet.herokuapp.com/>`_ instead of the
          full-size data. If True, `root` must contain a `tiny` directory with `train` and `val` directories inside.
        pin_memory: Whether to use the PyTorchs `pin memory` mechanism.
        use_cache: Whether to cache data in a `Cache` object.
        pre_cache: Whether to run caching before the first epoch.

    Returns:
        A list data loaders of the chosen splits.
    """
    if tiny:
        root = os.path.join(root, 'tiny')
    train_dir = os.path.join(root, 'train')
    test_dir = os.path.join(root, 'val')

    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_transform_list = list()
    if not tiny:
        val_transform_list.append(Resize(int(img_size * 8 / 7)))
        val_transform_list.append(CenterCrop(img_size))
    val_transform_list.append(ToTensor())
    val_transform_list.append(normalize)
    val_transform = Compose(val_transform_list)

    train_transform_list = list()
    if tiny:
        train_transform_list.append(RandomCrop(img_size, padding=8))
    else:
        train_transform_list.append(RandomResizedCrop(img_size))
    train_transform_list.append(RandomHorizontalFlip())
    train_transform_list.append(ToTensor())
    train_transform_list.append(normalize)
    train_transform = Compose(train_transform_list)

    loader_list = list()
    if 'train' in splits:
        train_set = ImageFolder(train_dir, train_transform if augment else val_transform)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
                                  pin_memory=pin_memory)
        loader_list.append(train_loader)

    if 'val' or 'test' in splits:
        val_test_set = ImageFolder(test_dir, val_transform)
        val_set, test_set = torch.utils.data.random_split(val_test_set, [25000, 25000])

        if 'test' in splits:
            if use_cache:
                test_set = Cashed(test_set, img_size, channels=3)
            test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=workers, pin_memory=pin_memory)
            if use_cache and pre_cache:
                print("Caching")
                for _ in tqdm(test_loader):
                    pass
                test_loader.dataset.set_use_cache(True)
                # test_loader.dataset.pin_memory()
            loader_list.append(test_loader)

        if 'val' in splits:
            if use_cache:
                val_set = Cashed(val_set, img_size, channels=3)
            val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=workers, pin_memory=pin_memory)
            if use_cache and pre_cache:
                print("Caching")
                for _ in tqdm(val_loader):
                    pass
                val_loader.dataset.set_use_cache(True)
                # val_loader.dataset.pin_memory()
            loader_list.append(val_loader)

    if len(loader_list) == 1:
        return loader_list[0]
    return loader_list


LoaderLists = Union[
    Union[List[Cashed], List[Memory]],
    Union[List[Cashed], List[DataLoader]],
    Union[List[Memory], List[DataLoader]],
    Union[List[Memory], List[DataLoader], List[Cashed]]]


def gtsrb(root: str,
          img_size: int = 32,
          batch_size: int = 32,
          workers: int = 6,
          splits: Union[str, Tuple[str]] = ('train', 'val'),
          pin_memory: bool = True) -> Union[LoaderTypes, Cashed, LoaderLists]:
    """Data loader for the `German Traffic Sign Recognition Benchmark
    <http://benchmark.ini.rub.de/?section=gtsrb&subsection=news>`_.

    Args:
        root: The root directory where the image data is stored. Must contain a `train`, `val` and `test` directory with
            training, validation and test data respectively.
        img_size: The size of the image.
        batch_size: The batch size.
        workers: The number of CPUs to use for when loading the data from disk.
        splits: Which splits of the data to return. Possible values are `train`, `val` and `test`.
        pin_memory: Whether to use the PyTorchs `pin memory` mechanism.

    Returns:
        A list data loaders of the chosen splits.
    """
    train_dir = os.path.join(root, 'train')
    val_dir = os.path.join(root, 'val')
    test_dir = os.path.join(root, 'test')

    normalize = Normalize([0.34038433, 0.3119956, 0.32119358], [0.05087305, 0.05426421, 0.05859348])
    if img_size > 32:
        val_transform = Compose([Resize(int(img_size * 8 / 7)),
                                 CenterCrop(img_size),
                                 ToTensor(),
                                 normalize])
        train_transform = Compose([RandomResizedCrop(img_size),
                                   RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10),
                                   ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                   ToTensor(),
                                   normalize])
    else:
        val_transform = Compose([Resize(img_size + 10),
                                 CenterCrop(img_size),
                                 ToTensor(),
                                 normalize])
        train_transform = Compose([RandomCrop(img_size, padding=4),
                                   RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
                                   ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                   ToTensor(),
                                   normalize])

    loader_list = list()
    if 'train' in splits:
        train_set = ImageFolder(train_dir, train_transform)

        weights = list()
        for c in range(43):
            dir_name = f"000{c}" if c > 9 else f"0000{c}"
            weights.append(len(os.listdir(os.path.join(train_dir, dir_name))[:-1]))
        weights = 1 / np.array(weights)
        weights = np.array([weights[t] for t in train_set.targets])
        sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.from_numpy(weights).double(), len(weights))

        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=workers,
                                  pin_memory=pin_memory)
        loader_list.append(train_loader)
    if 'val' in splits:
        val_set = ImageFolder(val_dir, val_transform)
        if img_size > 32:
            val_set = Cashed(val_set, img_size, channels=3)
            val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=workers,
                                                     pin_memory=pin_memory)
            for _ in val_loader:
                pass
            val_loader.dataset.set_use_cache(True)
            val_loader.dataset.pin_memory()
            loader_list.append(val_loader)
        else:
            val_set = Memory(val_set, img_size=img_size, channels=3)
            for _ in val_set:
                pass
            val_set.set_use_cache(True)
            val_set.pin_memory()
            loader_list.append(val_set)

    if 'test' in splits:
        test_set = ImageFolder(test_dir, val_transform)
        test_set = Memory(test_set, img_size=img_size, channels=3)
        for _ in test_set:
            pass
        test_set.set_use_cache(True)
        test_set.pin_memory()
        loader_list.append(test_set)

    if len(loader_list) == 1:
        return loader_list[0]
    return loader_list
