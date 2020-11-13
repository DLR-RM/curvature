import pickle
import random
import logging
import copy
import os
from typing import Union, Tuple

from tqdm import tqdm
import numpy as np
import torch
from torch.utils import data
from torchvision.datasets.folder import make_dataset, default_loader, IMG_EXTENSIONS
from torchvision import transforms
from torchvision.models import resnet50
import cloudpickle

from utils import seed_all_rng, setup


class PicklableWrapper(object):
    """
    Wrap an object to make it more picklable, note that it uses
    heavy weight serialization libraries that are slower than pickle.
    It's best to use it only on closures (which are usually not picklable).

    This is a simplified version of
    https://github.com/joblib/joblib/blob/master/joblib/externals/loky/cloudpickle_wrapper.py
    """

    def __init__(self, obj):
        self._obj = obj

    def __reduce__(self):
        s = cloudpickle.dumps(self._obj)
        return cloudpickle.loads, (s,)

    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)

    def __getattr__(self, attr):
        # Ensure that the wrapped object can be used seamlessly as the previous object.
        if attr not in ["_obj"]:
            return getattr(self._obj, attr)
        return getattr(self, attr)


class MapDataset(data.Dataset):
    """
    Map a function over the elements in a dataset.

    Args:
        dataset: a dataset where map function is applied.
        map_func: a callable which maps the element in dataset. map_func is
            responsible for error handling, when error happens, it needs to
            return None so the MapDataset will randomly use other
            elements from the dataset.
    """

    def __init__(self, dataset, map_func):
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            data = self._map_func(self._dataset[cur_idx])
            if data is not None:
                self._fallback_candidates.add(cur_idx)
                return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )


class DatasetFromList(data.Dataset):
    """
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(self, lst: list, copy: bool = True, serialize: bool = True):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
            serialize (bool): whether to hold memory using serialized objects, when
                enabled, data loader workers can use shared RAM from master
                process instead of making a copy.
        """
        self._lst = lst
        self._copy = copy
        self._serialize = serialize

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            logger = logging.getLogger(__name__)
            logger.info(
                "Serializing {} elements to byte tensors and concatenating them all ...".format(
                    len(self._lst)
                )
            )
            self._lst = [_serialize(x) for x in self._lst]
            self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst)
            logger.info("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024 ** 2))

    def __len__(self):
        if self._serialize:
            return len(self._addr)
        else:
            return len(self._lst)

    def __getitem__(self, idx):
        if self._serialize:
            start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
            end_addr = self._addr[idx].item()
            bytes = memoryview(self._lst[start_addr:end_addr])
            return pickle.loads(bytes)
        elif self._copy:
            return copy.deepcopy(self._lst[idx])
        else:
            return self._lst[idx]


class DatasetMapper:
    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, item):
        path, target = item
        sample = default_loader(path)
        if self.transform:
            sample = self.transform(sample)
        return sample, target


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)


def find_classes(dir):
    """
    Finds the class folders in a dataset.

    Args:
        dir (string): Root directory path.

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

    Ensures:
        No class is a subdirectory of another.
    """
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def imagenet(root: str,
             img_size: int = 224,
             batch_size: int = 32,
             augment: bool = True,
             shuffle: bool = True,
             workers: int = 6,
             splits: Union[str, Tuple[str, str], Tuple[str, str, str]] = ('train', 'val'),
             seed: int = 42):

    train_dir = os.path.join(root, 'train')
    val_test_dir = os.path.join(root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    val_transform_list = [transforms.Resize(int(img_size * 8 / 7)),
                          transforms.CenterCrop(img_size),
                          transforms.ToTensor(),
                          normalize]
    val_transform = transforms.Compose(val_transform_list)
    val_mapper = train_mapper = DatasetMapper(val_transform)

    if augment:
        train_transform_list = [transforms.RandomResizedCrop(img_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize]
        train_transform = transforms.Compose(train_transform_list)
        train_mapper = DatasetMapper(train_transform)

    loader_list = list()
    if "train" in splits:
        classes, class_to_idx = find_classes(train_dir)
        dataset = make_dataset(train_dir, class_to_idx, IMG_EXTENSIONS)
        dataset = DatasetFromList(dataset)
        dataset = MapDataset(dataset, train_mapper)
        loader_list.append(data.DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True,
                                           worker_init_fn=worker_init_reset_seed))
    if "val" or "test" in splits:
        classes, class_to_idx = find_classes(val_test_dir)
        val_test_set = make_dataset(val_test_dir, class_to_idx, IMG_EXTENSIONS)

        random.seed(seed)
        random.shuffle(val_test_set)
        val_set = val_test_set[:int(round(len(val_test_set) / 2))]
        test_set = val_test_set[int(round(len(val_test_set) / 2)):]

        if "val" in splits:
            val_set = DatasetFromList(val_set)
            val_set = MapDataset(val_set, val_mapper)
            loader_list.append(data.DataLoader(val_set, batch_size, num_workers=workers, pin_memory=True))
        if "test" in splits:
            test_set = DatasetFromList(test_set)
            test_set = MapDataset(test_set, val_mapper)
            loader_list.append(data.DataLoader(test_set, batch_size, num_workers=workers, pin_memory=True))

    if len(loader_list) == 1:
        return loader_list[0]
    return loader_list


if __name__ == "__main__":
    args = setup(required=False)

    val_loader = imagenet(os.path.join(args.root_dir, "datasets", "imagenet"), splits="val")
    model = resnet50(pretrained=True).to(args.device).eval()

    correct = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            logits = model(images.to(args.device, non_blocking=True))
            _, predicted = torch.max(logits.detach().cpu(), 1)
            correct += predicted.eq(labels).sum().item()

    print("Accuracy:", 100 * correct / 25000)

