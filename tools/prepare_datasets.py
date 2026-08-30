#!/usr/bin/env python3
"""Prepare the two benchmarks we have never run: ImageNet-A and 5-Datasets.

Why a tool rather than a few lines inside the training command. Both of these
have a first-use side effect that is destructive and easy to get wrong, and
both are worth failing fast on before any GPU time is spent:

  * Imagenet_A splits its ImageFolder 80/20 on first construction by MOVING
    files into imagenet-a/train and imagenet-a/test and then deleting the
    original class directories. If that runs half way and dies, the directory
    is left in a state no rerun can recover from, and the tar has to be
    extracted again. Doing it here, alone, means a failure costs a re-extract
    and nothing else.

  * 5-Datasets is five separate downloads (SVHN, MNIST, CIFAR-10, NotMNIST,
    FashionMNIST) from five hosts, one of which is a GitHub raw URL. Any of
    them can be slow or down. Better to find that out now than eleven minutes
    into a training run.

The split is random with no seed -- that is how the released code is written,
for ImageNet-R as well as ImageNet-A -- so it is fixed for good once it has
run, and every later run reads the same train/test directories. Re-splitting
would silently change what the numbers mean, which is why this tool refuses to
touch a directory that has already been split.

Usage:
    python tools/prepare_datasets.py --which all
    python tools/prepare_datasets.py --which imagenet-a --root /path/to/datasets
"""
import argparse
import os
import sys
import tarfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

IMAGENET_A_TAR = 'imagenet-a.tar'
IMAGENET_A_URL = 'https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar'


def default_root():
    """Same convention as training_scripts/train_any_4090.sh."""
    return os.path.join(os.path.dirname(REPO_ROOT), 'datasets')


def prepare_imagenet_a(root):
    from continual_datasets.continual_datasets import Imagenet_A

    fpath = os.path.join(root, 'imagenet-a')
    train_dir = os.path.join(fpath, 'train')
    test_dir = os.path.join(fpath, 'test')

    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        n_tr = sum(len(f) for _, _, f in os.walk(train_dir))
        n_te = sum(len(f) for _, _, f in os.walk(test_dir))
        print('  da chia tu truoc: %d anh train, %d anh test -- khong dung vao'
              % (n_tr, n_te))
    else:
        if not os.path.isdir(fpath):
            tar_path = os.path.join(root, IMAGENET_A_TAR)
            if not os.path.isfile(tar_path):
                print('  THIEU %s' % tar_path)
                print('  Tai bang:  wget -c -P %s %s' % (root, IMAGENET_A_URL))
                return False
            print('  giai nen %s ...' % tar_path)
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(root)

        classes = sorted(d for d in os.listdir(fpath)
                         if os.path.isdir(os.path.join(fpath, d)))
        n_img = sum(len(f) for _, _, f in os.walk(fpath))
        print('  truoc khi chia: %d thu muc lop, %d tep' % (len(classes), n_img))
        if len(classes) < 2:
            print('  KHONG DUNG: mong doi hang tram thu muc lop truc tiep trong '
                  '%s' % fpath)
            return False

        print('  chia 80/20 (di chuyen tep, khong sao chep) ...')
        Imagenet_A(root, train=True, download=True)

    train_set = Imagenet_A(root, train=True, download=True).data
    val_set = Imagenet_A(root, train=False, download=True).data
    print('  ImageNet-A: %d lop, %d anh train, %d anh test'
          % (len(val_set.classes), len(train_set), len(val_set)))
    return True


def prepare_five(root):
    from torchvision import datasets as tvd
    from continual_datasets.continual_datasets import (
        MNIST_RGB, FashionMNIST, NotMNIST, SVHN)

    # Exactly the five names datasets.py uses for '5-datasets', and exactly the
    # constructor calls get_dataset makes for each.
    jobs = [
        ('SVHN', lambda tr: SVHN(root, split='train' if tr else 'test',
                                 download=True)),
        ('MNIST', lambda tr: MNIST_RGB(root, train=tr, download=True)),
        ('CIFAR10', lambda tr: tvd.CIFAR10(root, train=tr, download=True)),
        ('NotMNIST', lambda tr: NotMNIST(root, train=tr, download=True)),
        ('FashionMNIST', lambda tr: FashionMNIST(root, train=tr, download=True)),
    ]

    ok = True
    for name, build in jobs:
        try:
            train_set, val_set = build(True), build(False)
            n_tr = len(getattr(train_set, 'data', train_set))
            n_te = len(getattr(val_set, 'data', val_set))
            print('  %-13s %6d train, %6d test' % (name, n_tr, n_te))
        except Exception as exc:                      # noqa: BLE001
            print('  %-13s HONG: %s: %s' % (name, type(exc).__name__, exc))
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=None,
                        help='thu muc du lieu (mac dinh: canh repo)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'imagenet-a', 'five'])
    args = parser.parse_args()

    root = args.root or default_root()
    os.makedirs(root, exist_ok=True)
    print('thu muc du lieu: %s\n' % root)

    ok = True
    if args.which in ('all', 'imagenet-a'):
        print('ImageNet-A')
        ok = prepare_imagenet_a(root) and ok
        print()
    if args.which in ('all', 'five'):
        print('5-Datasets')
        ok = prepare_five(root) and ok
        print()

    print('SAN SANG' if ok else 'CHUA XONG -- doc thong bao ben tren')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
