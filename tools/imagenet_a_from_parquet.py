#!/usr/bin/env python3
"""Rebuild the ImageNet-A image folder from the Hugging Face parquet mirror.

Why this exists. continual_datasets.Imagenet_A expects <root>/imagenet-a to be
an ImageFolder whose subdirectories are WordNet ids, which is what the official
imagenet-a.tar unpacks to. That tar lives on people.eecs.berkeley.edu, and that
host is unreachable from this machine -- DNS resolves to 128.32.139.28 but both
wget and curl time out on IPv4, while the rest of the internet answers in a
fifth of a second. The GitHub repository publishes no release assets.

huggingface.co/datasets/barkermrl/imagenet-a carries the same 7,500 images
across the same 200 classes, and its ClassLabel names are the WordNet ids, so
the directory layout this produces is the layout the tar would have produced.

Image bytes are copied out of the parquet verbatim rather than decoded and
re-encoded, so nothing is resampled or recompressed on the way through.

Usage:
    python tools/imagenet_a_from_parquet.py --parquet-dir /path/with/parquets
    python tools/imagenet_a_from_parquet.py --parquet-dir . --root /data
"""
import argparse
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_root():
    return os.path.join(os.path.dirname(REPO_ROOT), 'datasets')


def find_class_names(schema):
    """Pull the ClassLabel names out of the parquet's huggingface metadata.

    The metadata is a JSON blob whose shape has moved around across versions of
    `datasets`, so rather than hard-coding a path this walks it looking for the
    ClassLabel node. Failing loudly here is much cheaper than writing 7,500
    files into wrongly named directories.
    """
    meta = schema.metadata or {}
    blob = meta.get(b'huggingface')
    if blob is None:
        return None
    doc = json.loads(blob.decode('utf-8'))

    found = []

    def walk(node):
        if isinstance(node, dict):
            if node.get('_type') == 'ClassLabel' and 'names' in node:
                found.append(node['names'])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(doc)
    if not found:
        return None
    if len(found) > 1 and any(f != found[0] for f in found[1:]):
        raise SystemExit('nhieu ClassLabel khac nhau trong metadata, dung lai')
    return found[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet-dir', required=True,
                        help='thu muc chua cac tep .parquet da tai')
    parser.add_argument('--root', default=None,
                        help='thu muc du lieu (mac dinh: canh repo)')
    args = parser.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print('THIEU pyarrow. Cai bang:  .venv/bin/pip install pyarrow')
        return 1

    root = args.root or default_root()
    out_dir = os.path.join(root, 'imagenet-a')

    if os.path.isdir(os.path.join(out_dir, 'train')):
        print('%s da duoc chia 80/20 tu truoc -- khong dung vao' % out_dir)
        return 0
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        print('%s da ton tai va khong rong -- khong ghi de' % out_dir)
        return 1

    files = sorted(glob.glob(os.path.join(args.parquet_dir, '*.parquet')))
    if not files:
        print('khong thay tep .parquet nao trong %s' % args.parquet_dir)
        return 1
    print('%d tep parquet:' % len(files))
    for f in files:
        print('  %s  (%.1f MB)' % (os.path.basename(f),
                                   os.path.getsize(f) / 1e6))

    written = 0
    names = None
    for path in files:
        table = pq.read_table(path)
        if names is None:
            names = find_class_names(table.schema)
            if not names:
                print('KHONG DOC DUOC ten lop tu metadata cua parquet.')
                print('Dung lai thay vi doan ten thu muc.')
                return 1
            print('%d lop, tu %s den %s' % (len(names), names[0], names[-1]))
            if not names[0].startswith('n'):
                print('CANH BAO: ten lop khong phai WordNet id, kiem tra lai')

        images = table.column('image').to_pylist()
        labels = table.column('label').to_pylist()
        if len(images) != len(labels):
            print('so anh va so nhan khong khop trong %s' % path)
            return 1

        for image, label in zip(images, labels):
            wnid = names[label]
            class_dir = os.path.join(out_dir, wnid)
            os.makedirs(class_dir, exist_ok=True)

            stem = os.path.basename(image.get('path') or '')
            if not stem:
                stem = '%06d.jpg' % written
            dest = os.path.join(class_dir, stem)
            if os.path.exists(dest):                 # tranh dam ten giua 2 tep
                base, ext = os.path.splitext(stem)
                dest = os.path.join(class_dir, '%s_%06d%s' % (base, written, ext))

            with open(dest, 'wb') as handle:
                handle.write(image['bytes'])
            written += 1

        print('  %s -> tong %d anh' % (os.path.basename(path), written))

    n_dirs = len([d for d in os.listdir(out_dir)
                  if os.path.isdir(os.path.join(out_dir, d))])
    print('\nda ghi %d anh vao %d thu muc lop tai %s' % (written, n_dirs, out_dir))
    if written != 7500 or n_dirs != 200:
        print('CANH BAO: ban goc la 7500 anh trong 200 lop')
    print('Buoc tiep: python tools/prepare_datasets.py --which imagenet-a')
    return 0


if __name__ == '__main__':
    sys.exit(main())
