#!/usr/bin/env python3
"""Rebuild the ImageNet-A or ImageNet-R image folder from a parquet mirror.

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

Class names are looked for in two places, in order: the `huggingface` key of
the parquet schema metadata, then dataset_infos.json. The first is absent on
this particular mirror -- which is why the second exists -- and if neither
yields names the tool stops rather than inventing directory names, since
writing 7,500 files into wrongly labelled directories is far more expensive to
undo than one error message.

Rows are streamed in batches; reading both files whole would hold roughly
680 MB of image bytes in memory at once for no benefit.

Neither official tar is obtainable from its source any more. ImageNet-A times
out on IPv4 from some networks; ImageNet-R now redirects to
iris.eecs.berkeley.edu and answers 404 there. The two mirrors named in SPECS
carry the same images, and both store them as parquet, so one tool serves both.

Usage:
    python tools/imagenet_folder_from_parquet.py --which imagenet-r --parquet-dir DIR
    python tools/imagenet_folder_from_parquet.py --which imagenet-a --parquet-dir DIR
"""
import argparse
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH = 64

# So anh va so lop la cua ban goc, dung de doi chieu sau khi ghi xong. Khong
# co chung thi mot ban sao thieu mot manh parquet se di qua ma khong ai biet.
SPECS = {
    # Hai ban sao khong ghi nhan giong nhau. barkermrl luu mot so nguyen va
    # mot bang ten lop, nen phai tra bang. axiong luu thang WordNet id dang
    # chuoi, tuc da la ten thu muc can tao -- khong tra bang gi ca.
    'imagenet-a': {'out': 'imagenet-a', 'images': 7500, 'classes': 200,
                   'repo': 'barkermrl/imagenet-a', 'label_col': 'label',
                   'lookup': True,
                   'next': 'python tools/prepare_datasets.py --which imagenet-a'},
    'imagenet-r': {'out': 'imagenet-r', 'images': 30000, 'classes': 200,
                   'repo': 'axiong/imagenet-r', 'label_col': 'wnid',
                   'lookup': False,
                   'next': None},
}


def default_root():
    return os.path.join(os.path.dirname(REPO_ROOT), 'datasets')


def names_from_json(doc):
    """Find the ClassLabel names anywhere inside a parsed JSON document.

    The shape of both the parquet metadata blob and dataset_infos.json has
    moved around across versions of `datasets`, so this walks the tree looking
    for the ClassLabel node instead of hard-coding a path into it.
    """
    found = []

    def walk(node):
        if isinstance(node, dict):
            if 'names' in node and isinstance(node['names'], list) and (
                    node.get('_type') == 'ClassLabel' or 'num_classes' in node):
                found.append(node['names'])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(doc)
    if not found:
        return None
    if any(f != found[0] for f in found[1:]):
        raise SystemExit('nhieu danh sach lop khac nhau, dung lai')
    return found[0]


def names_from_parquet(schema):
    meta = schema.metadata or {}
    if meta:
        print('  khoa metadata trong parquet: %s'
              % [k.decode('utf-8', 'replace') for k in meta])
    else:
        print('  parquet khong mang metadata nao')
    blob = meta.get(b'huggingface')
    if blob is None:
        return None
    return names_from_json(json.loads(blob.decode('utf-8')))


def names_from_infos(parquet_dir, explicit):
    candidates = [explicit] if explicit else [
        os.path.join(parquet_dir, 'dataset_infos.json'),
        os.path.join(os.path.dirname(parquet_dir.rstrip('/\\')),
                     'dataset_infos.json'),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            print('  doc ten lop tu %s' % path)
            with open(path, encoding='utf-8') as handle:
                return names_from_json(json.load(handle))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--which', required=True, choices=sorted(SPECS),
                        help='bo du lieu can dung lai')
    parser.add_argument('--parquet-dir', required=True,
                        help='thu muc chua cac tep .parquet da tai')
    parser.add_argument('--names-json', default=None,
                        help='duong dan dataset_infos.json (neu de khac cho)')
    parser.add_argument('--root', default=None,
                        help='thu muc du lieu (mac dinh: canh repo)')
    args = parser.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print('THIEU pyarrow. Cai bang:  .venv/bin/python -m pip install pyarrow')
        return 1

    spec = SPECS[args.which]
    root = args.root or default_root()
    out_dir = os.path.join(root, spec['out'])

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
    for path in files:
        print('  %s  (%.1f MB)' % (os.path.basename(path),
                                   os.path.getsize(path) / 1e6))

    readers = [pq.ParquetFile(path) for path in files]

    label_col = spec['label_col']
    columns = list(readers[0].schema_arrow.names)
    missing = [c for c in ('image', label_col) if c not in columns]
    if missing:
        print('parquet thieu cot %s; cot co san: %s' % (missing, columns))
        return 1

    names = None
    if not spec['lookup']:
        print('  cot %s mang thang WordNet id, khong can bang ten lop'
              % label_col)
    else:
        names = names_from_parquet(readers[0].schema_arrow)
        if not names:
            names = names_from_infos(args.parquet_dir, args.names_json)
    if spec['lookup'] and not names:
        print('KHONG DOC DUOC ten lop tu parquet lan dataset_infos.json.')
        print('Tai kem tep do roi chay lai:')
        print('  wget -P %s https://huggingface.co/datasets/%s/resolve/main/'
              'dataset_infos.json' % (args.parquet_dir, spec['repo']))
        return 1

    if names:
        print('%d lop, tu %s den %s' % (len(names), names[0], names[-1]))
        if not names[0].startswith('n'):
            print('CANH BAO: ten lop khong phai WordNet id, kiem tra lai')

    written = 0
    for reader, path in zip(readers, files):
        for batch in reader.iter_batches(batch_size=BATCH,
                                         columns=['image', label_col]):
            images = batch.column('image').to_pylist()
            labels = batch.column(label_col).to_pylist()
            for image, label in zip(images, labels):
                cls = names[label] if names else str(label)
                if written == 0 and not cls.startswith('n'):
                    print('CANH BAO: %r khong giong WordNet id' % cls)
                class_dir = os.path.join(out_dir, cls)
                os.makedirs(class_dir, exist_ok=True)

                stem = os.path.basename(image.get('path') or '')
                if not stem:
                    stem = '%06d.jpg' % written
                dest = os.path.join(class_dir, stem)
                if os.path.exists(dest):        # tranh dam ten giua hai tep
                    base, ext = os.path.splitext(stem)
                    dest = os.path.join(class_dir,
                                        '%s_%06d%s' % (base, written, ext))

                with open(dest, 'wb') as handle:
                    handle.write(image['bytes'])
                written += 1
        print('  %s -> tong %d anh' % (os.path.basename(path), written))

    n_dirs = len([d for d in os.listdir(out_dir)
                  if os.path.isdir(os.path.join(out_dir, d))])
    print('\nda ghi %d anh vao %d thu muc lop tai %s' % (written, n_dirs, out_dir))
    if written != spec['images'] or n_dirs != spec['classes']:
        print('CANH BAO: ban goc la %d anh trong %d lop'
              % (spec['images'], spec['classes']))
    if spec['next']:
        print('Buoc tiep: %s' % spec['next'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
