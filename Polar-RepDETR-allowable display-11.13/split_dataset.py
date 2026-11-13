import argparse
import csv
import hashlib
import json
import math
import platform
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


SPLITS = ('train', 'val', 'test')
RATIOS = (7, 1, 2)
SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def allocate(total, weights, capacities=None):
    denominator = sum(weights)
    if denominator <= 0:
        raise ValueError('Allocation requires a positive population')
    capacities = list(weights) if capacities is None else list(capacities)
    if total < 0 or total > sum(capacities):
        raise ValueError('Allocation exceeds available capacity')
    numerators = [total * w for w in weights]
    result = [min(n // denominator, cap) for n, cap in zip(numerators, capacities)]
    while sum(result) < total:
        available = [i for i, cap in enumerate(capacities) if result[i] < cap]
        i = max(available, key=lambda k: (numerators[k] - result[k] * denominator, -k))
        result[i] += 1
    return result


def read_image_classes(path):
    if path is None:
        return None
    result = {}
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'image', 'class_id'} <= set(reader.fieldnames or []):
            raise ValueError('Image-class CSV requires image,class_id columns')
        for row in reader:
            key = row['image'].replace('.')
            if key in result:
                raise ValueError('Duplicate image in image-class CSV: ' + key)
            cls = int(row['class_id'])
            if cls < 0:
                raise ValueError('Class IDs must be nonnegative integers')
            result[key] = cls
    return result


def inspect(images, labels, image_classes, image_size=416):
    records, stems, hashes = [], set(), {}
    paths = sorted((p for p in images.rglob('*') if p.is_file() and p.suffix.lower() in SUFFIXES),
                   key=lambda p: p.relative_to(images).as_posix())
    for path in paths:
        relative = path.relative_to(images)
        key = relative.as_posix()
        stem = relative.with_suffix('').as_posix().casefold()
        if stem in stems:
            raise ValueError('Image names would share a label: ' + key)
        stems.add(stem)
        label = labels / relative.with_suffix('.txt')
        if not label.is_file():
            raise ValueError('Missing label for image: ' + key)
        classes = set()
        box_counts = Counter()
        for line in label.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            values = list(map(float, line.split()))
            if len(values) != 5 or not all(math.isfinite(x) for x in values):
                raise ValueError('Invalid detection annotation: ' + key)
            cls, x, y, w, h = values
            if cls != int(cls) or cls < 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError('Expected nonnegative class ID and normalized xywh: ' + key)
            classes.add(int(cls))
            box_counts[int(cls)] += 1
        if not classes:
            raise ValueError('Stratification requires an annotated defect class: ' + key)
        if image_classes is not None:
            if key not in image_classes:
                raise ValueError('Image missing from image-class CSV: ' + key)
            assigned = image_classes[key]
            if assigned not in classes:
                raise ValueError('Assigned class is absent from actual annotations: ' + key)
        elif len(classes) == 1:
            assigned = next(iter(classes))
        else:
            raise ValueError('Multi-class image requires --image-classes: ' + key)
        with Image.open(path) as image:
            if image.size != (image_size, image_size):
                raise ValueError('Unexpected image resolution: ' + key)
            image.verify()
        image_hash = digest(path)
        if image_hash in hashes:
            raise ValueError('Duplicate source image content: ' + key + ', ' + hashes[image_hash])
        hashes[image_hash] = key
        records.append({'image': key, 'label': relative.with_suffix('.txt').as_posix(),
                        'class_id': assigned, 'image_sha256': image_hash,
                        'label_sha256': digest(label),
                        'annotated_classes': ','.join(map(str, sorted(classes))),
                        'box_counts': json.dumps(dict(sorted(box_counts.items())))})
    if not records:
        raise ValueError('No source images found')
    actual_classes = {int(c) for r in records for c in r['annotated_classes'].split(',')}
    if actual_classes != {r['class_id'] for r in records}:
        raise ValueError('Every annotated class must have an image-level stratum in the supplied mapping')
    expected_labels = {r['label'] for r in records}
    found_labels = {p.relative_to(labels).as_posix() for p in labels.rglob('*.txt')}
    if 'classes.txt' not in expected_labels:
        found_labels.discard('classes.txt')
    if found_labels != expected_labels:
        raise ValueError('Image and label filenames are not paired exactly')
    if image_classes is not None and set(image_classes) != {r['image'] for r in records}:
        raise ValueError('Image-class CSV must cover the input images exactly')
    return records


def assign_splits(records, seed):
    counts = Counter(r['class_id'] for r in records)
    classes = sorted(counts)
    sizes = [counts[c] for c in classes]
    # Global split sizes are rounded first, then apportioned across classes.
    totals = allocate(len(records), RATIOS, [len(records)] * len(SPLITS))
    train = allocate(totals[0], sizes)
    val = allocate(totals[1], sizes, [n - t for n, t in zip(sizes, train)])
    rng = random.Random(seed)
    for cls, nt, nv in zip(classes, train, val):
        group = [r for r in records if r['class_id'] == cls]
        group.sort(key=lambda r: r['image'])
        rng.shuffle(group)
        for i, record in enumerate(group):
            record['split'] = 'train' if i < nt else 'val' if i < nt + nv else 'test'
    actual = [sum(r['split'] == s for r in records) for s in SPLITS]
    if actual != totals:
        raise RuntimeError('Internal split allocation mismatch')
    return records


def split_dataset(images, labels, output, seed=2026, image_classes=None, image_size=416,
                  min_train_per_class=255):
    images, labels, output = Path(images).resolve(), Path(labels).resolve(), Path(output).resolve()
    if not isinstance(image_size, int) or image_size < 1:
        raise ValueError('image_size must be a positive integer')
    if not images.is_dir() or not labels.is_dir():
        raise ValueError('Input image and label directories must exist')
    for source in (images, labels):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError('Output must be separate from both source directories')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output must be empty; source data will not be overwritten')
    records = inspect(images, labels, read_image_classes(Path(image_classes) if image_classes else None), image_size)
    assign_splits(records, seed)
    if not isinstance(min_train_per_class, int) or min_train_per_class < 0:
        raise ValueError('min_train_per_class must be a nonnegative integer')
    train_counts = Counter(r['class_id'] for r in records if r['split'] == 'train')
    for cls in sorted({r['class_id'] for r in records}):
        if train_counts[cls] < min_train_per_class:
            raise ValueError('Class ' + str(cls) + ' has insufficient training source images: '
                             + str(train_counts[cls]) + '; required minimum: ' + str(min_train_per_class))
    assigned_counts, presence_counts, instance_counts = {}, {}, {}
    classes = sorted({r['class_id'] for r in records})
    for split in SPLITS:
        group = [r for r in records if r['split'] == split]
        assigned = Counter(r['class_id'] for r in group)
        if any(assigned[c] == 0 for c in classes):
            raise ValueError('Every assigned class must be represented in ' + split)
        presence, instances = Counter(), Counter()
        for r in group:
            presence.update(map(int, r['annotated_classes'].split(',')))
            instances.update({int(k): v for k, v in json.loads(r['box_counts']).items()})
        assigned_counts[split] = {c: assigned[c] for c in classes}
        presence_counts[split] = {c: presence[c] for c in classes}
        instance_counts[split] = {c: instances[c] for c in classes}
    for kind in ('images', 'labels'):
        for split in SPLITS:
            (output / kind / split).mkdir(parents=True, exist_ok=True)
    for r in records:
        image_out = output / 'images' / r['split'] / r['image']
        label_out = output / 'labels' / r['split'] / r['label']
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(images / r['image'], image_out)
        shutil.copy2(labels / r['label'], label_out)
        if digest(image_out) != r['image_sha256'] or digest(label_out) != r['label_sha256']:
            raise RuntimeError('Copied file content changed: ' + r['image'])
    with (output / 'split_manifest.csv').open('w', encoding='utf-8', newline='') as f:
        fields = ['image', 'label', 'class_id', 'split', 'image_sha256', 'label_sha256',
                  'annotated_classes', 'box_counts']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    metadata = {'seed': seed, 'ratios': dict(zip(SPLITS, RATIOS)),
                'rng': 'random.Random', 'python_version': platform.python_version(),
                'ordering': 'relative filename lexical order; class ID ascending',
                'allocation': 'largest remainder: global totals, training classes, capacity-limited validation classes; test receives the remainder',
                'resolution': [image_size, image_size], 'resizing': False,
                'min_train_per_assigned_class': min_train_per_class,
                'assigned_image_counts': assigned_counts,
                'images_containing_class': presence_counts,
                'box_instance_counts': instance_counts,
                'balance_basis': 'proportional allocation by unique assigned image class; multi-class annotations are retained'}
    (output / 'split_config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--image-classes', type=Path)
    parser.add_argument('--image-size', type=int, default=416)
    parser.add_argument('--min-train-per-class', type=int, default=300)
    args = parser.parse_args()
    split_dataset(args.images, args.labels, args.output, args.seed, args.image_classes,
                  args.image_size, args.min_train_per_class)
    print('Dataset split complete; see split_manifest.csv and split_config.json.')


if __name__ == '__main__':
    main()
