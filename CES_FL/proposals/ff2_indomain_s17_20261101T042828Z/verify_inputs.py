"""Read-only verification of the prepared FF2 manifests/config; never trains."""
import collections
import datetime
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    started = time.monotonic()
    request = json.loads((ROOT / 'request.json').read_text())
    source = Path(request['model_source']) / 'CES_FL/src'
    assert sha(source / 'ces_fl_diagram_model_v021.py') == request['model_source_sha256']
    assert sha(request['pretraining_config']) == request['config_sha256']
    assert sha(request['pretraining_template']) == request['template_sha256']
    config = json.loads(Path(request['pretraining_config']).read_text())
    assert sha(config['weights']) == request['weights_sha256']
    assert config['mode'] == 'bce' and config['pairs'] is None and config['stage'] == 'pretrain'
    assert 'test' not in config and config['seed'] == 17
    template = json.loads(Path(request['pretraining_template']).read_text())
    changed = {k for k in set(template) | set(config) if template.get(k) != config.get(k)}
    assert changed == {'train', 'val', 'protocol_overlap_file'}
    sys.path.insert(0, str(source))
    import ces_fl_diagram_model_v021 as model
    rows, report = {}, {}
    for split, expected in request['data']['splits'].items():
        assert sha(expected['path']) == expected['sha256']
        # The actual model reader checks every path is a file, labels, required
        # provenance, duplicate original paths, and class coverage. No decoding.
        rows[split] = model.read_rows(expected['path'])
        labels = {str(k): v for k, v in collections.Counter(r['label'] for r in rows[split]).items()}
        assert len(rows[split]) == expected['rows'] and labels == expected['labels']
        report[split] = {'path': expected['path'], 'sha256': expected['sha256'],
                         'rows': len(rows[split]), 'labels': labels,
                         'every_image_path_is_file': True, 'all_images_decoded': False}
        print('SPLIT_PASS', split, len(rows[split]), labels, flush=True)
    overlaps = {}
    for left, right in itertools.combinations(rows, 2):
        model.check_splits(rows[left], rows[right], None)
        overlaps[left + ':' + right] = {'image_paths': 0, 'recorded_identity_ids': 0,
                                        'recorded_source_ids': 0}
    assert sha(request['data']['pairs']['path']) == request['data']['pairs']['sha256']
    _, pairs = model.read_pairs(request['data']['pairs']['path'], rows['train'], ['age'])
    assert pairs['approved'] == 16 and pairs['by_attribute'] == {'age': 16}
    result = {'status': 'PASS', 'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'command': [sys.executable, str(Path(__file__).resolve())],
              'splits': report, 'overlaps': overlaps, 'pairs': pairs,
              'config_sha256': request['config_sha256'], 'source_sha256': request['model_source_sha256'],
              'weights_sha256': request['weights_sha256'], 'training_started': False,
              'counterfactual_semantic_validity_verified': False, 'true_person_independence_verified': False,
              'elapsed_seconds': time.monotonic() - started}
    target = ROOT / 'INPUT_VERIFICATION.json'
    with target.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print('FF2_INPUT_VERIFICATION_PASS', str(target), flush=True)


if __name__ == '__main__':
    main()
