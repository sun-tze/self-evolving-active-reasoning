"""Safe entry point for future extraction; original frozen prepare.py is archival."""
import argparse
from pathlib import Path
from experiments.sun_c2_real.common import load_bundle
from .prepare import MANIFEST_SHA, main as prepare_main
from .provenance import verified_targets


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--bundle', type=Path, required=True)
    args, _ = parser.parse_known_args()
    manifest, _ = load_bundle(args.bundle, MANIFEST_SHA)
    verified_targets(args.bundle / 'private/targets.jsonl', manifest)
    prepare_main()


if __name__ == '__main__':
    main()
