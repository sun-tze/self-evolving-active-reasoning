"""Static installed-CLI check; never imports vLLM, torch or GPU code.

The controller bounds this process to 30 seconds. This verifies registration,
not runtime compatibility; actual startup and inference retain their own gates.
"""
import argparse
import ast
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import time

VERSION = '0.31.0'
ENTRYPOINT = 'vllm.entrypoints.cli.main:main'
MAX_SOURCE_BYTES = 2_000_000


def require(ok, message):
    if not ok:
        raise ValueError(message)


def source(path, limit=MAX_SOURCE_BYTES):
    require(path.is_file(), 'missing CLI source: ' + str(path))
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, 'oversized CLI source: ' + str(path))
    return data.decode('utf-8'), hashlib.sha256(data).hexdigest()


def registered_option(text):
    """Require a direct registration in EngineArgs.add_cli_args, not a mention."""
    tree = ast.parse(text)
    methods = [f for c in tree.body if isinstance(c, ast.ClassDef) and c.name == 'EngineArgs'
               for f in c.body if isinstance(f, ast.FunctionDef) and f.name == 'add_cli_args']
    require(len(methods) == 1, 'missing EngineArgs.add_cli_args')
    calls = [n.value for n in methods[0].body if isinstance(n, ast.Expr)
             and isinstance(n.value, ast.Call)]
    matches = [c for c in calls if isinstance(c.func, ast.Attribute)
               and isinstance(c.func.value, ast.Name) and c.func.value.id == 'attention_group'
               and c.func.attr == 'add_argument' and c.args
               and isinstance(c.args[0], ast.Constant) and c.args[0].value == '--attention-backend']
    require(len(matches) == 1, 'attention-backend option is not registered')
    call = matches[0]
    require(any(k.arg is None and isinstance(k.value, ast.Subscript)
                and isinstance(k.value.value, ast.Name) and k.value.value.id == 'attention_kwargs'
                and isinstance(k.value.slice, ast.Constant) and k.value.slice.value == 'backend'
                for k in call.keywords), 'attention-backend config binding differs')
    return call.lineno


def inspect_cli(executable):
    started = time.monotonic()
    prefix = Path(sys.prefix).resolve()
    executable = Path(executable)
    require(executable.resolve() == (prefix/'bin/vllm').resolve(), 'CLI executable outside pinned runtime')
    require(executable.is_file() and os.access(executable, os.X_OK), 'CLI executable unavailable')
    dist = metadata.distribution('vllm')  # Reading metadata does not import the package.
    require(dist.version == VERSION, 'unreviewed vLLM version')
    entries = [e.value for e in dist.entry_points if e.group == 'console_scripts' and e.name == 'vllm']
    require(entries == [ENTRYPOINT], 'unexpected vLLM entrypoint')
    script, exe_sha = source(executable, 16384)
    require(script.splitlines()[0] == '#!' + str(Path(sys.executable)), 'CLI interpreter differs from preflight')
    wrapper = ast.parse(script)
    require(any(isinstance(n, ast.ImportFrom) and n.module == 'vllm.entrypoints.cli.main'
                and any(a.name == 'main' and a.asname is None for a in n.names)
                for n in wrapper.body), 'CLI wrapper entrypoint differs')
    inventory = {}
    for rel in ('vllm/entrypoints/cli/main.py', 'vllm/engine/arg_utils.py'):
        path = Path(dist.locate_file(rel)).resolve()
        require(path.is_relative_to(prefix), 'CLI source outside pinned runtime')
        text, digest = source(path)
        inventory[rel] = digest
        if rel.endswith('arg_utils.py'):
            line = registered_option(text)
    return {'status': 'passed', 'method': 'installed_metadata_and_static_ast',
            'vllm_version': dist.version, 'entrypoint': ENTRYPOINT,
            'executable': str(executable), 'executable_sha256': exe_sha,
            'source_sha256': inventory, 'option': '--attention-backend',
            'registration_line': line, 'imports_vllm': False,
            'elapsed_seconds': time.monotonic() - started}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--executable', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = inspect_cli(args.executable)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('Installed vLLM version, executable and attention-backend registration verified.')
