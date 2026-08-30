#!/usr/bin/env python3
"""Check that every flag the run scripts send is accepted by every config.

This exists because of a failure that would have cost about five days. The
--rp_* family and --strict_exemplar_free were added to imr_lora.py and
cifar100_lora.py as the method was built, and never to ima_lora.py or
five_datasets_lora.py, because those two benchmarks had never been run.
Training on them would have completed normally; then every evaluation would
have died on argparse's `unrecognized arguments`, ten cells in, with all the
GPU time already spent.

Nothing in the repository connected the flags a shell script emits to the
options a config declares, so the mismatch was invisible until runtime. This
closes that gap: flags are read out of the scripts themselves, so the check
cannot drift out of date the way a hard-coded list would.

It is a static check -- it compares option strings, and needs neither a GPU nor
the datasets. Run it before launching anything long.

Usage:  python tools/check_run_args.py
"""
import argparse
import importlib
import io
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

TII_CONFIGS = ['imr_hideprompt_5e', 'cifar100_hideprompt_5e',
               'ima_hideprompt_5e', 'five_datasets_hideprompt_5e']
LORA_CONFIGS = ['imr_lora', 'cifar100_lora', 'ima_lora', 'five_datasets_lora']

# (script, marker inside the `main.py ...` line, configs that line runs under)
CHECKS = [
    ('training_scripts/train_any_4090.sh', 'CFG_TII', TII_CONFIGS),
    ('training_scripts/train_any_4090.sh', 'CFG_LORA', LORA_CONFIGS),
    ('training_scripts/eval_rp_head_any_4090.sh', 'CONFIG', LORA_CONFIGS),
]

FLAG = re.compile(r'(?<![\w-])--([A-Za-z][A-Za-z0-9_-]*)')


def invocation_flags(script_path, marker):
    """Flags on the `main.py` command line whose config name matches `marker`.

    The command spans several lines joined by trailing backslashes, and some
    flags arrive through shell variables (FUSE_FLAG and friends), so those
    assignments are scanned too.
    """
    text = io.open(os.path.join(REPO_ROOT, script_path),
                   encoding='utf-8', newline='').read()
    lines = text.splitlines()

    start = None
    for i, line in enumerate(lines):
        if 'main.py' in line and marker in line:
            start = i
            break
    if start is None:
        raise SystemExit('%s: khong thay loi goi main.py voi %s'
                         % (script_path, marker))

    block = []
    i = start
    while i < len(lines):
        block.append(lines[i])
        if not lines[i].rstrip().endswith('\\'):
            break
        i += 1
    body = '\n'.join(block)

    flags = set(FLAG.findall(body))

    # Flags that reach the command line through a shell variable.
    for name in re.findall(r'\$\{([A-Za-z_][A-Za-z0-9_]*_FLAG)\}', body):
        for assign in re.findall(r'%s=("([^"]*)"|\S*)' % name, text):
            flags |= set(FLAG.findall(assign[1]))
    return flags


def config_options(name):
    module = importlib.import_module('configs.%s' % name)
    parser = argparse.ArgumentParser(add_help=False)
    module.get_args_parser(parser)
    options = set()
    for action in parser._actions:                       # noqa: SLF001
        for string in action.option_strings:
            options.add(string.lstrip('-'))
    return options


def main():
    ok = True
    for script, marker, configs in CHECKS:
        flags = invocation_flags(script, marker)
        print('%s  [%s]  %d co' % (script, marker, len(flags)))
        for name in configs:
            have = config_options(name)
            missing = sorted(f for f in flags if f not in have)
            if missing:
                ok = False
                print('  %-30s THIEU %d: %s'
                      % (name, len(missing), ' '.join(missing)))
            else:
                print('  %-30s du' % name)
        print()

    print('TAT CA KHOP' if ok else 'CO CHO THIEU -- dung chay truoc khi vá')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
