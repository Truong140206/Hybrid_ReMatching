#!/usr/bin/env python3
"""Compile the report locally and report what a reader would actually notice.

Until 2026-08-28 no .tex in this repo had ever been compiled on the authoring
machine -- every file was handed to Overleaf unverified, and the four static
checkers in the handoff exist precisely because that was the only safety net
available. They catch unescaped characters, column-count mismatches, dangling
refs and swallowed backslashes. They cannot catch a table that runs off the
page, because that only exists once TeX has done the typesetting.

This runs pdflatex twice (the table of contents and every \\ref need the second
pass) and then reports the three classes of problem worth acting on:

  errors            -- the document did not build
  undefined refs    -- a \\ref or \\cite that resolves to nothing
  overfull boxes    -- content wider than the text block, i.e. visible bleed

Overfull boxes are filtered by how far they stick out. TeX complains about
fractions of a point routinely and nobody can see those; the default threshold
of 5pt is roughly where it becomes visible on the page.

Usage: python tools/build_tex.py reports/bao_cao_fusion.tex [more.tex ...]
       python tools/build_tex.py --overfull 1 reports/bao_cao_fusion.tex
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

OVERFULL = re.compile(r'Overfull \\([hv])box \(([0-9.]+)pt too wide\).*?'
                      r'at lines? (\d+)', re.S)
UNDEFINED = re.compile(r'(?:LaTeX Warning: )?(Reference|Citation) `([^\']+)\' '
                       r'on page \d+ undefined')
ERROR = re.compile(r'^! (.+)$', re.M)
ERROR_LINE = re.compile(r'^l\.(\d+)', re.M)


def find_pdflatex():
    found = shutil.which('pdflatex')
    if found:
        return found
    # MiKTeX installed for the current user is not on PATH in a fresh shell.
    for base in (os.environ.get('LOCALAPPDATA', ''),
                 os.environ.get('PROGRAMFILES', '')):
        if not base:
            continue
        guess = os.path.join(base, 'Programs', 'MiKTeX', 'miktex', 'bin',
                             'x64', 'pdflatex.exe')
        if os.path.isfile(guess):
            return guess
        guess = os.path.join(base, 'MiKTeX', 'miktex', 'bin', 'x64',
                             'pdflatex.exe')
        if os.path.isfile(guess):
            return guess
    return None


def compile_once(binary, tex_path, out_dir):
    proc = subprocess.run(
        [binary, '-interaction=nonstopmode', '-file-line-error',
         '-output-directory=' + out_dir, os.path.abspath(tex_path)],
        cwd=os.path.dirname(os.path.abspath(tex_path)) or '.',
        capture_output=True, text=True, errors='replace')
    return proc.stdout + proc.stderr


def report(name, log, threshold):
    problems = 0

    errors = ERROR.findall(log)
    if errors:
        problems += len(errors)
        print('  LOI (%d):' % len(errors))
        for message in errors[:12]:
            print('    ! ' + message.strip())
        lines = ERROR_LINE.findall(log)
        if lines:
            print('    tai dong: ' + ', '.join(lines[:12]))

    undefined = sorted(set(UNDEFINED.findall(log)))
    if undefined:
        problems += len(undefined)
        print('  THAM CHIEU TREO (%d):' % len(undefined))
        for kind, key in undefined:
            print('    %s `%s`' % (kind, key))

    wide = [(float(amount), int(line)) for _, amount, line
            in OVERFULL.findall(log) if float(amount) >= threshold]
    if wide:
        problems += len(wide)
        print('  TRAN LE (%d, nguong %.0fpt):' % (len(wide), threshold))
        for amount, line in sorted(wide, reverse=True)[:12]:
            print('    dong %-6d thua %6.1fpt' % (line, amount))

    if not problems:
        print('  sach: khong loi, khong ref treo, khong tran le qua %.0fpt'
              % threshold)
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('--overfull', type=float, default=5.0,
                        help='nguong tran le tinh bang pt (mac dinh 5)')
    parser.add_argument('--keep', metavar='DIR',
                        help='giu lai pdf va aux o thu muc nay')
    args = parser.parse_args()

    binary = find_pdflatex()
    if binary is None:
        print('Khong tim thay pdflatex. Cai MiKTeX roi chay lai.',
              file=sys.stderr)
        return 2
    print('pdflatex: %s\n' % binary)

    total = 0
    for tex_path in args.files:
        if not os.path.isfile(tex_path):
            print('%s: khong co file' % tex_path)
            total += 1
            continue
        out_dir = args.keep or tempfile.mkdtemp(prefix='tex-')
        if args.keep:
            os.makedirs(out_dir, exist_ok=True)
        print('=== %s ===' % tex_path)
        # Two passes: the first writes the .aux and .toc the second reads.
        compile_once(binary, tex_path, out_dir)
        log = compile_once(binary, tex_path, out_dir)
        total += report(tex_path, log, args.overfull)
        pdf = os.path.join(
            out_dir, os.path.splitext(os.path.basename(tex_path))[0] + '.pdf')
        print('  pdf: %s' % (pdf if os.path.isfile(pdf) else 'KHONG SINH RA'))
        if not args.keep:
            shutil.rmtree(out_dir, ignore_errors=True)
        print()
    return 1 if total else 0


if __name__ == '__main__':
    sys.exit(main())
