"""Dem so cot cua moi hang tabular. Bat loi thieu \\ hoac lech so &."""
import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def grab_braced(s, i):
    """s[i] phai la '{'. Tra ve (noi_dung, chi_so_sau_dau_dong)."""
    assert s[i] == '{'
    depth = 0
    for j in range(i, len(s)):
        if s[j] == '{':
            depth += 1
        elif s[j] == '}':
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise ValueError('dau ngoac khong dong')


def count_cols(spec):
    n = 0
    i = 0
    while i < len(spec):
        c = spec[i]
        if c in '@!><':
            if i + 1 < len(spec) and spec[i + 1] == '{':
                _, i = grab_braced(spec, i + 1)
                continue
            i += 1
        elif c in 'lcr':
            n += 1
            i += 1
        elif c in 'pmb':
            n += 1
            i += 1
            if i < len(spec) and spec[i] == '{':
                _, i = grab_braced(spec, i)
        elif c in 'LRMC':          # newcolumntype tu dinh nghia
            n += 1
            i += 1
        elif c == '*':
            i += 1
            rep, i = grab_braced(spec, i)
            sub, i = grab_braced(spec, i)
            n += int(rep) * count_cols(sub)
        else:
            i += 1
    return n


path = sys.argv[1]
s = io.open(path, encoding='utf-8').read()
s = re.sub(r'\\begin\{verbatim\}.*?\\end\{verbatim\}', '', s, flags=re.S)

problems = 0
for m in re.finditer(r'\\begin\{tabular\}', s):
    i = m.end()
    while i < len(s) and s[i] in ' \n':
        i += 1
    if i < len(s) and s[i] == '[':
        i = s.index(']', i) + 1
    spec, i = grab_braced(s, i)
    ncol = count_cols(spec)
    end = s.index('\\end{tabular}', i)
    body = s[i:end]
    lineno = s.count('\n', 0, m.start()) + 1

    rows = re.split(r'\\\\', body)
    for row in rows[:-1] + ([rows[-1]] if rows[-1].strip() else []):
        clean = re.sub(
            r'\\(toprule|midrule|bottomrule|addlinespace|hline)(\[[^\]]*\])?',
            '', row)
        clean = re.sub(r'\\cmidrule(\([^)]*\))?(\{[^}]*\})?', '', clean)
        if not clean.strip():
            continue
        namps = len(re.findall(r'(?<!\\)&', clean))
        need = ncol - 1
        if '\\multicolumn' in clean:
            for mc in re.finditer(r'\\multicolumn\{(\d+)\}', clean):
                need -= int(mc.group(1)) - 1
        if namps != need:
            print('  ! tabular bat dau dong %d (spec="%s" -> %d cot)'
                  % (lineno, spec, ncol))
            print('      hang co %d dau & , can %d' % (namps, need))
            print('      > %s' % clean.strip().replace('\n', ' ')[:100])
            problems += 1

print('%s: %d hang lech' % (path, problems))
