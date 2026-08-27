"""Tim (va sua) cac dong ket thuc bang dung MOT dau backslash.

Trong LaTeX, ngat hang cua bang la \\ (hai dau). Mot dau o cuoi dong gan nhu
luon la dau hieu bi nuot mat mot dau khi ghi file.
"""
import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BS = chr(92)
FIX = '--fix' in sys.argv
paths = [a for a in sys.argv[1:] if not a.startswith('--')]

for path in paths:
    s = io.open(path, encoding='utf-8').read()
    lines = s.split('\n')
    hits = []
    in_verb = False
    for i, line in enumerate(lines):
        if '\\begin{verbatim}' in line:
            in_verb = True
            continue
        if '\\end{verbatim}' in line:
            in_verb = False
            continue
        if in_verb:
            continue
        m = re.search(re.escape(BS) + '+$', line.rstrip(' \t'))
        if m and len(m.group(0)) % 2 == 1:
            hits.append(i)
    print('=== %s : %d dong nghi bi nuot ===' % (path, len(hits)))
    for i in hits:
        print('   dong %d: %s' % (i + 1, lines[i].strip()[-70:]))
    if FIX and hits:
        for i in hits:
            stripped = lines[i].rstrip(' \t')
            lines[i] = stripped + BS
        io.open(path, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines))
        print('   -> DA SUA %d dong' % len(hits))
