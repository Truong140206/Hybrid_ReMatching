import io
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

path = sys.argv[1]
# newline='' la BAT BUOC: che do universal-newline mac dinh se dich mot ky tu
# \r don le thanh \n khi doc, nen \ref bi cut thanh CR+'ef{' se bi che giau.
s = io.open(path, encoding='utf-8', newline='').read()

counts = Counter()
for kind, name in re.findall(r'\\(begin|end)\{([a-zA-Z*]+)\}', s):
    counts[name] += 1 if kind == 'begin' else -1
bad = {k: v for k, v in counts.items() if v != 0}
print('moi truong lech:', bad if bad else 'khong co')

MATH = ('verbatim', 'align', r'align\*', 'equation', r'equation\*',
        'tikzpicture', 'tabular')
body = s
for env in MATH:
    body = re.sub(r'\\begin\{' + env + r'\}.*?\\end\{' + env + r'\}',
                  '', body, flags=re.S)
body = re.sub(r'\\\[.*?\\\]', '', body, flags=re.S)
body = re.sub(r'\$[^$]*\$', '', body)
body = re.sub(r'(?<!\\)%.*', '', body)

for ch in ['_', '&', '#']:
    spots = list(re.finditer(r'(?<!\\)' + re.escape(ch), body))
    print('ky tu %s chua escape: %d' % (ch, len(spots)))
    for m in spots[:3]:
        print('    ...' + body[max(0, m.start() - 50):m.start() + 20].replace('\n', ' '))

labels = set(re.findall(r'\\label\{([^}]+)\}', s))
refs = set(re.findall(r'\\ref\{([^}]+)\}', s))
print('ref thieu label:', (refs - labels) or 'khong co')

ok = True
for m in re.finditer(r'\\begin\{verbatim\}(.*?)\\end\{verbatim\}', s, flags=re.S):
    non = sorted({c for c in m.group(1) if ord(c) > 127})
    if non:
        print('CANH BAO verbatim co ky tu non-ASCII:', non)
        ok = False
if ok:
    print('verbatim: ASCII thuan, OK')

print('tikz library:', set(re.findall(r'\\usetikzlibrary\{([^}]+)\}', s)))
print('dung cu phap calc trong tikz:',
      'CO - can \\usetikzlibrary{calc}' if re.search(r'\$\([^)]*\)\s*!', s) else 'khong')
print('goi duoc nap:', sorted(set(re.findall(r'\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}', s))))

# Ky tu dieu khien lac: dau hieu mot lop shell/python da dien giai \t \r
ctrl = [i for i, c in enumerate(s) if c in '\t\r\x0b\x0c']
print('ky tu tab/CR lac:', len(ctrl))
for i in ctrl[:3]:
    print('    ...' + s[max(0, i - 40):i + 20].replace('\n', ' '))

# Manh lenh LaTeX bi cut mat dau backslash (\textbf -> extbf, \ref -> ef{)
orphan = []
for word in ['extbf', 'exttt', 'extit', 'extsc', 'egin{', 'ewcommand']:
    orphan += [(word, m.start())
               for m in re.finditer(r'(?<![a-zA-Z\\])' + re.escape(word), s)]
print('manh lenh bi cut dau backslash:', len(orphan))
for w, i in orphan[:3]:
    print('    %s ...%s...' % (w, s[max(0, i - 40):i + 15].replace('\n', ' ')))
