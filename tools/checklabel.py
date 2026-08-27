import collections
import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

path = sys.argv[1]
s = io.open(path, encoding='utf-8').read()

labels = collections.Counter(re.findall(r'\\label\{([^}]+)\}', s))
refs = set(re.findall(r'\\(?:ref|autoref|eqref)\{([^}]+)\}', s))

dup = {k: v for k, v in labels.items() if v > 1}
print('nhan TRUNG LAP :', dup if dup else 'khong co')
print('tong so nhan   :', len(labels))
print('ref thieu label:', sorted(refs - set(labels)) or 'khong co')
orphan = sorted(set(labels) - refs)
print('label chua ai tham chieu (%d):' % len(orphan))
for o in orphan:
    print('   ', o)
