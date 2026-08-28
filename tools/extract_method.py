#!/usr/bin/env python3
"""Build a standalone method document out of the full report.

A supervisor asking for "the method" wants to judge the approach before the
results exist to argue about: what problem is being solved, what is proposed,
why it should work. That is Sections 2 and 3 of the report -- the analysis of
the two limitations, and the proposed method. Everything from the experimental
setup onward is a different conversation.

Generating this rather than maintaining a second copy matters: the two would
drift, and the version being reviewed would stop being the version being built.

The extracted range turns out to be almost self-contained -- of the four
cross-references it makes, three point inside itself. The fourth points at the
closed-directions section, which belongs to results, so that one sentence is
rewritten to stand on its own.

Usage: python tools/extract_method.py [OUT.tex]
"""
import io
import os
import sys

SOURCE = 'reports/bao_cao_fusion.tex'
DEFAULT_OUT = 'reports/phuong_phap_trich.tex'

START = '\\section{Phân tích hai hạn chế}'
END = '\\section{Thiết lập thí nghiệm}'
PREAMBLE_END = '\\begin{document}'

# The only reference that points outside the extracted range.
OUTBOUND = ('Mục \\ref{sec:closed} trình bày số liệu so sánh giữa hai '
            'phương án.')
OUTBOUND_FIX = ('Số liệu so sánh giữa hai phương án nằm ở phần khảo sát các '
                'hướng đã loại bỏ của báo cáo đầy đủ.')

FRAME = """\\begin{center}
{\\large\\bfseries Phần phương pháp}\\\\[2pt]
{\\small Trích từ báo cáo kỹ thuật, gồm phân tích hạn chế và phương pháp đề
xuất. Phần thiết lập thí nghiệm, kết quả và khảo sát loại bỏ thành phần nằm ở
báo cáo đầy đủ.}
\\end{center}

\\bigskip
"""


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    if not os.path.isfile(SOURCE):
        print('Khong thay %s -- chay tu goc kho ma nguon.' % SOURCE)
        return 2
    text = io.open(SOURCE, encoding='utf-8').read()

    for marker in (PREAMBLE_END, START, END):
        if marker not in text:
            print('Khong tim thay moc: %s' % marker.encode(
                'ascii', 'replace').decode('ascii'))
            return 1

    preamble = text[:text.index(PREAMBLE_END) + len(PREAMBLE_END)]
    body = text[text.index(START):text.index(END)]

    if OUTBOUND in body:
        body = body.replace(OUTBOUND, OUTBOUND_FIX, 1)
    else:
        print('Chu y: khong thay cau tham chieu ra ngoai; kiem lai ref treo.')

    # The report renumbers from 1, so the extract should too: what was Section 2
    # becomes Section 1 here, and a reader is not left wondering what came first.
    document = (preamble + '\n\n' + FRAME + '\n' + body.rstrip()
                + '\n\n\\end{document}\n')
    io.open(out_path, 'w', encoding='utf-8', newline='\n').write(document)

    print('Da ghi %s (%d dong)' % (out_path, document.count('\n')))
    print('Bien dich: py tools/build_tex.py %s' % out_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
