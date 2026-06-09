"""
Generate manuscript_ijacsa_blind.tex from manuscript_ijacsa.tex.
Copies main manuscript exactly, only anonymizing author-identifying content.
"""
import re
from pathlib import Path

root = Path(__file__).parent.parent
src  = root / 'ijacsa_paper' / 'manuscript_ijacsa.tex'
dst  = root / 'ijacsa_paper' / 'manuscript_ijacsa_blind.tex'

with open(src, encoding='utf-8') as f:
    text = f.read()

# 1. Replace header comment
text = text.replace(
    '%% IJACSA Manuscript — IEEEtran.cls (SAI official template)',
    '%% IJACSA Manuscript — DOUBLE-BLIND VERSION (authors anonymized for peer review)'
)

# 2. Replace full \author{...} block with anonymous placeholder
# The block starts at \author{% and ends just before \maketitle
author_marker = r'\author{%'
maketitle_marker = r'\maketitle'
a_start = text.find(author_marker)
a_end   = text.find(maketitle_marker)
# walk back to find closing newline before \maketitle
a_end   = text.rfind('\n', a_start, a_end) + 1

anon = (
    r'\author{\IEEEauthorblockN{Anonymous Author(s)}' + '\n'
    r'\IEEEauthorblockA{Anonymous Institution\\' + '\n'
    r'Anonymous Country}}' + '\n'
)
text = text[:a_start] + anon + text[a_end:]

# 3. Remove Author contributions paragraph
text = re.sub(
    r'\\textbf\{Author contributions:\}[^\n]*\n(?:[^\n]*\n){0,6}',
    '',
    text
)

# 4. Replace GitHub minipage + QR block with placeholder
text = re.sub(
    r'\\medskip\s*\\noindent\s*\\begin\{minipage\}.*?\\end\{minipage\}',
    r'\\medskip\n\\noindent\\textit{[Code repository URL omitted for blind review.]}',
    text, flags=re.DOTALL
)

# 5. Remove AI disclosure sentence (mentions "authors")
text = re.sub(
    r'\\textit\{AI disclosure:.*?\}\s*\n?',
    '',
    text, flags=re.DOTALL
)

with open(dst, 'w', encoding='utf-8') as f:
    f.write(text)

print(f'Blind version written to {dst}')
print(f'  Anonymous author block : {"Anonymous Author" in text}')
print(f'  GitHub URL removed     : {"Andre031222" not in text}')
print(f'  Author contribs removed: {"Author contributions" not in text}')
