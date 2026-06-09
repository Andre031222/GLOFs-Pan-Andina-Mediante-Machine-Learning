import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import re
tex = open('ijacsa_paper/manuscript_ijacsa.tex', encoding='utf-8').read()

print('=== FINAL CONSISTENCY REVIEW ===')

def check(pattern, label, should_exist=True):
    try:
        m = re.findall(pattern, tex)
    except re.error:
        m = [x for x in tex.split() if pattern.replace('\\','') in x]
    count = len(m)
    if should_exist:
        flag = '[OK]' if count else '[MISSING!]'
    else:
        flag = '[OK-absent]' if not count else '[RESIDUAL!]'
    print(f'  {flag} {label}: {count} occurrence(s)')

print('\n--- Numbers that SHOULD be present ---')
check('12,703', 'obs count 12703')
check('17 matched', '17 matched events')
check('0.787', 'BRF AUC 0.787')
check(r'\[0\.683', 'CI lower 0.683')
check('0.876', 'CI upper 0.876')
check('1:746', 'class ratio 1:746')
check('0.00134', 'prevalence 0.00134')
check('71 historical', '71 GLOF events')
check('1932--2023', 'date range 1932-2023')
check(r't=0\.356', 'Youden t=0.356')
check('11.8', 'Lift 11.8x at top-2%')
check('Carabaya', 'Carabaya row')
check('0.739', 'Peru LOCO AUC')
check('0.965', 'Ecuador LOCO AUC')
check('0.790', 'Bolivia LOCO AUC')
check('z.*0.47', 'DeLong z=0.47')
check('29', '29% relative improvement')
check('5,248', '5248 flagged lakes')
check('70.9', 'top 70.9% worst rank')
check('XGBoost', 'XGBoost in table')
check('0.795', 'XGBoost AUC 0.795')
check('8.7', 'BRF Lift 8.7x')
check('eleven study areas', 'eleven study areas')

print('\n--- Numbers that should NOT appear (old values) ---')
check('12,588', 'OLD obs count', False)
check('0.821', 'OLD BRF AUC 0.821', False)
check('0.737', 'OLD CI low 0.737', False)
check('1:787', 'OLD class ratio', False)
check('t=0.304', 'OLD Youden t=0.304', False)
check('52 historical', 'OLD 52 events', False)
check('5,030', 'OLD 5030 flagged', False)
check('46.8', 'OLD top 46.8%', False)
check('z=0.94', 'OLD DeLong z=0.94', False)
check('p=0.350', 'OLD DeLong p', False)
check(r'\textbf\{16\}', 'OLD bold n+=16', False)
check('43%', 'OLD 43% improvement', False)
