"""
scripts/check_ieee_format.py
============================
Automated format, table width, label, citation, and figure validator for IEEEtran paper.
"""

import os
import re
import sys

def audit_paper(tex_path, bib_path):
    print("=" * 70)
    print(f"IEEE FORMAT & LATEX LAYOUT AUDIT: {tex_path}")
    print("=" * 70)
    
    if not os.path.exists(tex_path):
        print(f"[ERROR] File not found: {tex_path}")
        return False
        
    with open(tex_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    errors = 0
    warnings = 0
    
    # 1. Document class check
    if r"\documentclass[conference]{IEEEtran}" not in content:
        print("[FAIL] Missing standard IEEEtran conference documentclass.")
        errors += 1
    else:
        print("[PASS] Documentclass is standard IEEEtran conference.")
        
    # 2. Check for manual margin manipulation / geometry package
    bad_packages = ['geometry', 'fullpage', 'a4wide']
    for pkg in bad_packages:
        if re.search(rf'\\usepackage(?:\[[^\]]*\])?\{{{pkg}\}}', content):
            print(f"[FAIL] Forbidden margin package detected: {pkg}")
            errors += 1
            
    margin_commands = [r'\textwidth', r'\oddsidemargin', r'\evensidemargin', r'\columnsep', r'\columnwidth']
    for cmd in margin_commands:
        if re.search(rf'\\setlength\{{{re.escape(cmd)}\}}', content) or re.search(rf'\\addtolength\{{{re.escape(cmd)}\}}', content):
            print(f"[FAIL] Manual dimension modification found for: {cmd}")
            errors += 1
            
    # 3. Check for markdown formatting remnants
    md_patterns = [
        (r'\*\*[^*]+\*\*', 'Markdown bold (**text**)'),
        (r'\\textbf\{\*\*[^*]+\*\*\}', 'Nested markdown in textbf'),
        (r'\\textit\{\*[^*]+\*\}', 'Nested markdown in textit'),
        (r'\\label\{[^}]*\\:[^}]*\}', 'Colon-escaped label (\\:)')
    ]
    for pattern, desc in md_patterns:
        matches = re.findall(pattern, content)
        if matches:
            print(f"[FAIL] Found {len(matches)} occurrences of {desc}: {matches[:3]}")
            errors += 1
            
    # 4. Check tables for proper tabularx / width controls
    table_stars = re.findall(r'\\begin\{table\*\}[\s\S]*?\\end\{table\*\}', content)
    table_singles = re.findall(r'\\begin\{table\}[\s\S]*?\\end\{table\}', content)
    tables = [('table*', t) for t in table_stars] + [('table', t) for t in table_singles]
    print(f"\n[INFO] Found {len(tables)} table environments:")
    for idx, (env_type, table_body) in enumerate(tables, start=1):
        cap_match = re.search(r'\\caption\{([^}]+)\}', table_body)
        caption = cap_match.group(1) if cap_match else f"Table {idx}"
        
        # Check if single column table uses tabularx with \columnwidth or proper columns
        is_full_width = (env_type == 'table*')
        uses_tabularx = r'\begin{tabularx}' in table_body
        uses_resizebox = r'\resizebox' in table_body
        
        print(f"  Table {idx} ({env_type}): \"{caption[:45]}...\"")
        if not is_full_width:
            if not uses_tabularx and not uses_resizebox and 'tabular' in table_body:
                # check column specs
                col_spec = re.search(r'\\begin\{tabular\}\{([^}]+)\}', table_body)
                if col_spec:
                    spec = col_spec.group(1).replace(' ', '')
                    if len(spec) >= 5 and 'p' not in spec and 'X' not in spec:
                        print(f"    [WARN] Single-column table with {len(spec)} unconstrained columns '{spec}' may overflow \\columnwidth.")
                        warnings += 1
                    else:
                        print(f"    [PASS] Standard column specification: '{spec}'")
            elif uses_tabularx:
                width_spec = re.search(r'\\begin\{tabularx\}\{([^}]+)\}', table_body)
                if width_spec:
                    print(f"    [PASS] Tabularx constrained to: {width_spec.group(1)}")
        else:
            if uses_tabularx:
                width_spec = re.search(r'\\begin\{tabularx\}\{([^}]+)\}', table_body)
                if width_spec:
                    print(f"    [PASS] Full-width tabularx constrained to: {width_spec.group(1)}")
                    
    # 5. Check figures
    figures = re.findall(r'\\begin\{(figure|figure\*)\}(.*?)\\end\{\1\}', content, re.DOTALL)
    print(f"\n[INFO] Found {len(figures)} figure environments:")
    for idx, (env_type, fig_body) in enumerate(figures, start=1):
        img_match = re.search(r'\\includegraphics(?:\[([^\]]*)\])?\{([^}]+)\}', fig_body)
        cap_match = re.search(r'\\caption\{([^}]+)\}', fig_body)
        caption = cap_match.group(1) if cap_match else f"Figure {idx}"
        
        if img_match:
            opts = img_match.group(1) or ""
            path = img_match.group(2)
            # check file exists
            exists = os.path.exists(path) or os.path.exists(os.path.join('paper', path))
            status = "FOUND" if exists else "MISSING"
            print(f"  Fig {idx} ({env_type}): {path} [{status}] opts: [{opts}]")
            if not exists:
                print(f"    [FAIL] Image file not found: {path}")
                errors += 1
            if env_type == 'figure' and 'textwidth' in opts:
                print(f"    [FAIL] Single-column figure uses \\textwidth instead of \\columnwidth!")
                errors += 1
        else:
            print(f"  [WARN] Fig {idx} has no \\includegraphics")
            warnings += 1
            
    # 6. Check citations and references
    labels = set(re.findall(r'\\label\{([^}]+)\}', content))
    refs = set(re.findall(r'\\ref\{([^}]+)\}', content))
    eqrefs = set(re.findall(r'\\eqref\{([^}]+)\}', content))
    missing_refs = (refs | eqrefs) - labels
    if missing_refs:
        print(f"\n[FAIL] Missing labels for references: {missing_refs}")
        errors += len(missing_refs)
    else:
        print(f"\n[PASS] All {len(refs)} references and {len(eqrefs)} eqrefs resolve to {len(labels)} declared labels.")
        
    if os.path.exists(bib_path):
        with open(bib_path, 'r', encoding='utf-8') as f:
            bib_content = f.read()
        bib_keys = set(re.findall(r'@\w+\{([^,]+),', bib_content))
        cites_raw = re.findall(r'\\cite\{([^}]+)\}', content)
        cites = set(c.strip() for raw in cites_raw for c in raw.split(','))
        missing_cites = cites - bib_keys
        if missing_cites:
            print(f"[FAIL] Missing BibTeX keys for citations: {missing_cites}")
            errors += len(missing_cites)
        else:
            print(f"[PASS] All {len(cites)} citations resolve to {len(bib_keys)} BibTeX entries.")
            
    print("=" * 70)
    print(f"AUDIT SUMMARY: {errors} Errors, {warnings} Warnings")
    print("=" * 70)
    return errors == 0

if __name__ == '__main__':
    tex_file = os.path.join('paper', 'paper.tex')
    bib_file = os.path.join('paper', 'references.bib')
    success = audit_paper(tex_file, bib_file)
    sys.exit(0 if success else 1)
