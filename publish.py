#!/usr/bin/env python3
"""publish.py — 一鍵更新 trading-dashboard GitHub Pages 網站

Usage:
  python publish.py                  # 跑所有啟用策略，然後 push
  python publish.py --only 00631L    # 只跑指定策略
  python publish.py --dry-run        # 產生檔案但不 push
"""

import argparse, json, os, shutil, subprocess, sys
from datetime import datetime, timezone, timedelta

REPO_ROOT   = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(REPO_ROOT, 'scripts')
DOCS_DIR    = os.path.join(REPO_ROOT, 'docs')
CONFIG_FILE = os.path.join(REPO_ROOT, 'strategies.json')
PYTHON_EXE  = r'C:\Users\Min\Miniconda3\envs\PW\python.exe'
TZ_TST      = timezone(timedelta(hours=8))


def generate_index(strategies, timestamps, docs_dir):
    enabled = [s for s in strategies if s['enabled']]
    cards_html = ''
    for s in enabled:
        ts  = timestamps.get(s['id'], '—')
        url = s['output_html']
        cards_html += f'''
    <a class="card" href="{url}">
      <div class="card-header">
        <span class="ticker">{s["id"]}</span>
        <span class="badge">Dashboard</span>
      </div>
      <div class="card-desc">{s["description"]}</div>
      <div class="card-footer">
        <span class="updated">更新：{ts}</span>
        <span class="arrow">→</span>
      </div>
    </a>'''
    now_str = datetime.now(TZ_TST).strftime('%Y-%m-%d %H:%M TST')
    html = f'''<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trading Dashboards</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {{--bg:#0f1118;--sf:#181c28;--sf2:#1e2333;--bd:#2a3042;--tx:#e2e6ef;--tx2:#8b93a8;--up:#ef4444;--bl:#3b82f6;}}
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:'Noto Sans TC',sans-serif;background:var(--bg);color:var(--tx);min-height:100vh;padding:40px 20px;}}
.site-header{{max-width:720px;margin:0 auto 36px;display:flex;align-items:baseline;gap:16px;border-bottom:1px solid var(--bd);padding-bottom:16px;}}
.site-header h1{{font-size:22px;font-weight:700;}} .site-header h1 span{{color:var(--up);}}
.gen-time{{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--tx2);margin-left:auto;}}
.grid{{max-width:720px;margin:0 auto;display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px;}}
.card{{display:block;text-decoration:none;background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:20px 22px;transition:border-color .15s,background .15s;}}
.card:hover{{border-color:var(--bl);background:var(--sf2);}}
.card-header{{display:flex;align-items:center;gap:10px;margin-bottom:8px;}}
.ticker{{font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;}}
.badge{{font-size:10px;padding:2px 8px;border-radius:3px;background:rgba(59,130,246,.15);color:var(--bl);border:1px solid rgba(59,130,246,.3);text-transform:uppercase;letter-spacing:.5px;}}
.card-desc{{font-size:13px;color:var(--tx2);margin-bottom:16px;line-height:1.5;}}
.card-footer{{display:flex;align-items:center;justify-content:space-between;border-top:1px solid var(--bd);padding-top:12px;}}
.updated{{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--tx2);}}
.arrow{{font-size:16px;color:var(--bl);transition:transform .15s;}} .card:hover .arrow{{transform:translateX(4px);}}
</style>
</head>
<body>
<header class="site-header">
  <h1>Trading <span>Dashboards</span></h1>
  <span class="gen-time">Generated: {now_str}</span>
</header>
<main class="grid">{cards_html}</main>
</body>
</html>'''
    with open(os.path.join(docs_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Generated: docs/index.html ({len(enabled)} card(s))")


parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true')
parser.add_argument('--only', metavar='ID')
args = parser.parse_args()

with open(CONFIG_FILE, encoding='utf-8') as f:
    config = json.load(f)

to_run = [s for s in config['strategies'] if s['enabled']]
if args.only:
    to_run = [s for s in to_run if s['id'] == args.only]
    if not to_run:
        print(f"ERROR: No enabled strategy '{args.only}'"); sys.exit(1)

os.makedirs(DOCS_DIR, exist_ok=True)
timestamps = {}

for s in to_run:
    print(f"\n{'='*60}\nRunning: {s['id']}\n{'='*60}")
    result = subprocess.run([PYTHON_EXE, os.path.join(SCRIPTS_DIR, s['script'])],
                            cwd=SCRIPTS_DIR, check=False)
    if result.returncode != 0:
        print(f"ERROR: {s['id']} failed (exit {result.returncode}). Aborting.")
        sys.exit(result.returncode)

    src = os.path.join(SCRIPTS_DIR, s['source_html'])
    dst = os.path.join(DOCS_DIR, s['output_html'])
    if not os.path.exists(src):
        print(f"ERROR: Expected output not found: {src}"); sys.exit(1)
    shutil.copy2(src, dst)
    print(f"Copied -> docs/{s['output_html']}")
    timestamps[s['id']] = datetime.now(TZ_TST).strftime('%Y-%m-%d %H:%M TST')

# Merge with cached timestamps so partial runs don't lose other entries
TIMESTAMPS_FILE = os.path.join(DOCS_DIR, 'timestamps.json')
all_ts = {}
if os.path.exists(TIMESTAMPS_FILE):
    with open(TIMESTAMPS_FILE, encoding='utf-8') as f:
        all_ts = json.load(f)
all_ts.update(timestamps)
with open(TIMESTAMPS_FILE, 'w', encoding='utf-8') as f:
    json.dump(all_ts, f, ensure_ascii=False, indent=2)

generate_index(config['strategies'], all_ts, DOCS_DIR)

today = datetime.now(TZ_TST).strftime('%Y-%m-%d')
if args.dry_run:
    print("\n[dry-run] Skipping git push.")
else:
    for cmd in [['git','add','docs/'], ['git','commit','-m',f'Update {today}'], ['git','push']]:
        print(f"$ {' '.join(cmd)}")
        r = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if r.returncode != 0 and cmd[1] != 'commit':
            print(f"ERROR: {' '.join(cmd)} failed"); sys.exit(r.returncode)
    print(f"\nDone! Site updates within ~60s.")
