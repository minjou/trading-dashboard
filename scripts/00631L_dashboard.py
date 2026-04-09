#!/usr/bin/env python3
"""
00631L v17 — K線 Dashboard 產生器
==================================
基於 00675L dashboard 翻寫，K棒紅漲綠跌（台灣慣例）。

指標面板：成交量、漲跌家數比、乖離率(90d)、CCI
條件判斷：週線熊、月線熊、價<MA30、ADL非多、Mom3M<0、CCI20<0、急跌保護

Usage:
  uv run --with finlab --with python-dotenv python3 WEB/00631L_dashboard.py
"""

import numpy as np
import pandas as pd
import json
from dotenv import load_dotenv

load_dotenv()

from finlab import data

TICKER = '00631L'

# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def calc_cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    ma_tp = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma_tp) / (0.015 * md.replace(0, np.nan))

def calc_sar(th, tl, tc, af_step=0.02, af_max=0.2):
    n = len(tc)
    sar = np.zeros(n); trend_arr = np.ones(n, dtype=int)
    af = af_step; trend = 1; ep = th.iloc[0]; sar[0] = tl.iloc[0]
    for i in range(1, n):
        prev_sar = sar[i-1]
        sar[i] = prev_sar + af * (ep - prev_sar)
        if trend == 1:
            sar[i] = min(sar[i], tl.iloc[i-1])
            if i >= 2: sar[i] = min(sar[i], tl.iloc[i-2])
            if tc.iloc[i] < sar[i]:
                trend = -1; sar[i] = ep; ep = tl.iloc[i]; af = af_step
            else:
                if th.iloc[i] > ep: ep = th.iloc[i]; af = min(af+af_step, af_max)
        else:
            sar[i] = max(sar[i], th.iloc[i-1])
            if i >= 2: sar[i] = max(sar[i], th.iloc[i-2])
            if tc.iloc[i] > sar[i]:
                trend = 1; sar[i] = ep; ep = th.iloc[i]; af = af_step
            else:
                if tl.iloc[i] < ep: ep = tl.iloc[i]; af = min(af+af_step, af_max)
        trend_arr[i] = trend
    return pd.Series(sar, index=tc.index), pd.Series(trend_arr, index=tc.index)

def calc_vhf(tc, period=28):
    num = tc.rolling(period).max() - tc.rolling(period).min()
    den = tc.diff().abs().rolling(period).sum()
    return num / den.replace(0, np.nan)

def wma(series, period):
    weights = np.arange(1, period+1, dtype=float)
    return series.rolling(period).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)

def to_daily(s, idx): return s.reindex(idx, method='ffill')

def confirm_state(condition, cd):
    if cd <= 1: return condition.astype(bool).fillna(False)
    return condition.astype(float).rolling(cd).min().fillna(0).astype(bool)

def filter_confirm(signal_bool, on_days, off_days):
    confirmed_on  = confirm_state(signal_bool, on_days)
    confirmed_off = confirm_state(~signal_bool, off_days)
    state = pd.Series(np.nan, index=signal_bool.index)
    state[confirmed_on] = 1.0; state[confirmed_off] = 0.0
    return state.ffill().fillna(0).astype(bool)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

print("--- Loading data... ---")
close_all = data.get('price:收盤價')

# Use TAIEX as primary index (ETF may have trading halts, e.g. stock splits)
taiex_close = data.get('taiex_total_index:收盤指數')['TAIEX'].dropna()
try:
    taiex_high = data.get('taiex_total_index:最高指數')['TAIEX']
    taiex_low  = data.get('taiex_total_index:最低指數')['TAIEX']
except Exception:
    taiex_high, taiex_low = taiex_close.copy(), taiex_close.copy()

# ETF data: use adjusted prices (handles stock splits), fill halted days with ffill
c_raw = data.get('etl:adj_close')[TICKER].dropna()
etf_start = c_raw.index[0]
daily_idx = taiex_close.loc[etf_start:].index
c = c_raw.reindex(daily_idx).ffill()

try:
    h_etf = data.get('etl:adj_high')[TICKER].reindex(daily_idx).ffill().fillna(c)
    l_etf = data.get('etl:adj_low')[TICKER].reindex(daily_idx).ffill().fillna(c)
    o_etf = data.get('etl:adj_open')[TICKER].reindex(daily_idx).ffill().fillna(c)
except Exception:
    h_etf, l_etf, o_etf = c.copy(), c.copy(), c.copy()

try:
    taiex_open_raw = data.get('taiex_total_index:開盤指數')['TAIEX']
    to_idx = taiex_open_raw.reindex(daily_idx, method='ffill')
except Exception:
    to_idx = taiex_close.shift(1).reindex(daily_idx, method='ffill')

tc = taiex_close.reindex(daily_idx, method='ffill').bfill()
th = taiex_high.reindex(daily_idx, method='ffill').fillna(tc)
tl = taiex_low.reindex(daily_idx, method='ffill').fillna(tc)
to_px = to_idx.fillna(tc)

# Volume (TAIEX 官方成交金額)
try:
    mkt_vol = data.get('market_transaction_info:成交金額')
    vol = mkt_vol['TAIEX'].reindex(daily_idx, method='ffill').fillna(0)
except Exception:
    vol_all = data.get('price:成交金額')
    vol = vol_all.sum(axis=1).reindex(daily_idx).fillna(0)

# ADL
close_diff = close_all.diff()
total_stocks = (~close_all.isna()).sum(axis=1)
rise_stocks = (close_diff > 0).sum(axis=1)
ADLs = rise_stocks / total_stocks - 0.5

# Weekly / Monthly
tc_m = tc.resample('ME').last().dropna()
if len(tc_m) > 0 and tc_m.index[-1] > tc.index[-1]:
    tc_m = tc_m.iloc[:-1]
t_c_w = tc.resample('W-FRI').last().dropna()
if len(t_c_w) > 0 and t_c_w.index[-1] > tc.index[-1]:
    t_c_w = t_c_w.iloc[:-1]

print(f"  {TICKER}: {c.index[0].date()} ~ {c.index[-1].date()} ({len(c)} days)")

# ── Indicators ──
sar_vals, sar_trend = calc_sar(th, tl, tc, 0.015, 0.3)
cci_40 = calc_cci(th, tl, tc, 40)
cci_20 = calc_cci(th, tl, tc, 20)
vhf_35 = calc_vhf(tc, 35)

# Moving Averages for chart
ma20 = tc.rolling(20).mean()
ma60 = tc.rolling(60).mean()
ma240 = tc.rolling(240).mean()

# Deviation (90d)
sma90 = tc.rolling(90).mean()
dev90 = ((tc - sma90) / sma90)

# ADL smoothed
adl_short = ADLs.rolling(20).mean()
adl_long = ADLs.rolling(55).mean()
adl_bull = (adl_short >= adl_long).reindex(daily_idx, method='ffill').fillna(False)

# Mom 3M
mom_3m = tc / tc.shift(63) - 1

# ── Conditions ──
c1 = (cci_40 <= -150) & (tc > tc.shift(2))
c2 = (tc > sar_vals) & (tc.shift(1) <= sar_vals.shift(1))

wf_w = wma(t_c_w, 12); ws_w = wma(t_c_w, 25)
cond_w_bear = to_daily(wf_w < ws_w, daily_idx).fillna(False)

t_ma6_m = tc_m.rolling(6).mean(); t_ma12_m = tc_m.rolling(12).mean()
cond_m_bear = to_daily(t_ma6_m < t_ma12_m, daily_idx).fillna(False)

bear_ma30 = tc.rolling(30).mean()
cond_price_below_ma30 = tc < bear_ma30
cond_adl_not_bull = ~adl_bull
cond_mom3m_neg = mom_3m < 0
cond_cci20_neg = cci_20 < 0

bear_raw = (cond_w_bear | cond_m_bear) & cond_price_below_ma30 & cond_adl_not_bull & cond_mom3m_neg
bear_confirmed = confirm_state(bear_raw, 1)
c4 = bear_confirmed & cond_cci20_neg

ret1d = tc / tc.shift(1) - 1
ret5d_pre = tc.shift(1) / tc.shift(6) - 1
c5_single = ret1d < -0.035
c5_compound = (ret1d < -0.025) & (ret5d_pre < -0.02)
c5 = c5_single | c5_compound

c6_raw = vhf_35 > 0.2
c6_confirm = filter_confirm(c6_raw, 3, 3)

entry = (c1 | c2).reindex(daily_idx).fillna(False)
exit_sig = (c4 | c5).reindex(daily_idx).fillna(False)

# Deviation gate
deeply_oversold = (dev90 < -0.05).fillna(False)
safe_entry = entry & (~c4.reindex(daily_idx).fillna(False) | deeply_oversold)

position = pd.Series(1.0, index=daily_idx)
position[exit_sig] = 0.0
position[safe_entry] = 1.0
position[~c6_confirm] = 0.0

print(f"  持倉比例: {position.mean():.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# BUILD JSON DATA
# ══════════════════════════════════════════════════════════════════════════════

# Display position: shift by 1 day (signal at close T, trade at open T+1)
# position[T]=0 means "exit signal on T", actual sell at open T+1
# So on day T we're still holding, on T+1 we're flat
display_pos = position.shift(1).fillna(1.0)

# Daily data with indicators
ohlc_data = []
for i, dt in enumerate(daily_idx):
    v = float(vol.iloc[i]) if vol.iloc[i] != 0 else 0
    ohlc_data.append({
        'd': dt.strftime('%Y-%m-%d'),
        'o': round(float(to_px.iloc[i]), 1),
        'h': round(float(th.iloc[i]), 1),
        'l': round(float(tl.iloc[i]), 1),
        'c': round(float(tc.iloc[i]), 1),
        'pos': int(display_pos.iloc[i]),
        'vol': round(v / 1e8, 1) if v > 0 else 0,
        'ma20': round(float(ma20.iloc[i]), 1) if not pd.isna(ma20.iloc[i]) else None,
        'ma60': round(float(ma60.iloc[i]), 1) if not pd.isna(ma60.iloc[i]) else None,
        'ma240': round(float(ma240.iloc[i]), 1) if not pd.isna(ma240.iloc[i]) else None,
        'adl': round(float(ADLs.reindex(daily_idx).iloc[i]), 4) if not pd.isna(ADLs.reindex(daily_idx).iloc[i]) else 0,
        'dev': round(float(dev90.iloc[i]) * 100, 2) if not pd.isna(dev90.iloc[i]) else None,
        'cci': round(float(cci_20.iloc[i]), 1) if not pd.isna(cci_20.iloc[i]) else None,
        'vhf': round(float(vhf_35.iloc[i]), 3) if not pd.isna(vhf_35.iloc[i]) else None,
        # 7 conditions
        'k1': bool(cond_w_bear.iloc[i]),
        'k2': bool(cond_m_bear.iloc[i]),
        'k3': bool(cond_price_below_ma30.iloc[i]),
        'k4': bool(cond_adl_not_bull.iloc[i]),
        'k5': bool(cond_mom3m_neg.iloc[i]) if not pd.isna(mom_3m.iloc[i]) else False,
        'k6': bool(cond_cci20_neg.iloc[i]) if not pd.isna(cci_20.iloc[i]) else False,
        'k7': bool(c5.reindex(daily_idx).fillna(False).iloc[i]),
    })

# Events
pos_diff = position.diff().fillna(0)
date_list = list(daily_idx)
date_pos = {dt: i for i, dt in enumerate(date_list)}

def next_trading_day(dt):
    idx = date_pos.get(dt)
    if idx is not None and idx + 1 < len(date_list):
        return date_list[idx + 1]
    return None

events = []
for dt in daily_idx:
    is_entry = bool(pos_diff.loc[dt] > 0.3)
    is_exit  = bool(pos_diff.loc[dt] < -0.3)
    if not is_entry and not is_exit: continue

    exec_dt = next_trading_day(dt)
    exec_price = round(float(to_px.loc[exec_dt]), 1) if exec_dt else None
    etf_exec_price = round(float(o_etf.loc[exec_dt]), 2) if exec_dt else None

    triggers = []
    if is_entry:
        if bool(c1.reindex(daily_idx).fillna(False).loc[dt]):
            triggers.append('C1: CCI(40)超賣 ≤-150 且反彈')
        if bool(c2.reindex(daily_idx).fillna(False).loc[dt]):
            triggers.append('C2: SAR 空轉多')
        if not triggers:
            triggers.append('C6 趨勢環境開啟 → 自動持倉')
    if is_exit:
        if bool(c4.reindex(daily_idx).fillna(False).loc[dt]):
            triggers.append('C4: 熊市結構出場')
        if bool(c5.reindex(daily_idx).fillna(False).loc[dt]):
            parts = []
            r1 = float(ret1d.loc[dt])
            if bool(c5_single.reindex(daily_idx).fillna(False).loc[dt]):
                parts.append(f'單日跌{r1*100:.2f}%<-3.5%')
            if bool(c5_compound.reindex(daily_idx).fillna(False).loc[dt]):
                r5 = float(ret5d_pre.loc[dt])
                parts.append(f'單日跌{r1*100:.2f}%<-2.5% 且前5日{r5*100:.2f}%<-2%')
            triggers.append('C5: 急跌 — ' + '; '.join(parts))
        if not bool(c6_confirm.loc[dt]):
            triggers.append('C6 趨勢環境關閉 → 空手')
        if not triggers:
            triggers.append('出場條件解除 → 等待重新進場')

    events.append({
        'd': dt.strftime('%Y-%m-%d'),
        'type': 'entry' if is_entry else 'exit',
        'tc': round(float(tc.loc[dt]), 1),
        'exec_d': exec_dt.strftime('%Y-%m-%d') if exec_dt else None,
        'exec_p': exec_price,
        'etf_exec_p': etf_exec_price,
        'triggers': triggers,
    })

# Segments
segments = []
entry_events = [e for e in events if e['type'] == 'entry' and e['etf_exec_p']]
exit_events  = [e for e in events if e['type'] == 'exit' and e['etf_exec_p']]

for ent in entry_events:
    matching_exit = None
    for ex in exit_events:
        if ex['exec_d'] and ent['exec_d'] and ex['exec_d'] > ent['exec_d']:
            matching_exit = ex
            break
    if matching_exit:
        buy_p = ent['etf_exec_p']
        sell_p = matching_exit['etf_exec_p']
        pnl_pct = (sell_p / buy_p - 1) * 100 if buy_p > 0 else 0
        segments.append({
            'entry_sig': ent['d'], 'entry_exec': ent['exec_d'],
            'exit_sig': matching_exit['d'], 'exit_exec': matching_exit['exec_d'],
            'buy_p': buy_p, 'sell_p': sell_p, 'pnl': round(pnl_pct, 2),
        })
        exit_events.remove(matching_exit)

if entry_events:
    last_ent = entry_events[-1]
    paired = {s['entry_sig'] for s in segments}
    if last_ent['d'] not in paired and last_ent['etf_exec_p']:
        buy_p = last_ent['etf_exec_p']
        sell_p = round(float(c.iloc[-1]), 2)
        pnl_pct = (sell_p / buy_p - 1) * 100 if buy_p > 0 else 0
        segments.append({
            'entry_sig': last_ent['d'], 'entry_exec': last_ent['exec_d'],
            'exit_sig': None, 'exit_exec': None,
            'buy_p': buy_p, 'sell_p': sell_p, 'pnl': round(pnl_pct, 2),
        })

# Status for header
# display_pos = actual holding today, position = today's signal (executes tomorrow)
actual_pos = int(display_pos.iloc[-1])  # what we actually hold right now
signal_pos = int(position.iloc[-1])     # what the signal says (executes tomorrow)
if signal_pos > actual_pos:
    tomorrow_action = 'buy'
elif signal_pos < actual_pos:
    tomorrow_action = 'sell'
else:
    tomorrow_action = 'hold'

last_date = daily_idx[-1].strftime('%Y-%m-%d')
last_dev = round(float(dev90.iloc[-1]) * 100, 2) if not pd.isna(dev90.iloc[-1]) else None
last_cci = round(float(cci_20.iloc[-1]), 1) if not pd.isna(cci_20.iloc[-1]) else None
last_adl = round(float(ADLs.reindex(daily_idx).iloc[-1]) * 100, 1) if not pd.isna(ADLs.reindex(daily_idx).iloc[-1]) else None
last_vhf = round(float(vhf_35.iloc[-1]), 3) if not pd.isna(vhf_35.iloc[-1]) else None

status_data = {
    'date': last_date,
    'pos': actual_pos,
    'action': tomorrow_action,
    'dev': last_dev,
    'cci': last_cci,
    'adl': last_adl,
    'vhf': last_vhf,
    'taiex': round(float(tc.iloc[-1]), 1),
}

print(f"  事件: {len(events)} 筆, 損益段: {len(segments)} 段")
print(f"  狀態: {'持倉' if actual_pos else '空手'}, 明日: {tomorrow_action}")

# ══════════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

html_template = r"""
<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>00631L v17 — K線 Dashboard</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--bg:#0f1118;--sf:#181c28;--sf2:#1e2333;--bd:#2a3042;--tx:#e2e6ef;--tx2:#8b93a8;
--up:#ef4444;--up2:rgba(239,68,68,.12);--dn:#22c55e;--dn2:rgba(34,197,94,.12);
--am:#f59e0b;--bl:#3b82f6}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Noto Sans TC',sans-serif;background:var(--bg);color:var(--tx);overflow:hidden;height:100vh}
.hd{padding:10px 20px 8px;border-bottom:1px solid var(--bd);display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.hd h1{font-size:17px;font-weight:700} .hd h1 b{color:var(--up)}
.st{display:flex;gap:12px;align-items:center;margin-left:auto;font:12px 'JetBrains Mono',monospace}
.sb2{padding:6px 16px;border-radius:5px;font-weight:700;font-size:15px;letter-spacing:1px}
.sb2.long{background:rgba(239,68,68,.25);color:#ff6b6b;border:2px solid rgba(239,68,68,.5);text-shadow:0 0 8px rgba(239,68,68,.3)}
.sb2.flat{background:rgba(139,147,168,.15);color:#b0b8cc;border:2px solid rgba(139,147,168,.3)}
.sb2.act{padding:6px 16px;border-radius:5px;font-weight:700;font-size:15px;letter-spacing:1px}
.sb2.act.buy{background:rgba(239,68,68,.25);color:#ff6b6b;border:2px solid rgba(239,68,68,.5)}
.sb2.act.sell{background:rgba(34,197,94,.25);color:#4ade80;border:2px solid rgba(34,197,94,.5)}
.sb2.act.hold{background:rgba(100,116,139,.12);color:#94a3b8;border:2px solid rgba(100,116,139,.25)}
.si{color:var(--tx2);font-size:11px}
.si b{color:var(--tx);font-weight:500}
.ctl{padding:6px 20px;display:flex;gap:4px;flex-wrap:wrap;border-bottom:1px solid var(--bd);background:var(--sf)}
.yb{padding:4px 12px;border:1px solid var(--bd);border-radius:4px;background:transparent;color:var(--tx2);font:12px 'JetBrains Mono',monospace;cursor:pointer;transition:all .12s}
.yb:hover{border-color:var(--tx2);color:var(--tx)} .yb.on{background:var(--up);color:#fff;border-color:var(--up);font-weight:500}
.mn{display:flex;height:calc(100vh - 116px);overflow:hidden}
.cc{flex:1;min-width:0;display:flex;flex-direction:column}
.cb{padding:4px 16px;display:flex;justify-content:space-between;align-items:center;font-size:10px;color:var(--tx2)}
.lg{display:flex;gap:10px} .li{display:flex;align-items:center;gap:3px} .ld{width:7px;height:7px;border-radius:2px}
.cw{flex:1;position:relative;min-height:0} .cw canvas{position:absolute;top:0;left:0;width:100%;height:100%;cursor:crosshair}
.tip{position:fixed;background:var(--sf2);border:1px solid var(--bd);border-radius:5px;padding:6px 10px;font:11px 'JetBrains Mono',monospace;color:var(--tx);pointer-events:none;display:none;z-index:100;white-space:nowrap;line-height:1.6}
.rsb{width:360px;min-width:360px;border-left:1px solid var(--bd);background:var(--sf);display:flex;flex-direction:column;overflow:hidden}
.sh{padding:10px 14px 6px;border-bottom:1px solid var(--bd);font-size:12px;font-weight:500;color:var(--tx2)}
.el{flex:1;overflow-y:auto}
.ei{padding:7px 14px;cursor:pointer;transition:background .1s;border-left:3px solid transparent;display:flex;align-items:center;gap:7px}
.ei:hover{background:var(--sf2)} .ei.on{background:var(--sf2);border-left-color:var(--up)} .ei.on.xt{border-left-color:var(--dn)}
.em{width:20px;height:20px;display:flex;align-items:center;justify-content:center;border-radius:3px;font-size:10px;font-weight:700;flex-shrink:0}
.em.en{background:var(--up2);color:var(--up)} .em.ex{background:var(--dn2);color:var(--dn)}
.ed{font:11px 'JetBrains Mono',monospace} .ep{color:var(--tx2);font-size:10px}
.dt{border-top:1px solid var(--bd);padding:12px 14px;display:none;max-height:55vh;overflow-y:auto} .dt.show{display:block}
.dh{font-size:13px;font-weight:700;margin-bottom:8px;display:flex;align-items:center;gap:5px}
.tg{font-size:9px;padding:2px 6px;border-radius:3px;font-weight:500} .tg.e{background:var(--up2);color:var(--up)} .tg.x{background:var(--dn2);color:var(--dn)}
.ds{margin-bottom:10px} .ds h4{font-size:9px;text-transform:uppercase;letter-spacing:1px;color:var(--tx2);margin-bottom:4px;font-weight:500}
.tr{padding:4px 7px;background:var(--sf);border-left:2px solid var(--am);border-radius:0 3px 3px 0;margin-bottom:2px;font-size:11px;line-height:1.4}
.ig{display:grid;grid-template-columns:1fr 1fr;gap:2px}
.ic{padding:4px 7px;background:var(--bg);border-radius:3px;font-size:10px;display:flex;justify-content:space-between;gap:3px}
.il{color:var(--tx2)} .iv{font-family:'JetBrains Mono',monospace;font-weight:500} .iv.on{color:var(--up)} .iv.off{color:var(--dn)}
.cg{display:flex;gap:3px;flex-wrap:wrap}
.cc2{padding:2px 6px;border-radius:3px;font:10px 'JetBrains Mono',monospace;font-weight:500}
.cc2.on{background:rgba(239,68,68,.12);color:var(--up)} .cc2.off{background:var(--bg);color:var(--tx2)}
::-webkit-scrollbar{width:4px} ::-webkit-scrollbar-track{background:transparent} ::-webkit-scrollbar-thumb{background:var(--bd);border-radius:2px}
</style></head><body>
<div class="hd">
<h1><b>00631L</b> v17</h1>
<div class="st">
<div class="sb2 __STATUS_POS_CLASS__">__STATUS_POS_TEXT__</div>
<div class="si">__STATUS_DATE__ TAIEX <b>__STATUS_TAIEX__</b></div>
<div class="si">乖離<b>__STATUS_DEV__%</b> CCI<b>__STATUS_CCI__</b> ADL<b>__STATUS_ADL__%</b> VHF<b>__STATUS_VHF__</b></div>
<div class="sb2 act __STATUS_ACTION_CLASS__">__STATUS_ACTION__</div>
</div>
</div>
<div class="ctl" id="tabs"></div>
<div class="mn">
<div class="cc">
<div class="cb">
<div id="cT" style="font-weight:500"></div>
<div class="lg">
<div class="li"><div class="ld" style="background:rgba(59,130,246,.25)"></div>持倉</div>
<div class="li"><div class="ld" style="background:rgba(239,68,68,.08)"></div>空手</div>
</div>
</div>
<div class="cw" id="cW"><canvas id="cv"></canvas></div>
</div>
<div class="rsb"><div class="sh" id="sT">進出場事件</div><div class="el" id="eL"></div><div class="dt" id="dp"></div></div>
</div>
<div class="tip" id="tip"></div>
<script>
const OD=__OHLC_JSON__,EV=__EVENTS_JSON__,SG=__SEGMENTS_JSON__;
let cY=null,sE=null,yD=[],yE=[],ySG=[];
const cv=document.getElementById('cv'),cx=cv.getContext('2d'),tip=document.getElementById('tip');
const yrs=[...new Set(OD.map(r=>r.d.slice(0,4)))].sort();
const tabs=document.getElementById('tabs');
yrs.forEach(y=>{const b=document.createElement('button');b.className='yb';b.textContent=y;b.onclick=()=>go(y);tabs.appendChild(b)});
const yrEv=[...new Set(EV.map(e=>e.d.slice(0,4)))];

/* ── Constants ── */
const PR=52,PL=8,PT=8,PB=22,PGAP=4;
const KLINE_FRAC=0.78,VOL_FRAC=0.18;
const MIN_BARS=20,ZOOM_STEP=0.15;

let cm={},vStart=0,vEnd=0; /* visible window indices into yD */
let isDragging=false,dragStartX=0,dragStartVS=0;

function go(y){
  cY=y;sE=null;
  document.querySelectorAll('.yb').forEach(b=>b.classList.toggle('on',b.textContent===y));
  document.getElementById('dp').classList.remove('show');
  yD=OD.filter(r=>r.d.startsWith(y));
  yE=EV.filter(e=>e.d.startsWith(y)||((e.exec_d||'').startsWith(y)));
  ySG=SG.filter(s=>((s.entry_exec||'').startsWith(y)||(s.exit_exec||'').startsWith(y)||(s.entry_sig||'').startsWith(y)));
  document.getElementById('cT').textContent=y+' TAIEX';
  document.getElementById('sT').textContent=y+' 進出場 ('+yE.length+')';
  vStart=0;vEnd=yD.length; /* reset zoom */
  draw();renderList();
}

function sz(){const w=document.getElementById('cW');const W=w.clientWidth,H=w.clientHeight;const d=devicePixelRatio||1;cv.width=W*d;cv.height=H*d;cv.style.width=W+'px';cv.style.height=H+'px';cx.setTransform(d,0,0,d,0,0);return{W,H}}

function draw(){
  const{W,H}=sz();if(!yD.length||W<10||H<10)return;
  const usable=H-PT-PB-PGAP;
  const kH=Math.round(usable*KLINE_FRAC),vH=Math.round(usable*VOL_FRAC);
  const kP={y:PT,h:kH},vP={y:PT+kH+PGAP,h:vH};
  const cw=W-PL-PR;
  /* visible slice */
  const vs=Math.max(0,Math.floor(vStart)),ve=Math.min(yD.length,Math.ceil(vEnd));
  const vLen=ve-vs;if(vLen<1)return;
  const vD=yD.slice(vs,ve); /* visible data */
  const d2i={};yD.forEach((r,i)=>d2i[r.d]=i);
  const bw=Math.max(1,Math.min(10,(cw/vLen)*0.6));
  const xS=i=>PL+((i-vs+0.5)/vLen)*cw; /* i is index into full yD */

  cx.clearRect(0,0,W,H);

  /* ── K-Line Panel (large) ── */
  const allH=vD.map(r=>r.h),allL=vD.map(r=>r.l);
  const pMn=Math.min(...allL)*0.995,pMx=Math.max(...allH)*1.005;
  const yS=v=>kP.y+(1-(v-pMn)/(pMx-pMn))*kP.h;
  cm={cw,xS,bw,W,H,vs,ve,vLen,kline:{pMn,pMx,yS,p:kP}};

  // position background (visible range)
  let ss=vs;
  for(let i=vs+1;i<=ve;i++){
    const end=i>=ve;
    if(end||yD[i].pos!==yD[ss].pos){
      cx.fillStyle=yD[ss].pos?'rgba(59,130,246,0.18)':'rgba(239,68,68,0.12)';
      const x0=Math.max(PL,xS(ss)-bw),x1=Math.min(PL+cw,xS(end?i-1:i)+bw);
      cx.fillRect(x0,kP.y,x1-x0,kP.h);
      if(!end)ss=i;
    }
  }

  // grid: auto step based on visible range
  {
    const range=pMx-pMn;
    const gridStep=range<3000?500:range<6000?1000:2000;
    const gridStart=Math.ceil(pMn/gridStep)*gridStep;
    cx.strokeStyle='rgba(80,90,115,0.5)';cx.lineWidth=0.7;
    cx.fillStyle='#8b93a8';cx.font='9px JetBrains Mono';cx.textAlign='left';
    for(let v=gridStart;v<=pMx;v+=gridStep){
      const y=yS(v);
      cx.beginPath();cx.moveTo(PL,y);cx.lineTo(W-PR,y);cx.stroke();
      cx.fillText(v.toLocaleString(),W-PR+3,y+3);
    }
  }

  // candlesticks (visible range only)
  for(let i=vs;i<ve;i++){
    const r=yD[i],x=xS(i),bull=r.c>=r.o;
    const bodyT=yS(Math.max(r.o,r.c)),bodyB=yS(Math.min(r.o,r.c));
    const bodyH=Math.max(1,bodyB-bodyT);
    const color=bull?'#ef4444':'#22c55e';
    cx.strokeStyle=color;cx.lineWidth=0.8;
    cx.beginPath();cx.moveTo(x,yS(r.h));cx.lineTo(x,yS(r.l));cx.stroke();
    cx.fillStyle=color;cx.fillRect(x-bw,bodyT,bw*2,bodyH);
  }

  // MA lines
  const maLabels=[];
  function drawMA(key,label,color,lw){
    cx.strokeStyle=color;cx.lineWidth=lw;cx.beginPath();
    let started=false;
    for(let i=vs;i<ve;i++){
      const v=yD[i][key];if(v===null||v===undefined)continue;
      const x=xS(i),y=yS(v);
      if(!started){cx.moveTo(x,y);started=true}else cx.lineTo(x,y);
    }
    cx.stroke();
    const last=yD[ve-1][key];
    if(last!==null&&last!==undefined) maLabels.push({label,color,val:last,y:yS(last)});
  }
  drawMA('ma240','MA240','rgba(255,165,0,0.5)',1.5);
  drawMA('ma60','MA60','rgba(162,130,246,0.7)',1.2);
  drawMA('ma20','MA20','rgba(0,188,212,0.7)',1);
  // draw MA labels on left side, avoid overlap
  maLabels.sort((a,b)=>a.y-b.y);
  for(let i=1;i<maLabels.length;i++){if(maLabels[i].y-maLabels[i-1].y<12)maLabels[i].y=maLabels[i-1].y+12}
  maLabels.forEach(m=>{
    cx.fillStyle=m.color;cx.font='bold 9px JetBrains Mono';cx.textAlign='left';
    cx.fillText(m.label+' '+Math.round(m.val),PL+2,m.y-3);
  });

  // segment P&L labels (only if visible)
  ySG.forEach(sg=>{
    const ei=d2i[sg.entry_exec],xi=sg.exit_exec?d2i[sg.exit_exec]:null;
    if(ei===undefined)return;
    const endI=xi!==undefined?xi:yD.length-1;
    if(endI<vs||ei>=ve)return; /* skip if entirely outside view */
    const clampS=Math.max(vs,ei),clampE=Math.min(ve-1,endI);
    const midI=Math.round((clampS+clampE)/2);
    let maxH=0;for(let j=clampS;j<=clampE;j++){if(yD[j].h>maxH)maxH=yD[j].h;}
    const y=yS(maxH)-8;
    cx.font='bold 10px JetBrains Mono';cx.textAlign='center';
    cx.fillStyle=sg.pnl>=0?'#ef4444':'#22c55e';
    cx.fillText((sg.pnl>=0?'+':'')+sg.pnl.toFixed(1)+'%',xS(midI),Math.max(kP.y+10,y));
  });

  // signal day markers (dashed vertical line + trigger label on signal date T)
  yE.forEach(ev=>{
    const si=d2i[ev.d];
    if(si===undefined||si<vs||si>=ve)return;
    const sx=xS(si),isE=ev.type==='entry',isSel=sE&&sE.d===ev.d;
    const color='#f59e0b'; /* amber for signal */
    // dashed vertical line
    cx.save();cx.setLineDash([4,3]);cx.strokeStyle=color;cx.lineWidth=isSel?1.5:0.8;cx.globalAlpha=isSel?0.9:0.5;
    cx.beginPath();cx.moveTo(sx,kP.y);cx.lineTo(sx,kP.y+kP.h);cx.stroke();cx.restore();
    // trigger label
    const trig=ev.triggers&&ev.triggers[0]?ev.triggers[0]:'';
    const short=trig.replace(/^C\d+[：:]\s*/,'').slice(0,8);
    const label=(isE?'⚡進場':'⚡出場')+(short?' '+short:'');
    cx.font='bold 9px JetBrains Mono';cx.textAlign='center';
    cx.fillStyle=color;cx.globalAlpha=isSel?1:0.7;
    const ly=isE?kP.y+kP.h-4:kP.y+11;
    cx.fillText(label,sx,ly);cx.globalAlpha=1;
  });

  // execution arrows (solid on execution date T+1)
  yE.forEach(ev=>{
    const xi=ev.exec_d?d2i[ev.exec_d]:null;
    if(xi===undefined||xi===null||xi<vs||xi>=ve)return;
    const isE=ev.type==='entry',isSel=sE&&sE.d===ev.d;
    const color=isE?'#ef4444':'#22c55e';
    const ey=yS(ev.exec_p||ev.tc);
    const ms=isSel?10:7;
    cx.fillStyle=color;
    cx.beginPath();
    if(isE){cx.moveTo(xS(xi),ey-ms);cx.lineTo(xS(xi)-ms*.8,ey+ms*.5);cx.lineTo(xS(xi)+ms*.8,ey+ms*.5)}
    else{cx.moveTo(xS(xi),ey+ms);cx.lineTo(xS(xi)-ms*.8,ey-ms*.5);cx.lineTo(xS(xi)+ms*.8,ey-ms*.5)}
    cx.closePath();cx.fill();
    if(isSel){cx.strokeStyle=color;cx.lineWidth=2;cx.beginPath();cx.arc(xS(xi),ey,ms+4,0,Math.PI*2);cx.stroke()}
  });

  cx.fillStyle='#8b93a8';cx.font='9px JetBrains Mono';cx.textAlign='left';
  cx.fillText('TAIEX',PL+2,kP.y+11);
  // zoom info
  if(vLen<yD.length){
    cx.fillStyle='#f59e0b';cx.font='9px JetBrains Mono';cx.textAlign='right';
    cx.fillText(vD[0].d+' ~ '+vD[vD.length-1].d+' ('+vLen+'d)',W-PR-2,kP.y+11);
  }

  /* ── Volume Panel (visible range) ── */
  {
    const vols=vD.map(r=>r.vol).filter(v=>v>0);
    if(vols.length){
      const vMx=Math.max(...vols)*1.05;

      cx.fillStyle='rgba(24,28,40,0.5)';cx.fillRect(PL,vP.y,cw,vP.h);
      cx.strokeStyle='rgba(42,48,66,0.3)';cx.lineWidth=0.5;cx.strokeRect(PL,vP.y,cw,vP.h);

      for(let i=vs;i<ve;i++){
        const r=yD[i];if(!r.vol)continue;
        const bull=r.c>=r.o;
        cx.fillStyle=bull?'rgba(239,68,68,0.5)':'rgba(34,197,94,0.5)';
        const barH=Math.max(1,(r.vol/vMx)*vP.h);
        cx.fillRect(xS(i)-bw,vP.y+vP.h-barH,bw*2,barH);
      }

      cx.fillStyle='#8b93a8';cx.font='8px JetBrains Mono';cx.textAlign='left';
      cx.fillText('Vol',PL+2,vP.y+9);
      cx.fillText(Math.round(vMx)+'億',W-PR+3,vP.y+9);
    }
  }

  /* ── X-axis date labels ── */
  cx.fillStyle='#8b93a8';cx.font='9px JetBrains Mono';cx.textAlign='center';
  if(vLen<=60){
    /* zoomed in: show day labels */
    const step=vLen<=30?1:2;
    for(let i=vs;i<ve;i+=step){
      const r=yD[i];
      cx.fillText(r.d.slice(5),xS(i),H-PB+14);
    }
  } else {
    let lm='';
    for(let i=vs;i<ve;i++){const m=yD[i].d.slice(5,7);if(m!==lm){lm=m;cx.fillText(m+'月',xS(i),H-PB+14)}}
  }
}

function renderList(){
  const el=document.getElementById('eL');
  if(!yE.length){el.innerHTML='<div style="padding:14px;color:var(--tx2);font-size:11px">此年度無事件</div>';return}
  el.innerHTML=yE.map((ev,i)=>{const isE=ev.type==='entry';return`<div class="ei ${isE?'':'xt'}" data-i="${i}"><div class="em ${isE?'en':'ex'}">${isE?'▲':'▼'}</div><div><div class="ed">${ev.exec_d||ev.d} <span style="color:var(--tx2);font-size:9px">${isE?'買進':'賣出'}</span></div><div class="ep">${ev.triggers[0]||''}</div></div></div>`}).join('');
  el.querySelectorAll('.ei').forEach(it=>it.onclick=()=>pick(yE[+it.dataset.i]))
}

function pick(ev){
  sE=ev;draw();
  document.querySelectorAll('.ei').forEach(el=>el.classList.remove('on'));
  yE.forEach((e,i)=>{if(e.d===ev.d){const items=document.querySelectorAll('.ei');if(items[i])items[i].classList.add('on')}});
  showDt(ev)
}

function showDt(ev){
  const dp=document.getElementById('dp');dp.classList.add('show');
  const isE=ev.type==='entry';
  const seg=isE?SG.find(s=>s.entry_sig===ev.d):SG.find(s=>s.exit_sig===ev.d);
  const row=OD.find(r=>r.d===ev.d);
  let h=`<div class="dh">${ev.d} <span class="tg ${isE?'e':'x'}">${isE?'進場':'出場'}</span></div>`;
  h+=`<div class="ds"><h4>成交</h4><div class="ig"><div class="ic"><span class="il">訊號日</span><span class="iv">${ev.d}</span></div><div class="ic"><span class="il">成交日</span><span class="iv">${ev.exec_d||'N/A'}</span></div><div class="ic"><span class="il">TAIEX</span><span class="iv">${ev.tc}</span></div><div class="ic"><span class="il">ETF開盤價</span><span class="iv">${ev.etf_exec_p||'N/A'}</span></div></div></div>`;
  if(seg){const pc=seg.pnl;h+=`<div class="ds"><h4>損益</h4><div class="ig"><div class="ic"><span class="il">買</span><span class="iv">${seg.buy_p}</span></div><div class="ic"><span class="il">賣</span><span class="iv">${seg.sell_p}</span></div><div class="ic"><span class="il">損益</span><span class="iv ${pc>=0?'on':'off'}">${pc>=0?'+':''}${pc.toFixed(2)}%</span></div><div class="ic"><span class="il">期間</span><span class="iv">${seg.entry_exec}~${seg.exit_exec||'持倉中'}</span></div></div></div>`}
  h+=`<div class="ds"><h4>觸發</h4>${ev.triggers.map(t=>`<div class="tr">${t}</div>`).join('')}</div>`;
  if(row){
    const KL=['k1','k2','k3','k4','k5','k6','k7'];
    const KN=['週線熊','月線熊','價<MA30','ADL非多','Mom3M<0','CCI20<0','急跌保護'];
    h+=`<div class="ds"><h4>條件</h4><div class="cg">${KL.map((k,i)=>`<div class="cc2 ${row[k]?'on':'off'}">${KN[i]}</div>`).join('')}</div></div>`;
    h+=`<div class="ds"><h4>指標</h4><div class="ig"><div class="ic"><span class="il">乖離90d</span><span class="iv${row.dev!==null&&row.dev<-5?' on':row.dev!==null&&row.dev>15?' on':''}">${row.dev!==null?row.dev+'%':'N/A'}</span></div><div class="ic"><span class="il">CCI20</span><span class="iv${row.cci!==null&&(row.cci>200||row.cci<-200)?' on':''}">${row.cci!==null?row.cci:'N/A'}</span></div><div class="ic"><span class="il">ADL</span><span class="iv${Math.abs(row.adl)>0.3?' on':''}">${(row.adl*100).toFixed(1)}%</span></div><div class="ic"><span class="il">VHF(35)</span><span class="iv${row.vhf!==null&&row.vhf>0.2?' on':''}">${row.vhf!==null?row.vhf:'N/A'}</span></div><div class="ic"><span class="il">成交量</span><span class="iv">${row.vol}億</span></div></div></div>`;
  }
  dp.innerHTML=h
}

function showDayInfo(r){
  const dp=document.getElementById('dp');dp.classList.add('show');
  const KL=['k1','k2','k3','k4','k5','k6','k7'];
  const KN=['週線熊','月線熊','價<MA30','ADL非多','Mom3M<0','CCI20<0','急跌保護'];
  const condOn=KL.reduce((n,k)=>n+(r[k]?1:0),0);
  let h=`<div class="dh">${r.d} <span style="font-size:11px;color:${r.pos?'#3b82f6':'#8b93a8'}">${r.pos?'持倉':'空手'}</span></div>`;
  h+=`<div class="ds"><h4>TAIEX</h4><div class="ig"><div class="ic"><span class="il">開</span><span class="iv">${r.o}</span></div><div class="ic"><span class="il">高</span><span class="iv">${r.h}</span></div><div class="ic"><span class="il">低</span><span class="iv">${r.l}</span></div><div class="ic"><span class="il">收</span><span class="iv">${r.c}</span></div></div></div>`;
  h+=`<div class="ds"><h4>指標</h4><div class="ig"><div class="ic"><span class="il">乖離90d</span><span class="iv${r.dev!==null&&(r.dev<-5||r.dev>15)?' on':''}">${r.dev!==null?r.dev+'%':'N/A'}</span></div><div class="ic"><span class="il">CCI(20)</span><span class="iv${r.cci!==null&&(r.cci>200||r.cci<-200)?' on':''}">${r.cci!==null?r.cci:'N/A'}</span></div><div class="ic"><span class="il">ADL</span><span class="iv${Math.abs(r.adl)>0.3?' on':''}">${(r.adl*100).toFixed(1)}%</span></div><div class="ic"><span class="il">VHF(35)</span><span class="iv${r.vhf!==null&&r.vhf>0.2?' on':''}">${r.vhf!==null?r.vhf:'N/A'}</span></div><div class="ic"><span class="il">成交量</span><span class="iv">${r.vol}億</span></div></div></div>`;
  h+=`<div class="ds"><h4>熊市條件 (${condOn}/7)</h4><div class="cg">${KL.map((k,i)=>`<div class="cc2 ${r[k]?'on':'off'}">${KN[i]}</div>`).join('')}</div></div>`;
  dp.innerHTML=h;
  // deselect events
  sE=null;draw();
  document.querySelectorAll('.ei').forEach(el=>el.classList.remove('on'));
}

/* ── Zoom (scroll wheel) & Pan (drag) ── */
cv.addEventListener('wheel',e=>{
  e.preventDefault();
  if(!yD.length)return;
  const rect=cv.getBoundingClientRect(),mx=e.clientX-rect.left;
  const ratio=(mx-PL)/cm.cw; /* 0~1 position of mouse */
  const curLen=vEnd-vStart;
  const zoomIn=e.deltaY<0;
  const factor=zoomIn?(1-ZOOM_STEP):(1+ZOOM_STEP);
  let newLen=Math.round(curLen*factor);
  newLen=Math.max(MIN_BARS,Math.min(yD.length,newLen));
  if(newLen===curLen)return;
  /* keep mouse position anchored */
  const anchor=vStart+ratio*curLen;
  vStart=Math.round(anchor-ratio*newLen);
  vEnd=vStart+newLen;
  if(vStart<0){vStart=0;vEnd=newLen}
  if(vEnd>yD.length){vEnd=yD.length;vStart=vEnd-newLen}
  draw();
},{passive:false});

cv.addEventListener('mousedown',e=>{
  if(vEnd-vStart>=yD.length)return; /* not zoomed, don't pan */
  isDragging=true;dragStartX=e.clientX;dragStartVS=vStart;
  cv.style.cursor='grabbing';
});
window.addEventListener('mousemove',e=>{
  if(!isDragging)return;
  const dx=e.clientX-dragStartX;
  const barsPerPx=(vEnd-vStart)/cm.cw;
  let shift=Math.round(-dx*barsPerPx);
  let ns=dragStartVS+shift;
  const curLen=vEnd-vStart;
  ns=Math.max(0,Math.min(yD.length-curLen,ns));
  vStart=ns;vEnd=ns+curLen;
  draw();
});
window.addEventListener('mouseup',()=>{
  if(isDragging){isDragging=false;cv.style.cursor='crosshair'}
});

/* ── Tooltip ── */
cv.addEventListener('mousemove',e=>{
  if(!yD.length||!cm.xS){tip.style.display='none';return}
  const rect=cv.getBoundingClientRect(),mx=e.clientX-rect.left;
  const ratio=(mx-PL)/cm.cw;const idx=Math.round(cm.vs+ratio*cm.vLen-0.5);
  if(idx<cm.vs||idx>=cm.ve){tip.style.display='none';return}
  const r=yD[idx];
  const KN=['週線熊','月線熊','價<MA30','ADL非多','Mom3M<0','CCI20<0','急跌'];
  const KL=['k1','k2','k3','k4','k5','k6','k7'];
  const condOn=KL.reduce((n,k)=>n+(r[k]?1:0),0);

  let html=`<b>${r.d}</b> ${r.pos?'<span style="color:#3b82f6">持倉</span>':'<span style="color:var(--tx2)">空手</span>'}`;
  html+=`<br>開${r.o} 高${r.h} 低${r.l} 收${r.c}`;
  const mas=[];
  if(r.ma20)mas.push(`<span style="color:#00bcd4">M20:${r.ma20}</span>`);
  if(r.ma60)mas.push(`<span style="color:#a282f6">M60:${r.ma60}</span>`);
  if(r.ma240)mas.push(`<span style="color:#ffa500">M240:${r.ma240}</span>`);
  if(mas.length)html+=`<br>${mas.join(' ')}`;

  // Indicators with extreme alerts
  const devC=r.dev!==null?(r.dev<-5?'color:#ef4444':r.dev>15?'color:#ef4444':'color:var(--tx)'):'';
  const cciC=r.cci!==null?(r.cci<-200?'color:#ef4444':r.cci>200?'color:#ef4444':'color:var(--tx)'):'';
  const adlC=Math.abs(r.adl)>0.3?'color:#ef4444':'color:var(--tx)';
  const vhfC=r.vhf!==null?(r.vhf>0.2?'color:#3b82f6':'color:var(--tx)'):'';
  html+=`<br><span style="${devC}">乖離:${r.dev!==null?r.dev+'%':'N/A'}</span> <span style="${cciC}">CCI:${r.cci!==null?r.cci:'N/A'}</span> <span style="${adlC}">ADL:${(r.adl*100).toFixed(1)}%</span> <span style="${vhfC}">VHF:${r.vhf!==null?r.vhf:'N/A'}</span>`;
  if(condOn>0)html+=`<br>熊市條件: <span style="color:#ef4444">${condOn}/7</span>`;

  tip.style.display='block';
  let tx=e.clientX+14,ty=e.clientY-60;
  if(tx+280>window.innerWidth)tx=e.clientX-280;
  if(ty<0)ty=e.clientY+14;
  tip.style.left=tx+'px';tip.style.top=ty+'px';
  tip.innerHTML=html;
});
cv.addEventListener('mouseleave',()=>tip.style.display='none');

cv.addEventListener('click',e=>{
  if(!yD.length||!cm.xS||isDragging)return;
  const rect=cv.getBoundingClientRect(),mx=e.clientX-rect.left;
  const ratio=(mx-PL)/cm.cw;const idx=Math.round(cm.vs+ratio*cm.vLen-0.5);
  if(idx<cm.vs||idx>=cm.ve)return;
  const r=yD[idx];
  // Check if clicked near an event marker first
  const d2i={};yD.forEach((ri,i)=>d2i[ri.d]=i);
  const my=e.clientY-rect.top;
  let evMatch=null,minD=25;
  yE.forEach(ev=>{
    const xi=ev.exec_d?d2i[ev.exec_d]:d2i[ev.d];
    if(xi===undefined)return;
    const ex=cm.xS(xi),ey=cm.kline.yS(ev.exec_p||ev.tc);
    const dist=Math.sqrt((mx-ex)**2+(my-ey)**2);
    if(dist<minD){minD=dist;evMatch=ev}
  });
  if(evMatch){pick(evMatch)} else {showDayInfo(r)}
});

window.addEventListener('resize',()=>draw());
go(yrEv.length?yrEv[yrEv.length-1]:yrs[yrs.length-1]);
</script></body></html>
"""

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

html_out = html_template
html_out = html_out.replace('__OHLC_JSON__', json.dumps(ohlc_data, ensure_ascii=False))
html_out = html_out.replace('__EVENTS_JSON__', json.dumps(events, ensure_ascii=False))
html_out = html_out.replace('__SEGMENTS_JSON__', json.dumps(segments, ensure_ascii=False))

# Status bar replacements
html_out = html_out.replace('__STATUS_POS_CLASS__', 'long' if status_data['pos'] else 'flat')
html_out = html_out.replace('__STATUS_POS_TEXT__', '持倉中' if status_data['pos'] else '空手')
html_out = html_out.replace('__STATUS_DATE__', status_data['date'])
html_out = html_out.replace('__STATUS_TAIEX__', str(status_data['taiex']))
html_out = html_out.replace('__STATUS_DEV__', str(status_data['dev'] if status_data['dev'] is not None else 'N/A'))
html_out = html_out.replace('__STATUS_CCI__', str(status_data['cci'] if status_data['cci'] is not None else 'N/A'))
html_out = html_out.replace('__STATUS_ADL__', str(status_data['adl'] if status_data['adl'] is not None else 'N/A'))
html_out = html_out.replace('__STATUS_VHF__', str(status_data['vhf'] if status_data['vhf'] is not None else 'N/A'))

action_text = {'buy': '明日開盤買進', 'sell': '明日開盤賣出', 'hold': '明日不交易'}[status_data['action']]
html_out = html_out.replace('__STATUS_ACTION__', action_text)
html_out = html_out.replace('__STATUS_ACTION_CLASS__', status_data['action'])

import os
output_path = os.path.join(os.path.dirname(__file__), '00631L_v17_dashboard.html')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_out)

print(f'\n已產生: {output_path}')
