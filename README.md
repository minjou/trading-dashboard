# Trading Dashboards

Taiwan ETF trading dashboards — hosted on GitHub Pages.

## Usage

```bash
python publish.py                  # Run all enabled strategies + push
python publish.py --only 00631L    # Run only one strategy
python publish.py --dry-run        # Generate files locally, no push
```

## Adding a new strategy

1. Place the script in `scripts/`
2. Add an entry to `strategies.json` with `"enabled": true`
3. Run `python publish.py`

## Structure

```
docs/          ← GitHub Pages root (auto-generated, do not edit manually)
scripts/       ← Dashboard Python scripts
publish.py     ← Main orchestration script
strategies.json ← Strategy configuration
```
