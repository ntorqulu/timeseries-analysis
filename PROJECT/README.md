# Rodalies delay prediction

## Setup

```bash
conda env create -f environment.yml
conda activate jupyter_env
pip install pyarrow
```

Copy `credentials.json` into the project root. Then authenticate:

```bash
python src/download.py
```

The first run opens a browser OAuth flow and catches the token in `token.json`. Subsequent runs refresh authomatically.

## Download

**Everything available on Drive:**

```bash
python src/download.py
```

**Specific tables:**

```bash
python src/download.py --tables trains timetables
```

**Date range:**

```bash
python src/download.py --start 2026_03_15 --end 2026_03_21
```

**Re-download existing files:**

```bash
python src/download.py --force
```

Downloaded files are saved to `data/` (gitignored).

## EDA R1