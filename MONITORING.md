# Daily Signal Logging & Live Forward-Test

This project keeps a **permanent, append-only record of every daily signal**, so
that its accuracy can be measured on data it genuinely could not have seen —
because each signal is written *before* the price move it predicts exists.

That record is produced by one small script:

```bash
python -m src.monitor.record_signal
```

Each run fetches the latest fully closed daily candle, computes the same signal
the dashboard shows, and appends one row per model to
`data/signal_log/live_signals.csv`. It is **idempotent**: running it more than
once for the same candle does nothing the second time, so you can never create a
duplicate. It never rewrites or deletes past rows.

> **This is a one-time manual setup — it is NOT automatic out of the box.**
> Nothing schedules this for you. Until you set up one of the options below, the
> live forward-test on the dashboard will stay at "Accumulating — 0/20 days".
> The dashboard does **not** need to be open for logging to happen; that is the
> whole point of running it as a scheduled job.

The dashboard reads this log and, once at least **20 days** have accumulated,
shows a "Live forward-test accuracy" section that is leakage-immune by
construction — separate from, and more trustworthy than, the Phase 4/5 backtest
numbers.

---

## Option A — Windows Task Scheduler (recommended for Windows)

This runs the recorder once a day at a fixed time. BTC daily candles close at
**00:00 UTC**, so any time after that captures the most recent closed candle;
09:00 local is a safe, simple choice.

### Step 1 — create a wrapper batch file

Create `record_signal.bat` in the project root (next to `README.md`), replacing
`C:\path\to\trading` with your actual repo path:

```bat
@echo off
REM --- Daily BTC signal recorder ---
cd /d C:\path\to\trading
call venv\Scripts\activate.bat
python -m src.monitor.record_signal --log-level INFO >> data\signal_log\record_signal.out 2>&1
```

- `cd /d` sets the **working directory** to the repo root (the script writes to
  `data/signal_log/` relative to this).
- `call venv\Scripts\activate.bat` activates the virtual environment. If your
  venv lives elsewhere, point to its `Scripts\activate.bat`.
- The `>>` redirect keeps a rolling log of each run's output for troubleshooting.

Double-click the `.bat` once to confirm it runs and appends a row to
`data\signal_log\live_signals.csv`.

### Step 2 — open Task Scheduler

Press `Win + R`, type `taskschd.msc`, press Enter.

### Step 3 — create the task

1. In the right-hand pane, click **Create Task…** (not "Basic Task" — we want the
   full options).
2. **General** tab:
   - Name: `BTC daily signal recorder`.
   - Select **Run whether user is logged on or not** (so it runs headless).
   - Tick **Run with highest privileges** only if your Python needs it (usually
     not required).
3. **Triggers** tab → **New…**:
   - Begin the task: **On a schedule**.
   - Settings: **Daily**, recur every **1** day.
   - Start time: **09:00:00** (or any time after 00:00 UTC).
   - Click **OK**.
4. **Actions** tab → **New…**:
   - Action: **Start a program**.
   - Program/script: browse to your `record_signal.bat`
     (e.g. `C:\path\to\trading\record_signal.bat`).
   - **Start in (optional)**: set this to the repo root
     `C:\path\to\trading` — this is the working directory and matters.
   - Click **OK**.
5. **Conditions** tab: untick **Start the task only if the computer is on AC
   power** if you want it to run on battery (laptops).
6. **Settings** tab: tick **Run task as soon as possible after a scheduled start
   is missed** so a day the machine was off is caught up on next boot.
7. Click **OK**. Enter your Windows password if prompted (needed for "run whether
   logged on or not").

### Step 4 — test it

In Task Scheduler, right-click the task → **Run**. Then open
`data\signal_log\live_signals.csv` and confirm a new row for today's candle
appears. Run it again — the file must **not** grow (idempotency).

---

## Option B — WSL / Linux cron (alternative)

If you run the project under WSL or Linux, a cron entry is the simplest option.

### Step 1 — edit your crontab

```bash
crontab -e
```

### Step 2 — add one line

Run daily at 09:00 local, activating the venv and using an absolute working
directory (adjust the path to your checkout):

```cron
0 9 * * * cd /home/mutant/full-stack-web3-cu/trading && ./venv/bin/python -m src.monitor.record_signal >> data/signal_log/record_signal.out 2>&1
```

- `cd …` sets the working directory so relative paths resolve.
- `./venv/bin/python` uses the project's virtualenv without needing to `activate`.
- Output (and any errors) append to `data/signal_log/record_signal.out`.

### Step 3 — verify

```bash
# Run it once by hand first:
cd /home/mutant/full-stack-web3-cu/trading && ./venv/bin/python -m src.monitor.record_signal
# Confirm a row was written:
cat data/signal_log/live_signals.csv
# Run again — row count must not change (idempotent):
./venv/bin/python -m src.monitor.record_signal && wc -l data/signal_log/live_signals.csv
```

> **WSL note:** cron only runs while WSL is running. If your machine is often off
> or WSL isn't kept alive, prefer Option A (Windows Task Scheduler), which the
> OS wakes for you and can catch up missed runs.

---

## What gets logged

`data/signal_log/live_signals.csv` (append-only, one row per model per candle):

| Column | Meaning |
|---|---|
| `candle_date` | Date of the closed candle the signal predicts **from** (the dedup key) |
| `model` | `lr` or `lgb` |
| `prob_up` | The model's P(up) |
| `signal` | `BUY` / `SELL` / `SILENT` at the confidence threshold |
| `threshold` | Confidence threshold used |
| `close` | Close price at `candle_date` (known at record time — leaks nothing) |
| `recorded_at_utc` | When the row was written |

The realized outcome is **deliberately not recorded** — it lies in the future at
record time. The dashboard looks it up later from price history, which is exactly
what makes the resulting accuracy honest.

## How the accuracy is scored

For each logged directional signal at date `t`, the outcome is the move to
`t + horizon` days, classified with the **same dead-zone label logic** used at
training time (a move within ±`dead_zone_pct` is "no clear move" and is not
scored). `SILENT` signals make no claim and are never scored. A signal is only
scored once its outcome candle exists — more recent signals are simply "not old
enough yet".

See `DASHBOARD.md` for how this appears on the dashboard, and `PHASES.md` for the
full project methodology.
