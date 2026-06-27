# The Spending Paradox — Health Analytics Dashboard

Single-screen Streamlit dashboard for MSBA382. The health problem: **life
expectancy plateaus even as spending rises.** Chart 1 shows the diminishing
returns; three pillars show what actually moves life expectancy afterward.

## The story (one screen)

- **The plateau** — life expectancy vs spending. Steep, then flat. The US sits
  off the curve: highest spend, lower life.
- **Pillar 1 · Reach** — health-service access (UHC) tracks life expectancy
  tightly. Getting people covered is what works (global r ≈ 0.84).
- **Pillar 2 · Prevention** — among rich countries, obesity vs value for money.
  Heavier populations get less life per dollar (r ≈ −0.47).
- **Pillar 3 · Effectiveness** — years gained/lost vs what spending predicts.
  The US is last despite spending the most (−6.2 yrs).
- **Map** — life expectancy by country (geographic view).

Reach gets people in, prevention keeps them out of the system, effectiveness
makes the spending count.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Reads the four CSVs in `data/`. No internet needed.

## Password

Create `.streamlit/secrets.toml` with `password = "your-password"` (falls back
to `health2026`). On Streamlit Cloud set it under App settings → Secrets.

## Publish (Streamlit Community Cloud)

Push this folder **including `data/`** to a public GitHub repo, then
share.streamlit.io → New app → pick the repo and `app.py` → add the password
secret → Deploy.

## Submission data file

Sidebar → Download merged data (CSV).

## Files
`app.py`, `data.py`, `smoke_test.py`, `requirements.txt`, `data/` (4 CSVs).

Relationships are correlational; national outcomes also reflect inequality,
diet, and demographics.
