# Health spending vs life expectancy

A Streamlit consultant dashboard exploring whether higher health spending actually
buys longer lives, and what does, across ~190 countries.

**The argument, in two beats**

1. **Diminishing returns.** Life expectancy rises with health spending but flattens
   fast. The biggest gains per dollar happen at low spending levels. The United States
   is the world's largest *overspending underperformer*: it spends the most per person
   yet lives about 5.4 years less than the global curve predicts for that spending.
2. **What still moves the needle.** Once spending plateaus, three pillars explain who
   beats their spending: prevention and primary care (child and maternal mortality),
   public health and environment (safe water), and behavioral risk (obesity, smoking),
   the last of which shows up clearly among high-income countries.

A standardized regression on the same data ranks the levers: child mortality first,
then spending, clean water, obesity, maternal mortality, smoking (R² ≈ 0.87, mean
error ≈ 2 years).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Default password is `msba382` (see `.streamlit/secrets.toml`).

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. On https://share.streamlit.io, create an app pointing at `app.py`.
3. Under **Settings → Secrets**, add:
   ```toml
   password = "your-password"
   ```
   Do **not** commit `.streamlit/secrets.toml` (it is in `.gitignore`).

## Files

```
app.py             Streamlit UI: password gate, Dashboard tab, Model tab
data_prep.py       Data layer: cross-section, residuals, pillars, regression
requirements.txt   Dependencies
data/              The 10 source CSVs (Our World in Data exports)
.streamlit/        Theme config and secrets template
```

## Data sources

All files are Our World in Data exports, drawing on the underlying providers:

| File | Indicator | Original source |
|------|-----------|-----------------|
| life-expectancy-vs-health-expenditure.csv | Life expectancy, world region | UN WPP, World Bank |
| annual-healthcare-expenditure-per-capita.csv | Health spend per capita (PPP int$) | WHO GHED via World Bank |
| child-mortality-igme.csv | Under-5 mortality | UN IGME |
| maternal-mortality-ratio-who-gho.csv | Maternal mortality ratio | WHO GHO |
| share-of-children-immunized-dtp3.csv | DTP3 immunization | WHO / UNICEF |
| proportion-using-safely-managed-drinking-water.csv | Safe drinking water | WHO / UNICEF JMP |
| share-of-adults-who-smoke.csv | Adult smoking | WHO GHO |
| share-of-adults-defined-as-obese.csv | Adult obesity | WHO GHO |
| gdp-per-capita-worldbank.csv | GDP per capita | World Bank |
| female-and-male-life-expectancy-at-birth-in-years.csv | Life expectancy by sex | UN WPP |

## Notes and honest limits

- Analysis is **cross-sectional**, so relationships are associations, not proof of cause.
- The efficiency residual flags *overperformers* only above a spending floor, so the
  list reflects genuinely efficient systems (Japan, South Korea, Costa Rica) rather than
  very low-spend countries that look efficient only because the curve is steep there.
- DTP3 immunization is excluded from the regression because it is highly collinear with
  child mortality; child mortality carries the prevention signal.
- The behavioral pillar is shown on high-income countries because, pooled globally,
  obesity correlates positively with life expectancy through income confounding.
