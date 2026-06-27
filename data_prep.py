"""
Data layer for the health-spending vs life-expectancy dashboard.

Pure pandas / numpy / scikit-learn. No Streamlit here, so it can be tested on
its own. app.py imports these functions and wraps the slow ones in
st.cache_data.

Sources (all Our World in Data, underlying World Bank / WHO / UN / UN IGME):
  hero      life-expectancy-vs-health-expenditure.csv       life expectancy + region
  spend     annual-healthcare-expenditure-per-capita.csv    health spend / capita (PPP int$)
  child     child-mortality-igme.csv                        under-5 mortality
  maternal  maternal-mortality-ratio-who-gho.csv            maternal mortality ratio
  dtp3      share-of-children-immunized-dtp3.csv             DTP3 immunization
  water     proportion-using-safely-managed-drinking-water.csv
  smoking   share-of-adults-who-smoke.csv                   adult smoking (age-std)
  obesity   share-of-adults-defined-as-obese.csv            adult obesity
  gdp       gdp-per-capita-worldbank.csv                    GDP / capita
  sex       female-and-male-life-expectancy-at-birth-in-years.csv  LE by sex
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# logical name -> (filename, source column, clean name)
INDICATORS = {
    "child_mort": ("child-mortality-igme.csv", "Child mortality rate", "child_mort"),
    "maternal": ("maternal-mortality-ratio-who-gho.csv",
                 "Maternal mortality ratio (per 100 000 live births)", "maternal"),
    "dtp3": ("share-of-children-immunized-dtp3.csv",
             "Diphtheria/tetanus/pertussis (DTP3)", "dtp3"),
    "water": ("proportion-using-safely-managed-drinking-water.csv",
              "Share of the population using safely managed drinking water", "water"),
    "smoking": ("share-of-adults-who-smoke.csv",
                "Share of adults who smoke or use tobacco (age-standardized)", "smoking"),
    "gdp": ("gdp-per-capita-worldbank.csv", "GDP per capita", "gdp"),
}

# Friendly labels for the UI
LABELS = {
    "child_mort": "Child mortality (under-5, per 1,000)",
    "maternal": "Maternal mortality (per 100k births)",
    "dtp3": "DTP3 immunization (%)",
    "water": "Safely managed drinking water (%)",
    "smoking": "Adult smoking (%)",
    "obesity": "Adult obesity (%)",
    "gdp": "GDP per capita (int$)",
    "log_spend": "Health spend per capita (log)",
}

PILLARS = ["child_mort", "maternal", "dtp3", "water", "smoking", "obesity"]
MODEL_FEATURES = ["log_spend", "child_mort", "maternal", "water", "smoking", "obesity"]


def _iso(df):
    """Keep real countries only: ISO-3 codes, drop OWID aggregates / regions."""
    code = df["Code"].astype(str)
    return df[code.str.match(r"^[A-Z]{3}$") & ~code.str.startswith("OWID")].copy()


def _read(name):
    return pd.read_csv(os.path.join(DATA_DIR, name))


def _load_indicator(filename, col, new):
    d = _iso(_read(filename)).rename(columns={col: new})
    return d[["Code", "Year", new]].dropna(subset=[new])


def _obesity_col(df):
    return [c for c in df.columns if "Obesity" in c][0]


def available_years():
    """Years where life expectancy and broad spend overlap for >=120 countries."""
    hero = _iso(_read("life-expectancy-vs-health-expenditure.csv"))
    spend = _load_indicator(
        "annual-healthcare-expenditure-per-capita.csv",
        "Current health expenditure per capita, PPP (current international $)", "spend")
    le = hero.dropna(subset=["Life expectancy"])[["Code", "Year", "Life expectancy"]]
    m = le.merge(spend, on=["Code", "Year"])
    cov = m.groupby("Year")["Code"].nunique()
    years = sorted(int(y) for y in cov[cov >= 120].index)
    return years


def build_cross_section(year):
    """
    Return one row per country for the given year:
    LE, spend, region, log_spend, predicted LE, residual, the pillar indicators,
    gdp, life expectancy by sex, and a spend band.
    """
    hero = _iso(_read("life-expectancy-vs-health-expenditure.csv"))
    le = hero.dropna(subset=["Life expectancy"]).rename(
        columns={"Life expectancy": "LE",
                 "World region according to OWID": "region"})
    le = le[["Code", "Entity", "Year", "LE", "region"]]

    spend = _load_indicator(
        "annual-healthcare-expenditure-per-capita.csv",
        "Current health expenditure per capita, PPP (current international $)", "spend")

    base = (le[le["Year"] == year]
            .merge(spend[spend["Year"] == year][["Code", "spend"]], on="Code")
            .dropna(subset=["LE", "spend"]))
    base = base[base["spend"] > 0].copy()

    # diminishing-returns fit: LE ~ ln(spend)
    base["log_spend"] = np.log(base["spend"])
    b1, b0 = np.polyfit(base["log_spend"], base["LE"], 1)
    base["pred_LE"] = b0 + b1 * base["log_spend"]
    base["residual"] = base["LE"] - base["pred_LE"]

    base = base.set_index("Code")

    # pillars: most recent value per country at or before `year` (within 12y)
    def latest(filename, col, new):
        d = _load_indicator(filename, col, new)
        d = d[(d["Year"] <= year) & (d["Year"] >= year - 12)]
        return d.sort_values("Year").groupby("Code").tail(1).set_index("Code")[new]

    for key, (fn, col, new) in INDICATORS.items():
        base[key] = latest(fn, col, new)

    ob = _iso(_read("share-of-adults-defined-as-obese.csv"))
    ocol = _obesity_col(ob)
    base["obesity"] = latest("share-of-adults-defined-as-obese.csv", ocol, "obesity")

    sx = _iso(_read("female-and-male-life-expectancy-at-birth-in-years.csv"))
    sx = sx[sx["Year"] == year].set_index("Code")
    base["LE_men"] = sx["Men"]
    base["LE_women"] = sx["Women"]

    # spend band (drives the high-income behavioral slice + the filter)
    def band(s):
        if s < 1000:
            return "Low (<$1k)"
        if s < 3000:
            return "Middle ($1k-3k)"
        return "High (>$3k)"
    base["band"] = base["spend"].apply(band)

    return base.reset_index()


def fit_info(base):
    """Refit the log curve on the (possibly filtered) frame for plotting."""
    b1, b0 = np.polyfit(base["log_spend"], base["LE"], 1)
    xs = np.linspace(base["spend"].min(), base["spend"].max(), 100)
    ys = b0 + b1 * np.log(xs)
    return xs, ys, b0, b1


def overperformers(base, spend_floor=1500, n=6):
    sub = base[base["spend"] > spend_floor]
    return sub.nlargest(n, "residual")


def underperformers(base, spend_floor=2500, n=6):
    sub = base[base["spend"] > spend_floor]
    return sub.nsmallest(n, "residual")


def run_regression(base):
    """
    Standardized linear regression predicting life expectancy.
    Returns (coef_df sorted by |effect|, r2, mae, predicted-vs-actual df).
    DTP3 is intentionally excluded: it is collinear with child mortality.
    """
    reg = base[["Entity", "LE"] + MODEL_FEATURES].dropna()
    X = reg[MODEL_FEATURES]
    y = reg["LE"]
    Xz = (X - X.mean()) / X.std()
    lr = LinearRegression().fit(Xz, y)
    pred = lr.predict(Xz)

    coef = (pd.DataFrame({"feature": MODEL_FEATURES, "coef": lr.coef_})
            .assign(abs_coef=lambda d: d["coef"].abs())
            .sort_values("abs_coef", ascending=False))
    coef["label"] = coef["feature"].map(LABELS)

    pva = reg[["Entity"]].copy()
    pva["actual"] = y.values
    pva["predicted"] = pred
    pva["Code"] = reg.index.map(lambda i: base.loc[i, "Code"]) if "Code" in base else None

    return coef, r2_score(y, pred), mean_absolute_error(y, pred), pva
