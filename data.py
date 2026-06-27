"""
Data layer for the health spending paradox dashboard.

Reads four local OWID CSVs (in ./data), cleans to real countries, merges into
one country-year table, and adds:
  value_gap  -> years lived above/below what spending predicts (EFFECTIVENESS)
  life_per_1k-> life expectancy per $1,000 spent

Story: spending buys life only until it plateaus. Three levers move it after
that — REACH (access), PREVENTION (obesity), EFFECTIVENESS (value gap).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

FILES = {
    "health_pc": "annual-healthcare-expenditure-per-capita.csv",
    "life_exp": "life-expectancy-unwpp.csv",
    "uhc": "universal-health-coverage-index.csv",
    "obesity": "share-of-adults-defined-as-obese.csv",
}

DEFAULT_HIGHLIGHT = ["United States", "Japan"]
RICH_THRESHOLD = 4000


def _tidy(path: Path, name: str, keep_code: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    val = [c for c in df.columns if c not in ("Entity", "Code", "Year")][0]
    df = df[df["Code"].notna() & df["Code"].str.fullmatch(r"[A-Z]{3}")]
    cols = ["Entity", "Code", "Year", val] if keep_code else ["Entity", "Year", val]
    return df[cols].rename(columns={val: name})


def _add_value_gap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["value_gap"] = np.nan
    for _, g in df.groupby("Year"):
        m = g["health_pc"].notna() & g["life_exp"].notna() & (g["health_pc"] > 0)
        if m.sum() < 8:
            continue
        b, a = np.polyfit(np.log(g.loc[m, "health_pc"]), g.loc[m, "life_exp"], 1)
        df.loc[g.loc[m].index, "value_gap"] = (
            g.loc[m, "life_exp"] - (a + b * np.log(g.loc[m, "health_pc"])))
    return df


def load_data(data_dir: Path | None = None) -> pd.DataFrame:
    d = data_dir or DATA_DIR
    spine = _tidy(d / FILES["health_pc"], "health_pc", keep_code=True)
    df = spine.merge(_tidy(d / FILES["life_exp"], "life_exp"), on=["Entity", "Year"])
    df = df.merge(_tidy(d / FILES["uhc"], "uhc"), on=["Entity", "Year"], how="left")
    df = df.merge(_tidy(d / FILES["obesity"], "obesity"), on=["Entity", "Year"], how="left")
    df["life_per_1k"] = df["life_exp"] / (df["health_pc"] / 1000.0)
    df = _add_value_gap(df)
    return df.sort_values(["Entity", "Year"]).reset_index(drop=True)


def year_bounds(df: pd.DataFrame) -> tuple[int, int]:
    return int(df["Year"].min()), int(df["Year"].max())


def cross_section(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df[(df["Year"] == year)
             & df["health_pc"].notna() & df["life_exp"].notna()].copy()
