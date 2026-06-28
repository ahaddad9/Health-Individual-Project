"""
Health spending vs life expectancy, a consultant dashboard (single file).

Beat 1: more money buys fewer extra years (diminishing returns), and the US is
the world's biggest overspending underperformer.
Beat 2: three pillars still move life expectancy once spending plateaus:
prevention (child mortality), environment (clean water), and reach (service coverage).

Everything (data layer + UI) lives in this one file so there is no second
module to keep in sync. Reads the CSVs in ./data.

Run locally:  streamlit run app.py
Deploy:       push to GitHub, point Streamlit Community Cloud at app.py.
Password:     set [password] in Cloud "Secrets" (defaults to msba382).
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Health spend vs life expectancy",
                   page_icon="\U0001FAC0", layout="wide")

ACCENT = "#185FA5"
DANGER = "#D85A30"
TEAL = "#0F6E56"
CURVE = "#888780"

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
    "pm25": ("pm25-air-pollution.csv",
             "PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)",
             "pm25"),
    "reach": ("universal-health-coverage-index.csv",
              "UHC Service Coverage Index (SDG 3.8.1)", "reach"),
    "gdp": ("gdp-per-capita-worldbank.csv", "GDP per capita", "gdp"),
}

LABELS = {
    "child_mort": "Child mortality (under-5, per 1,000)",
    "maternal": "Maternal mortality (per 100k births)",
    "dtp3": "DTP3 immunization (%)",
    "water": "Safely managed drinking water (%)",
    "smoking": "Adult smoking (%)",
    "obesity": "Adult obesity (%)",
    "pm25": "PM2.5 air pollution (ug/m3)",
    "prevention": "Prevention index (0-100)",
    "environment": "Environment index (0-100)",
    "reach": "Reach index (0-100)",
    "gdp": "GDP per capita (int$)",
    "log_spend": "Health spend per capita (log)",
}

MODEL_FEATURES = ["prevention", "environment", "reach"]

PLATEAU_BANDS = [(0, 200, "<$200"), (200, 500, "$200-500"), (500, 1000, "$500-1k"),
                 (1000, 2000, "$1k-2k"), (2000, 4000, "$2k-4k"),
                 (4000, 8000, "$4k-8k"), (8000, 1e12, ">$8k")]


# ----------------------------------------------------------------------------
# Data layer
# ----------------------------------------------------------------------------
def _iso(df):
    code = df["Code"].astype(str)
    return df[code.str.match(r"^[A-Z]{3}$") & ~code.str.startswith("OWID")].copy()


def _read(name):
    return pd.read_csv(os.path.join(DATA_DIR, name))


def _load_indicator(filename, col, new):
    d = _iso(_read(filename)).rename(columns={col: new})
    return d[["Code", "Year", new]].dropna(subset=[new])


def _obesity_col(df):
    return [c for c in df.columns if "Obesity" in c][0]


@st.cache_data(show_spinner=False)
def available_years():
    hero = _iso(_read("life-expectancy-vs-health-expenditure.csv"))
    spend = _load_indicator(
        "annual-healthcare-expenditure-per-capita.csv",
        "Current health expenditure per capita, PPP (current international $)", "spend")
    le = hero.dropna(subset=["Life expectancy"])[["Code", "Year", "Life expectancy"]]
    m = le.merge(spend, on=["Code", "Year"])
    cov = m.groupby("Year")["Code"].nunique()
    return sorted(int(y) for y in cov[cov >= 120].index)


@st.cache_data(show_spinner=False)
def build_cross_section(year):
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

    base["log_spend"] = np.log(base["spend"])
    b1, b0 = np.polyfit(base["log_spend"], base["LE"], 1)
    base["pred_LE"] = b0 + b1 * base["log_spend"]
    base["residual"] = base["LE"] - base["pred_LE"]
    base = base.set_index("Code")

    def latest(filename, col, new):
        d = _load_indicator(filename, col, new)
        d = d[(d["Year"] <= year) & (d["Year"] >= year - 12)]
        return d.sort_values("Year").groupby("Code").tail(1).set_index("Code")[new]

    for key, (fn, col, new) in INDICATORS.items():
        try:
            base[key] = latest(fn, col, new)
        except (FileNotFoundError, KeyError, IndexError):
            base[key] = np.nan

    try:
        ob = _iso(_read("share-of-adults-defined-as-obese.csv"))
        base["obesity"] = latest("share-of-adults-defined-as-obese.csv",
                                  _obesity_col(ob), "obesity")
    except (FileNotFoundError, KeyError, IndexError):
        base["obesity"] = np.nan

    try:
        sx = _iso(_read("female-and-male-life-expectancy-at-birth-in-years.csv"))
        sx = sx[sx["Year"] == year].set_index("Code")
        base["LE_men"] = sx["Men"]
        base["LE_women"] = sx["Women"]
    except (FileNotFoundError, KeyError, IndexError):
        base["LE_men"] = np.nan
        base["LE_women"] = np.nan

    # pillar composite indices: z-score blend, oriented so higher = healthier,
    # then rescaled to 0-100. Reach is already a 0-100 composite (UHC index).
    def _z(s):
        return (s - s.mean()) / s.std()

    def _rescale(s):
        return 100 * (s - s.min()) / (s.max() - s.min())

    base["prevention"] = _rescale((-_z(base["child_mort"]) - _z(base["maternal"])
                                   + _z(base["dtp3"])) / 3)
    base["environment"] = _rescale((_z(base["water"]) - _z(base["pm25"])) / 2)

    def band(s):
        if s < 1000:
            return "Low (<$1k)"
        if s < 3000:
            return "Middle ($1k-3k)"
        return "High (>$3k)"
    base["band"] = base["spend"].apply(band)

    return base.reset_index()


def fit_info(base):
    b1, b0 = np.polyfit(base["log_spend"], base["LE"], 1)
    xs = np.linspace(base["spend"].min(), base["spend"].max(), 100)
    ys = b0 + b1 * np.log(xs)
    return xs, ys, b0, b1


def overperformers(base, spend_floor=1500, n=6):
    return base[base["spend"] > spend_floor].nlargest(n, "residual")


def plateau_table(base):
    rows = []
    for lo, hi, lab in PLATEAU_BANDS:
        sub = base[(base["spend"] >= lo) & (base["spend"] < hi)]
        if len(sub):
            rows.append({"band": lab, "mean_LE": sub["LE"].mean(), "n": len(sub)})
    out = pd.DataFrame(rows)
    out["gain"] = out["mean_LE"].diff()
    return out


def run_regression(base):
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
    return coef, r2_score(y, pred), mean_absolute_error(y, pred), pva


# ----------------------------------------------------------------------------
# Password gate
# ----------------------------------------------------------------------------
def check_password():
    def entered():
        pw = st.secrets.get("password", "msba382")
        st.session_state["ok"] = st.session_state.get("pw_input", "") == pw

    if st.session_state.get("ok"):
        return True
    st.markdown("### \U0001FAC0 Health spending vs life expectancy")
    st.caption("A consultant view on where health budgets actually buy years of life.")
    st.text_input("Password", type="password", key="pw_input", on_change=entered)
    if "ok" in st.session_state and not st.session_state["ok"]:
        st.error("Incorrect password.")
    st.stop()


check_password()

# ----------------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------------
YRS = available_years()
st.sidebar.header("Filters")
year = st.sidebar.slider("Year", min(YRS), max(YRS), max(YRS))
base = build_cross_section(year)

all_regions = sorted(base["region"].dropna().unique())
regions = st.sidebar.multiselect("Region", all_regions, default=all_regions)
all_bands = ["Low (<$1k)", "Middle ($1k-3k)", "High (>$3k)"]
bands = st.sidebar.multiselect("Spend band", all_bands, default=all_bands)
st.sidebar.caption(
    "Data: Our World in Data, drawing on World Bank, WHO, UN and UN IGME. "
    "Each country is its most recent value at or before the selected year.")

flt = base[base["region"].isin(regions) & base["band"].isin(bands)].copy()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("Health spending vs life expectancy")
st.caption(f"{len(flt)} countries shown, {year}. More spending buys fewer extra "
           "years, and these are the levers that still raise it.")

tab_dash, tab_model = st.tabs(["Dashboard", "Model"])

# ============================================================================
# DASHBOARD TAB
# ============================================================================
with tab_dash:
    us_row = base[base["Code"] == "USA"]
    us_gap = float(us_row["residual"].iloc[0]) if len(us_row) else np.nan
    women_m = flt["LE_women"].mean()
    men_m = flt["LE_men"].mean()
    gender_gap = women_m - men_m

    def card(label, value, sub, dot, num_color="#1f2937"):
        return f"""<div style="border:1px solid #e8e8e8;border-radius:10px;
            padding:16px 18px;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.04)">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="color:#6b7280;font-size:13px">{label}</span>
            <span style="width:9px;height:9px;border-radius:50%;background:{dot};
                  display:inline-block"></span>
          </div>
          <div style="font-size:30px;font-weight:700;color:{num_color};
                line-height:1.15;margin-top:8px">{value}</div>
          <div style="font-size:12px;color:#9ca3af;margin-top:4px;
                min-height:18px">{sub}</div>
        </div>"""

    c = st.columns(4)
    c[0].markdown(card("Avg life expectancy", f"{flt['LE'].mean():.1f} yrs",
                       "across countries shown", "#185FA5"), unsafe_allow_html=True)
    c[1].markdown(card("Median health spend", f"${flt['spend'].median():,.0f}",
                       "per person, int$", "#0F6E56"), unsafe_allow_html=True)
    c[2].markdown(card("US efficiency gap", f"{us_gap:+.1f} yrs",
                       "vs what its spending predicts", "#D85A30",
                       num_color="#D85A30"), unsafe_allow_html=True)
    c[3].markdown(f"""<div style="border:1px solid #e8e8e8;border-radius:10px;
        padding:16px 18px;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.04)">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span style="color:#6b7280;font-size:13px">Gender gap (W-M)</span>
        <span style="width:9px;height:9px;border-radius:50%;background:#D4537E;
              display:inline-block"></span>
      </div>
      <div style="display:flex;align-items:center;gap:10px;margin-top:8px">
        <div style="font-size:30px;font-weight:700;color:#1f2937;line-height:1">
          +{gender_gap:.1f} yrs</div>
        <div style="display:flex;gap:6px">
          <span style="background:#FBE4EE;color:#C13C77;font-weight:700;font-size:13px;
            border-radius:999px;padding:4px 11px">&#9792; {women_m:.1f}</span>
          <span style="background:#E4EEFB;color:#1F62B0;font-weight:700;font-size:13px;
            border-radius:999px;padding:4px 11px">&#9794; {men_m:.1f}</span>
        </div>
      </div>
      <div style="font-size:12px;color:#9ca3af;margin-top:6px;min-height:18px">
        women vs men, years</div>
    </div>""", unsafe_allow_html=True)

    st.write("")
    st.divider()

    # --- Beat 1: diminishing returns ---
    st.subheader("Diminishing returns: spending vs life expectancy")
    st.caption("Each dot is a country. The dashed line is the global fit of life "
               "expectancy on log spending. Countries below it get less life than "
               "their spending predicts; those above get more. Albania, highlighted, "
               "matches US life expectancy on a fraction of the spending.")

    col_a, col_b = st.columns([3, 2])

    with col_a:
        fig = px.scatter(
            flt, x="spend", y="LE", color="region", log_x=True, hover_name="Entity",
            hover_data={"spend": ":$,.0f", "LE": ":.1f", "residual": ":+.1f",
                        "region": False},
            labels={"spend": "Health spend per capita (int$, log scale)",
                    "LE": "Life expectancy (years)", "region": "Region"})
        fig.update_traces(marker=dict(size=8, opacity=0.75))
        xs, ys, _, _ = fit_info(base)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Global fit",
                                 line=dict(color=CURVE, dash="dash", width=2),
                                 hoverinfo="skip"))
        if len(us_row):
            u = us_row.iloc[0]
            fig.add_trace(go.Scatter(
                x=[u["spend"]], y=[u["LE"]], mode="markers+text",
                text=["US " + f"{u['residual']:+.1f}"], textposition="bottom center",
                textfont=dict(color=DANGER),
                marker=dict(color=DANGER, size=15, symbol="star"),
                name="US", hoverinfo="skip"))
        over = overperformers(base, spend_floor=1500, n=5)
        over = over[over["Entity"].isin(flt["Entity"])]
        if len(over):
            fig.add_trace(go.Scatter(
                x=over["spend"], y=over["LE"], mode="markers",
                marker=dict(color=TEAL, size=11, symbol="diamond",
                            line=dict(color="white", width=1)),
                name="Efficient overperformers", text=over["Entity"],
                hoverinfo="text"))

        # comparator: same life expectancy as the US on a fraction of the spend
        alb = base[base["Code"] == "ALB"]
        if len(alb) and len(us_row):
            a = alb.iloc[0]
            uu = us_row.iloc[0]
            fig.add_trace(go.Scatter(
                x=[a["spend"], uu["spend"]], y=[a["LE"], uu["LE"]], mode="lines",
                line=dict(color="#6b7280", dash="dot", width=1.5),
                hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(
                x=[a["spend"]], y=[a["LE"]], mode="markers+text", text=["Albania"],
                textposition="top center", textfont=dict(color="#2E7D32"),
                marker=dict(color="#2E7D32", size=13,
                            line=dict(color="white", width=1)),
                name="Albania (same life, far less spend)", hoverinfo="skip"))
            fig.add_annotation(
                x=np.log10(np.sqrt(a["spend"] * uu["spend"])),
                y=(a["LE"] + uu["LE"]) / 2 + 1.6, xref="x", yref="y",
                text=f"{uu['spend'] / a['spend']:.0f}x less spending, same life",
                showarrow=False, font=dict(color="#6b7280", size=11))
        fig.update_layout(height=430, legend=dict(orientation="h", y=-0.32),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Where the returns run out**")
        pl = plateau_table(base)
        mid = {"<$200": 120, "$200-500": 350, "$500-1k": 730, "$1k-2k": 1450,
               "$2k-4k": 2900, "$4k-8k": 5800, ">$8k": 11000}
        pl["mid"] = pl["band"].map(mid)
        line = px.line(pl, x="mid", y="mean_LE", line_shape="spline",
                       labels={"mid": "Health spend per capita (int$)",
                               "mean_LE": "Avg life expectancy"})
        line.update_traces(line=dict(color=ACCENT, width=3))
        line.update_xaxes(tickvals=list(pl["mid"]), ticktext=list(pl["band"]),
                          tickangle=-30)
        line.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10),
                           yaxis_range=[pl["mean_LE"].min() - 6,
                                        pl["mean_LE"].max() + 3])
        st.plotly_chart(line, use_container_width=True)

    top = pl["mean_LE"].iloc[-1]
    g_last = pl["gain"].iloc[-1] if len(pl) > 1 else 0
    st.caption(f"Average life expectancy by spending band. The line climbs fast at low "
               f"spending, then goes flat. The top band, which includes the US, reaches "
               f"about {top:.0f} years and sits only {g_last:.1f} above the band below "
               f"it. Past a few thousand dollars per person, extra spending adds almost "
               f"nothing.")

    st.divider()

    # --- Beat 2: three pillar relationships ---
    st.subheader("What moves life expectancy")
    st.caption("Each pillar is a composite index built from several indicators, "
               "plotted against life expectancy. Scoring higher on each index goes "
               "with longer lives.")

    p1, p2, p3 = st.columns(3)

    def pillar_chart(col, xcol, title, note):
        d = flt.dropna(subset=[xcol, "LE"])
        with col:
            st.markdown(f"**{title}**")
            if len(d) < 5:
                st.info("Not enough countries in the current filter.")
                return
            f = px.scatter(d, x=xcol, y="LE", hover_name="Entity",
                           trendline="ols", trendline_color_override=ACCENT,
                           labels={xcol: LABELS[xcol], "LE": "Life expectancy"})
            f.update_traces(marker=dict(size=6, opacity=0.6, color=ACCENT))
            f.update_layout(height=300, showlegend=False,
                            margin=dict(l=5, r=5, t=5, b=5))
            st.plotly_chart(f, use_container_width=True)
            st.caption(note)

    pillar_chart(p1, "prevention", "Prevention",
                 "Child mortality, maternal mortality, and vaccination combined. "
                 "Higher means stronger prevention.")
    pillar_chart(p2, "environment", "Environment",
                 "Safe drinking water and clean air (PM2.5) combined. Higher means "
                 "a healthier environment.")
    pillar_chart(p3, "reach", "Reach",
                 "WHO universal health coverage index, a blend of 14 service "
                 "indicators.")

    st.markdown(
        """<div style="border-left:4px solid #185FA5;background:#f4f8fc;
        border-radius:0 8px 8px 0;padding:12px 16px;margin-top:8px">
        <b>What this means:</b> putting money into these levers, fewer child
        deaths, clean water, and wider service coverage, is what adds years of
        life. Spending on its own does not.</div>""", unsafe_allow_html=True)

    st.divider()

    # --- Map ---
    st.subheader("Geographic view")
    metric = st.radio("Colour by", ["Life expectancy", "Efficiency gap (residual)"],
                      horizontal=True, label_visibility="collapsed")
    if metric.startswith("Life"):
        mcol, scale, mid = "LE", "Blues", None
    else:
        mcol, scale, mid = "residual", "RdBu", 0
    mp = px.choropleth(flt, locations="Code", color=mcol, hover_name="Entity",
                       color_continuous_scale=scale, color_continuous_midpoint=mid,
                       labels={"LE": "Life exp.", "residual": "Residual (yrs)"})
    mp.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0),
                     geo=dict(showframe=False))
    st.plotly_chart(mp, use_container_width=True)

# ============================================================================
# MODEL TAB
# ============================================================================
with tab_model:
    st.subheader("What predicts life expectancy")
    st.caption("Standardized regression of the three pillar indices on life expectancy "
               "for the selected year. Every predictor is on the same scale, so bar "
               "length shows how much improving each pillar moves life expectancy. This "
               "is association from one year of data, not proof of cause.")

    coef, r2, mae, pva = run_regression(base)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        bar = px.bar(coef.sort_values("coef"), x="coef", y="label", orientation="h",
                     color="coef", color_continuous_scale="RdBu",
                     color_continuous_midpoint=0,
                     labels={"coef": "Standardized effect on life expectancy",
                             "label": ""})
        bar.update_layout(height=320, coloraxis_showscale=False,
                          margin=dict(l=5, r=5, t=5, b=5))
        st.plotly_chart(bar, use_container_width=True)
    c2.metric("R\u00b2", f"{r2:.2f}", help="Share of variation in life expectancy explained.")
    c3.metric("Mean error", f"{mae:.1f} yrs", help="Average miss, in years.")
    c2.caption("Together the three pillars explain most of the variation in life "
               "expectancy across countries.")

    st.divider()
    st.subheader("Predicted vs actual")
    st.caption("If the model were perfect, every country would sit on the diagonal. "
               "The US scores near the top on all three pillars, so the model expects "
               "more years than it actually reaches: it sits below the line, despite "
               "spending far more than any other country. The extra money did not buy "
               "a higher position.")

    pva = pva.copy()
    pva["is_us"] = pva["Entity"] == "United States"
    sc = px.scatter(pva, x="predicted", y="actual", hover_name="Entity", color="is_us",
                    color_discrete_map={True: DANGER, False: ACCENT},
                    labels={"predicted": "Predicted life expectancy",
                            "actual": "Actual life expectancy", "is_us": "US"})
    sc.update_traces(marker=dict(size=7, opacity=0.7))
    lo = float(min(pva["predicted"].min(), pva["actual"].min()))
    hi = float(max(pva["predicted"].max(), pva["actual"].max()))
    sc.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                            line=dict(color=CURVE, dash="dash"),
                            name="Perfect fit", hoverinfo="skip"))
    sc.update_layout(height=420, showlegend=False, margin=dict(l=5, r=5, t=5, b=5))
    st.plotly_chart(sc, use_container_width=True)
