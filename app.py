"""
Health spending vs life expectancy — a consultant dashboard.

Beat 1: more money buys fewer extra years (diminishing returns), and the US is
the world's biggest overspending underperformer.
Beat 2: once spending plateaus, three pillars still move life expectancy —
prevention, environment, and (among high spenders) behavioral risk.

Run locally:  streamlit run app.py
Deploy:       push to GitHub, point Streamlit Community Cloud at app.py.
Password:     set [password] in .streamlit/secrets.toml or Cloud secrets.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data_prep as dp

st.set_page_config(page_title="Health spend vs life expectancy",
                   page_icon="\U0001FAC0", layout="wide")

ACCENT = "#185FA5"
DANGER = "#D85A30"
TEAL = "#0F6E56"
CURVE = "#888780"

# ----------------------------------------------------------------------------
# Password gate (the landing page)
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
# Data (cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load(year):
    return dp.build_cross_section(year)


@st.cache_data(show_spinner=False)
def years():
    return dp.available_years()


YRS = years()

# ----------------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------------
st.sidebar.header("Filters")
year = st.sidebar.slider("Year", min(YRS), max(YRS), max(YRS))
base = load(year)

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
st.caption(f"{len(flt)} countries shown · {year} · more spending buys fewer extra "
           "years, and the levers that still move the needle")

tab_dash, tab_model = st.tabs(["Dashboard", "Model"])

# ============================================================================
# DASHBOARD TAB
# ============================================================================
with tab_dash:
    # --- KPI row (outlined cards) ---
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
    c[3].markdown(card(
        "Gender gap (W\u2212M)", f"{gender_gap:+.1f} yrs",
        f'<span style="color:#D4537E;font-weight:600">\u2640 {women_m:.1f}</span>'
        f'&nbsp;&nbsp;<span style="color:#185FA5;font-weight:600">\u2642 {men_m:.1f}</span>',
        "#D4537E"), unsafe_allow_html=True)

    st.write("")
    st.divider()

    # --- Beat 1: diminishing returns hero ---
    st.subheader("Diminishing returns: spending vs life expectancy")
    st.caption("Each dot is a country. The curve is the global fit of life "
               "expectancy on log spending. Countries below the curve get less "
               "life than their spending predicts; those above get more.")

    fig = px.scatter(
        flt, x="spend", y="LE", color="region", log_x=True,
        hover_name="Entity",
        hover_data={"spend": ":$,.0f", "LE": ":.1f", "residual": ":+.1f",
                    "region": False},
        labels={"spend": "Health spend per capita (int$, log scale)",
                "LE": "Life expectancy (years)", "region": "Region"},
    )
    fig.update_traces(marker=dict(size=8, opacity=0.75))

    # global fitted curve (fit on the full year set, not the filtered subset)
    xs, ys, _, _ = dp.fit_info(base)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Global fit",
                             line=dict(color=CURVE, dash="dash", width=2),
                             hoverinfo="skip"))

    # flag the US
    if len(us_row):
        u = us_row.iloc[0]
        fig.add_trace(go.Scatter(
            x=[u["spend"]], y=[u["LE"]], mode="markers+text",
            text=["US " + f"{u['residual']:+.1f}"], textposition="bottom center",
            textfont=dict(color=DANGER),
            marker=dict(color=DANGER, size=15, symbol="star"),
            name="US", hoverinfo="skip"))

    # label the efficient overperformers
    over = dp.overperformers(base, spend_floor=1500, n=5)
    over = over[over["Entity"].isin(flt["Entity"])]
    if len(over):
        fig.add_trace(go.Scatter(
            x=over["spend"], y=over["LE"], mode="markers",
            marker=dict(color=TEAL, size=11, symbol="diamond",
                        line=dict(color="white", width=1)),
            name="Efficient overperformers",
            text=over["Entity"], hoverinfo="text"))

    fig.update_layout(height=460, legend=dict(orientation="h", y=-0.25),
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # the plateau, shown with the data: average life expectancy by spend band
    st.markdown("##### Where the returns run out")
    pl = dp.plateau_table(base)
    pbar = px.bar(pl, x="band", y="mean_LE", text="mean_LE",
                  color="mean_LE", color_continuous_scale="Blues",
                  labels={"band": "Health spend per capita (band)",
                          "mean_LE": "Avg life expectancy"})
    pbar.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                       cliponaxis=False)
    pbar.update_layout(height=300, coloraxis_showscale=False,
                       yaxis_range=[max(0, pl["mean_LE"].min() - 8),
                                    pl["mean_LE"].max() + 3],
                       margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(pbar, use_container_width=True)
    low = pl["mean_LE"].iloc[0]
    top = pl["mean_LE"].iloc[-1]
    g_last = pl["gain"].iloc[-1] if len(pl) > 1 else 0
    st.caption(f"Each bar averages the life expectancy of countries in that spending "
               f"band. It climbs steeply from about {low:.0f} years in the "
               f"lowest-spending countries to about {top:.0f} in high-spending ones, "
               f"then flattens: the top band ({pl['band'].iloc[-1]}, which includes the "
               f"US) sits only about {g_last:.1f} years above the band below it. Past a "
               f"few thousand dollars per person, more spending buys almost no extra years.")

    st.divider()

    # --- Beat 2: three pillar relationships ---
    st.subheader("What moves life expectancy")
    st.caption("Each pillar plotted against life expectancy. Prevention and "
               "environment hold across all countries; behavioral risk is shown "
               "on high-income countries only, where it isn't masked by income.")

    p1, p2, p3 = st.columns(3)

    def pillar_chart(col, xcol, title, note, hi_income=False):
        d = flt.dropna(subset=[xcol, "LE"])
        if hi_income:
            d = d[d["band"] == "High (>$3k)"]
        with col:
            st.markdown(f"**{title}**")
            if len(d) < 5:
                st.info("Not enough countries in the current filter.")
                return
            f = px.scatter(d, x=xcol, y="LE", hover_name="Entity",
                           trendline="ols", trendline_color_override=ACCENT,
                           labels={xcol: dp.LABELS[xcol], "LE": "Life expectancy"})
            f.update_traces(marker=dict(size=6, opacity=0.6, color=ACCENT))
            f.update_layout(height=300, showlegend=False,
                            margin=dict(l=5, r=5, t=5, b=5))
            st.plotly_chart(f, use_container_width=True)
            st.caption(note)

    pillar_chart(p1, "child_mort", "Prevention",
                 "Lower child mortality \u2192 higher life expectancy. "
                 "Source: child-mortality-igme.csv")
    pillar_chart(p2, "water", "Environment",
                 "More safe water \u2192 higher life expectancy. "
                 "Source: proportion-using-safely-managed-drinking-water.csv")
    pillar_chart(p3, "reach", "Reach",
                 "More health-service coverage \u2192 higher life expectancy. "
                 "Source: uhc-coverage-index.csv")

    st.markdown(
        """<div style="border-left:4px solid #185FA5;background:#f4f8fc;
        border-radius:0 8px 8px 0;padding:12px 16px;margin-top:8px">
        <b>The consulting takeaway:</b> targeted investment in these levers
        &mdash; preventing child deaths, clean water, and broad service
        coverage &mdash; is what buys years of life. Raw spending on its own
        does not.</div>""", unsafe_allow_html=True)

    st.divider()

    # --- Map ---
    st.subheader("Geographic view")
    metric = st.radio("Colour by", ["Life expectancy", "Efficiency gap (residual)"],
                      horizontal=True, label_visibility="collapsed")
    if metric.startswith("Life"):
        col, scale, mid = "LE", "Blues", None
    else:
        col, scale, mid = "residual", "RdBu", 0
    mp = px.choropleth(flt, locations="Code", color=col, hover_name="Entity",
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
    st.caption("Standardized linear regression on the selected year. Coefficients "
               "are comparable because every predictor is on the same scale, so the "
               "bar length is how much each lever moves life expectancy. "
               "Cross-sectional, so this is association, not proof of cause.")

    coef, r2, mae, pva = dp.run_regression(base)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        bar = px.bar(coef.sort_values("coef"), x="coef", y="label",
                     orientation="h", color="coef",
                     color_continuous_scale="RdBu", color_continuous_midpoint=0,
                     labels={"coef": "Standardized effect on life expectancy",
                             "label": ""})
        bar.update_layout(height=320, coloraxis_showscale=False,
                          margin=dict(l=5, r=5, t=5, b=5))
        st.plotly_chart(bar, use_container_width=True)
    c2.metric("R\u00b2", f"{r2:.2f}", help="Share of variation in life expectancy explained.")
    c3.metric("Mean error", f"{mae:.1f} yrs", help="Average miss, in years of life expectancy.")
    c2.caption("Spending matters, but child mortality, clean water and obesity "
               "carry more of the prediction.")

    st.divider()
    st.subheader("Predicted vs actual")
    st.caption("If the model were perfect, every country would sit on the diagonal. "
               "The US sits below it: the model expects more life for its profile.")

    pva = pva.copy()
    pva["is_us"] = pva["Entity"] == "United States"
    sc = px.scatter(pva, x="predicted", y="actual", hover_name="Entity",
                    color="is_us",
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
