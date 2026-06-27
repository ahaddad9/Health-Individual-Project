"""
The Spending Paradox — why life expectancy plateaus, and what actually moves it.

Single screen, presentation-ready:
  KPIs            -> the paradox in numbers
  The plateau     -> spending alone hits diminishing returns
  Pillar 1 REACH  -> can people actually access care? (UHC)
  Pillar 2 PREVENTION -> do they stay healthy? (obesity)
  Pillar 3 EFFECTIVENESS -> does spending convert into life? (value gap)
  Map             -> life expectancy around the world

Data: Our World in Data (WHO / World Bank / UN). See data.py.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import data

st.set_page_config(page_title="The Spending Paradox", page_icon="🩺", layout="wide")

CORAL, BLUE, GREEN, TREND = "#D85A30", "#185FA5", "#4F9D69", "#5F5E5A"
HL = [CORAL, BLUE, "#1D9E75", "#534AB7", "#BA7517"]


def check_password():
    expected = st.secrets.get("password", "health2026")
    if st.session_state.get("auth_ok"):
        return
    st.markdown("<h2 style='text-align:center'>🩺 The Spending Paradox</h2>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:gray'>Why does life "
                "expectancy plateau, even when spending keeps rising?</p>",
                unsafe_allow_html=True)
    pw = st.text_input("Password", type="password",
                       label_visibility="collapsed", placeholder="Enter password")
    if pw == expected:
        st.session_state["auth_ok"] = True
        st.rerun()
    elif pw:
        st.error("Incorrect password.")
    st.stop()


@st.cache_data(show_spinner="Loading data…")
def get_data():
    return data.load_data()


def log_fit(x, y):
    m = (x > 0) & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return None, None
    b, a = np.polyfit(np.log(x[m]), y[m], 1)
    xs = np.linspace(x[m].min(), x[m].max(), 100)
    return xs, a + b * np.log(xs)


def money(v):
    return "—" if pd.isna(v) else f"${v:,.0f}"


def main():
    check_password()
    df = get_data()
    lo, hi = data.year_bounds(df)

    st.title("The Spending Paradox")
    st.caption("Spending buys life expectancy — until it doesn't. After the "
               "plateau, three levers actually move the needle: reach, "
               "prevention, and effectiveness.")

    with st.sidebar:
        st.header("Filters")
        year = st.slider("Year", lo, hi, 2019 if lo <= 2019 <= hi else hi)
        countries = sorted(df["Entity"].unique())
        highlights = st.multiselect("Highlight countries", countries,
                    default=[c for c in data.DEFAULT_HIGHLIGHT if c in countries])
        st.divider()
        st.download_button("⬇ Download merged data (CSV)",
                           df.to_csv(index=False).encode("utf-8"),
                           "health_spending_merged.csv", "text/csv",
                           help="Your submission data file.")
        st.caption(f"“Rich” = spending ≥ ${data.RICH_THRESHOLD:,}/person (PPP).")

    c = data.cross_section(df, year).dropna(subset=["health_pc", "life_exp"])
    cmap = {ct: HL[i % len(HL)] for i, ct in enumerate(highlights)}

    def hl_layer(sub, x, y):
        h = sub[sub["Entity"].isin(highlights)]
        return go.Scatter(x=h[x], y=h[y], mode="markers+text", text=h["Entity"],
            textposition="top center", textfont=dict(size=11),
            marker=dict(size=13, color=[cmap[e] for e in h["Entity"]],
                        line=dict(width=1.5, color="white")),
            hoverinfo="skip", showlegend=False)

    def style(fig, h=330, **kw):
        fig.update_layout(height=h, margin=dict(l=10, r=10, t=10, b=10),
                          showlegend=False, **kw)
        return fig

    # ---------- KPIs ----------
    st.subheader(f"The paradox at a glance · {year}")
    k = st.columns(4)
    if len(highlights) >= 2:
        c1, c2 = highlights[0], highlights[1]
        r1, r2 = c[c["Entity"] == c1], c[c["Entity"] == c2]
        g1 = r1["value_gap"].iloc[0] if len(r1) else np.nan
        l1 = r1["life_exp"].iloc[0] if len(r1) else np.nan
        l2 = r2["life_exp"].iloc[0] if len(r2) else np.nan
        k[0].metric(f"{c1} · spend/person", money(r1["health_pc"].iloc[0] if len(r1) else np.nan))
        k[1].metric(f"{c2} · spend/person", money(r2["health_pc"].iloc[0] if len(r2) else np.nan))
        k[2].metric(f"{c1} · value for money", "—" if pd.isna(g1) else f"{g1:+.1f} yrs",
                    help="Years lived above/below what its spending predicts.")
        k[3].metric(f"Life-expectancy gap ({c1}−{c2})",
                    "—" if pd.isna(l1) or pd.isna(l2) else f"{l1 - l2:+.1f} yrs")
    else:
        k[0].info("Pick two highlight countries in the sidebar.")

    # ---------- The plateau ----------
    st.divider()
    st.markdown("### The plateau problem")
    st.caption("Each dot is a country. Life expectancy rises fast with the "
               "first health dollars, then flattens. Past the bend, spending "
               "more buys almost nothing — and the biggest spender sits off the curve.")
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=c["health_pc"], y=c["life_exp"], mode="markers",
        marker=dict(size=8, color="#9DC3E6", line=dict(width=0.5, color="white")),
        text=c["Entity"],
        hovertemplate="<b>%{text}</b><br>$%{x:,.0f}<br>%{y:.1f}y<extra></extra>"))
    xs, ys = log_fit(c["health_pc"].to_numpy(), c["life_exp"].to_numpy())
    if xs is not None:
        f1.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                     line=dict(color=TREND, dash="dash", width=2)))
    f1.add_trace(hl_layer(c, "health_pc", "life_exp"))
    st.plotly_chart(style(f1, 400, xaxis_type="log",
        xaxis_title="Health spending per person (PPP $, log scale)",
        yaxis_title="Life expectancy (years)"), use_container_width=True)

    # ---------- Three pillars ----------
    st.divider()
    st.markdown("### Three levers that actually move life expectancy")

    p = st.columns(2)
    # Pillar 1 — Reach
    with p[0]:
        st.markdown("#### 1 · Reach — can people access care?")
        st.caption("Health-service coverage (UHC) vs life expectancy. Access "
                   "tracks lifespan tightly — getting people covered is what works.")
        f = go.Figure()
        f.add_trace(go.Scatter(x=c["uhc"], y=c["life_exp"], mode="markers",
            marker=dict(size=7, color=BLUE, opacity=0.55,
                        line=dict(width=0.4, color="white")),
            text=c["Entity"],
            hovertemplate="<b>%{text}</b><br>Access %{x:.0f}<br>%{y:.1f}y<extra></extra>"))
        f.add_trace(hl_layer(c.dropna(subset=["uhc"]), "uhc", "life_exp"))
        st.plotly_chart(style(f, xaxis_title="Access index (UHC)",
            yaxis_title="Life expectancy"), use_container_width=True)

    # Pillar 2 — Prevention
    with p[1]:
        st.markdown("#### 2 · Prevention — do people stay healthy?")
        st.caption("Among rich countries (access already high), obesity vs value "
                   "for money. Heavier populations get less life per dollar.")
        rich = c[c["health_pc"] >= data.RICH_THRESHOLD].dropna(subset=["obesity", "value_gap"])
        f = go.Figure()
        f.add_trace(go.Scatter(x=rich["obesity"], y=rich["value_gap"], mode="markers",
            marker=dict(size=9, color=CORAL, opacity=0.65,
                        line=dict(width=0.4, color="white")),
            text=rich["Entity"],
            hovertemplate="<b>%{text}</b><br>Obesity %{x:.0f}%%<br>%{y:+.1f}y<extra></extra>"))
        b, a = np.polyfit(rich["obesity"], rich["value_gap"], 1)
        lx = np.linspace(rich["obesity"].min(), rich["obesity"].max(), 50)
        f.add_trace(go.Scatter(x=lx, y=a + b*lx, mode="lines", hoverinfo="skip",
                    line=dict(color=TREND, dash="dash", width=2)))
        f.add_trace(hl_layer(rich, "obesity", "value_gap"))
        st.plotly_chart(style(f, xaxis_title="Adult obesity (%)",
            yaxis_title="Value for money (yrs)"), use_container_width=True)

    # Pillar 3 — Effectiveness
    st.markdown("#### 3 · Effectiveness — does spending convert into life?")
    st.caption("Years of life each rich country gains above (green) or loses "
               "(red) versus what its spending predicts. Big budgets sink to the "
               "bottom; the US is last despite spending the most.")
    eff = c[c["health_pc"] >= data.RICH_THRESHOLD].dropna(subset=["value_gap"]).sort_values("value_gap")
    if len(eff):
        colors = ["#C0504D" if v < 0 else GREEN for v in eff["value_gap"]]
        f4 = go.Figure(go.Bar(x=eff["value_gap"], y=eff["Entity"], orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>%{x:+.1f} yrs vs predicted<extra></extra>"))
        st.plotly_chart(style(f4, max(300, 16*len(eff)),
            xaxis_title="Years lived vs. spending prediction"), use_container_width=True)

    # ---------- Map ----------
    st.divider()
    st.markdown("### Life expectancy around the world")
    st.caption(f"Geographic view, {year}. Hover any country for details.")
    fm = go.Figure(go.Choropleth(
        locations=c["Code"], z=c["life_exp"], text=c["Entity"],
        colorscale="RdYlGn", marker_line_color="white", marker_line_width=0.4,
        colorbar=dict(title="Life exp", thickness=12),
        hovertemplate="<b>%{text}</b><br>%{z:.1f} yrs<extra></extra>"))
    fm.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0),
                     geo=dict(showframe=False, projection_type="natural earth"))
    st.plotly_chart(fm, use_container_width=True)

    st.divider()
    st.caption("Sources: Our World in Data — health spending per capita "
               "(WHO/World Bank), life expectancy (UN WPP), UHC service coverage "
               "(WHO), adult obesity (WHO). Relationships are correlational; "
               "national outcomes also reflect inequality, diet, and demographics.")


if __name__ == "__main__":
    main()
