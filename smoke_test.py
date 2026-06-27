"""Offline test of the data pipeline against the bundled local files."""
import numpy as np
import data


def main():
    df = data.load_data()
    assert {"health_pc", "life_exp", "uhc", "obesity", "value_gap"} <= set(df.columns)
    assert "World" not in df["Entity"].values, "aggregates not filtered"

    d = data.cross_section(df, 2019)
    rich = d[d["health_pc"] >= data.RICH_THRESHOLD]

    # two-world story must hold
    assert d["life_exp"].corr(d["uhc"]) > 0.7, "access should track life expectancy globally"
    assert rich["life_exp"].corr(np.log(rich["health_pc"])) < 0, \
        "among rich countries, more spending should NOT buy more life"
    assert rich["value_gap"].corr(rich["obesity"]) < -0.3, \
        "obesity should track the rich-country shortfall"

    us = d[d["Entity"] == "United States"].iloc[0]
    assert us["value_gap"] < -3, "US should be a major over-payer"

    print("All pipeline + narrative checks passed.")
    print(f"  global access corr   : {d['life_exp'].corr(d['uhc']):.2f}")
    print(f"  rich spend corr      : {rich['life_exp'].corr(np.log(rich['health_pc'])):.2f}")
    print(f"  rich obesity corr    : {rich['value_gap'].corr(rich['obesity']):.2f}")
    print(f"  US value gap         : {us['value_gap']:+.1f} yrs")


if __name__ == "__main__":
    main()
