from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd

DATA_FILE = Path(__file__).with_name("matches.csv")
OUTPUT_DIR = Path(__file__).parent

REQUIRED_COLUMNS = [
    "gameDuration",
    "winner",
    "firstBlood",
    "firstDragon",
    "firstBaron",
]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson score interval for a Bernoulli proportion."""
    if total <= 0:
        return (math.nan, math.nan)

    p = successes / total
    denominator = 1 + (z**2 / total)
    center = (p + (z**2 / (2 * total))) / denominator
    margin = (
        z
        * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
        / denominator
    )
    return center - margin, center + margin


def load_matches(path: Path = DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=REQUIRED_COLUMNS)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["gameDuration", "winner"]).copy()
    df = df[df["winner"].isin([1, 2])]
    df["duration_min"] = df["gameDuration"] / 60.0
    return df


def objective_summary(df: pd.DataFrame, column: str) -> dict[str, float]:
    recorded = df[df[column].isin([1, 2])].copy()
    won_after_objective = recorded[column].eq(recorded["winner"])

    total = int(len(recorded))
    wins = int(won_after_objective.sum())
    rate = wins / total if total else math.nan
    low, high = wilson_interval(wins, total)

    return {
        "metric": column,
        "games": total,
        "wins_by_objective_team": wins,
        "win_rate": rate,
        "ci95_low": low,
        "ci95_high": high,
    }


def duration_summary(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 25, 30, 35, 40, float("inf")]
    labels = ["<25", "25-30", "30-35", "35-40", "40+"]

    working = df.copy()
    working["duration_bin"] = pd.cut(
        working["duration_min"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    working["team1_win"] = working["winner"].eq(1)

    records = []
    for label in labels:
        group = working[working["duration_bin"] == label]
        total = int(len(group))
        wins = int(group["team1_win"].sum())
        rate = wins / total if total else math.nan
        low, high = wilson_interval(wins, total)
        records.append(
            {
                "duration_bin_min": label,
                "games": total,
                "team1_win_rate": rate,
                "ci95_low": low,
                "ci95_high": high,
            }
        )

    return pd.DataFrame(records)


def save_duration_plot(summary: pd.DataFrame) -> None:
    x = range(len(summary))
    y = summary["team1_win_rate"] * 100
    lower = (summary["team1_win_rate"] - summary["ci95_low"]) * 100
    upper = (summary["ci95_high"] - summary["team1_win_rate"]) * 100

    plt.figure(figsize=(9, 5))
    plt.errorbar(x, y, yerr=[lower, upper], fmt="o", capsize=4)
    plt.axhline(50, linewidth=1)
    plt.xticks(list(x), summary["duration_bin_min"])
    plt.xlabel("Game duration (minutes)")
    plt.ylabel("Team 1 win rate (%)")
    plt.title("Team 1 Win Rate by Game Duration")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "winrate_by_duration.png", dpi=160)
    plt.close()


def save_objective_plot(label: str, row: dict[str, float], filename: str) -> None:
    rate = row["win_rate"] * 100
    low = (row["win_rate"] - row["ci95_low"]) * 100
    high = (row["ci95_high"] - row["win_rate"]) * 100

    plt.figure(figsize=(6, 5))
    plt.bar([label], [rate])
    plt.errorbar([0], [rate], yerr=[[low], [high]], fmt="none", capsize=5)
    plt.axhline(50, linewidth=1)
    plt.ylim(0, 100)
    plt.ylabel("Win rate of team securing objective (%)")
    plt.title(f"{label} and Match Outcome")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close()


def main() -> None:
    df = load_matches()

    team1_rate = df["winner"].eq(1).mean()
    print(f"Games analyzed: {len(df):,}")
    print(f"Median game duration: {df['duration_min'].median():.2f} minutes")
    print(f"Team 1 win rate: {team1_rate:.2%}")

    duration = duration_summary(df)
    objectives = pd.DataFrame(
        [
            objective_summary(df, "firstBlood"),
            objective_summary(df, "firstDragon"),
            objective_summary(df, "firstBaron"),
        ]
    )

    print("\nObjective associations")
    for row in objectives.to_dict(orient="records"):
        print(
            f"{row['metric']}: {row['win_rate']:.2%} "
            f"(95% CI {row['ci95_low']:.2%}-{row['ci95_high']:.2%}, "
            f"n={int(row['games']):,})"
        )

    duration.to_csv(OUTPUT_DIR / "duration_summary.csv", index=False)
    objectives.to_csv(OUTPUT_DIR / "objective_summary.csv", index=False)

    save_duration_plot(duration)
    for label, metric, filename in [
        ("First Blood", "firstBlood", "first_blood_impact.png"),
        ("First Dragon", "firstDragon", "dragon_impact.png"),
        ("First Baron", "firstBaron", "baron_impact.png"),
    ]:
        row = objective_summary(df, metric)
        save_objective_plot(label, row, filename)

    print("\nGenerated:")
    print("- duration_summary.csv")
    print("- objective_summary.csv")
    print("- winrate_by_duration.png")
    print("- first_blood_impact.png")
    print("- dragon_impact.png")
    print("- baron_impact.png")
    print("\nThese are descriptive associations, not causal effects.")


if __name__ == "__main__":
    main()
