import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Load dataset
    file_name = "matches.csv"
    df = pd.read_csv(file_name)

    print("=== First 5 Rows ===")
    print(df.head())
    print()

    print("=== Columns ===")
    print(df.columns.tolist())
    print()

    # Keep only needed columns if they exist
    required_columns = ["champion", "win", "kills", "deaths", "assists"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print("Error: Missing required columns:", missing_columns)
        print("Please check your CSV file column names.")
        return

    # Drop missing rows
    df = df.dropna(subset=required_columns)

    # Make sure numeric columns are numeric
    numeric_cols = ["win", "kills", "deaths", "assists"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)

    # Convert win column to integer if possible
    df["win"] = df["win"].astype(int)

    # Calculate KDA
    # Avoid division by zero by replacing 0 deaths with 1
    df["kda"] = (df["kills"] + df["assists"]) / df["deaths"].replace(0, 1)

    # Overall win rate
    overall_win_rate = df["win"].mean()
    print(f"=== Overall Win Rate ===\n{overall_win_rate:.2%}\n")

    # Champion statistics
    champion_stats = (
        df.groupby("champion")
        .agg(
            games_played=("champion", "count"),
            win_rate=("win", "mean"),
            avg_kills=("kills", "mean"),
            avg_deaths=("deaths", "mean"),
            avg_assists=("assists", "mean"),
            avg_kda=("kda", "mean"),
        )
        .sort_values(by="win_rate", ascending=False)
    )

    print("=== Top 10 Champions by Win Rate ===")
    print(champion_stats.head(10))
    print()

    # Save full stats
    champion_stats.to_csv("champion_stats_summary.csv")

    # Filter champions with at least 5 games to reduce noise
    filtered_stats = champion_stats[champion_stats["games_played"] >= 5]

    # Top 10 win rates
    top10_winrate = filtered_stats.sort_values(by="win_rate", ascending=False).head(10)

    # Top 10 KDA
    top10_kda = filtered_stats.sort_values(by="avg_kda", ascending=False).head(10)

    # Plot 1: Top 10 champion win rates
    plt.figure(figsize=(10, 6))
    plt.bar(top10_winrate.index, top10_winrate["win_rate"])
    plt.title("Top 10 Champion Win Rates")
    plt.xlabel("Champion")
    plt.ylabel("Win Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("top10_champion_winrates.png")
    plt.show()

    # Plot 2: Top 10 champion average KDA
    plt.figure(figsize=(10, 6))
    plt.bar(top10_kda.index, top10_kda["avg_kda"])
    plt.title("Top 10 Champion Average KDA")
    plt.xlabel("Champion")
    plt.ylabel("Average KDA")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("top10_champion_kda.png")
    plt.show()

    # Correlation-style insight
    avg_win_by_kda = df.groupby(pd.cut(df["kda"], bins=5))["win"].mean()
    print("=== Win Rate by KDA Range ===")
    print(avg_win_by_kda)
    print()

    print("Analysis complete.")
    print("Generated files:")
    print("- champion_stats_summary.csv")
    print("- top10_champion_winrates.png")
    print("- top10_champion_kda.png")


if __name__ == "__main__":
    main()
