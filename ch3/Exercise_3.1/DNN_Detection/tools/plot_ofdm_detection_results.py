from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def label_for_row(row):
    method = str(row["method"])
    mode = str(row.get("mode", ""))
    modulation = str(row.get("modulation", ""))
    pilots = int(row["pilots"])
    if method in {"ls", "lmmse"}:
        return f"{method.upper()}, P={pilots}"
    if mode == "single":
        return f"Single FC-DNN, {modulation.upper()}, P={pilots}"
    if mode == "eight":
        return f"8 FC-DNNs, {modulation.upper()}, P={pilots}"
    return f"{method}, {mode}, {modulation}, P={pilots}"


def plot_csv(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    required = {"snr_db", "ber", "method", "pilots"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"{csv_path} is missing columns: {sorted(missing)}")

    task_name = csv_path.stem.replace("results_", "")
    df = df.copy()
    df["snr_db"] = pd.to_numeric(df["snr_db"], errors="coerce")
    df["ber"] = pd.to_numeric(df["ber"], errors="coerce")
    df = df.dropna(subset=["snr_db", "ber"])

    dedup_cols = [c for c in ["task", "method", "modulation", "mode", "pilots", "snr_db"] if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols, keep="last")

    if df.empty:
        raise SystemExit(f"{csv_path} has no plottable rows after filtering.")

    plt.figure(figsize=(8.0, 5.2))

    group_cols = ["method", "mode", "modulation", "pilots"]
    available_group_cols = [c for c in group_cols if c in df.columns]

    for _, group in df.groupby(available_group_cols, dropna=False):
        group = group.sort_values("snr_db")
        first = group.iloc[0]
        label = label_for_row(first)
        plt.semilogy(
            group["snr_db"].to_numpy(),
            group["ber"].to_numpy(),
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=label,
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(f"OFDM Signal Detection BER - {task_name}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend(fontsize=8)
    plt.tight_layout()

    png_path = out_dir / f"ber_{task_name}.png"
    pdf_path = out_dir / f"ber_{task_name}.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="../outputs/ofdm_detection",
        help="Directory containing results_*.csv files.",
    )
    parser.add_argument(
        "--pattern",
        default="results_*.csv",
        help="CSV filename glob pattern.",
    )
    args = parser.parse_args()

    base = Path(args.output_dir)
    out_dir = base / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(base.glob(args.pattern))
    if not csv_files:
        raise SystemExit(f"No CSV files matched: {base / args.pattern}")

    for csv_path in csv_files:
        plot_csv(csv_path, out_dir)


if __name__ == "__main__":
    main()
