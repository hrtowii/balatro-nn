import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_scalars_csv(csv_path, image_path):
    df = pd.read_csv(csv_path, index_col=0)
    n = len(df.columns)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3*n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, df.columns):
        ax.plot(df.index, df[col], label=col)
        ax.set_ylabel(col)
        ax.grid(True)
        ax.legend(loc="best")
    axes[-1].set_xlabel("step")
    plt.tight_layout()
    plt.savefig(image_path)
    print(f"Saved plot to {image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="scalars.csv", help="CSV file to plot")
    parser.add_argument("--image", default="image.png", help="Output image file")
    args = parser.parse_args()
    plot_scalars_csv(args.csv, args.image)
