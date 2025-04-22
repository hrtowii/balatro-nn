#!/usr/bin/env python3
# pip install tensorboard pandas matplotlib
import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_scalars(run_dir, size_guidance=None):
    """
    Given a run directory (containing events.out.tfevents.*),
    return a pandas DataFrame indexed by `step` and columns=tag names.
    """
    # find the events file
    event_files = glob.glob(os.path.join(run_dir, "dqn_balatro_*/events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"no events files found in {run_dir}")

    # size_guidance: how many events to store (None means default)
    if size_guidance is None:
        size_guidance = {
            "scalars": 0,    # 0 == load all
            "histograms": 0,
            "images": 0,
            "audio": 0,
            "graph": 0,
            "meta_graph": 0,
            "graph_def": 0
        }

    # build the accumulator
    dfs = []
    for event_file in event_files:
        ea = EventAccumulator(event_file, size_guidance=size_guidance)
        print("Loading events from", run_dir, "this may take a second...")
        ea.Reload()
        # list of all scalar tags
        tags = ea.Tags().get("scalars", [])
        if not tags:
            raise ValueError("no scalar tags found in the event file")

        print("Found scalar tags:", tags)

        # build one DataFrame per tag
        df = pd.DataFrame()
        for tag in tags:
            events = ea.Scalars(tag)  # list of ScalarEvent(step, value, wall_time)
            steps  = [e.step for e in events]
            vals   = [e.value for e in events]
            tmp    = pd.DataFrame({ tag: vals, "step": steps })
            tmp    = tmp.drop_duplicates("step").set_index("step")
            if df.empty:
                df = tmp
            else:
                df = df.join(tmp, how="outer")
        # sort by step and forward‐fill any missing values
        df = df.sort_index().ffill()
        dfs.append(df)
        df_all = pd.concat(dfs)
        df_all = df_all.groupby(df_all.index).first().sort_index().ffill()
        return df_all

def main():
    p = argparse.ArgumentParser(
        description="Read all scalars out of a TensorBoard run and dump to CSV/plot."
    )
    p.add_argument(
        "run_dir",
        help="Path to a single TensorBoard run directory (contains events.out.*)."
    )
    p.add_argument(
        "--csv",
        help="CSV file to write the scalars to",
        default="scalars.csv"
    )
    p.add_argument(
        "--plot",
        help="If given, plot each series in a separate subplot",
        action="store_true"
    )
    args = p.parse_args()

    # load
    df = load_scalars(args.run_dir)

    # write CSV
    df.to_csv(args.csv)
    print(f"All scalars written to {args.csv}")

    # optional plotting
    if args.plot:
        n = len(df.columns)
        fig, axes = plt.subplots(n, 1, figsize=(8, 3*n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, df.columns):
            ax.plot(df.index, df[col], label=col)
            ax.set_ylabel(col)
            ax.grid(True)
            ax.legend(loc="best")
        axes[-1].set_xlabel("step")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
