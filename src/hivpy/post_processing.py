import argparse
import os
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from titlecase import titlecase


def graph_output(output_dir, output_stats, graph_outputs):

    for out in graph_outputs:
        if out in output_stats.columns:

            plt.plot(output_stats["Date"], output_stats[out])
            title_out = titlecase(out)

            plt.xlabel("Date")
            plt.ylabel(title_out)
            plt.title("{0} Over Time".format(title_out))
            plt.savefig(os.path.join(output_dir, "{0} Over Time".format(title_out)))
            plt.close()


def compare_output(output_dir, df1, df2, graph_outputs, label1="HIVpy", label2="SAS"):

    for out in graph_outputs:
        if out in df1.columns and out in df2.columns:

            _, ax = plt.subplots()
            plt.plot(df1["Date"], df1[out], label=label1)
            plt.plot(df2["Date"], df2[out], label=label2)

            title_out = titlecase(out)
            plt.xlabel("Date")
            plt.ylabel(title_out)
            plt.title("Comparison of {0} Over Time".format(title_out))
            ax.legend()
            plt.savefig(os.path.join(output_dir, "Comparison of {0} Over Time".format(title_out)), bbox_inches='tight')
            plt.close()


def run_post():
    """
    Run post-processing on a HIV model simulation output.
    """
    parser = argparse.ArgumentParser(description="run post-processing")
    parser.add_argument("config", type=pathlib.Path, help="config containing graph outputs")
    parser.add_argument("output_dir", type=pathlib.Path, help="output directory to save graphs")
    parser.add_argument("input", nargs="+", type=pathlib.Path, help="csv to plot")
    args = parser.parse_args()

    try:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
        graph_outputs = config["EXPERIMENT"]["graph_outputs"]

        input = []
        for file in args.input:
            input.append(pd.read_csv(file))

        if len(input) == 1:
            print("graphing outputs")
            graph_output(args.output_dir, input[0], graph_outputs)
        if len(input) == 2:
            print("comparing outputs")
            compare_output(args.output_dir, input[0], input[1], graph_outputs)

    except yaml.YAMLError as err:
        print("Error parsing yaml file {}".format(err))
    # FIXME
    except Exception as err:
        print(type(err), err)


if __name__ == "__main__":
    run_post()
