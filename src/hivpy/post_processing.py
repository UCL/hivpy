import argparse
import os
import pathlib
import traceback

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sas7bdat import SAS7BDAT as s7b
from titlecase import titlecase


def graph_output(output_dir, output_stats, graph_outputs, sas_cols=False):
    """
    Plot all graph output columns in the output statistics dataframe.
    """
    for out in graph_outputs:
        if out in output_stats.columns:

            # FIXME: can we assume pre-formatted date columns?
            if sas_cols:
                plt.plot(format_sas_date(output_stats, "cald"), output_stats[out])
            else:
                plt.plot(pd.to_datetime(output_stats["Date"], format="(%Y, %m, %d)"), output_stats[out])
            title_out = titlecase(out)

            plt.xlabel("Date")
            plt.ylabel(title_out)
            plt.title("{0} Over Time".format(title_out))
            plt.savefig(os.path.join(output_dir, "{0} Over Time".format(title_out)))
            plt.close()


def compare_output(output_dir, output_stats, graph_outputs, label1="HIVpy", label2="SAS"):
    """
    Plot all graph output columns of all runs in output statistics on one graph for comparison.
    The first dataframe is expected to contain HIVpy data by default, and the second is expected
    to contain SAS data by default.
    """
    for out in graph_outputs:

        _, ax = plt.subplots()

        for i in range(len(output_stats)):
            df = output_stats[i]
            if out in df.columns:

                # assumes the date columns have already been pre-formatted
                if i == 0:
                    plt.plot(df["Date"], df[out], label=label1)
                elif i == 1:
                    plt.plot(df["Date"], df[out], label=label2)
                else:
                    # plot additional files without label for now
                    plt.plot(df["Date"], df[out])

        title_out = titlecase(out)
        plt.xlabel("Date")
        plt.ylabel(title_out)
        plt.title("Comparison of {0} Over Time".format(title_out))
        ax.legend()
        plt.savefig(os.path.join(output_dir, "Comparison of {0} Over Time".format(title_out)), bbox_inches='tight')
        plt.close()


def compare_avg_output(output_dir, output_stats, graph_outputs, grouped_avg):
    """
    Plot the averaged data for each graph output (along with standard deviation and 25th-75th percentiles).
    Plot all graph output columns of all runs in output statistics on one graph for comparison to the averages.
    """
    # first df is assumed to contain the average statistics
    avg_df = output_stats[0]
    for out in graph_outputs:

        # FIXME: get relevant information out of grouped_avg once grouping by dates
        # save a clean copy of the average values
        _, ax = plt.subplots()
        # FIXME: may want to assume pre-formatted date columns
        plt.plot(pd.to_datetime(avg_df["Date"], format="(%Y, %m, %d)"), avg_df[out],
                 color="teal", label="mean")
        # standard deviation
        ax.fill_between(pd.to_datetime(avg_df["Date"], format="(%Y, %m, %d)"),
                        avg_df[out]+grouped_avg[out].std(), avg_df[out]-grouped_avg[out].std(),
                        color="teal", alpha=0.15)
        # quantiles
        ax.fill_between(pd.to_datetime(avg_df["Date"], format="(%Y, %m, %d)"),
                        grouped_avg[out].quantile(0.25), grouped_avg[out].quantile(0.75),
                        color="darkslategrey", alpha=0.15)

        title_out = titlecase(out)
        plt.xlabel("Date")
        plt.ylabel(title_out)
        plt.title("Clean Comparison of {0} Over Time".format(title_out))
        ax.legend()
        plt.savefig(os.path.join(output_dir, "Clean Comparison of {0} Over Time".format(title_out)),
                    bbox_inches='tight')

        # now save a 'dirty' copy including all the other runs
        for df in output_stats[1:]:
            if out in df.columns:
                plt.plot(pd.to_datetime(df["Date"], format="(%Y, %m, %d)"), df[out],
                         linestyle="dashed", alpha=0.35)

        title_out = titlecase(out)
        plt.xlabel("Date")
        plt.ylabel(title_out)
        plt.title("Comparison of {0} Over Time".format(title_out))
        ax.legend()
        plt.savefig(os.path.join(output_dir, "Comparison of {0} Over Time".format(title_out)), bbox_inches='tight')
        plt.close()


def read_s7b(path: str):
    """
    Return a dataframe from the path to a sas7bdat file.
    """
    try:
        with s7b(path) as reader:
            sas_df = reader.to_data_frame()
    except Exception as err:
        print(type(err), err)
        print(traceback.format_exc())
    return sas_df


def format_sas_date(df, date_col):
    """
    Return a formatted datetime column from the SAS cald float without modifying the original.
    """
    sas_date = df[date_col].copy().map("{:.2f}".format)
    sas_date = sas_date.map(lambda x: "{0}.{1}".format(x.split(".")[0], int(int(x.split(".")[1])*0.12)+1))
    return pd.to_datetime(sas_date, format="%Y.%m")


def aggregate_data(in_path, out_path):
    """
    Find all csv output files in the input path directory and create a new output file
    in the output path directory containing the average values of the provided files.
    """
    # FIXME: this file could probably use a better name
    avg_path = os.path.join(out_path, "aggregate_data.csv")
    # delete aggregate data if it already exists
    if os.path.exists(avg_path):
        os.remove(avg_path)
    if os.path.exists(os.path.join(out_path, "aggregate_sas_data.csv")):
        os.remove(os.path.join(out_path, "aggregate_sas_data.csv"))

    out_files = []
    idx_group = None
    if os.path.isdir(in_path):

        # look for output files in folder
        for file in [os.path.join(root, file) for root, _, files
                     in os.walk(in_path) for file in files]:
            # append files with csv extension
            if os.path.splitext(file)[1] == ".csv":
                out_files.append(file)
        # no eligible output files in folder
        if len(out_files) == 0:
            raise RuntimeError(
                "No csv output files found in given directory.")
        print("Found {0} csv output files.".format(len(out_files)))

        # concatenate all output file dataframes
        avg = pd.read_csv(out_files[0], index_col=[0])
        for out in out_files[1:]:
            df = pd.read_csv(out, index_col=[0])
            avg = pd.concat((avg, df))

        # FIXME: try to groupby date instead
        idx_group = avg.groupby(avg.index)
        # FIXME: consider use of mean vs median
        df_means = idx_group.mean(numeric_only=True)
        # re-insert date column
        # NOTE: this assumes dates of all rows are the same
        df_means.insert(0, "Date", pd.read_csv(out_files[0], index_col=[0])["Date"])

        # save aggregate data to file
        df_means.to_csv(avg_path, index=True, sep=",")
        out_files.insert(0, avg_path)

    return out_files, idx_group


# FIXME: merge this function with the one above and make them more generic
def aggregate_sas_data(in_path, out_path):
    """
    Find all sas7bdat output files in the input path directory and create a new output file
    in the output path directory containing the average values of the provided files.
    """
    avg_path = os.path.join(out_path, "aggregate_sas_data.csv")
    if os.path.exists(avg_path):
        os.remove(avg_path)

    out_files = []
    idx_group = None
    if os.path.isdir(in_path):

        # look for output files in folder
        for file in [os.path.join(root, file) for root, _, files
                     in os.walk(in_path) for file in files]:
            # append files with sas7bdat extension
            if os.path.splitext(file)[1] == ".sas7bdat":
                out_files.append(file)
        # no eligible output files in folder
        if len(out_files) == 0:
            raise RuntimeError(
                "No sas7bdat output files found in given directory.")
        print("Found {0} sas7bdat output files.".format(len(out_files)))

        # concatenate all output file dataframes
        avg = read_s7b(out_files[0])
        # drop first row to remove unneeded nans
        avg.drop(index=0, inplace=True)
        for out in out_files[1:]:
            df = read_s7b(out)
            # drop first row to remove unneeded nans
            df.drop(index=0, inplace=True)
            if int(df["inc_cat"][1]) == 1:
                avg = pd.concat((avg, df))

        # FIXME: try to groupby date instead
        idx_group = avg.groupby(avg.index)
        # FIXME: consider use of mean vs median
        df_means = idx_group.mean(numeric_only=True)

        # save aggregate data to file
        df_means.to_csv(avg_path, index=True, sep=",")
        out_files.insert(0, avg_path)

    return out_files, idx_group


def run_post():
    """
    Run post-processing on a HIV model simulation output.
    """
    # argument management
    parser = argparse.ArgumentParser(description="run post-processing")
    parser.add_argument("config", type=pathlib.Path, help="config containing graph outputs")
    parser.add_argument("input_dir", type=pathlib.Path, help="input directory with csv files to plot")
    parser.add_argument("output_dir", type=pathlib.Path, help="output directory to save graphs")
    args = parser.parse_args()

    try:
        # open config file and find graph output column names
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
        graph_out_columns = config["EXPERIMENT"]["graph_outputs"]

        # create output directory if it doesn't exist, otherwise overwrite existing
        if not os.path.exists(args.output_dir):
            os.makedirs(os.path.join(args.output_dir, "graph_outputs"))
        # find output csv files in input directory and get grouped data from output files
        out_files, grouped_avg = aggregate_data(args.input_dir, args.output_dir)

        # read output files into dataframes
        input_dfs = []
        for f in out_files:
            input_dfs.append(pd.read_csv(f, index_col=[0]))
        # graph aggregate data vs individual runs
        compare_avg_output(os.path.join(args.output_dir, "graph_outputs"), input_dfs, graph_out_columns, grouped_avg)

    except yaml.YAMLError as err:
        print("Error parsing yaml file {}".format(err))
    # FIXME: can this be less generic?
    except Exception as err:
        print(type(err), err)
        print(traceback.format_exc())


if __name__ == "__main__":
    run_post()
