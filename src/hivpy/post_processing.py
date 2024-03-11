import argparse
import math
import os
import pathlib
import traceback

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sas7bdat import SAS7BDAT as s7b
from titlecase import titlecase


def graph_output(output_dir, output_stats, graph_outputs):
    """
    Plot all graph output columns in the output statistics dataframe.
    """
    for out in graph_outputs:
        if out in output_stats.columns:

            plt.plot(output_stats["Date"], output_stats[out])
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
        if out in avg_df.columns:
            plt.plot(avg_df["Date"], avg_df[out], color="teal", label="mean")
            # standard deviation
            ax.fill_between(avg_df["Date"], avg_df[out]+grouped_avg[out].std(),
                            avg_df[out]-grouped_avg[out].std(), color="teal", alpha=0.15)
            # quantiles
            ax.fill_between(avg_df["Date"], grouped_avg[out].quantile(0.25), grouped_avg[out].quantile(0.75),
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
                plt.plot(df["Date"], df[out], linestyle="dashed", alpha=0.35)

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


def compare_early_epidemic_periods(hivpy_avg, sas_avg, output_dir):
    """
    Plot comparison graphs of early epidemic periods between average HIVpy and SAS model runs.
    """
    # rename the SAS columns used for graphing
    sas_avg.rename(columns={"cald": "Date",
                            "s_alive1549": "Population (15-49)",
                            "s_alive1549_m": "Population (15-49, male)",
                            "s_alive1549_w": "Population (15-49, female)",
                            "s_dead_all": "Deaths (tot)",
                            "m15r": "Partner sex balance (15-24, male)",
                            "w15r": "Partner sex balance (15-24, female)",
                            "m25r": "Partner sex balance (25-34, male)",
                            "w25r": "Partner sex balance (25-34, female)",
                            "m35r": "Partner sex balance (35-44, male)",
                            "w35r": "Partner sex balance (35-44, female)"}, inplace=True)

    # format date
    hivpy_avg["Date"] = pd.to_datetime(hivpy_avg["Date"], format="(%Y, %m, %d)")
    sas_avg["Date"] = format_sas_date(sas_avg, "Date")

    start_date = 1989
    end_date = 1996
    # drop all rows not in the early epidemic date range of 1989-1995
    hivpy_avg.drop(hivpy_avg[hivpy_avg["Date"] < pd.to_datetime(start_date, format="%Y")].index, inplace=True)
    hivpy_avg.drop(hivpy_avg[(hivpy_avg["Date"] >= pd.to_datetime(end_date, format="%Y"))].index, inplace=True)
    sas_avg.drop(sas_avg[sas_avg["Date"] < pd.to_datetime(start_date, format="%Y")].index, inplace=True)
    sas_avg.drop(sas_avg[(sas_avg["Date"] >= pd.to_datetime(end_date, format="%Y"))].index, inplace=True)

    # calculate additional SAS outputs
    # total deaths by sex and age bracket
    sas_avg["Deaths (15-24, male)"] = sas_avg["s_dead1519m_all"].values + sas_avg["s_dead2024m_all"].values
    sas_avg["Deaths (15-24, female)"] = sas_avg["s_dead1519w_all"].values + sas_avg["s_dead2024w_all"].values
    sas_avg["Deaths (25-34, male)"] = sas_avg["s_dead2529m_all"].values + sas_avg["s_dead3034m_all"].values
    sas_avg["Deaths (25-34, female)"] = sas_avg["s_dead2529w_all"].values + sas_avg["s_dead3034w_all"].values
    sas_avg["Deaths (35-44, male)"] = sas_avg["s_dead3539m_all"].values + sas_avg["s_dead4044m_all"].values
    sas_avg["Deaths (35-44, female)"] = sas_avg["s_dead3539w_all"].values + sas_avg["s_dead4044w_all"].values
    sas_avg["Deaths (45-54, male)"] = sas_avg["s_dead4549m_all"].values + sas_avg["s_dead5054m_all"].values
    sas_avg["Deaths (45-54, female)"] = sas_avg["s_dead4549w_all"].values + sas_avg["s_dead5054w_all"].values

    # non-HIV deaths by age bracket
    sas_avg["Non-HIV deaths (15-24)"] = (sas_avg["Deaths (15-24, male)"].values +
                                         sas_avg["Deaths (15-24, female)"].values -
                                         sas_avg["s_death_hiv_age_1524"].values)
    sas_avg["Non-HIV deaths (25-34)"] = (sas_avg["Deaths (25-34, male)"].values +
                                         sas_avg["Deaths (25-34, female)"].values -
                                         sas_avg["s_death_hiv_age_2534"].values)
    sas_avg["Non-HIV deaths (35-44)"] = (sas_avg["Deaths (35-44, male)"].values +
                                         sas_avg["Deaths (35-44, female)"].values -
                                         sas_avg["s_death_hiv_age_3544"].values)
    sas_avg["Non-HIV deaths (45-54)"] = (sas_avg["Deaths (45-54, male)"].values +
                                         sas_avg["Deaths (45-54, female)"].values -
                                         sas_avg["s_death_hiv_age_4554"].values)

    # total non-HIV deaths and death rate
    sas_avg["Non-HIV deaths (tot)"] = sas_avg["Deaths (tot)"].values - sas_avg["s_death_hiv"].values
    sas_avg["Non-HIV deaths (ratio)"] = (sas_avg["Non-HIV deaths (tot)"].values /
                                         (sas_avg["s_alive_m"].values + sas_avg["s_alive_w"].values))

    # stp ratios
    sas_avg["At least 1 short term partner (ratio)"] = (sas_avg["s_newp_ge1"].values /
                                                        (sas_avg["s_alive_m"].values +
                                                         sas_avg["s_alive_w"].values))
    sas_avg["Short term partners (15-49, male)"] = ((sas_avg["s_m_1524_newp"].values +
                                                     sas_avg["s_m_2534_newp"].values +
                                                     sas_avg["s_m_3544_newp"].values) /
                                                    sas_avg["Population (15-49, male)"].values)
    sas_avg["Short term partners (15-49, female)"] = ((sas_avg["s_w_1524_newp"].values +
                                                       sas_avg["s_w_2534_newp"].values +
                                                       sas_avg["s_w_3544_newp"].values) /
                                                      sas_avg["Population (15-49, female)"].values)

    # log sex balance
    for balance_col in ["Partner sex balance (15-24, male)", "Partner sex balance (15-24, female)",
                        "Partner sex balance (25-34, male)", "Partner sex balance (25-34, female)",
                        "Partner sex balance (35-44, male)", "Partner sex balance (35-44, female)"]:
        sas_avg[balance_col] = sas_avg[balance_col].map(lambda x: math.log(x, 10) if x > 0 else None)

    # plot population comparison
    compare_output(os.path.join(output_dir, "graph_outputs", "model_comparison"),
                   [hivpy_avg, sas_avg], ["Population (15-49)", "Population (15-49, male)",
                                          "Population (15-49, female)", "Deaths (tot)",
                                          "Non-HIV deaths (15-24)", "Non-HIV deaths (25-34)",
                                          "Non-HIV deaths (35-44)", "Non-HIV deaths (45-54)",
                                          "Non-HIV deaths (tot)", "Non-HIV deaths (ratio)",
                                          "At least 1 short term partner (ratio)",
                                          "Short term partners (15-49, male)",
                                          "Short term partners (15-49, female)",
                                          "Partner sex balance (15-24, male)",
                                          "Partner sex balance (15-24, female)",
                                          "Partner sex balance (25-34, male)",
                                          "Partner sex balance (25-34, female)",
                                          "Partner sex balance (35-44, male)",
                                          "Partner sex balance (35-44, female)"])


def run_post():
    """
    Run post-processing on a HIV model simulation output.
    """
    # argument management
    parser = argparse.ArgumentParser(description="run post-processing")
    parser.add_argument("config", type=pathlib.Path, help="config containing graph outputs")
    parser.add_argument("output_dir", type=pathlib.Path, help="output directory to save graphs")
    parser.add_argument("-hi", "--hivpy_input",
                        type=pathlib.Path, help="input directory with csv files to plot")
    parser.add_argument("-si", "--sas_input",
                        type=pathlib.Path, help="input directory with sas7bdat files to plot")
    parser.add_argument("-eec", "--early_epidemic_comparison", default=False, action="store_true",
                        help="early epidemic comparison flag")

    args = parser.parse_args()
    if not (args.hivpy_input or args.sas_input):
        parser.error("No input provided, please add at least one of hivpy_input or sas_input.")
    if args.early_epidemic_comparison and not (args.hivpy_input and args.sas_input):
        parser.error("Early epidemic comparison requires both hivpy_input and sas_input.")

    try:
        # open config file and find graph output column names
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
        graph_out_columns = config["EXPERIMENT"]["graph_outputs"]

        # create output directory if it doesn't exist, otherwise overwrite existing
        if not os.path.exists(os.path.join(args.output_dir, "graph_outputs", "model_comparison")):
            os.makedirs(os.path.join(args.output_dir, "graph_outputs", "model_comparison"))

        if args.hivpy_input and args.sas_input:
            # get hivpy df
            if os.path.isdir(args.hivpy_input):
                out_files, _ = aggregate_data(args.hivpy_input, args.output_dir)
                hivpy_df = pd.read_csv(out_files[0], index_col=[0])
            elif os.path.isfile(args.hivpy_input):
                hivpy_df = pd.read_csv(args.hivpy_input, index_col=[0])
            # get sas df
            if os.path.isdir(args.sas_input):
                out_files, _ = aggregate_sas_data(args.sas_input, args.output_dir)
                sas_df = pd.read_csv(out_files[0])
            elif os.path.isfile(args.sas_input):
                sas_df = read_s7b(str(args.sas_input))
                # drop first row to remove unneeded nans
                sas_df.drop(index=0, inplace=True)

            # early epidemic comparison
            if args.early_epidemic_comparison:
                compare_early_epidemic_periods(hivpy_df, sas_df, args.output_dir)
            # generic comparison
            else:
                hivpy_df["Date"] = pd.to_datetime(hivpy_df["Date"], format="(%Y, %m, %d)")
                # FIXME: could use a full column name translation function here
                sas_df.rename(columns={"cald": "Date"}, inplace=True)
                sas_df["Date"] = format_sas_date(sas_df, "Date")
                compare_output(os.path.join(args.output_dir, "graph_outputs", "model_comparison"),
                               [hivpy_df, sas_df], graph_out_columns)

        elif args.hivpy_input:
            # graph with averages
            if os.path.isdir(args.hivpy_input):
                # find output csv files in input directory and get grouped data from output files
                out_files, grouped_avg = aggregate_data(args.hivpy_input, args.output_dir)
                # read output files into dataframes
                # FIXME: just get this from aggregate data
                input_dfs = []
                for f in out_files:
                    df = pd.read_csv(f, index_col=[0])
                    df["Date"] = pd.to_datetime(df["Date"], format="(%Y, %m, %d)")
                    input_dfs.append(df)
                # graph aggregate data vs individual runs
                compare_avg_output(os.path.join(args.output_dir, "graph_outputs"), input_dfs,
                                   graph_out_columns, grouped_avg)
            # single-run graph
            elif os.path.isfile(args.hivpy_input):
                hivpy_df = pd.read_csv(args.hivpy_input, index_col=[0])
                hivpy_df["Date"] = pd.to_datetime(hivpy_df["Date"], format="(%Y, %m, %d)")
                graph_output(os.path.join(args.output_dir, "graph_outputs"), hivpy_df, graph_out_columns)

        elif args.sas_input:
            # graph with averages
            if os.path.isdir(args.sas_input):
                # get grouped sas data
                out_files, grouped_avg = aggregate_sas_data(args.sas_input, args.output_dir)
                # read output files into dataframes
                # FIXME: just get this from aggregate data
                input_dfs = []
                for f in out_files:
                    if f == os.path.join(args.output_dir, "aggregate_sas_data.csv"):
                        df = pd.read_csv(f)
                    else:
                        df = read_s7b(f)
                        # drop first row to remove unneeded nans
                        df.drop(index=0, inplace=True)
                    # FIXME: could use a full column name translation function here
                    df.rename(columns={"cald": "Date"}, inplace=True)
                    df["Date"] = format_sas_date(df, "Date")
                    input_dfs.append(df)
                # graph aggregate data vs individual runs
                compare_avg_output(os.path.join(args.output_dir, "graph_outputs"), input_dfs,
                                   graph_out_columns, grouped_avg)
            # single-run graph
            elif os.path.isfile(args.sas_input):
                sas_df = read_s7b(str(args.sas_input))
                # drop first row to remove unneeded nans
                sas_df.drop(index=0, inplace=True)
                # FIXME: could use a full column name translation function here
                sas_df.rename(columns={"cald": "Date"}, inplace=True)
                sas_df["Date"] = format_sas_date(sas_df, "Date")
                graph_output(os.path.join(args.output_dir, "graph_outputs"), sas_df, graph_out_columns)

    except yaml.YAMLError as err:
        print("Error parsing yaml file {}".format(err))
    # FIXME: can this be less generic?
    except Exception as err:
        print(type(err), err)
        print(traceback.format_exc())


if __name__ == "__main__":
    run_post()
