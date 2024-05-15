## Simulation Output Post-processing

Each time a HIVpy model run is completed, some basic post-processing is run at the end to produce graphs of the column contents specified by a `graph_outputs` list under `EXPERIMENT` in the model YAML config file. Any valid column name in `graph_outputs` will be used for plotting.

### Usage

```bash
python src/hivpy/post_processing.py <config> <output_dir> [-hi hipvy_input] [-si sas_input] [-eec]
```

- `config` - Path to configuration file containing names of expected graph output columns.
- `output_dir` - Path to output directory where graph PNGs will be saved.
- `-hi / --hivpy_input <input_dir>` - (Optional.) Path to directory with CSV files or a single file containing data to plot.
- `-si / --sas_input <input_dir>` - (Optional.) Path to directory with SAS7BDAT files or a single file containing data to plot.
- `-eec / --early_epidemic_comparison` - (Optional.) Flag to run an early epidemic comparison including only data from 1989 to 1995.

**Note**: At least one of `-hi` or `-si` must be present in order for a post-processing run to succeed and both must be present if running with the `-eec` flag.

### Plot a Single Simulation Run

Run post-processing with only one of `-hi` or `-si` where the input path is a single file. This will create a separate graph for each column in `graph_outputs` in the output directory under a `graph_outputs/` subdirectory.

### Plot an Average of Multiple Simulation Runs

Run post-processing with only one of `-hi` or `-si` where the input path is a directory containing multiple output files from several model runs. This will create a new CSV file in your output directory containing data averaged across all output files in your input directory.

Two types of graph will be produced for each graph output column: a 'clean' plot containing just the averaged data, and another plot containing the averaged data along with the data for all runs.

### Plot a Comparison Between HIVpy and SAS Simulation Runs

Run post-processing with both `-hi` and `-si` to create comparison graphs including data from both models in the output directory under a `graph_outputs/model_comparison/` subdirectory. Include the `-eec` flag to plot early epidemic comparisons of a few relevant, pre-selected outputs.
