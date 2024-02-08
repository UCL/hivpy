## Simulation Output Post-processing

Each time a HIVpy model run is completed, some basic post-processing is run at the end to produce graphs of the column contents specified by a `graph_outputs` list under `EXPERIMENT` in the model YAML config file. Any valid column name in `graph_outputs` will be used for plotting.

### Create an Average of Multiple Simulation Runs

Once you have multiple CSV output files from multiple model runs, ensure they are all in one directory. This will be your input directory for data aggregation.

Next, run post-processing from the terminal with the following command:
```bash
python <path>/<to>/post_processing.py <configuration file> <input directory> <output directory>
```
This will produce a new CSV file in your output directory containing data averaged across all CSV output files in your input directory.

Two types of graph will be produced for each graph output column: a 'clean' plot containing just the averaged data, and another plot containing the averaged data along with the data for all runs.
