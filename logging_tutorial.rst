Logging tutorial
================

While running the model, it is sometimes useful to monitor the progress
of certain characteristics across time. This can be used, for instance,
when debugging or to verify that a recent change has had the desired behaviour.

This can be achieved by modifying the ``LOGGING`` section of the configuration
that is passed to the simulation tool.

Logging configuration
---------------------
**TODO:** Explain the basic options (directory, prefix, level)

Specifying what to log
----------------------
By default, no variables are logged. To control which variables will be included,
list them under the ``column`` key in the logging configuration.

For example, the following configuration will print the values of the ``age``
and ``sex`` columns for the whole population at every time step:

.. code-block:: yaml
   :caption: Basic logging configuration

    LOGGING:
        log_directory: log
        logfile_prefix: hivpy
        log_file_level: DEBUG
        console_log_level: WARNING
        columns: [age, sex]

If any of the columns are not recognised (because of a misspelling, for example),
the log will display a warning at the start of the simulation, but the simulation
will run as normal (with that column name ignored in the log output).

Selecting a subpopulation
-------------------------
Instead of showing values for the whole population, you can select a subset to monitor.
This can be done with the ``if`` key and by providing the criteria for selecting
the subset of interest.

For example, this configuration will only log values for female individuals:

.. code-block:: yaml
   :caption: Selecting by a single variable

    LOGGING:
        log_directory: log
        logfile_prefix: hivpy
        log_file_level: DEBUG
        console_log_level: WARNING
        columns: [age, sex]
        if:
            sex: female


Multiple variables can be specified at the same time by providing more keys to ``if``.
For instance, to select male, HIV- individuals:

.. code-block:: yaml
   :caption: Selecting by multiple variable

    LOGGING:
        log_directory: log
        logfile_prefix: hivpy
        log_file_level: DEBUG
        console_log_level: WARNING
        columns: [age, sex]
        if:
            sex: male
            hiv: False


The examples so far have been matching on exact values (such as ``female`` or ``False``),
but often we want to match a range of values, particularly for numerical variables.
To do this, use one of these special keys:

* ``eq``: equal to the value provided
* ``lt``: less than the value provided
* ``gt``: greater than the value provided
* ``lte``, ``gte``: as above but also allowing equality


The configuration below selects female, HIV+ individuals between 23 and 30 years old
(both inclusive):

.. code-block:: yaml
   :caption: More complex criteria

    LOGGING:
        # (the other configuration keys are omitted for brevity) 
        if:
            sex: female
            hiv: True
            age:
                gte: 23
                lte: 30

