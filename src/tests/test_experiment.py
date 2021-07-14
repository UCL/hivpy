from hivpy.experiment import create_experiment_from_file, run_experiment


def test_dummy_workflow():
    """Check that we can run a sample experiment from start to end.
    
    This is only a placeholder test and should be replaced when we have
    implemented actual functionality.
    """
    dummy_config = create_experiment_from_file('myconfig.yaml')
    run_experiment(dummy_config)
