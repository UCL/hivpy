## Setup and Package Installation with PyCharm

The following steps outline how to set up a new, clean installation of the HIVpy package. Instruction specifics are based on the assumption that PyCharm v2023 is used.

1. Open PyCharm and create a new project by cloning this repository.
    - Select **Git** > **Clone** and enter https://github.com/UCL/hivpy.git as the URL.
        - If **Git** is not in the menu, select **VCS** > **Get from Version Control** instead.
    - Change the directory to be wherever you'd like to keep the files.

    For further details, see the official PyCharm [tutorial](https://www.jetbrains.com/help/pycharm/manage-projects-hosted-on-github.html) for cloning from GitHub.

2. Find and open the cloned project if it does not open automatically. Then, look for the `src` folder and ensure that it has been marked as source.
    - The folder icon should be blue and hovering your mouse over it will display the text **Sources root**.
        - If this is not the case, right-click the folder and select **Mark Directory as** > **Sources Root**.

3. Create a new virtual environment.
    - Select **File** > **Settings** > **Project: hivpy** > **Python Interpreter** > **Add Interpreter** > **Add Local Interpreter**.
    - The default interpreter settings do not need to be changed, but do ensure that Python 3.7 or newer is being used as the base interpreter for your environment.

    For further details, see the official PyCharm [tutorial](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html) for configuring a virtual environment.

4. Activate the virtual environment and download all necessary libraries.
    - Your can check that your environment is running by finding the name of your environment in brackets in front of your current directory in the PyCharm terminal.
        - If your terminal is not already open, select **View** > **Tool Windows** > **Terminal**.
        - If the environment does not activate automatically, run `venv\Scripts\activate` as a PyCharm terminal command, replacing `venv` with the name of your environment.
    - Once the environment is active, run `pip install -e .[dev]`.

5. Switch to a branch you'd like to work on or test.
    - Select **Git** > **Branches**. Click the branch you'd like to switch to and select **Checkout**.
        - If you would like to switch to a different branch but have uncommitted work, run `git stash` as a PyCharm terminal command.
        - Once you switch back to the branch your stashed work originally belonged to, run `git stash pop`.
        - For further details, see this git stash [tutorial](https://www.atlassian.com/git/tutorials/saving-changes/git-stash).

6. Run `run_model hivpy.yaml` as a PyCharm terminal command to run the model given the default configuration settings.

Note: If your virtual environment does not automatically start up each time you open the cloned HIVpy repository, you must manually activate it at the start of each session during which you run the model.
