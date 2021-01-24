# -*- coding: utf-8 -*-
"""
Created on Fri Nov 06 16:53:39 2015

@author: Brian
"""


import copy
import os
import shutil
import parameterorg

# ========================================================================


def get_environment_directory_path(
        experiment_directory_path,
        environment_name):
    return os.path.join(experiment_directory_path, environment_name)

# ========================================================================


def determine_environment_name_and_dir(
    repeat_number, experiment_directory, template_experiment_name_format_string
):
    environment_name = template_experiment_name_format_string.format(
        repeat_number)
    environment_dir = os.path.join(experiment_directory, environment_name)

    return environment_name, environment_dir


def create_environment(
        justify_parameters,
    environment_wide_variable_defns,
    user_cell_group_defns,
):
    an_environment = parameterorg.make_environment_given_user_cell_group_defns(
        user_cell_group_defns=user_cell_group_defns,
        justify_parameters=justify_parameters,
        **environment_wide_variable_defns
    )

    return an_environment


def run_template_experiments(
        environment_wide_variable_defns,
    user_cell_group_defns_per_subexperiment,
        num_experiment_repeats=1,
    justify_parameters=True,
):
    template_experiment_name_format_string = "RPT={}"
    for repeat_number in range(num_experiment_repeats):
        for subexperiment_index, user_cell_group_defns in enumerate(
            copy.deepcopy(user_cell_group_defns_per_subexperiment)
        ):
            an_environment = create_environment(
                justify_parameters,
                environment_wide_variable_defns,
                user_cell_group_defns,
            )

            an_environment.execute_system_dynamics()
