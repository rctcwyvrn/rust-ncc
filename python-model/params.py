# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 22:10:15 2015

@author: brian
"""

import copy

import numpy as np

import environment
import geometry

output_mech_labels = [
    "x",
    "y",
    "edge_lengths",
    "sum_forces_x",
    "sum_forces_y",
    "edge_forces_plus_x",
    "edge_forces_plus_y",
    "edge_forces_minus_x",
    "edge_forces_minus_y",
    "rgtp_forces_x",
    "rgtp_forces_y",
    "cyto_forces_x",
    "cyto_forces_y",
    "local_strains",
    "uiv_x",
    "uiv_y",
]

output_chem_labels = [
    "rac_acts",
    "rac_inacts",
    "rac_cyto",
    "rho_act",
    "rho_inacts",
    "rho_cyto",
    "x_coa",
    "x_cil",
    "k_mem_on",
    "kdgdi_rho",
    "kgtp_rac",
    "kgtp_rho",
    "kdgtp_rac",
    "kdgtp_rho",
    "rac_rands",
]

output_info_labels = output_mech_labels + output_chem_labels

for index, label in enumerate(output_info_labels):
    exec("{}_ix = {}".format(label, repr(index)))

num_info_labels = len(output_mech_labels + output_chem_labels)

info_indices_dict = {
    x: i for i, x in enumerate(output_mech_labels + output_chem_labels)
}

g_rate_labels = ["kgtp", "kdgtp", "kgdi", "kdgdi"]

g_var_labels = [
    "exponent",
    "threshold",
    "diffusion",
    "eta",
    "length",
    "stiffness",
    "force",
    "factor",
]


# -----------------------------------------------------------------
rho_gtpase_parameter_labels = [
    "tot_rac",
    "tot_rho",
    "init_act_rgtp",
    "init_inact_rgtp",
    "diffusion_rgtp",
    "kgtp_rac",
    "kdgtp_rac",
    "kgtp_rho",
    "kdgtp_rho",
    "kgtp_rac_auto",
    "kgtp_rho_auto",
    "kdgtp_rho_on_rac",
    "kdgtp_rac_on_rho",
    "k_mem_off",
    "k_mem_on",
    "kdgdi_rho",
    "threshold_rac_activity",
    "threshold_rho_activity",
    "halfmax_tension_inhib",
    "tension_inhib",
]

mech_parameter_labels = [
    "rest_edge_len",
    "rest_area",
    "stiffness_edge",
    "stiffness_cyto",
    "eta",
    "max_protrusive_velocity",
    "max_force_rac",
    "max_force_rho",
    "threshold_force_rac_activity",
    "threshold_force_rho_activity",
    "force_adh_const",
]

interaction_parameter_labels = [
    "cil",
    "coa",
    "close_criterion",
    "los_factor",
    "coa_sensing_dist_at_value",
]

randomization_parameter_labels = [
    "randomization_scheme",
    "randomization_time_mean",
    "randomization_time_variance_factor",
    "randomization_magnitude",
    "randomization_node_percentage",
    "randomization_type",
]

model_run_parameter_labels = [
    "nverts",
    "rgtp_distrib_def",
    "init_verts",
    "cell_r",
]

all_parameter_labels = (
    rho_gtpase_parameter_labels
    + mech_parameter_labels
    + interaction_parameter_labels
    + randomization_parameter_labels
    + model_run_parameter_labels
)


# -----------------------------------------------------------------

polygon_model_parameters = {"cell_r": 2e-05, "nverts": None}

user_rho_gtpase_biochemistry_parameters = {
    "kdgdi_multiplier": [1, 2],
    "init_act_rgtp": [0, 1],
    "threshold_rho_activity_multiplier": [0.01, 1],
    "kgtp_rac_autoact_multiplier": [1, 1000],
    "tot_rac": [2e6, 3e6],
    "kdgtp_rho_multiplier": [1, 2000],
    "coa_sensing_dist_at_value": 0.00011,
    "halfmax_tension_inhib": [0.01, 0.99],
    "tension_inhib": [1.0, 100.0],
    "init_cyto_rgtp": [0, 1],
    "hill_exponent": 3,
    "kgtp_rac_multiplier": [1, 500],
    "tot_rho": [0.5e6, 1.5e6],
    "diffusion_rgtp": [2e-14, 4.5e-13],
    "kdgtp_rac_multiplier": [1, 2000],
    "kgtp_rho_multiplier": [1, 500],
    "kgdi_multiplier": [1, 2],
    "kgtp_rho_autoact_multiplier": [1, 1000],
    "init_inact_rgtp": [0, 1],
    "kgtp_rac_on_rho": [1, 2000],
    "kdgtp_rho_on_rac": [1, 2000],
    "threshold_rac_activity_multiplier": [0.01, 1],
}


user_interaction_parameters = {
    "x_cils": None,
    "interaction_factors_coa_per_celltype": None,
    "close_criterion": [0.25e-6 ** 2, 5e-6 ** 2],
    "los_factor": [0.0, 1000.0],
}

user_mechanical_parameters = {
    "stiffness_cyto": None,
    "char_l3d": 1e-05,
    "max_force_rac": [0.001 * 10e3, 5 * 10e3],
    "force_adh_const": [0, 100],
    "stiffness_edge": [1, 8000],
    "force_rho_multiplier": [0, 1],
    "eta": [0.01 * 1e5, 1e100],
}

user_randomization_parameters = {
    "randomization_magnitude": None,
    "randomization_scheme": None,
    "randomization_time_mean": None,
    "randomization_time_variance_factor": None,
    "randomization_node_percentage": [0.01, 0.5],
    "randomization_type": None,
}

user_model_run_parameters = {
    "rgtp_distrib_def": None,
    "nverts": [3, 100],
}

all_user_parameters_with_justifications = {}
for params in [
    polygon_model_parameters,
    user_rho_gtpase_biochemistry_parameters,
    user_interaction_parameters,
    user_mechanical_parameters,
    user_randomization_parameters,
    user_model_run_parameters,
]:
    all_user_parameters_with_justifications.update(params)

# -----------------------------------------------------------------


def verify_user_parameters(justify_parameters, user_params):
    global all_user_parameters_with_justifications

    for key in list(user_params.keys()):
        try:
            justification = all_user_parameters_with_justifications[key]
        except BaseException:
            raise Exception("Unknown parameter given: {}".format(key))

        if justify_parameters and justification is not None:
            value = user_params[key]

            if isinstance(justification, list):
                assert len(justification) == 2
                if not (justification[0] <= value <= justification[1]):
                    raise Exception(
                        "Parameter {} violates justification ({}) with value {}".format(
                            key, justification, value))
            elif value != justification:
                raise Exception(
                    "Parameter {} violates justification ({}) with value {}".format(
                        key, justification, value))


# -----------------------------------------------------------------


def make_cell_group_params(justify_parameters, user_params):
    verify_user_parameters(justify_parameters, user_params)
    cell_params = {}

    kgtp = 1e-4  # per second
    kdgtp = 1e-4  # per second

    cell_params["nverts"] = user_params["nverts"]
    tot_rac, tot_rho = user_params["tot_rac"], user_params["tot_rho"]

    cell_params["tot_rac"] = tot_rac
    cell_params["tot_rho"] = tot_rho

    assert (
        user_params["init_cyto_rgtp"]
        + user_params["init_inact_rgtp"]
        + user_params["init_act_rgtp"]
        == 1
    )

    cell_params["init_cyto_rgtp"] = user_params[
        "init_cyto_rgtp"
    ]
    cell_params["init_inact_rgtp"] = user_params[
        "init_inact_rgtp"
    ]
    cell_params["init_act_rgtp"] = user_params[
        "init_act_rgtp"
    ]

    # --------------
    cell_params["kgtp_rac"] = (
        kgtp_rac_unmodified * user_params["kgtp_rac_multiplier"]
    )  # per second
    cell_params["kdgtp_rac"] = (
        kdgtp_rac_unmodified * user_params["kdgtp_rac_multiplier"]
    )  # per second
    # --------------
    cell_params["kgtp_rho"] = (
        kgtp_rho_unmodified * user_params["kgtp_rho_multiplier"]
    )  # per second
    cell_params["kdgtp_rho"] = (
        kdgtp_rho_unmodified * user_params["kdgtp_rho_multiplier"]
    )  # per second
    # --------------
    cell_params["kgtp_rac_auto"] = (
        user_params["kgtp_rac_autoact_multiplier"] *
        kgtp_rac_unmodified)  # per second
    cell_params["kgtp_rho_auto"] = (
        user_params["kgtp_rho_autoact_multiplier"] *
        kgtp_rho_unmodified)  # per second
    # --------------
    cell_params["kdgtp_rho_on_rac"] = (
        kdgtp_rac_unmodified
        * user_params["kdgtp_rho_on_rac"]
    )  # per second
    cell_params["kdgtp_rac_on_rho"] = (
        kdgtp_rho_unmodified
        * user_params["kgtp_rac_on_rho"]
    )  # per second
    # --------------
    kgdi = 0.15
    kdgdi = 0.02
    cell_params["k_mem_off"] = (
        kgdi * user_params["kgdi_multiplier"]
    )  # per second
    cell_params["k_mem_on"] = (
        kdgdi * user_params["kdgdi_multiplier"]
    )  # per second
    # --------------
    cell_params["threshold_rac_activity"] = (
        user_params["threshold_rac_activity_multiplier"] * tot_rac
    )
    cell_params["threshold_rho_activity"] = (
        user_params["threshold_rho_activity_multiplier"] * tot_rho
    )
    # --------------
    cell_params["diffusion_rgtp"] = user_params[
        "diffusion_rgtp"
    ]
    # --------------
    cell_params["hill_exponent"] = user_params["hill_exponent"]
    cell_params[
        "halfmax_tension_inhib"
    ] = user_params["halfmax_tension_inhib"]
    cell_params[
        "tension_inhib"
    ] = user_params["tension_inhib"]
    cell_params["coa_sensing_dist_at_value"] = user_params[
        "coa_sensing_dist_at_value"
    ]
    cell_params["los_factor"] = user_params[
        "los_factor"
    ]
    nverts, cell_r = (
        user_params["nverts"],
        user_params["cell_r"],
    )
    cell_params["nverts"], cell_params["cell_r"] = (
        nverts, cell_r, )

    cell_node_thetas = np.pi * np.linspace(0, 2, endpoint=False, num=nverts)
    cell_verts = np.transpose(
        np.array(
            [
                cell_r * np.cos(cell_node_thetas),
                cell_r * np.sin(cell_node_thetas),
            ]
        )
    )
    edge_vectors = geometry.calculate_edge_vectors(cell_verts)
    edge_lengths = geometry.calculate_2D_vector_mags(edge_vectors)

    rest_edge_len = np.average(edge_lengths)

    char_l3d = user_params["char_l3d"]
    cell_params["eta"] = user_params["eta"] * \
        char_l3d
    cell_params["stiffness_edge"] = (
        user_params["stiffness_edge"] * char_l3d
    )
    cell_params["stiffness_cyto"] = user_params[
        "stiffness_cyto"
    ]
    cell_params["rest_edge_len"] = rest_edge_len
    cell_params["max_force_rac"] = (
        user_params["max_force_rac"] * rest_edge_len * 200e-9
    )
    cell_params["max_force_rho"] = (
        user_params["force_rho_multiplier"]
        * cell_params["max_force_rac"]
    )
    cell_params["max_protrusive_velocity"] = (
        cell_params["max_force_rac"] / cell_params["eta"]
    )
    cell_params["threshold_force_rac_activity"] = (
        user_params["threshold_rac_activity_multiplier"] * tot_rac
    )
    cell_params["threshold_force_rho_activity"] = (
        user_params["threshold_rho_activity_multiplier"] * tot_rho
    )
    cell_params["force_adh_const"] = user_params["force_adh_const"]
    # --------------
    cell_params["close_criterion"] = user_params[
        "close_criterion"
    ]
    # --------------
    randomization_scheme = user_params["randomization_scheme"]
    cell_params["randomization_scheme"] = randomization_scheme

    cell_params["randomization_time_mean"] = user_params[
        "randomization_time_mean"
    ]
    cell_params["randomization_time_variance_factor"] = user_params[
        "randomization_time_variance_factor"
    ]
    cell_params["randomization_magnitude"] = user_params[
        "randomization_magnitude"
    ]
    cell_params["randomization_node_percentage"] = user_params[
        "randomization_node_percentage"
    ]
    cell_params["randomization_type"] = user_params[
        "randomization_type"
    ]

    return cell_params


# ==============================================================


def expand_x_cils_array(
    num_cell_groups, cell_group_defs, this_cell_group_def
):
    x_cils_def = this_cell_group_def["x_cils"]

    num_defs = len(
        list(
            x_cils_def.keys()))

    if num_defs != num_cell_groups:
        raise Exception(
            "Number of cell groups does not equal number of keys in x_cils_def."
        )

    x_cils = []
    for cgi in range(num_cell_groups):
        cg = cell_group_defs[cgi]
        cg_name = cg["group_name"]
        intercellular_contact_factor_mag = x_cils_def[
            cg_name]

        x_cils += (
            cell_group_defs[cgi]["num_cells"]
        ) * [intercellular_contact_factor_mag]

    return np.array(x_cils)


# ==============================================================


def expand_interaction_factors_coa_per_celltype_array(
    num_cell_groups, cell_group_defs, this_cell_group_def
):
    interaction_factors_coa_per_celltype_def = this_cell_group_def[
        "interaction_factors_coa_per_celltype"
    ]

    num_defs = len(list(interaction_factors_coa_per_celltype_def.keys()))

    if num_defs != num_cell_groups:
        raise Exception(
            "Number of cell groups does not equal number of keys in x_cils_def."
        )

    interaction_factors_coa_per_celltype = []
    for cgi in range(num_cell_groups):
        cg = cell_group_defs[cgi]
        cg_name = cg["group_name"]
        cg_nverts = this_cell_group_def["params"]["nverts"]
        x_coa_strength = (
            interaction_factors_coa_per_celltype_def[cg_name] / cg_nverts
        )

        interaction_factors_coa_per_celltype += (
            cell_group_defs[cgi]["num_cells"]) * [x_coa_strength]

    return np.array(interaction_factors_coa_per_celltype)


# ==============================================================


def find_undefined_labels(cell_group_params):
    given_labels = list(cell_group_params.keys())
    undefined_labels = []
    global all_parameter_labels

    for label in all_parameter_labels:
        if label not in given_labels:
            undefined_labels.append(label)

    return undefined_labels


# ==============================================================


def make_environment_given_user_cell_group_defs(
        num_tsteps=0,
        user_cell_group_defs=None,
        char_t=(1 / 0.5),
        justify_parameters=True,
):

    if user_cell_group_defs is None:
        user_cell_group_defs = []
    num_cell_groups = len(user_cell_group_defs)

    for cell_group_def_ix, user_cell_group_def in enumerate(
            user_cell_group_defs):
        user_cell_group_params = copy.deepcopy(
            user_cell_group_def["params"]
        )
        user_cell_group_params[
            "x_cils"
        ] = expand_x_cils_array(
            num_cell_groups, user_cell_group_defs, user_cell_group_def
        )
        user_cell_group_params[
            "interaction_factors_coa_per_celltype"
        ] = expand_interaction_factors_coa_per_celltype_array(
            num_cell_groups, user_cell_group_defs, user_cell_group_def
        )
        cell_group_params = make_cell_group_params(
            justify_parameters, user_cell_group_params
        )

        user_cell_group_def.update(
            [("params", cell_group_params)])

    the_environment = environment.Environment(
        num_tsteps=num_tsteps,
        cell_group_defs=user_cell_group_defs,
        char_t=char_t,
    )

    return the_environment
