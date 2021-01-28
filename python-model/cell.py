import utilities as utils
import numba as nb
import numpy as np

import chemistry
import dynamics
import geometry
import mechanics
import params

"""
Cell.

"""


class NaNError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


# =============================================


@nb.jit(nopython=True)
def is_angle_between_range(a, b, angle):
    mod_a, mod_b = a % (2 * np.pi), b % (2 * np.pi)
    mod_angle = angle % (2 * np.pi)

    if mod_b < mod_a:
        if (0 <= mod_angle <= mod_b) or (mod_a <= mod_angle <= 2 * np.pi):
            return 1
        else:
            return 0
    else:
        if mod_a <= mod_angle <= mod_b:
            return 1
        else:
            return 0


# =============================================


def calculate_biased_distrib_factors(
        nverts, bias_range, bias_strength, bias_type, node_directions
):
    # index_directions = np.linspace(0, 2*np.pi, num=nverts)
    distrib_factors = np.zeros(nverts, dtype=np.float64)
    alpha, beta = bias_range

    biased_nodes = np.array([is_angle_between_range(
        alpha, beta, angle) for angle in node_directions])
    num_biased_nodes = int(np.sum(biased_nodes))
    num_unbiased_nodes = nverts - num_biased_nodes

    if bias_type == "random":
        biased_distrib_factors = (
                bias_strength *
                utils.calculate_normalized_randomization_factors(
                    num_biased_nodes))
        unbiased_distrib_factors = (
                                           1 - bias_strength
                                   ) * \
                                   utils.calculate_normalized_randomization_factors(
                                       nverts - num_biased_nodes)
    elif bias_type == "uniform":
        biased_distrib_factors = (
                bias_strength
                * (1.0 / num_biased_nodes)
                * np.ones(num_biased_nodes, dtype=np.float64)
        )
        unbiased_distrib_factors = (
                (1 - bias_strength)
                * bias_strength
                * (1.0 / num_unbiased_nodes)
                * np.ones(num_unbiased_nodes, dtype=np.float64)
        )
    else:
        raise Exception("Got unknown bias type: {}.".format(bias_type))

    nverts_biased = 0
    nverts_unbiased = 0

    for ni, biased in enumerate(biased_nodes):
        if biased == 1:
            distrib_factors[ni] = biased_distrib_factors[nverts_biased]
            nverts_biased += 1
        else:
            distrib_factors[ni] = unbiased_distrib_factors[nverts_unbiased]
            nverts_unbiased += 1

    assert nverts_biased + nverts_unbiased == nverts

    return distrib_factors


# =============================================

# @nb.jit(nopython=True)
def generate_random_multipliers_fixed_number(nverts, threshold, magnitude):
    rfs = np.ones(nverts, dtype=np.float64)

    num_random_nodes = int(nverts * threshold)
    random_node_indices = np.random.choice(
        np.arange(nverts), size=num_random_nodes, replace=False
    )

    rfs[random_node_indices] = magnitude * \
                               np.ones(num_random_nodes, dtype=np.float64)

    return rfs


# =============================================


# @nb.jit(nopython=True)
def generate_random_multipliers_random_number(
        nverts, threshold, randoms, magnitude):
    rfs = np.ones(nverts, dtype=np.float64)

    for i in range(nverts):
        if randoms[i] < threshold:
            rfs[i] = magnitude
        else:
            continue

    return rfs


# ---------------------------------------------
def verify_parameter_completeness(parameters_dict):
    if not isinstance(parameters_dict, dict):
        raise Exception(
            "parameters_dict is not a dict, instead: {}".type(parameters_dict)
        )

    given_keys = list(parameters_dict.keys())

    for key in given_keys:
        if key not in params.all_parameter_labels:
            raise Exception(
                "Parameter {} not in standard parameter list!".format(key))

    for key in params.all_parameter_labels:
        if key not in given_keys:
            raise Exception("Parameter {} not in given keys!".format(key))

    return


# =============================================

class Cell:
    def __init__(
            self,
            cell_label,
            cell_group_ix,
            cell_ix,
            num_tsteps,
            char_t,
            num_env_cells,
            num_int_steps,
            parameters_dict,
    ):
        self.cell_label = cell_label
        self.cell_group_ix = cell_group_ix
        self.cell_ix = cell_ix

        self.num_tpoints = num_tsteps + 1
        self.num_tsteps = num_tsteps
        self.num_int_steps = num_int_steps

        self.curr_tpoint = 0

        self.nverts = parameters_dict["nverts"]
        self.num_env_cells = num_env_cells
        verify_parameter_completeness(parameters_dict)

        self.tot_rac = parameters_dict["tot_rac"]
        self.tot_rho = parameters_dict["tot_rho"]

        # initializing output arrays
        self.last_state = np.zeros(
            (
                self.num_tpoints + 1,
                self.nverts,
                len(params.output_info_labels),
            )
        )
        self.curr_state = np.zeros(
            (
                self.nverts,
                len(params.output_info_labels),
            )
        )

        vertex_eta = parameters_dict["eta"] / self.nverts
        self.char_l = 1e-6
        self.char_t = char_t
        self.char_f = 1e-9

        self.vertex_eta = vertex_eta / (
                self.char_f / (self.char_l / self.char_t))

        if parameters_dict["init_verts"].shape[0] != self.nverts:
            raise Exception(
                "Number of node coords given for initialization does not "
                "match number of nodes! Given: {}, required: {}.".format(
                    parameters_dict["init_verts"].shape[0],
                    self.nverts))

        self.curr_verts = parameters_dict[
                                    "init_verts"] / self.char_l
        self.radius_resting = parameters_dict["cell_r"] / self.char_l
        self.rest_edge_len = parameters_dict[
                                       "rest_edge_len"] / self.char_l
        edgeplus_lengths = geometry.calculate_edgeplus_lengths(
            self.curr_verts)
        self.init_average_edge_lengths = np.average(
            geometry.calculate_average_edge_length_around_nodes(
                edgeplus_lengths))
        self.rest_area = parameters_dict["rest_area"] / (self.char_l ** 2)

        self.diffusion_rgtp = parameters_dict[
                                          "diffusion_rgtp"] * (
                                              self.char_t / (self.char_l ** 2))

        self.stiffness_edge = (parameters_dict["stiffness_edge"]) / (
                self.char_f / self.char_l
        )
        self.stiffness_cyto = parameters_dict[
                                         "stiffness_cyto"] / (
                                         self.char_f)

        # ======================================================

        self.max_force_rac = parameters_dict["max_force_rac"] / self.char_f
        self.max_force_rho = parameters_dict["max_force_rho"] / self.char_f
        self.threshold_force_rac_activity = parameters_dict[
                                                "threshold_rac_activity"
                                            ] / (self.tot_rac * self.nverts)
        self.threshold_force_rho_activity = parameters_dict[
                                                "threshold_rho_activity"
                                            ] / (self.tot_rho * self.nverts)

        # ======================================================

        self.close_criterion = parameters_dict[
                                                   "close_criterion"
                                               ] / (self.char_l ** 2)
        self.close_criterion_0_until = (
                self.close_criterion * 9
        )
        self.close_criterion_1_at = \
            self.close_criterion
        self.closeness_dist_criteria = np.sqrt(
            self.close_criterion)
        self.force_adh_constant = parameters_dict["force_adh_const"]

        self.kgtp_rac = parameters_dict[
                                     "kgtp_rac"] * self.char_t
        self.kgtp_rac_auto = (
                parameters_dict["kgtp_rac_auto"] * self.char_t
        )

        self.halfmax_rgtp_frac = parameters_dict[
                                         "threshold_rac_activity"] / (
                                             self.tot_rac * self.nverts *
                                             self.init_average_edge_lengths)

        self.kdgtp_rac = parameters_dict[
                                      "kdgtp_rac"] * self.char_t
        self.kdgtp_rho_on_rac = (
                parameters_dict[
                    "kdgtp_rho_on_rac"] * self.char_t
        )

        self.halfmax_rgtp_frac = parameters_dict[
                                                    "threshold_rho_activity"
                                                ] / (
                                                        self.tot_rho *
                                                        self.nverts *
                                                        self.init_average_edge_lengths)

        self.halfmax_tension_inhib = parameters_dict[
            "halfmax_tension_inhib"
        ]
        self.tension_inhib = parameters_dict[
            "tension_inhib"
        ]
        self.k_mem_off = parameters_dict["k_mem_off"] * self.char_t
        self.k_mem_on = parameters_dict["k_mem_on"] * \
                         self.char_t * (1.0 / self.nverts)

        # ======================================================

        self.kgtp_rho = parameters_dict[
                                     "kgtp_rho"] * self.char_t
        self.kgtp_rho_auto = (
                parameters_dict["kgtp_rho_auto"] * self.char_t
        )

        self.halfmax_rgtp_frac = parameters_dict[
                                         "threshold_rho_activity"] / (
                                             self.tot_rho * self.nverts *
                                             self.init_average_edge_lengths)

        self.kdgtp_rho = parameters_dict[
                                      "kdgtp_rho"] * self.char_t
        self.kdgtp_rac_on_rho = (
                parameters_dict[
                    "kdgtp_rac_on_rho"] * self.char_t
        )

        self.halfmax_rgtp_frac = parameters_dict[
                                                    "threshold_rac_activity"
                                                ] / (
                                                        self.tot_rac *
                                                        self.nverts *
                                                        self.init_average_edge_lengths)

        self.x_cils = \
            parameters_dict[
                "x_cils"]

        # ==============================================================
        self.interaction_factors_coa_per_celltype = parameters_dict[
            "interaction_factors_coa_per_celltype"
        ]

        self.coa_sensing_dist_at_value = (
                parameters_dict["coa_sensing_dist_at_value"] / self.char_l
        )

        self.coa_distrib_exp = (
                np.log(0.5)
                / self.coa_sensing_dist_at_value
        )
        self.los_factor = parameters_dict[
            "los_factor"]
        # =============================================================

        self.phase_var_indices = [
            params.rac_acts_ix,
            params.rac_inacts_ix,
            params.rho_act_ix,
            params.rho_inacts_ix,
            params.x_ix,
            params.y_ix,
        ]
        self.num_phase_vars = len(self.phase_var_indices)
        self.total_num_phase_vars = self.num_phase_vars * \
                                          self.nverts

        self.initialize_phase_var_indices()

        # =============================================================

        self.randomization_scheme = parameters_dict["randomization_scheme"]
        self.randomization_time_mean = int(
            parameters_dict["randomization_time_mean"] * 60.0 / char_t
        )
        self.randomization_time_variance_factor = parameters_dict[
            "randomization_time_variance_factor"
        ]
        self.next_randomization_event_tpoint = 1200
        self.randomization_magnitude = parameters_dict[
            "randomization_magnitude"]
        self.rac_rands = \
            self.renew_rac_rands(
                0)
        self.randomization_node_percentage = parameters_dict[
            "randomization_node_percentage"
        ]

        # =============================================================

        self.all_cellwide_phase_var_indices = [
            params.rac_cyto_ix,
            params.rho_cyto_ix,
        ]
        self.ode_cellwide_phase_var_indices = []
        self.num_all_cellwide_phase_vars = len(
            self.all_cellwide_phase_var_indices)
        self.num_ode_cellwide_phase_vars = len(
            self.ode_cellwide_phase_var_indices)

        self.initialize_all_cellwide_phase_var_indices()
        self.initialize_ode_cellwide_phase_var_indices()

        # =============================================================

        self.pars_indices = [
            params.kgtp_rac_ix,
            params.kgtp_rho_ix,
            params.kdgtp_rac_ix,
            params.kdgtp_rho_ix,
            params.k_mem_on_ix,
            params.kdgdi_rho_ix,
            params.local_strains_ix,
            params.x_cils_ix,
        ]

        self.initialize_pars_indices()

        # =============================================================
        self.init_inact_rgtp = parameters_dict[
            "init_inact_rgtp"
        ]
        self.init_act_rgtp = parameters_dict[
            "init_act_rgtp"
        ]
        self.rgtp_distrib_def_for_randomization = [
            "unbiased random",
            0.0,
            0.0,
        ]
        self.rgtp_distrib_def = parameters_dict[
            "rgtp_distrib_def"]

        self.last_trim_timestep = -1

    # -----------------------------------------------------------------
    def insert_state_array_into_system_history(self, state_array, tstep):
        phase_vars, ode_cellwide_phase_vars = dynamics.unpack_state_array(
            self.num_phase_vars, state_array)
        access_ix = self.get_system_history_access_ix(tstep)
        self.curr_state[
        access_ix, :, self.phase_var_indices
        ] = phase_vars
        self.curr_state[
            access_ix, 0, self.ode_cellwide_phase_var_indices
        ] = ode_cellwide_phase_vars

    # -----------------------------------------------------------------
    def initialize_cell(
            self,
            intercellular_squared_dist_array,
            line_segment_intersection_matrix,
    ):
        init_verts = self.curr_verts
        rgtp_distrib_def = self.rgtp_distrib_def
        init_inact_rgtp = \
            self.init_inact_rgtp
        init_act_rgtp = \
            self.init_act_rgtp
        init_cyto_rgtp = 1.0 - init_inact_rgtp - init_act_rgtp

        self.last_state = copy.deepcopy(self.curr_state)

        self.curr_state[:, [params.x_ix, params.y_ix]] = np.transpose(
            init_verts)
        self.curr_state[:, params.edge_lengths_ix] = \
            self.rest_edge_len * np.ones(self.nverts)

        verts = init_verts
        cell_centroid = geometry.calculate_cluster_centroid(init_verts)

        self.set_rgtpase_distribution(
            rgtp_distrib_def,
            init_cyto_rgtp,
            init_inact_rgtp,
            init_act_rgtp,
            np.array([np.arctan2(y, x) for x, y in
                      init_verts - cell_centroid]),
        )

        rac_acts = self.curr_state[:, params.rac_acts_ix]
        rho_acts = self.curr_state[:, params.rho_act_ix]

        x_coas = np.zeros(self.nverts, dtype=np.float64)
        random_order_cell_indices = np.arange(self.num_env_cells)
        self.curr_state[:, params.x_coa_ix] = \
            chemistry.calculate_x_coas(
                self.cell_ix,
                self.nverts,
                random_order_cell_indices,
                self.coa_distrib_exp,
                self.interaction_factors_coa_per_celltype,
                intercellular_squared_dist_array,
                line_segment_intersection_matrix,
                self.los_factor,
            )

        x_cils = np.zeros(self.nverts)

        np.zeros(
            (self.nverts, self.num_env_cells), dtype=np.int64
        )
        np.zeros(
            (self.nverts,
             self.num_env_cells,
             2),
            dtype=np.float64)
        np.zeros(
            (self.nverts, self.num_env_cells, 2), dtype=np.int64
        )
        np.zeros(
            (self.nverts, self.num_env_cells), dtype=np.int64
        )
        close_point_smoothness_factors = np.zeros(
            (self.nverts, self.num_env_cells), dtype=np.float64
        )
        np.zeros(
            (self.num_env_cells, 2), dtype=np.float64
        )
        np.zeros(
            (self.num_env_cells,
             self.nverts,
             2),
            dtype=np.float64)

        sum_forces, edge_forces_plus, edge_forces_minus, rgtpase_forces, \
        cyto_forces, \
        edge_strains, local_strains, uivs = \
            mechanics.calculate_forces(
                self.nverts,
                verts,
                rac_acts,
                rho_acts,
                self.rest_edge_len,
                self.stiffness_edge,
                self.threshold_force_rac_activity,
                self.threshold_force_rho_activity,
                self.max_force_rac,
                self.max_force_rho,
                self.rest_area,
                self.stiffness_cyto,
            )

        self.curr_state[:, params.local_strains_ix] = local_strains

        self.curr_state[:, params.k_mem_on_ix] = self.k_mem_on * np.ones(
            self.nverts, dtype=np.float64)
        self.curr_state[:, params.kdgdi_rho_ix] = self.kdgdi_rho * np.ones(
            self.nverts, dtype=np.float64)

        edgeplus_lengths = geometry.calculate_edgeplus_lengths(verts)
        avg_edge_lengths = geometry.calculate_average_edge_length_around_nodes(
            edgeplus_lengths
        )

        conc_rac_acts = chemistry.calculate_concentrations(
            self.nverts, rac_acts, avg_edge_lengths
        )

        conc_rho_acts = chemistry.calculate_concentrations(
            self.nverts, rho_acts, avg_edge_lengths
        )

        self.curr_state[:, params.rac_rands_ix] = self.rac_rands

        self.curr_state[:, params.kgtp_rac_ix] = chemistry.calculate_kgtp_rac(
            conc_rac_acts,
            self.halfmax_rgtp_frac,
            self.kgtp_rac,
            self.kgtp_rac_auto,
            x_coas,
            self.rac_rands,
            x_cils,
            close_point_smoothness_factors,
        )

        self.curr_state[:, params.kgtp_rho_ix] = chemistry.calculate_kgtp_rho(
            self.nverts,
            conc_rho_acts,
            x_cils,
            self.halfmax_rgtp_frac,
            self.kgtp_rho,
            self.kgtp_rho_auto,
        )

        self.curr_state[:, params.kdgtp_rac_ix] = chemistry.calculate_kdgtp_rac(
            self.nverts,
            conc_rho_acts,
            self.halfmax_rgtp_frac,
            self.kdgtp_rac,
            self.kdgtp_rho_on_rac,
            x_cils,
            self.halfmax_tension_inhib,
            self.tension_inhib,
            np.array([ls if ls > 0 else 0.0 for ls in local_strains]),
        )

        self.curr_state[:, params.kdgtp_rho_ix] = chemistry.calculate_kdgtp_rho(
            self.nverts,
            conc_rac_acts,
            self.halfmax_rgtp_frac,
            self.kdgtp_rho,
            self.kdgtp_rac_on_rho,
        )

        # update mechanics parameters
        self.curr_state[:, [params.sum_forces_x_ix, params.sum_forces_y_ix]] \
            = np.transpose(sum_forces)
        self.curr_state[:, [params.edge_forces_plus_x_ix,
                            params.edge_forces_plus_y_ix]] = \
            np.transpose(edge_forces_plus)
        self.curr_state[:, [params.edge_forces_minus_x_ix,
                            params.edge_forces_minus_y_ix]] = \
            np.transpose(edge_forces_minus)
        self.curr_state[:, [params.rgtp_forces_x_ix, params.rgtp_forces_y_ix]] \
            = np.transpose(rgtpase_forces)
        self.curr_state[:, [params.cyto_forces_x_ix,
                            params.cyto_forces_y_ix]] = \
            np.transpose(cyto_forces)

        self.curr_state[[params.uiv_x_ix, params.uiv_y_ix]] = \
            np.transpose(uivs)

        self.curr_state[params.x_cils_ix] = x_cils

        self.curr_verts = verts

    # -----------------------------------------------------------------
    def initialize_phase_var_indices(self):
        for index, sys_info_ix in enumerate(self.phase_var_indices):
            label = params.output_info_labels[sys_info_ix]
            setattr(self, "" + label + "_ix", index)

    # -----------------------------------------------------------------
    def initialize_pars_indices(self):
        for index, sys_info_ix in enumerate(self.pars_indices):
            label = params.output_info_labels[sys_info_ix]
            setattr(self, "" + label + "_ix", index)

    # -----------------------------------------------------------------
    def initialize_ode_cellwide_phase_var_indices(self):
        for index, sys_info_ix in enumerate(
                self.ode_cellwide_phase_var_indices):
            label = params.output_info_labels[sys_info_ix]
            setattr(self, "cellwide_" + label + "_ix", index)

    # -----------------------------------------------------------------
    def initialize_all_cellwide_phase_var_indices(self):
        for index, sys_info_ix in enumerate(
                self.all_cellwide_phase_var_indices):
            label = params.output_info_labels[sys_info_ix]
            setattr(self, "cellwide_" + label + "_ix", index)

    # -----------------------------------------------------------------
    def calculate_when_randomization_event_occurs(self):
        return self.curr_tpoint + 1200

    # -----------------------------------------------------------------
    def set_rgtpase_distribution(
            self,
            rgtp_distrib_def,
            init_cyto_rgtp,
            init_inact_rgtp,
            init_act_rgtp,
            node_directions,
            tpoint=0,
    ):
        rgtpase_distrib = []
        distrib_type, bias_direction_range, bias_strength = \
            rgtp_distrib_def

        cellwide_distrib_factors = np.zeros(self.nverts)
        cellwide_distrib_factors[0] = 1

        access_ix = self.get_system_history_access_ix(tpoint)

        for rgtpase_label in ["rac", "rho"]:
            for label in params.output_chem_labels[:7]:
                if rgtpase_label in label:
                    if "act" in label:
                        if "inact" in label:
                            frac_factor = init_inact_rgtp
                        else:
                            frac_factor = init_act_rgtp

                        if distrib_type == "unbiased random":
                            rgtpase_distrib = (
                                    frac_factor *
                                    utils.calculate_normalized_randomization_factors(
                                        self.nverts))
                        elif distrib_type == "biased random":
                            if rgtpase_label == "rac":
                                rgtpase_distrib = (
                                        frac_factor
                                        * calculate_biased_distrib_factors(
                                    self.nverts,
                                    bias_direction_range,
                                    bias_strength,
                                    "random",
                                    node_directions,
                                )
                                )
                            elif rgtpase_label == "rho":
                                rgtpase_distrib = (
                                        frac_factor *
                                        utils.calculate_normalized_randomization_factors(
                                            self.nverts))
                                # rgtpase_distrib = 
                                # frac_factor*calculate_biased_distrib_factors(self.nverts, bias_direction_range + np.pi, bias_strength, 'random')
                        elif distrib_type == "unbiased uniform":
                            rgtpase_distrib = (
                                    frac_factor *
                                    np.ones(
                                        self.nverts) /
                                    self.nverts)
                        elif distrib_type == "biased uniform":
                            if rgtpase_label == "rac":
                                rgtpase_distrib = (
                                        frac_factor
                                        * calculate_biased_distrib_factors(
                                    self.nverts,
                                    bias_direction_range,
                                    bias_strength,
                                    "uniform",
                                    node_directions,
                                )
                                )
                            elif rgtpase_label == "rho":
                                # rgtpase_distrib = 
                                # frac_factor*gu.calculate_normalized_randomization_factors(self.nverts)
                                rgtpase_distrib = (
                                        frac_factor *
                                        calculate_biased_distrib_factors(
                                            self.nverts,
                                            np.array(
                                                [
                                                    bias_direction_range[0] +
                                                    np.pi,
                                                    bias_direction_range[1] +
                                                    np.pi,
                                                ]),
                                            bias_strength,
                                            "uniform",
                                            node_directions,
                                        ))
                            elif rgtpase_label == "rho":
                                # rgtpase_distrib = 
                                # frac_factor*gu.calculate_normalized_randomization_factors(self.nverts)
                                rgtpase_distrib = 1e-5 * np.ones(
                                    self.nverts, dtype=np.float64
                                )
                                rgtpase_distrib[int(self.nverts / 2)] = 1.0
                                if (self.nverts % 2) == 1:
                                    rgtpase_distrib[int(
                                        self.nverts / 2) + 1] = 1.0
                                    if self.nverts > 3:
                                        rgtpase_distrib[
                                            int(self.nverts / 2) - 1
                                            ] = 1.0
                                rgtpase_distrib = (
                                        frac_factor
                                        * rgtpase_distrib
                                        / np.sum(rgtpase_distrib)
                                )
                        elif distrib_type == "biased nodes":
                            biased_nodes = bias_direction_range
                            distrib_factors = np.array(
                                [1.0 if x in biased_nodes else 0.0 for x in
                                 range(self.nverts)])
                            distrib_factors /= np.sum(distrib_factors)
                            if rgtpase_label == "rac_":
                                rgtpase_distrib = frac_factor * distrib_factors
                            elif rgtpase_label == "rho_":
                                # rgtpase_distrib = 
                                # frac_factor*gu.calculate_normalized_randomization_factors(self.nverts)
                                rgtpase_distrib = frac_factor * \
                                                  np.roll(distrib_factors, 8)
                        else:
                            raise Exception(
                                "Invalid initial RhoGTPase distribution type "
                                "provided! ({})".format(
                                    distrib_type
                                )
                            )

                    elif "cyto" in label:
                        frac_factor = init_cyto_rgtp
                        rgtpase_distrib = frac_factor * cellwide_distrib_factors
                    else:
                        continue

                    self.curr_state[access_ix, :, eval(
                        "parameterorg." + label + "_ix")] = rgtpase_distrib

    # -----------------------------------------------------------------

    def renew_rac_rands(self, tstep):
        if self.randomization_scheme is not None:
            possible_rfs = np.array(
                [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                  1.0, 0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 1.0],
                 [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                  0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 1.0]]
            )
        else:
            possible_rfs = np.array(np.zeros((6, 16)))
        ix = int((tstep / 1200) % 6)
        return possible_rfs[ix] * self.randomization_magnitude

    # -----------------------------------------------------------------
    def set_next_state(
            self,
            next_state_array,
            this_cell_ix,
            num_cells,
            intercellular_squared_dist_array,
            line_segment_intersection_matrix,
            all_cells_verts,
            close_point_smoothness_factors,
    ):

        self.curr_tpoint += 1
        curr_tpoint = self.curr_tpoint
        nverts = self.nverts

        assert new_tpoint < self.num_tpoints

        self.insert_state_array_into_system_history(
            next_state_array, new_tpoint)

        verts = np.transpose(self.curr_state[[params.x_ix, params.y_ix]])

        edge_displacement_vectors_plus = geometry.calculate_edge_vectors(
            verts)
        edge_lengths = geometry.calculate_2D_vector_mags(
            edge_displacement_vectors_plus)

        self.curr_state[:, params.edge_lengths_ix] = edge_lengths

        geometry.calculate_centroids(all_cells_verts)

        x_cils = \
            chemistry.calculate_x_cils(
                this_cell_ix,
                nverts,
                num_cells,
                self.x_cils,
                close_point_smoothness_factors,
            )

        # ==================================
        if self.curr_tpoint == self.next_randomization_event_tpoint:
            self.next_randomization_event_tpoint += 1200

            # randomization event has occurred, so renew Rac kgtp rate
            # multipliers
            self.rac_rands = \
                self.renew_rac_rands(
                    self.curr_tpoint)

        # store the Rac randomization factors for this timestep
        self.curr_state[
        curr_tpoint,
        :,
        params.rac_rands_ix,
        ] = self.rac_rands

        # ==================================
        rac_acts = self.curr_state[
                   curr_tpoint,
                   :,
                   params.rac_acts_ix,
                   ]
        rho_acts = self.curr_state[
                               curr_tpoint,
                               :,
                               params.rho_act_ix,
                               ]
        random_order_cell_indices = np.arange(num_cells)

        x_coas = chemistry.calculate_x_coas(
            this_cell_ix,
            nverts,
            random_order_cell_indices,
            self.coa_distrib_exp,
            self.interaction_factors_coa_per_celltype,
            intercellular_squared_dist_array,
            line_segment_intersection_matrix,
            self.los_factor,
        )
        # print(np.max(x_coas))

        self.curr_state[:, params.x_coa_ix] = x_coas
        self.curr_state[:, params.x_cil_ix] = \
            x_cils

        rac_cyto = (
                1
                - np.sum(rac_acts)
                - np.sum(
            self.curr_state[
            curr_tpoint,
            :,
            params.rac_inacts_ix,
            ]
        )
        )
        rho_cyto = (
                1
                - np.sum(rho_acts)
                - np.sum(
            self.curr_state[
            curr_tpoint,
            :,
            params.rho_inacts_ix,
            ]
        )
        )

        insertion_array = np.zeros(self.nverts)
        insertion_array[0] = 1

        self.curr_state[
        curr_tpoint,
        :,
        params.rac_cyto_ix,
        ] = (rac_cyto * insertion_array)
        self.curr_state[
        curr_tpoint,
        :,
        params.rho_cyto_ix,
        ] = (rho_cyto * insertion_array)

        sum_forces, edge_forces_plus, edge_forces_minus, rgtp_forces, \
        cyto_forces, \
        edge_strains, local_strains, unit_inside_pointing_vecs = \
            mechanics.calculate_forces(
                self.nverts,
                verts,
                rac_acts,
                rho_acts,
                self.rest_edge_len,
                self.stiffness_edge,
                self.threshold_force_rac_activity,
                self.threshold_force_rho_activity,
                self.max_force_rac,
                self.max_force_rho,
                self.rest_area,
                self.stiffness_cyto,
            )

        self.curr_state[:, params.local_strains_ix] = local_strains

        # update chemistry parameters
        self.curr_state[:, params.k_mem_on_ix] = \
            self.k_mem_on * np.ones(nverts, dtype=np.float64)
        self.curr_state[:, params.kdgdi_rho_ix] = \
            self.kdgdi_rho * np.ones(nverts, dtype=np.float64)

        edgeplus_lengths = geometry.calculate_edgeplus_lengths(verts)
        avg_edge_lengths = geometry.calculate_average_edge_length_around_nodes(
            edgeplus_lengths
        )

        conc_rac_acts = chemistry.calculate_concentrations(
            self.nverts, rac_acts, avg_edge_lengths
        )

        conc_rho_acts = chemistry.calculate_concentrations(
            self.nverts, rho_acts, avg_edge_lengths
        )

        self.curr_state[
        curr_tpoint,
        :,
        params.rac_rands_ix,
        ] = self.rac_rands

        kgtp_racs = chemistry.calculate_kgtp_rac(
            conc_rac_acts,
            self.halfmax_rgtp_frac,
            self.kgtp_rac,
            self.kgtp_rac_auto,
            x_coas,
            self.rac_rands,
            x_cils,
            close_point_smoothness_factors,
        )

        kdgtps_rac = chemistry.calculate_kdgtp_rac(
            self.nverts,
            conc_rho_acts,
            self.halfmax_rgtp_frac,
            self.kdgtp_rac,
            self.kdgtp_rho_on_rac,
            x_cils,
            self.halfmax_tension_inhib,
            self.tension_inhib,
            np.array([ls if ls > 0 else 0.0 for ls in local_strains]),
        )

        kdgtps_rho = chemistry.calculate_kdgtp_rho(
            self.nverts,
            conc_rac_acts,
            self.halfmax_rgtp_frac,
            self.kdgtp_rho,
            self.kdgtp_rac_on_rho,
        )

        kgtp_rhos = chemistry.calculate_kgtp_rho(
            self.nverts,
            conc_rho_acts,
            x_cils,
            self.halfmax_rgtp_frac,
            self.kgtp_rho,
            self.kgtp_rho_auto,
        )

        self.curr_state[:, params.kgtp_rac_ix] = kgtp_racs
        self.curr_state[:, params.kgtp_rho_ix] = kgtp_rhos

        self.curr_state[:, params.kdgtp_rac_ix] = kdgtps_rac

        self.curr_state[:, params.kdgtp_rho_ix] = kdgtps_rho

        # update mechanics parameters
        self.curr_state[[params.sum_forces_x_ix, params.sum_forces_y_ix]] = \
            np.transpose(sum_forces)
        self.curr_state[:, [params.edge_forces_plus_x_ix,
                            params.edge_forces_plus_y_ix]] =\
            np.transpose(edge_forces_plus)
        self.curr_state[:, [params.edge_forces_minus_x_ix,
                            params.edge_forces_minus_y_ix]] = np.transpose(
            edge_forces_minus)
        self.curr_state[:, [params.rgtp_forces_x_ix,
                            params.rgtp_forces_y_ix]] =\
            np.transpose(rgtp_forces)
        self.curr_state[:, [params.cyto_forces_x_ix, params.cyto_forces_y_ix]]\
            = np.transpose(cyto_forces)

        self.curr_state[:, [params.uiv_x_ix,
                            params.uiv_y_ix]] = \
            np.transpose(unit_inside_pointing_vecs)

        self.curr_state[:, params.x_cils_ix] = x_cils
        self.curr_verts = verts

    # -----------------------------------------------------------------
    def pack_rhs_arguments(
            self,
            t,
            all_cells_verts,
            close_point_smoothness_factors,
    ):
        access_ix = self.get_system_history_access_ix(t)
        num_cells = all_cells_verts.shape[0]
        nverts = self.nverts

        x_cils = \
            chemistry.calculate_x_cils(
                this_cell_ix,
                nverts,
                num_cells,
                self.x_cils,
                close_point_smoothness_factors,
            )

        transduced_x_coas = self.curr_state[
                                 access_ix, :, params.x_coa_ix
                                 ]

        return (
            this_cell_ix,
            self.nverts,
            self.num_phase_vars,
            self.rac_acts_ix,
            self.rest_edge_len,
            self.rac_inacts_ix,
            self.rho_act_ix,
            self.rho_inacts_ix,
            self.x_ix,
            self.y_ix,
            self.kgtp_rac,
            self.kdgtp_rac,
            self.kgtp_rho,
            self.kdgtp_rho,
            self.kgtp_rac_auto,
            self.kgtp_rho_auto,
            self.kdgtp_rho_on_rac,
            self.kdgtp_rac_on_rho,
            self.k_mem_off,
            self.k_mem_on,
            self.halfmax_rgtp_frac,
            self.halfmax_rgtp_frac,
            self.halfmax_rgtp_frac,
            self.halfmax_rgtp_frac,
            self.diffusion_rgtp,
            self.vertex_eta,
            self.stiffness_edge,
            self.threshold_force_rac_activity,
            self.threshold_force_rho_activity,
            self.max_force_rac,
            self.max_force_rho,
            self.rest_area,
            self.stiffness_cyto,
            transduced_x_coas,
            close_point_smoothness_factors,
            x_cils,
            self.halfmax_tension_inhib,
            self.tension_inhib,
            self.rac_rands,
        )

    # -----------------------------------------------------------------
    def execute_step(
            self,
            this_cell_ix,
            nverts,
            all_cells_verts,
            intercellular_squared_dist_array,
            line_segment_intersection_matrix,
    ):
        dynamics.print_var = True

        num_cells = all_cells_verts.shape[0]

        are_nodes_inside_other_cells = \
            geometry.check_if_nodes_inside_other_cells(
                this_cell_ix, nverts, num_cells,
                all_cells_verts)
        close_point_on_other_cells_to_each_node_exists, \
        close_point_on_other_cells_to_each_node, \
        close_point_on_other_cells_to_each_node_indices, \
        close_point_on_other_cells_to_each_node_projection_factors, \
        close_point_smoothness_factors = \
            geometry.do_close_points_to_each_node_on_other_cells_exist(
                num_cells, nverts, this_cell_ix,
                all_cells_verts[this_cell_ix],
                intercellular_squared_dist_array,
                self.close_criterion_0_until,
                self.close_criterion_1_at,
                all_cells_verts, are_nodes_inside_other_cells, )

        state_array = dynamics.pack_state_array_from_system_history(
            self.phase_var_indices,
            self.ode_cellwide_phase_var_indices,
            self.curr_state,
            self.get_system_history_access_ix(self.curr_tpoint),
        )

        rhs_args = self.pack_rhs_arguments(
            self.curr_tpoint,
            all_cells_verts,
            close_point_smoothness_factors,
        )

        output_array = dynamics.eulerint(
            dynamics.cell_dynamics, state_array, np.array([0, 1]),
            rhs_args, self.num_int_steps)

        next_state_array = output_array[1]

        self.set_next_state(
            next_state_array,
            this_cell_ix,
            num_cells,
            intercellular_squared_dist_array,
            line_segment_intersection_matrix,
            all_cells_verts,
            close_point_smoothness_factors,
        )

# ===============================================================
