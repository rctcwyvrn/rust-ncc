# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:27:54 2015
@author: Brian
"""

import numba as nb
import numpy as np

# -----------------------------------------------------------------
@nb.jit(nopython=True)
def hill_function3(thresh, sig):
    pow_sig = sig ** 3.0
    pow_thresh = thresh ** 3.0

    return pow_sig / (pow_thresh + pow_sig)


# -----------------------------------------------------------------
# @nb.jit(nopython=True)


def calculate_kgtp_rac(
        conc_rac_acts,
        halfmax_rgtp_frac,
        kgtp_rac,
        kgtp_rac_auto,
        x_coas,
        randomization_factors,
        intercellular_contact_factors,
        close_point_smoothness_factors,
):
    num_vertices = conc_rac_acts.shape[0]
    result = np.empty(num_vertices, dtype=np.float64)

    for i in range(num_vertices):
        i_plus1 = (i + 1) % num_vertices
        i_minus1 = (i - 1) % num_vertices

        cil_factor = (
                             intercellular_contact_factors[i]
                             + intercellular_contact_factors[i_plus1]
                             + intercellular_contact_factors[i_minus1]
                     ) / 3.0
        smooth_factor = np.max(close_point_smoothness_factors[i])
        x_coa = x_coas[i]

        if cil_factor > 0.0 or smooth_factor > 1e-6:
            x_coa = 0.0

        rac_autoact_hill_effect = hill_function3(
            halfmax_rgtp_frac,
            conc_rac_acts[i])
        kgtp_rac_autoact = (
                kgtp_rac_auto
                * rac_autoact_hill_effect
        )

        if kgtp_rac_autoact > 1.25 * kgtp_rac_auto:
            kgtp_rac_autoact = 1.25 * kgtp_rac_auto

        kgtp_rac_base = (
                                1.0 + randomization_factors[
                            i] + x_coa) * kgtp_rac
        result[i] = kgtp_rac_base + kgtp_rac_autoact

    return result


# -----------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_kgtp_rho(
        nverts,
        conc_rho_act,
        intercellular_contact_factors,
        halfmax_rgtp_frac,
        kgtp_rho,
        kgtp_rho_auto,
):
    result = np.empty(nverts)
    for i in range(nverts):
        kgtp_rho_autoact = kgtp_rho_auto * hill_function3(halfmax_rgtp_frac,
                                                          conc_rho_act[i]
                                                          )

        i_plus1 = (i + 1) % nverts
        i_minus1 = (i - 1) % nverts

        cil_factor = (
                             intercellular_contact_factors[i]
                             + intercellular_contact_factors[i_plus1]
                             + intercellular_contact_factors[i_minus1]
                     ) / 3.0

        result[i] = (
                            1.0 + cil_factor
                    ) * kgtp_rho + kgtp_rho_autoact
    return result


# -----------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_kdgtp_rac(
        nverts,
        conc_rho_acts,
        halfmax_rgtp_frac,
        kdgtp_rac,
        kdgtp_rho_on_rac,
        intercellular_contact_factors,
        halfmax_tension_inhib,
        tension_inhib,
        local_strains,
):
    result = np.empty(nverts, dtype=np.float64)

    global_tension = np.sum(local_strains) / nverts
    if global_tension < 0.0:
        global_tension = 0.0
    strain_inhibition = tension_inhib * \
                        hill_function3(
                            halfmax_tension_inhib,
                            global_tension
                            )

    for i in range(nverts):
        kdgtp_rho_mediated_rac_inhib = (
                kdgtp_rho_on_rac
                * hill_function3(
            halfmax_rgtp_frac,
            conc_rho_acts[i],
        )
        )

        i_plus1 = (i + 1) % nverts
        i_minus1 = (i - 1) % nverts

        cil_factor = (
                             intercellular_contact_factors[i]
                             + intercellular_contact_factors[i_plus1]
                             + intercellular_contact_factors[i_minus1]
                     ) / 3.0

        result[i] = (
                            1.0 + cil_factor + strain_inhibition
                    ) * kdgtp_rac + kdgtp_rho_mediated_rac_inhib

    return result


# -----------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_kdgtp_rho(
        nverts,
        conc_rac_acts,
        halfmax_rgtp_frac,
        kdgtp_rho,
        kdgtp_rac_on_rho,
):
    result = np.empty(nverts, dtype=np.float64)

    for i in range(nverts):
        kdgtp_rac_mediated_rho_inhib = (
                kdgtp_rac_on_rho
                * hill_function3(
            halfmax_rgtp_frac,
            conc_rac_acts[i],
        )
        )

        result[i] = kdgtp_rho + kdgtp_rac_mediated_rho_inhib

    return result


# -----------------------------------------------------------------
# @nb.jit(nopython=True)
def calculate_concentrations(nverts, species, avg_edge_lengths):
    result = np.empty(nverts, dtype=np.float64)

    for i in range(nverts):
        result[i] = species[i] / avg_edge_lengths[i]

    return result


# -----------------------------------------------------------------
# @nb.jit(nopython=True)
def calculate_flux_terms(
        nverts, concentrations, diffusion_rgtp, edgeplus_lengths
):
    result = np.empty(nverts, dtype=np.float64)

    for i in range(nverts):
        i_plus1_ix = (i + 1) % nverts

        result[i] = (
                - diffusion_rgtp
                * (concentrations[i_plus1_ix] - concentrations[i])
                / edgeplus_lengths[i]
        )

    return result


# -----------------------------------------------------------------
# @nb.jit(nopython=True)
def calculate_diffusion(
        nverts, concentrations, diffusion_rgtp, edgeplus_lengths
):
    result = np.empty(nverts, dtype=np.float64)

    fluxes = calculate_flux_terms(
        nverts,
        concentrations,
        diffusion_rgtp,
        edgeplus_lengths,
    )

    for i in range(nverts):
        i_minus1_ix = (i - 1) % nverts

        result[i] = fluxes[i_minus1_ix] - fluxes[i]

    return result


# -----------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_intercellular_contact_factors(
        this_cell_ix,
        nverts,
        num_cells,
        intercellular_contact_factor_magnitudes,
        close_point_smoothness_factors,
):
    intercellular_contact_factors = np.zeros(nverts, dtype=np.float64)

    for other_ci in range(num_cells):
        if other_ci != this_cell_ix:
            for ni in range(nverts):
                current_ic_mag = intercellular_contact_factors[ni]

                new_ic_mag = (
                        intercellular_contact_factor_magnitudes[other_ci]
                        * close_point_smoothness_factors[ni][other_ci]
                )

                if new_ic_mag > current_ic_mag:
                    intercellular_contact_factors[ni] = new_ic_mag

    return intercellular_contact_factors


# -----------------------------------------------------------------
# @nb.jit(nopython=True)
def calculate_x_coas(
        this_cell_ix,
        nverts,
        random_order_cell_indices,
        coa_distrib_exp,
        cell_dependent_x_coa_strengths,
        intercellular_dist_squared_matrix,
        line_segment_intersection_matrix,
        los_factor,
):
    x_coas = np.zeros(nverts, dtype=np.float64)
    too_close_dist_squared = 1e-6

    for ni in range(nverts):
        this_node_x_coa = x_coas[ni]

        this_node_relevant_line_seg_intersection_slice = \
            line_segment_intersection_matrix[
                ni]
        this_node_relevant_dist_squared_slice = \
            intercellular_dist_squared_matrix[ni]

        for other_ci in random_order_cell_indices:
            if other_ci != this_cell_ix:
                signal_strength = cell_dependent_x_coa_strengths[other_ci]

                this_node_other_cell_relevant_line_seg_intersection_slice = \
                    this_node_relevant_line_seg_intersection_slice[
                        other_ci]
                relevant_dist_squared_slice = \
                    this_node_relevant_dist_squared_slice[
                        other_ci]
                for other_ni in range(nverts):
                    line_segment_between_node_intersects_polygon = \
                        this_node_other_cell_relevant_line_seg_intersection_slice[
                            other_ni]
                    intersection_factor = (
                            1.0
                            / (
                                    line_segment_between_node_intersects_polygon + 1.0)
                            ** los_factor
                    )

                    dist_squared_between_nodes = \
                        relevant_dist_squared_slice[other_ni]

                    # print("====================")
                    # print("(ci: {}, vi: {}, ovi: {}, oci: {}):".format(
                    #     this_cell_ix, ni, other_ci, other_ni))
                    # print("dist: {}".format(np.sqrt(
                    # dist_squared_between_nodes)))
                    # print("num_intersects: {}".format(
                    #     line_segment_between_node_intersects_polygon))
                    # print("los_factor: {}".format(
                    #     intersection_factor))

                    x_coa = 0.0
                    if dist_squared_between_nodes > too_close_dist_squared:
                        x_coa = (
                                np.exp(
                                    coa_distrib_exp
                                    * np.sqrt(dist_squared_between_nodes)
                                )
                                * intersection_factor
                        )
                    # old_x_coa = this_node_x_coa
                    this_node_x_coa += x_coa * signal_strength
                    # print("x_coa: {}".format(x_coa))
                    # print("signal_strength: {}".format(signal_strength))
                    # print(
                    #     "new = {} + {}".format(old_x_coa,
                    #                            x_coa * signal_strength))

        x_coas[ni] = this_node_x_coa

    return x_coas


# -------------------------------------------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_chemoattractant_shielding_effect_factors(
        this_cell_ix,
        nverts,
        num_cells,
        intercellular_dist_squared_matrix,
        line_segment_intersection_matrix,
        chemoattractant_shielding_effect_length,
):
    chemoattractant_shielding_effect_factors = np.zeros(
        nverts, dtype=np.float64)

    for ni in range(nverts):
        this_node_relevant_line_seg_intersection_slice = \
            line_segment_intersection_matrix[
                ni]
        this_node_relevant_dist_squared_slice = \
            intercellular_dist_squared_matrix[ni]

        sum_weights = 0.0

        for other_ci in range(num_cells):
            if other_ci != this_cell_ix:
                this_node_other_cell_relevant_line_seg_intersection_slice = \
                    this_node_relevant_line_seg_intersection_slice[
                        other_ci]
                this_node_other_cell_relevant_dist_squared_slice = \
                    this_node_relevant_dist_squared_slice[
                        other_ci]

                for other_ni in range(nverts):
                    if (
                            this_node_other_cell_relevant_line_seg_intersection_slice[
                                other_ni] == 0):
                        ds = this_node_other_cell_relevant_dist_squared_slice[
                            other_ni]
                        sum_weights += np.exp(
                            np.log(0.25)
                            * (ds / chemoattractant_shielding_effect_length)
                        )

        chemoattractant_shielding_effect_factors[ni] = 1.0 / \
                                                       (1.0 + sum_weights)

    return chemoattractant_shielding_effect_factors
