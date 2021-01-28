# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:22:43 2015

@author: Brian
"""


import hardio as fw
import utilities as general_utilities
import numba as nb
import numpy as np

import chemistry
import geometry
import mechanics


# ----------------------------------------------------------------------------------------
def pack_state_array(
        phase_var_indices,
        ode_cellwide_phase_var_indices,
        system_info_at_tstep):
    phase_var_array = (
        np.transpose(system_info_at_tstep[:, phase_var_indices])
    ).flatten()
    ode_cellwide_phase_var_array = system_info_at_tstep[
        0, ode_cellwide_phase_var_indices
    ]

    return np.append(phase_var_array, ode_cellwide_phase_var_array)


# ----------------------------------------------------------------------------------------
def unpack_state_array(num_phase_var_indices, state_array):
    # reversing append
    node_phase_var_array = state_array
    ode_cellwide_phase_vars = np.array([])

    # reversing flatten
    phase_vars = np.split(
        node_phase_var_array,
        num_phase_var_indices)

    return phase_vars, ode_cellwide_phase_vars


# ----------------------------------------------------------------------------------------
def pack_state_array_from_system_history(
        phase_var_indices,
        ode_cellwide_phase_var_indices,
        system_info,
        access_ix):
    system_info_at_tstep = system_info[access_ix]
    state_array = pack_state_array(
        phase_var_indices,
        ode_cellwide_phase_var_indices,
        system_info_at_tstep)

    return state_array


# ----------------------------------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_sum(num_elements, sequence):
    result = 0
    for i in range(num_elements):
        result += sequence[i]

    return result


# ----------------------------------------------------------------------------------------
def eulerint(f, current_state, tpoints, args, num_int_steps):
    num_tpoint_pairs = tpoints.shape[0] - 1
    tpoint_pairs = np.zeros((num_tpoint_pairs, 2), dtype=np.float64)
    states = np.zeros(
        (tpoints.shape[0],
         current_state.shape[0]),
        dtype=np.float64)

    states[0] = current_state

    for i in range(num_tpoint_pairs):
        j = 2 * i
        tpoint_pairs[i] = tpoints[j:j + 2]

    for i in range(tpoint_pairs.shape[0]):
        init_t, end_t = tpoint_pairs[i]
        current_state = states[i]
        dt = (end_t - init_t) / num_int_steps

        for j in range(num_int_steps):
            current_state = current_state + dt * f(current_state, *args)

        states[i + 1] = current_state

    return states

# @nb.jit()


def cell_dynamics(
    state_array,
    nverts,
    num_phase_vars,
    rac_acts_ix,
    rest_edge_len,
    rac_inacts_ix,
    rho_act_ix,
    rho_inacts_ix,
    x_ix,
    y_ix,
    kgtp_rac,
    kdgtp_rac,
    kgtp_rho,
    kdgtp_rho,
    kgtp_rac_auto,
    kgtp_rho_auto,
    kdgtp_rho_on_rac,
    kdgtp_rac_on_rho,
    k_mem_off,
    k_mem_on,
    halfmax_rgtp_frac,
    diffusion_rgtp,
        eta,
        stiffness_edge,
    threshold_force_rac_activity,
    threshold_force_rho_activity,
    max_force_rac,
    max_force_rho,
        rest_area,
    stiffness_cyto,
    transduced_x_coas,
        close_point_smoothness_factors,
    intercellular_contact_factors,
    halfmax_tension_inhib,
    tension_inhib,
        rac_rands,
):

    phase_vars = state_array

    rac_mem_active_start_ix = rac_acts_ix * nverts
    rac_mem_active_end_ix = rac_mem_active_start_ix + nverts

    rac_acts = phase_vars[
        rac_mem_active_start_ix:rac_mem_active_end_ix
    ]

    rac_mem_inactive_start_ix = rac_inacts_ix * nverts
    rac_mem_inactive_end_ix = rac_mem_inactive_start_ix + nverts

    rac_inacts = phase_vars[
        rac_mem_inactive_start_ix:rac_mem_inactive_end_ix
    ]

    rho_mem_active_start_ix = rho_act_ix * nverts
    rho_mem_active_end_ix = rho_mem_active_start_ix + nverts

    rho_acts = phase_vars[
        rho_mem_active_start_ix:rho_mem_active_end_ix
    ]

    rho_mem_inactive_start_ix = rho_inacts_ix * nverts
    rho_mem_inactive_end_ix = rho_mem_inactive_start_ix + nverts

    rho_inactss = phase_vars[
        rho_mem_inactive_start_ix:rho_mem_inactive_end_ix
    ]

    x_start_ix = x_ix * nverts
    x_end_ix = x_start_ix + nverts

    x = phase_vars[x_start_ix:x_end_ix]

    y_start_ix = y_ix * nverts
    y_end_ix = y_start_ix + nverts

    y = phase_vars[y_start_ix:y_end_ix]

    verts = general_utilities.make_verts_array_given_xs_and_ys(
        nverts, x, y
    )

    rac_cyto = (
        1
        - calculate_sum(nverts, rac_acts)
        - calculate_sum(nverts, rac_inacts)
    )
    rho_cyto = (
        1
        - calculate_sum(nverts, rho_acts)
        - calculate_sum(nverts, rho_inactss)
    )

    F, edge_forces_plus, edge_forces_minus, rgtp_forces, cyto_forces, edge_strains, local_strains, \
    uivs = mechanics.calculate_forces(
        nverts,
        verts,
        rac_acts,
        rho_acts,
        rest_edge_len,
        stiffness_edge,
        threshold_force_rac_activity,
        threshold_force_rho_activity,
        max_force_rac,
        max_force_rho,
        rest_area,
        stiffness_cyto,
    )

    sum_forces_x = F[:, 0]
    sum_forces_y = F[:, 1]

    only_tensile_local_strains = np.zeros_like(local_strains)
    for i in range(nverts):
        local_strain = local_strains[i]
        if local_strain > 0:
            only_tensile_local_strains[i] = local_strain

    edgeplus_lengths = geometry.calculate_edgeplus_lengths(verts)
    avg_edge_lengths = geometry.calculate_average_edge_length_around_nodes(
        edgeplus_lengths
    )

    conc_rac_acts = chemistry.calculate_concentrations(
        nverts, rac_acts, avg_edge_lengths
    )

    kgtps_rac = chemistry.calculate_kgtp_rac(
        conc_rac_acts,
        halfmax_rgtp_frac,
        kgtp_rac,
        kgtp_rac_auto,
        transduced_x_coas,
        rac_rands,
        intercellular_contact_factors,
        close_point_smoothness_factors,
    )

    conc_rho_acts = chemistry.calculate_concentrations(
        nverts, rho_acts, avg_edge_lengths
    )

    kdgtps_rac = chemistry.calculate_kdgtp_rac(
        nverts,
        conc_rho_acts,
        halfmax_rgtp_frac,
        kdgtp_rac,
        kdgtp_rho_on_rac,
        intercellular_contact_factors,
        halfmax_tension_inhib,
        tension_inhib,
        only_tensile_local_strains,
    )

    kdgdis_rac = k_mem_on * np.ones(nverts, dtype=np.float64)

    kgtps_rho = chemistry.calculate_kgtp_rho(
        nverts,
        conc_rho_acts,
        intercellular_contact_factors,
        halfmax_rgtp_frac,
        kgtp_rho,
        kgtp_rho_auto,
    )

    kdgtps_rho = chemistry.calculate_kdgtp_rho(
        nverts,
        conc_rac_acts,
        halfmax_rgtp_frac,
        kdgtp_rho,
        kdgtp_rac_on_rho,
    )

    kdgdis_rho = kdgdi_rho * np.ones(nverts, dtype=np.float64)

    conc_rac_inacts = chemistry.calculate_concentrations(
        nverts, rac_inacts, avg_edge_lengths
    )
    conc_rho_inactss = chemistry.calculate_concentrations(
        nverts, rho_inactss, avg_edge_lengths
    )

    diffusion_rac_acts = chemistry.calculate_diffusion(
        nverts,
        conc_rac_acts,
        diffusion_rgtp,
        edgeplus_lengths,
    )
    diffusion_rac_inacts = chemistry.calculate_diffusion(
        nverts,
        conc_rac_inacts,
        diffusion_rgtp,
        edgeplus_lengths,
    )
    diffusion_rho_act = chemistry.calculate_diffusion(
        nverts,
        conc_rho_acts,
        diffusion_rgtp,
        edgeplus_lengths,
    )
    diffusion_rho_inacts = chemistry.calculate_diffusion(
        nverts,
        conc_rho_inactss,
        diffusion_rgtp,
        edgeplus_lengths,
    )

    delta_rac_activated = np.zeros(nverts, dtype=np.float64)
    delta_rac_inactivated = np.zeros(nverts, dtype=np.float64)

    delta_rac_cytosol_to_membrane = np.zeros(nverts, dtype=np.float64)

    delta_rho_activated = np.zeros(nverts, dtype=np.float64)
    delta_rho_inactivated = np.zeros(nverts, dtype=np.float64)

    delta_rho_cytosol_to_membrane = np.zeros(nverts, dtype=np.float64)

    delta_x = np.zeros(nverts, dtype=np.float64)
    delta_y = np.zeros(nverts, dtype=np.float64)
    new_verts = np.zeros((nverts, 2), dtype=np.float64)
    np.zeros(2, dtype=np.float64)
    np.zeros(2, dtype=np.float64)

    for ni in range(nverts):
        old_coord = verts[ni]

        new_verts[ni][0] = old_coord[0] + sum_forces_x[ni] / eta
        new_verts[ni][1] = old_coord[1] + sum_forces_y[ni] / eta

    # # calculate volume exclusion effects
    # num_bisection_iterations = 2
    # max_movement_mag = max_force_rac / eta
    # success_condition_stay_out = 0
    # success_condition_stay_in = 1
    #
    # are_new_nodes_inside_other_cell = np.zeros(nverts, dtype=np.int64)
    # for other_ci in range(num_cells):
    #     if other_ci != this_cell_ix:
    #         are_new_nodes_inside_other_cell = geometry.are_points_inside_polygon(
    #             new_verts, all_cells_verts[other_ci]
    #         )
    #
    #         for ni in range(nverts):
    #             if are_new_nodes_inside_other_cell[ni] != success_condition_stay_out:
    #                 new_verts[ni] = determine_volume_exclusion_effects(
    #                     verts[ni],
    #                     new_verts[ni],
    #                     uivs[ni],
    #                     all_cells_verts[other_ci],
    #                     num_bisection_iterations,
    #                     max_movement_mag,
    #                     success_condition_stay_out,
    #                 )
    #
    # if exists_space_physical_bdry_polygon == 1:
    #     are_new_nodes_inside_space_physical_bdry_polygon = geometry.are_points_inside_polygon(
    #         new_verts, space_physical_bdry_polygon
    #     )
    #
    #     for ni in range(nverts):
    #         if (
    #             are_new_nodes_inside_space_physical_bdry_polygon[ni]
    #             != success_condition_stay_in
    #         ):
    #             new_verts[ni] = determine_volume_exclusion_effects(
    #                 verts[ni],
    #                 new_verts[ni],
    #                 uivs[ni],
    #                 space_physical_bdry_polygon,
    #                 num_bisection_iterations,
    #                 max_movement_mag,
    #                 success_condition_stay_in,
    #             )
    poly_area = geometry.calculate_polygon_area(verts)
    fw.write(["------------------------------",
              "poly: {}".format([list(x) for x in verts]),
              "rac_acts: {}".format(list(rac_acts)),
              "rac_inacts: {}".format(list(rac_inacts)),
              "rho_acts: {}".format(list(rho_acts)),
              "rho_inacts: {}".format(list(rho_inactss)),
              "tot_forces: {}".format([list([x, y]) for x, y in zip(sum_forces_x, sum_forces_y)]),
              "uivs: {}".format([list(x) for x in
                                 uivs]),
              "rgtp_forces: {}".format([list(x) for x in rgtp_forces]),
              "edge_forces: {}".format([list(x) for x in edge_forces_plus]),
              "cyto_forces: {}".format([list(x) for x in cyto_forces]),
              "kgtps_rac: {}".format(list(kgtps_rac)),
              "kdgtps_rac: {}".format(list(kdgtps_rac)),
              "kgtps_rho: {}".format(list(kgtps_rho)),
              "kdgtps_rho: {}".format(list(kdgtps_rho)),
              "conc_rac_acts: {}".format(list(conc_rac_acts)),
              "x_cils: {}".format(list(intercellular_contact_factors)),
              "x_coas: {}".format(list(transduced_x_coas)),
              "rac_act_net_fluxes: {}".format(list(
                  diffusion_rac_acts)),
              "edge_strains: {}".format(list(
                  edge_strains)),
              "poly_area: {}".format([poly_area] * nverts),
              "------------------------------",
              ])

    for ni in range(nverts):
        new_coord = new_verts[ni]
        old_coord = verts[ni]

        delta_x[ni] = new_coord[0] - old_coord[0]
        delta_y[ni] = new_coord[1] - old_coord[1]

    for ni in range(nverts):
        # finish assigning chemistry variables
        delta_rac_activated[ni] = kgtps_rac[ni] * rac_inacts[ni]
        delta_rac_inactivated[ni] = kdgtps_rac[ni] * rac_acts[ni]

        delta_rac_on = kdgdis_rac[ni] * rac_cyto
        delta_rac_off = k_mem_off * rac_inacts[ni]
        delta_rac_cytosol_to_membrane[ni] = delta_rac_on - delta_rac_off

        delta_rho_activated[ni] = kgtps_rho[ni] * rho_inactss[ni]
        delta_rho_inactivated[ni] = kdgtps_rho[ni] * rho_acts[ni]

        delta_rho_on = k_mem_on[ni] * rho_cyto
        delta_rho_off = k_mem_off * rho_inactss[ni]
        delta_rho_cytosol_to_membrane[ni] = delta_rho_on - delta_rho_off

    # set up ode array
    ode_array = np.empty(num_phase_vars * nverts)

    for i in range(nverts):
        ode_array[i] = (
            delta_rac_activated[i]
            - delta_rac_inactivated[i]
            + diffusion_rac_acts[i]
        )

        ode_array[i + nverts] = (
            delta_rac_inactivated[i]
            - delta_rac_activated[i]
            + diffusion_rac_inacts[i]
            + delta_rac_cytosol_to_membrane[i]
        )

        ode_array[i + 2 * nverts] = (
            delta_rho_activated[i]
            - delta_rho_inactivated[i]
            + diffusion_rho_act[i]
        )

        ode_array[i + 3 * nverts] = (
            delta_rho_inactivated[i]
            - delta_rho_activated[i]
            + diffusion_rho_inacts[i]
            + delta_rho_cytosol_to_membrane[i]
        )

        ode_array[i + 4 * nverts] = delta_x[i]

        ode_array[i + 5 * nverts] = delta_y[i]

    return ode_array


# -----------------------------------------------------------------
@nb.jit(nopython=True)
def determine_volume_exclusion_effects(
    old_coord,
    new_coord,
    unit_inside_pointing_vector,
    polygon,
    num_bisection_iterations,
    max_movement_mag,
    success_exclusion_condition,
):

    #    if success_exclusion_condition == 1:
    #        movement_mag = geometry.calculate_2D_vector_mag(old_coord - new_coord)
    #        return old_coord + movement_mag*unit_inside_pointing_vector

    min_x, max_x, min_y, max_y = geometry.calculate_polygon_bb(
        polygon)

    old_coord_status = geometry.is_point_in_polygon_given_bb(
        old_coord, polygon, min_x, max_x, min_y, max_y
    )

    # we know that the new coord is not in the polygon, now, so we test the
    # old_coord
    if old_coord_status != success_exclusion_condition:
        while old_coord_status != success_exclusion_condition:
            old_coord = old_coord + max_movement_mag * unit_inside_pointing_vector
            # num_bisection_iterations = int(num_bisection_iterations*1.5)
            old_coord_status = geometry.is_point_in_polygon_given_bb(
                old_coord, polygon, min_x, max_x, min_y, max_y
            )

    # if we have reached here, then we know that the old_coord is in the
    # polygon, and the new coord is not in the polygon
    a = old_coord
    b = new_coord
    np.zeros(2, dtype=np.float64)

    for i in range(num_bisection_iterations):
        test_coord = 0.5 * (a + b)

        if (
            geometry.is_point_in_polygon_given_bb(
                test_coord, polygon, min_x, max_x, min_y, max_y
            )
            == success_exclusion_condition
        ):
            a = test_coord
        else:
            b = test_coord

    return a
