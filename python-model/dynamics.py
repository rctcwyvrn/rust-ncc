# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:22:43 2015

@author: Brian
"""


import writer as fw
import utilities as general_utilities
import numba as nb
import numpy as np

import chemistry
import geometry
import mechanics


# ----------------------------------------------------------------------------------------
def pack_state_array(
        nodal_phase_var_indices,
        ode_cellwide_phase_var_indices,
        system_info_at_tstep):
    nodal_phase_var_array = (
        np.transpose(system_info_at_tstep[:, nodal_phase_var_indices])
    ).flatten()
    ode_cellwide_phase_var_array = system_info_at_tstep[
        0, ode_cellwide_phase_var_indices
    ]

    return np.append(nodal_phase_var_array, ode_cellwide_phase_var_array)


# ----------------------------------------------------------------------------------------
def unpack_state_array(num_nodal_phase_var_indices, state_array):
    # reversing append
    node_phase_var_array = state_array
    ode_cellwide_phase_vars = np.array([])

    # reversing flatten
    nodal_phase_vars = np.split(
        node_phase_var_array,
        num_nodal_phase_var_indices)

    return nodal_phase_vars, ode_cellwide_phase_vars


# ----------------------------------------------------------------------------------------
def pack_state_array_from_system_history(
        nodal_phase_var_indices,
        ode_cellwide_phase_var_indices,
        system_info,
        access_index):
    system_info_at_tstep = system_info[access_index]
    state_array = pack_state_array(
        nodal_phase_var_indices,
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
        this_cell_index,
    num_nodes,
    num_nodal_phase_vars,
        nodal_rac_membrane_active_index,
    length_edge_resting,
    nodal_rac_membrane_inactive_index,
    nodal_rho_membrane_active_index,
    nodal_rho_membrane_inactive_index,
    nodal_x_index,
    nodal_y_index,
    kgtp_rac_baseline,
    kdgtp_rac_baseline,
    kgtp_rho_baseline,
    kdgtp_rho_baseline,
    kgtp_rac_autoact_baseline,
    kgtp_rho_autoact_baseline,
    kdgtp_rho_mediated_rac_inhib_baseline,
    kdgtp_rac_mediated_rho_inhib_baseline,
    kgdi_rac,
    kdgdi_rac,
    kgdi_rho,
    kdgdi_rho,
    threshold_rac_autoact,
    threshold_rho_autoact,
    threshold_rho_mediated_rac_inhib,
    threshold_rac_mediated_rho_inhib,
    exponent_rac_autoact,
    exponent_rho_autoact,
    exponent_rho_mediated_rac_inhib,
    exponent_rac_mediated_rho_inhib,
    diffusion_const_active,
    diffusion_const_inactive,
        eta,
        stiffness_edge,
    threshold_force_rac_activity,
    threshold_force_rho_activity,
    max_force_rac,
    max_force_rho,
        area_resting,
    stiffness_cytoplasmic,
    transduced_coa_signals,
        close_point_smoothness_factors,
    intercellular_contact_factors,
    tension_mediated_rac_inhibition_half_strain,
    tension_mediated_rac_inhibition_magnitude,
        randomization_rac_kgtp_multipliers,
):

    nodal_phase_vars = state_array

    rac_mem_active_start_index = nodal_rac_membrane_active_index * num_nodes
    rac_mem_active_end_index = rac_mem_active_start_index + num_nodes

    rac_membrane_actives = nodal_phase_vars[
        rac_mem_active_start_index:rac_mem_active_end_index
    ]

    rac_mem_inactive_start_index = nodal_rac_membrane_inactive_index * num_nodes
    rac_mem_inactive_end_index = rac_mem_inactive_start_index + num_nodes

    rac_membrane_inactives = nodal_phase_vars[
        rac_mem_inactive_start_index:rac_mem_inactive_end_index
    ]

    rho_mem_active_start_index = nodal_rho_membrane_active_index * num_nodes
    rho_mem_active_end_index = rho_mem_active_start_index + num_nodes

    rho_membrane_actives = nodal_phase_vars[
        rho_mem_active_start_index:rho_mem_active_end_index
    ]

    rho_mem_inactive_start_index = nodal_rho_membrane_inactive_index * num_nodes
    rho_mem_inactive_end_index = rho_mem_inactive_start_index + num_nodes

    rho_membrane_inactives = nodal_phase_vars[
        rho_mem_inactive_start_index:rho_mem_inactive_end_index
    ]

    nodal_x_start_index = nodal_x_index * num_nodes
    nodal_x_end_index = nodal_x_start_index + num_nodes

    nodal_x = nodal_phase_vars[nodal_x_start_index:nodal_x_end_index]

    nodal_y_start_index = nodal_y_index * num_nodes
    nodal_y_end_index = nodal_y_start_index + num_nodes

    nodal_y = nodal_phase_vars[nodal_y_start_index:nodal_y_end_index]

    node_coords = general_utilities.make_node_coords_array_given_xs_and_ys(
        num_nodes, nodal_x, nodal_y
    )

    rac_cytosolic_gdi_bound = (
        1
        - calculate_sum(num_nodes, rac_membrane_actives)
        - calculate_sum(num_nodes, rac_membrane_inactives)
    )
    rho_cytosolic_gdi_bound = (
        1
        - calculate_sum(num_nodes, rho_membrane_actives)
        - calculate_sum(num_nodes, rho_membrane_inactives)
    )

    F, EFplus, EFminus, F_rgtpase, F_cytoplasmic, edge_strains, local_strains, \
    unit_inside_pointing_vectors = mechanics.calculate_forces(
        num_nodes,
        node_coords,
        rac_membrane_actives,
        rho_membrane_actives,
        length_edge_resting,
        stiffness_edge,
        threshold_force_rac_activity,
        threshold_force_rho_activity,
        max_force_rac,
        max_force_rho,
        area_resting,
        stiffness_cytoplasmic,
    )

    F_x = F[:, 0]
    F_y = F[:, 1]

    only_tensile_local_strains = np.zeros_like(local_strains)
    for i in range(num_nodes):
        local_strain = local_strains[i]
        if local_strain > 0:
            only_tensile_local_strains[i] = local_strain

    edgeplus_lengths = geometry.calculate_edgeplus_lengths(node_coords)
    avg_edge_lengths = geometry.calculate_average_edge_length_around_nodes(
        edgeplus_lengths
    )

    conc_rac_membrane_actives = chemistry.calculate_concentrations(
        num_nodes, rac_membrane_actives, avg_edge_lengths
    )

    kgtps_rac = chemistry.calculate_kgtp_rac(
        conc_rac_membrane_actives,
        exponent_rac_autoact,
        threshold_rac_autoact,
        kgtp_rac_baseline,
        kgtp_rac_autoact_baseline,
        transduced_coa_signals,
        randomization_rac_kgtp_multipliers,
        intercellular_contact_factors,
        close_point_smoothness_factors,
    )

    conc_rho_membrane_actives = chemistry.calculate_concentrations(
        num_nodes, rho_membrane_actives, avg_edge_lengths
    )

    kdgtps_rac = chemistry.calculate_kdgtp_rac(
        num_nodes,
        conc_rho_membrane_actives,
        exponent_rho_mediated_rac_inhib,
        threshold_rho_mediated_rac_inhib,
        kdgtp_rac_baseline,
        kdgtp_rho_mediated_rac_inhib_baseline,
        intercellular_contact_factors,
        tension_mediated_rac_inhibition_half_strain,
        tension_mediated_rac_inhibition_magnitude,
        only_tensile_local_strains,
    )

    kdgdis_rac = kdgdi_rac * np.ones(num_nodes, dtype=np.float64)

    kgtps_rho = chemistry.calculate_kgtp_rho(
        num_nodes,
        conc_rho_membrane_actives,
        intercellular_contact_factors,
        exponent_rho_autoact,
        threshold_rho_autoact,
        kgtp_rho_baseline,
        kgtp_rho_autoact_baseline,
    )

    kdgtps_rho = chemistry.calculate_kdgtp_rho(
        num_nodes,
        conc_rac_membrane_actives,
        exponent_rac_mediated_rho_inhib,
        threshold_rac_mediated_rho_inhib,
        kdgtp_rho_baseline,
        kdgtp_rac_mediated_rho_inhib_baseline,
    )

    kdgdis_rho = kdgdi_rho * np.ones(num_nodes, dtype=np.float64)

    conc_rac_membrane_inactives = chemistry.calculate_concentrations(
        num_nodes, rac_membrane_inactives, avg_edge_lengths
    )
    conc_rho_membrane_inactives = chemistry.calculate_concentrations(
        num_nodes, rho_membrane_inactives, avg_edge_lengths
    )

    diffusion_rac_membrane_active = chemistry.calculate_diffusion(
        num_nodes,
        conc_rac_membrane_actives,
        diffusion_const_active,
        edgeplus_lengths,
    )
    diffusion_rac_membrane_inactive = chemistry.calculate_diffusion(
        num_nodes,
        conc_rac_membrane_inactives,
        diffusion_const_inactive,
        edgeplus_lengths,
    )
    diffusion_rho_membrane_active = chemistry.calculate_diffusion(
        num_nodes,
        conc_rho_membrane_actives,
        diffusion_const_active,
        edgeplus_lengths,
    )
    diffusion_rho_membrane_inactive = chemistry.calculate_diffusion(
        num_nodes,
        conc_rho_membrane_inactives,
        diffusion_const_active,
        edgeplus_lengths,
    )

    delta_rac_activated = np.zeros(num_nodes, dtype=np.float64)
    delta_rac_inactivated = np.zeros(num_nodes, dtype=np.float64)

    delta_rac_cytosol_to_membrane = np.zeros(num_nodes, dtype=np.float64)

    delta_rho_activated = np.zeros(num_nodes, dtype=np.float64)
    delta_rho_inactivated = np.zeros(num_nodes, dtype=np.float64)

    delta_rho_cytosol_to_membrane = np.zeros(num_nodes, dtype=np.float64)

    delta_nodal_x = np.zeros(num_nodes, dtype=np.float64)
    delta_nodal_y = np.zeros(num_nodes, dtype=np.float64)
    new_node_coords = np.zeros((num_nodes, 2), dtype=np.float64)
    np.zeros(2, dtype=np.float64)
    np.zeros(2, dtype=np.float64)

    for ni in range(num_nodes):
        old_coord = node_coords[ni]

        new_node_coords[ni][0] = old_coord[0] + F_x[ni] / eta
        new_node_coords[ni][1] = old_coord[1] + F_y[ni] / eta

    # # calculate volume exclusion effects
    # num_bisection_iterations = 2
    # max_movement_mag = max_force_rac / eta
    # success_condition_stay_out = 0
    # success_condition_stay_in = 1
    #
    # are_new_nodes_inside_other_cell = np.zeros(num_nodes, dtype=np.int64)
    # for other_ci in range(num_cells):
    #     if other_ci != this_cell_index:
    #         are_new_nodes_inside_other_cell = geometry.are_points_inside_polygon(
    #             new_node_coords, all_cells_node_coords[other_ci]
    #         )
    #
    #         for ni in range(num_nodes):
    #             if are_new_nodes_inside_other_cell[ni] != success_condition_stay_out:
    #                 new_node_coords[ni] = determine_volume_exclusion_effects(
    #                     node_coords[ni],
    #                     new_node_coords[ni],
    #                     unit_inside_pointing_vectors[ni],
    #                     all_cells_node_coords[other_ci],
    #                     num_bisection_iterations,
    #                     max_movement_mag,
    #                     success_condition_stay_out,
    #                 )
    #
    # if exists_space_physical_bdry_polygon == 1:
    #     are_new_nodes_inside_space_physical_bdry_polygon = geometry.are_points_inside_polygon(
    #         new_node_coords, space_physical_bdry_polygon
    #     )
    #
    #     for ni in range(num_nodes):
    #         if (
    #             are_new_nodes_inside_space_physical_bdry_polygon[ni]
    #             != success_condition_stay_in
    #         ):
    #             new_node_coords[ni] = determine_volume_exclusion_effects(
    #                 node_coords[ni],
    #                 new_node_coords[ni],
    #                 unit_inside_pointing_vectors[ni],
    #                 space_physical_bdry_polygon,
    #                 num_bisection_iterations,
    #                 max_movement_mag,
    #                 success_condition_stay_in,
    #             )
    poly_area = geometry.calculate_polygon_area(node_coords)
    fw.write(["------------------------------",
              "poly: {}".format([list(x) for x in node_coords]),
              "rac_acts: {}".format(list(rac_membrane_actives)),
              "rac_inacts: {}".format(list(rac_membrane_inactives)),
              "rho_acts: {}".format(list(rho_membrane_actives)),
              "rho_inacts: {}".format(list(rho_membrane_inactives)),
              "tot_forces: {}".format([list([x, y]) for x, y in zip(F_x, F_y)]),
              "uivs: {}".format([list(x) for x in
                                 unit_inside_pointing_vectors]),
              "rgtp_forces: {}".format([list(x) for x in F_rgtpase]),
              "edge_forces: {}".format([list(x) for x in EFplus]),
              "cyto_forces: {}".format([list(x) for x in F_cytoplasmic]),
              "kgtps_rac: {}".format(list(kgtps_rac)),
              "kdgtps_rac: {}".format(list(kdgtps_rac)),
              "kgtps_rho: {}".format(list(kgtps_rho)),
              "kdgtps_rho: {}".format(list(kdgtps_rho)),
              "conc_rac_acts: {}".format(list(conc_rac_membrane_actives)),
              "x_cils: {}".format(list(intercellular_contact_factors)),
              "x_coas: {}".format(list(transduced_coa_signals)),
              "rac_act_net_fluxes: {}".format(list(
                  diffusion_rac_membrane_active)),
              "edge_strains: {}".format(list(
                  edge_strains)),
              "poly_area: {}".format([poly_area] * num_nodes),
              "------------------------------",
              ])

    for ni in range(num_nodes):
        new_coord = new_node_coords[ni]
        old_coord = node_coords[ni]

        delta_nodal_x[ni] = new_coord[0] - old_coord[0]
        delta_nodal_y[ni] = new_coord[1] - old_coord[1]

    for ni in range(num_nodes):
        # finish assigning chemistry variables
        delta_rac_activated[ni] = kgtps_rac[ni] * rac_membrane_inactives[ni]
        delta_rac_inactivated[ni] = kdgtps_rac[ni] * rac_membrane_actives[ni]

        delta_rac_on = kdgdis_rac[ni] * rac_cytosolic_gdi_bound
        delta_rac_off = kgdi_rac * rac_membrane_inactives[ni]
        delta_rac_cytosol_to_membrane[ni] = delta_rac_on - delta_rac_off

        delta_rho_activated[ni] = kgtps_rho[ni] * rho_membrane_inactives[ni]
        delta_rho_inactivated[ni] = kdgtps_rho[ni] * rho_membrane_actives[ni]

        delta_rho_on = kdgdis_rho[ni] * rho_cytosolic_gdi_bound
        delta_rho_off = kgdi_rho * rho_membrane_inactives[ni]
        delta_rho_cytosol_to_membrane[ni] = delta_rho_on - delta_rho_off

    # set up ode array
    ode_array = np.empty(num_nodal_phase_vars * num_nodes)

    for i in range(num_nodes):
        ode_array[i] = (
            delta_rac_activated[i]
            - delta_rac_inactivated[i]
            + diffusion_rac_membrane_active[i]
        )

        ode_array[i + num_nodes] = (
            delta_rac_inactivated[i]
            - delta_rac_activated[i]
            + diffusion_rac_membrane_inactive[i]
            + delta_rac_cytosol_to_membrane[i]
        )

        ode_array[i + 2 * num_nodes] = (
            delta_rho_activated[i]
            - delta_rho_inactivated[i]
            + diffusion_rho_membrane_active[i]
        )

        ode_array[i + 3 * num_nodes] = (
            delta_rho_inactivated[i]
            - delta_rho_activated[i]
            + diffusion_rho_membrane_inactive[i]
            + delta_rho_cytosol_to_membrane[i]
        )

        ode_array[i + 4 * num_nodes] = delta_nodal_x[i]

        ode_array[i + 5 * num_nodes] = delta_nodal_y[i]

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

    min_x, max_x, min_y, max_y = geometry.calculate_polygon_bounding_box(
        polygon)

    old_coord_status = geometry.is_point_in_polygon_given_bounding_box(
        old_coord, polygon, min_x, max_x, min_y, max_y
    )

    # we know that the new coord is not in the polygon, now, so we test the
    # old_coord
    if old_coord_status != success_exclusion_condition:
        while old_coord_status != success_exclusion_condition:
            old_coord = old_coord + max_movement_mag * unit_inside_pointing_vector
            # num_bisection_iterations = int(num_bisection_iterations*1.5)
            old_coord_status = geometry.is_point_in_polygon_given_bounding_box(
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
            geometry.is_point_in_polygon_given_bounding_box(
                test_coord, polygon, min_x, max_x, min_y, max_y
            )
            == success_exclusion_condition
        ):
            a = test_coord
        else:
            b = test_coord

    return a
