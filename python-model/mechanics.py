# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:26:37 2015

@author: Brian
"""

import numba as nb
import numpy as np

import geometry


# -----------------------------------------------------------------
# @nb.jit(nopython=True)
def capped_linear_function(max_x, x):
    if x > max_x:
        return 1.0
    else:
        return x / max_x


# -----------------------------------------------------------------
@nb.jit(nopython=True)
def linear_function(max_x, x):
    return x / max_x


# -----------------------------------------------------------------
@nb.jit(nopython=True)
def calculate_phys_space_bdry_contact_factors(
        this_cell_coords, space_physical_bdry_polygons
):

    if space_physical_bdry_polygons.size == 0:
        return np.zeros(len(this_cell_coords))
    else:
        return geometry.are_points_inside_polygons(
            this_cell_coords, space_physical_bdry_polygons
        )

# -----------------------------------------------------------------
# @nb.jit(nopython=True)
def calculate_cytoplasmic_force(
    nverts,
    this_cell_coords,
    rest_area,
    stiffness_cyto,
    uivs,
):
    current_area = abs(geometry.calculate_polygon_area(this_cell_coords))

    area_strain = (current_area - rest_area) / rest_area
    force_mag = area_strain * stiffness_cyto / nverts

    return geometry.multiply_vectors_by_scalar(
        uivs, force_mag
    )


# -----------------------------------------------------------------
#@nb.jit(nopython=True)
def calculate_spring_edge_forces(
    nverts, this_cell_coords, stiffness_edge, rest_edge_len
):

    edge_vectors_to_plus = np.empty((nverts, 2), dtype=np.float64)
    edge_vectors_to_minus = np.empty((nverts, 2), dtype=np.float64)

    for i in range(nverts):
        i_plus_1 = (i + 1) % nverts
        i_minus_1 = (i - 1) % nverts
        edge_vector_to_plus = geometry.calculate_vector_from_p1_to_p2_given_vectors(
            this_cell_coords[i], this_cell_coords[i_plus_1])
        edge_vector_to_minus = geometry.calculate_vector_from_p1_to_p2_given_vectors(
            this_cell_coords[i], this_cell_coords[i_minus_1])

        edge_vectors_to_plus[i, 0] = edge_vector_to_plus[0]
        edge_vectors_to_plus[i, 1] = edge_vector_to_plus[1]

        edge_vectors_to_minus[i, 0] = edge_vector_to_minus[0]
        edge_vectors_to_minus[i, 1] = edge_vector_to_minus[1]

    plus_dirn_edge_length = geometry.calculate_2D_vector_mags(
        edge_vectors_to_plus)

    minus_dirn_edge_length = geometry.calculate_2D_vector_mags(
        edge_vectors_to_minus)

    edge_strains_plus = np.empty(nverts, dtype=np.float64)
    edge_strains_minus = np.empty(nverts, dtype=np.float64)
    local_average_strains = np.empty(nverts, dtype=np.float64)

    for i in range(nverts):
        edge_strain_plus = (
            plus_dirn_edge_length[i] - rest_edge_len
        ) / rest_edge_len
        edge_strain_minus = (
            minus_dirn_edge_length[i] - rest_edge_len
        ) / rest_edge_len

        edge_strains_plus[i] = edge_strain_plus
        edge_strains_minus[i] = edge_strain_minus

        local_average_strains[i] = 0.5 * \
            edge_strain_plus + 0.5 * edge_strain_minus

    unit_edge_disp_vecs_plus = geometry.normalize_vectors(edge_vectors_to_plus)
    unit_edge_disp_vecs_minus = geometry.normalize_vectors(
        edge_vectors_to_minus)

    edge_forces_plus_mags = np.zeros(nverts, dtype=np.float64)
    edge_forces_minus_mags = np.zeros(nverts, dtype=np.float64)
    for i in range(nverts):
        edge_forces_plus_mags[i] = edge_strains_plus[i] * stiffness_edge
        edge_forces_minus_mags[i] = edge_strains_minus[i] * stiffness_edge

    edge_forces_plus = geometry.multiply_vectors_by_scalars(
        unit_edge_disp_vecs_plus, edge_forces_plus_mags)

    edge_forces_minus = geometry.multiply_vectors_by_scalars(
        unit_edge_disp_vecs_minus, edge_forces_minus_mags
    )

    return local_average_strains, edge_forces_plus, edge_forces_minus, edge_strains_plus


# -----------------------------------------------------------------
@nb.jit(nopython=True)
def determine_rac_rho_domination(rac_acts, rho_acts):
    nverts = rac_acts.shape[0]

    domination_array = np.empty(nverts, dtype=np.int64)

    for ni in range(nverts):
        if rac_acts[ni] < rho_acts[ni]:
            domination_array[ni] = 0
        else:
            domination_array[ni] = 1

    return domination_array


# -----------------------------------------------------------------


# @nb.jit(nopython=True)
def calculate_rgtpase_mediated_forces(
    nverts,
        rac_acts,
    rho_acts,
    threshold_force_rac_activity,
    threshold_force_rho_activity,
    max_force_rac,
    max_force_rho,
    uivs,
):
    rgtpase_mediated_force_mags = np.zeros(nverts, dtype=np.float64)

    for ni in range(nverts):
        rac_activity = rac_acts[ni]
        rho_activity = rho_acts[ni]

        if rac_activity > rho_activity:
            mag_frac = capped_linear_function(
                2 * threshold_force_rac_activity, rac_activity - rho_activity
            )
            force_mag = max_force_rac * mag_frac
        else:
            force_mag = (
                -1
                * max_force_rho
                * capped_linear_function(
                    2 * threshold_force_rho_activity, rho_activity - rac_activity
                )
            )

        rgtpase_mediated_force_mags[ni] = -1 * force_mag

    np.empty((nverts, 2), dtype=np.float64)
    result = geometry.multiply_vectors_by_scalars(
        uivs, rgtpase_mediated_force_mags
    )

    return result

# -----------------------------------------------------------------------


@nb.jit(nopython=True)
def calculate_external_forces(
    nverts,
    num_cells,
    this_ci,
    close_point_on_other_cells_to_each_node_exists,
        close_point_on_other_cells_to_each_node_indices,
        all_cells_node_forces,
        uivs,
):
    F_external = np.zeros((nverts, 2), dtype=np.float64)

    for ni in range(nverts):
        np.zeros(2, dtype=np.float64)
        uiv = uivs[ni]

        for ci in range(num_cells):
            if ci != this_ci:
                if close_point_on_other_cells_to_each_node_exists[ni][ci] == 1:
                    close_ni_a, close_ni_b = close_point_on_other_cells_to_each_node_indices[
                        ni][ci]

                    if close_ni_a == close_ni_b:
                        # closest point is another single node
                        close_point_force = all_cells_node_forces[ci][close_ni_a]
                        force_proj_mag = geometry.calculate_projection_of_a_on_b(
                            close_point_force, uiv)

                        if force_proj_mag < 0.0:
                            force_proj_mag = 0.0

                        F_external[ni] += force_proj_mag * uiv
                    else:
                        # closest point is on a line segment between two nodes
                        close_ni_a_force = all_cells_node_forces[ci][close_ni_a]
                        close_ni_b_force = all_cells_node_forces[ci][close_ni_b]

                        a_proj_mag = geometry.calculate_projection_of_a_on_b(
                            close_ni_a_force, uiv
                        )
                        b_proj_mag = geometry.calculate_projection_of_a_on_b(
                            close_ni_b_force, uiv
                        )

                        if a_proj_mag < 0.0:
                            a_proj_mag = 0.0
                        if b_proj_mag < 0.0:
                            b_proj_mag = 0.0

                        F_external[ni] += (a_proj_mag + b_proj_mag) * uiv

    return F_external


# ----------------------------------------------------------------------------
# @nb.jit(nopython=True)
def calculate_forces(
    nverts,
        this_cell_coords,
    rac_acts,
    rho_acts,
    rest_edge_len,
    stiffness_edge,
    threhsold_force_rac_activity,
    threhsold_force_rho_activity,
    max_force_rac,
    max_force_rho,
        rest_area,
    stiffness_cyto,
):

    uivs = geometry.calculate_unit_inside_pointing_vecs(
        this_cell_coords)

    rgtpase_mediated_forces = calculate_rgtpase_mediated_forces(
        nverts,
        rac_acts,
        rho_acts,
        threhsold_force_rac_activity,
        threhsold_force_rho_activity,
        max_force_rac,
        max_force_rho,
        uivs,
    )

    cyto_forces = calculate_cytoplasmic_force(
        nverts,
        this_cell_coords,
        rest_area,
        stiffness_cyto,
        uivs,
    )

    local_strains, edge_forces_plus, edge_forces_minus, edge_strains = calculate_spring_edge_forces(
        nverts, this_cell_coords, stiffness_edge, rest_edge_len
    )

    F_internal = rgtpase_mediated_forces + edge_forces_plus + edge_forces_minus + cyto_forces

    # F_external = calculate_external_forces(nverts, num_cells, this_ci, close_point_on_other_cells_to_each_node_exists, close_point_on_other_cells_to_each_node, close_point_on_other_cells_to_each_node_indices, close_point_on_other_cells_to_each_node_projection_factors, all_cells_centres, all_cells_node_forces, closeness_dist_criteria, uivs)

    F = F_internal  # + F_external

    return (
        F,
        edge_forces_plus,
        edge_forces_minus,
        rgtpase_mediated_forces,
        cyto_forces,
        edge_strains,
        local_strains,
        uivs,
    )


# =============================================================================
