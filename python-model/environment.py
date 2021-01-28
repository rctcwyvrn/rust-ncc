import copy
import time

import numba as nb
import numpy as np

import cell
import geometry
import params

from hardio import Writer

"""
Environment of cells.s
"""

MODE_EXECUTE = 0
MODE_OBSERVE = 1


@nb.jit(nopython=True)
def custom_floor(fp_number, roundoff_distance):
    a = int(fp_number)
    b = a + 1

    if abs(fp_number - b) < roundoff_distance:
        return b
    else:
        return a


# -----------------------------------------------------------------


def calc_bb_centre(bbox):
    x = bbox[0] + (bbox[1] - bbox[0]) * 0.5
    y = bbox[2] + (bbox[3] - bbox[2]) * 0.5

    return np.array([x, y])


# -----------------------------------------------------------------


class Environment:
    """Implementation of coupled map lattice model of a cell.
    """

    def __init__(
            self,
            num_tsteps,
            cell_group_defs,
            char_t,
    ):
        self.cell_group_defs = cell_group_defs

        self.curr_tpoint = 0
        self.char_t = char_t

        self.num_tsteps = num_tsteps
        self.num_tpoints = num_tsteps + 1
        self.timepoints = np.arange(0, self.num_tpoints)
        self.num_int_steps = 10

        self.num_cell_groups = len(self.cell_group_defs)
        self.num_cells = np.sum(
            [
                cell_group_def["num_cells"]
                for cell_group_def in self.cell_group_defs
            ],
            dtype=np.int64,
        )
        self. writer = Writer(num_cells, num_tsteps, num_int_steps, cil, coa)

        self.env_cells = self.make_cells()
        nverts_per_cell = np.array(
            [x.nverts for x in self.env_cells], dtype=np.int64
        )
        self.nverts = nverts_per_cell[0]
        for n in nverts_per_cell[1:]:
            if n != self.nverts:
                raise Exception(
                    "There exists a cell with number of nodes different "
                    "from other cells!"
                )
        self.cell_indices = np.arange(self.num_cells)
        self.exec_orders = np.zeros(
            (self.num_tpoints, self.num_cells), dtype=np.int64
        )

        self.all_geometry_tasks = np.array(
            geometry.create_dist_and_line_segment_interesection_test_args(
                self.num_cells, self.nverts
            ),
            dtype=np.int64,
        )
        self.geometry_tasks_per_cell = np.array(
            [
                geometry.create_dist_and_line_segment_interesection_test_args_relative_to_specific_cell(
                    ci, self.num_cells, self.nverts
                )
                for ci in range(self.num_cells)
            ],
            dtype=np.int64,
        )

        environment_cells_verts = np.array(
            [x.curr_verts for x in self.env_cells]
        )
        cells_bb_array = \
            geometry.calc_init_cell_bbs(
            self.num_cells, environment_cells_verts)
        (
            cells_node_distance_matrix,
            cells_line_segment_intersection_matrix,
        ) = geometry.init_lseg_intersects_and_dist_sq_matrices_old(
            self.num_cells,
            self.nverts,
            cells_bb_array,
            environment_cells_verts,
        )
        for (ci, cell) in enumerate(self.env_cells):
            cell.initialize_cell(cells_node_distance_matrix[ci],
                                 cells_line_segment_intersection_matrix[ci])

        self.mode = MODE_EXECUTE
        self.animation_settings = None

    def extend_simulation_runtime(self, new_num_tsteps):
        self.num_tsteps = new_num_tsteps
        self.num_tpoints = self.num_tsteps + 1
        self.timepoints = np.arange(0, self.num_tpoints)

        for a_cell in self.env_cells:
            a_cell.num_tpoints = self.num_tpoints

        old_exec_orders = np.copy(self.exec_orders)
        self.exec_orders = np.zeros(
            (self.num_tpoints, self.num_cells), dtype=np.int64
        )
        self.exec_orders[: old_exec_orders.shape[0]] = old_exec_orders

    def simulation_complete(self):
        return self.num_tsteps == self.curr_tpoint

    # -----------------------------------------------------------------

    def make_cells(self):
        env_cells = []

        ci_offset = 0
        for cell_group_ix, cell_group_def in enumerate(
                self.cell_group_defs):
            cells_in_group, init_cell_bbs = self.create_cell_group(
                self.num_tsteps, cell_group_def, cell_group_ix, ci_offset
            )
            ci_offset += len(cells_in_group)
            env_cells += cells_in_group

        return np.array(env_cells)

    # -----------------------------------------------------------------

    def create_cell_group(
            self, num_tsteps, cell_group_def, cell_group_ix, ci_offset
    ):
        group_name = cell_group_def["group_name"]
        num_cells = cell_group_def["num_cells"]
        cell_group_bb = cell_group_def["cell_group_bbox"]

        cell_params = copy.deepcopy(cell_group_def["params"])
        cell_r = cell_params["cell_r"]
        nverts = cell_params["nverts"]

        rgtp_distrib_defs = cell_group_def["rgtp_distrib_defs"]
        cells_with_bias = list(rgtp_distrib_defs.keys())

        init_cell_bbs = self.calculate_cell_bbs(
            num_cells,
            cell_r,
            cell_group_bb,
        )

        cells_in_group = []

        for ci, bbox in enumerate(init_cell_bbs):
            bias_def = rgtp_distrib_defs["default"]

            if ci in cells_with_bias:
                bias_def = rgtp_distrib_defs[ci]

            (
                init_verts,
                rest_edge_len,
                rest_area,
            ) = self.create_default_init_cell_verts(
                bbox, cell_r, nverts
            )

            cell_params.update(
                [
                    ("rgtp_distrib_def", bias_def),
                    ("init_verts", init_verts),
                    ("rest_edge_len", rest_edge_len),
                    ("rest_area", rest_area),
                ]
            )

            ci = ci_offset + cell_number

            undefined_labels = params.find_undefined_labels(
                cell_params)
            if len(undefined_labels) > 0:
                raise Exception(
                    "The following labels are not yet defined: {}".format(
                        undefined_labels
                    )
                )

            new_cell = cell.Cell(
                str(group_name) + "_" + str(ci),
                cell_group_ix,
                ci,
                num_tsteps,
                self.char_t,
                self.num_cells,
                self.num_int_steps,
                cell_params,
            )

            cells_in_group.append(new_cell)

        return cells_in_group, init_cell_bbs

    # -----------------------------------------------------------------

    @staticmethod
    def calculate_cell_bbs(
            num_cells,
            cell_r,
            cell_group_bb,
    ):

        cell_bbs = np.zeros((num_cells, 4), dtype=np.float64)
        xmin, xmax, ymin, ymax = cell_group_bb
        x_length = xmax - xmin
        y_length = ymax - ymin

        cell_diameter = 2 * cell_r

        # check if cells can fit in given bounding box
        total_cell_group_area = num_cells * (np.pi * cell_r ** 2)
        cell_group_bb_area = abs(x_length * y_length)

        if total_cell_group_area > cell_group_bb_area:
            raise Exception(
                "Cell group bounding box is not big enough to contain all "
                "cells given cell_r constraint."
            )
        num_cells_along_x = custom_floor(x_length / cell_diameter, 1e-6)
        num_cells_along_y = custom_floor(y_length / cell_diameter, 1e-6)

        cell_x_coords = (
                xmin +
                np.sign(x_length) *
                np.arange(num_cells_along_x) *
                cell_diameter)
        cell_y_coords = (
                ymin +
                np.sign(y_length) *
                np.arange(num_cells_along_y) *
                cell_diameter)
        x_step = np.sign(x_length) * cell_diameter
        y_step = np.sign(y_length) * cell_diameter

        xi = 0
        yi = 0
        for ci in range(num_cells):
            cell_bbs[ci] = [
                cell_x_coords[xi],
                cell_x_coords[xi] + x_step,
                cell_y_coords[yi],
                cell_y_coords[yi] + y_step,
            ]

            if yi == (num_cells_along_y - 1):
                yi = 0
                xi += 1
            else:
                yi += 1

        return cell_bbs

    # -----------------------------------------------------------------

    @staticmethod
    def create_default_init_cell_verts(
            bbox, cell_r, nverts
    ):
        cell_centre = calc_bb_centre(bbox)

        cell_node_thetas = np.pi * \
                           np.linspace(0, 2, endpoint=False, num=nverts)
        cell_verts = np.transpose(
            np.array(
                [
                    cell_r * np.cos(cell_node_thetas),
                    cell_r * np.sin(cell_node_thetas),
                ]
            )
        )

        # rotation_theta = np.random.rand()*2*np.pi
        # cell_verts = np.array([
        # geometry.rotate_2D_vector_CCW_by_theta(rotation_theta, x) for x in
        # cell_verts], dtype=np.float64)
        cell_verts = np.array(
            [[x + cell_centre[0], y + cell_centre[1]] for x, y in
             cell_verts],
            dtype=np.float64,
        )

        edge_vectors = geometry.calculate_edge_vectors(cell_verts)

        edge_lengths = geometry.calculate_2D_vector_mags(edge_vectors)

        rest_edge_len = np.average(edge_lengths)

        rest_area = geometry.calculate_polygon_area(cell_verts)
        if rest_area < 0:
            raise Exception("Resting area was calculated to be negative.")
        return cell_verts, rest_edge_len, rest_area

    # -----------------------------------------------------------------
    def execute_system_dynamics_in_random_sequence(
            self,
            t,
            cells_node_distance_matrix,
            cells_bb_array,
            cells_line_segment_intersection_matrix,
            environment_cells_verts,
            environment_cells_node_forces,
            environment_cells,
    ):
        execution_sequence = self.cell_indices
        np.random.shuffle(execution_sequence)

        self.exec_orders[t] = np.copy(execution_sequence)

        fw.write(["=============================="])

        for ci in execution_sequence:
            current_cell = environment_cells[ci]

            fw.write(["++++++++++++++++++++++++++++++", "ci: {}".format(ci)])
            current_cell.execute_step(
                ci,
                self.nverts,
                environment_cells_verts,
                environment_cells_node_forces,
                cells_node_distance_matrix[ci],
                cells_line_segment_intersection_matrix[ci],
            )
            fw.write(["++++++++++++++++++++++++++++++"])

            this_cell_coords = current_cell.curr_verts
            this_cell_forces = current_cell.curr_node_forces

            environment_cells_verts[ci] = this_cell_coords
            environment_cells_node_forces[ci] = this_cell_forces

            cells_bb_array[ci] = \
                geometry.calculate_polygon_bb(this_cell_coords)
            geometry.update_line_segment_intersection_and_dist_squared_matrices(
                4,
                self.geometry_tasks_per_cell[ci],
                environment_cells_verts,
                cells_bb_array,
                cells_node_distance_matrix,
                cells_line_segment_intersection_matrix,
                sequential=True,
            )

        fw.write(["=============================="])

        return (
            cells_node_distance_matrix,
            cells_bb_array,
            cells_line_segment_intersection_matrix,
            environment_cells_verts,
            environment_cells_node_forces,
        )

    # -----------------------------------------------------------------

    def execute_system_dynamics(
            self,
    ):
        simulation_st = time.time()
        num_cells = self.num_cells
        nverts = self.nverts

        environment_cells = self.env_cells
        all_cell_verts = np.array([x.curr_verts for x in
                                                  environment_cells])
        environment_cells_node_forces = np.array([x.curr_node_forces for x in
                                                  environment_cells])

        cell_bbs = \
            geometry.calc_init_cell_bbs(
            num_cells, all_cell_verts)
        (
            cells_node_distance_matrix,
            cells_line_segment_intersection_matrix,
        ) = geometry.init_lseg_intersects_and_dist_sq_matrices_old(
            num_cells,
            nverts,
            cell_bbs,
            all_cell_verts,
        )

        cell_group_indices = []

        for a_cell in self.env_cells:
            cell_group_indices.append(a_cell.cell_group_ix)

        fw.write(["******************************",
                  "num_tsteps: {}".format(self.num_tsteps),
                  "num_cells: {}".format(self.num_cells),
                  "num_int_steps: {}".format(self.num_int_steps),
                  "******************************"])

        if self.curr_tpoint == 0 or self.curr_tpoint < self.num_tsteps:
            for t in self.timepoints[self.curr_tpoint: -1]:

                (
                    cells_node_distance_matrix,
                    cell_bbs,
                    cells_line_segment_intersection_matrix,
                    all_cell_verts,
                    environment_cells_node_forces,
                ) = self.execute_system_dynamics_in_random_sequence(
                    t,
                    cells_node_distance_matrix,
                    cell_bbs,
                    cells_line_segment_intersection_matrix,
                    all_cell_verts,
                    environment_cells_node_forces,
                    environment_cells,
                )
                self.curr_tpoint += 1
        else:
            raise Exception("max_t has already been reached.")

        simulation_et = time.time()

        simulation_time = np.round(simulation_et - simulation_st, decimals=2)

        print(
            ("Time taken to complete simulation: {}s".format(simulation_time))
        )

# -----------------------------------------------------------------
