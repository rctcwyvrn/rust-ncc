import os
import copy
import orjson

HEADER_LABELS = ["num_tsteps", "num_cells", "num_int_steps",
                 "char_quant_eta", "char_quant_f", "char_quant_l",
                 "char_quant_t", "char_quant_l3d",
                 "char_quant_k_mem_off",
                 "char_quant_k_mem_on", "char_quant_kgtp",
                 "char_quant_kdgtp", "cell_r", "rest_edge_len",
                 "rest_area", "stiffness_edge", "const_protrusive",
                 "const_retractive", "stiffness_cyto",
                 "k_mem_on_vertex",
                 "k_mem_off", "diffusion_rgtp", "init_rac",
                 "init_rho",
                 "halfmax_vertex_rgtp_act",
                 "halfmax_vertex_rgtp_conc",
                 "tot_rac", "tot_rho", "kgtp_rac", "kgtp_rac_auto",
                 "kdgtp_rac", "kdgtp_rho_on_rac",
                 "halfmax_tension_inhib",
                 "tension_inhib", "kgtp_rho", "kgtp_rho_auto",
                 "kdgtp_rho",
                 "kdgtp_rac_on_rho", "randomization", "rand_avg_t",
                 "rand_std_t", "rand_mag", "num_rand_vs",
                 "total_rgtp",
                 "coa_los_penalty", "coa_range", "coa_mag",
                 "coa_distrib_exp", "cil_zero_at", "cil_one_at",
                 "cil_mag"]
BASICS = ["poly", "rac_acts", "rac_inacts", "rho_acts",
          "rho_inacts", "tot_forces"]
BASIC_COLORS = ["black", "blue", "slateblue", "red", "indianred",
                "tab:blue"]
GEOMETRY = ["uivs"]
GEOMETRY_COLORS = ["black"]
RAC_RATES = ["kgtps_rac", "kdgtps_rac"]
RAC_RATE_COLORS = ["deepskyblue", "violet"]
RHO_RATES = ["kgtps_rho", "kdgtps_rho"]
RHO_RATE_COLORS = ["orangered", "lightblue"]
FORCES = ["rgtp_forces", "edge_forces", "cyto_forces"]
FORCE_COLORS = ["darkolivegreen", "steelblue", "mediumvioletred"]
CALC_KGTPS_RAC = ["conc_rac_acts", "x_cils", "x_coas"]
CALC_KGTPS_RAC_COLORS = ["blueviolet", "gold", "lightseagreen"]
DIFFUSION = ["rac_act_net_fluxes"]
DIFFUSION_COLORS = ["teal"]
OTHERS = ["edge_strains", "poly_area"]
OTHER_COLORS = ["tab:orange", "green"]

WRITE_FOLDER = "B:\\rust-ncc\\model-comparison\\py-out\\"
WRITE_FILE_NAME_TEMPLATE = "out_euler_T={}_E{}_NC={}_CIL={}_COA={}.dat"


def validate_data(req_labels, data):
    for req_label, d in zip(req_labels, data):
        label, value = d
        if req_label != label:
            raise Exception("Expected label {}, found {}".format(req_label,
                                                                 label))


def validate_buffer_length(expected_length, buf, buf_description):
    if len(buf) != expected_length:
        raise Exception("{} expected {} elements. Found: {}".format(
            buf_description, expected_length, len(buf)))


class Writer:
    def __init__(self, num_cells, num_tsteps, num_int_steps, cil, coa):
        global HEADER_LABELS
        global BASICS
        global GEOMETRY
        global RAC_RATES
        global RHO_RATES
        global FORCES
        global CALC_KGTPS_RAC
        global DIFFUSION
        global OTHERS
        global WRITE_FOLDER
        global WRITE_FILE_NAME_TEMPLATE
        self.header_labels = copy.deepcopy(HEADER_LABELS)
        self.basics = copy.deepcopy(BASICS)
        self.geometry = copy.deepcopy(GEOMETRY)
        self.rac_rates = copy.deepcopy(RAC_RATES)
        self.rho_rates = copy.deepcopy(RHO_RATES)
        self.forces = copy.deepcopy(FORCES)
        self.calc_kgtps_rac = copy.deepcopy(CALC_KGTPS_RAC)
        self.diffusion = copy.deepcopy(DIFFUSION)
        self.others = copy.deepcopy(OTHERS)
        self.data_labels = self.basics + self.geometry + self.forces + \
                           self.rac_rates + self.rho_rates + \
                           self.calc_kgtps_rac + self.diffusion + self.others
        self.num_cells = num_cells
        self.num_tsteps = num_tsteps
        self.num_int_steps = num_int_steps
        self.cil = cil
        self.coa = coa

        self.write_folder = WRITE_FOLDER
        self.write_file_name_template = WRITE_FILE_NAME_TEMPLATE
        self.write_file_path_template = self.write_folder + \
                                        self.write_file_name_template
        self.write_file_path = ""
        self.write_file_label = ""

        self.int_step_buffer = []
        self.current_cell_ix = 0
        self.empty_tstep_data = dict([("cell_ix", 0), ("int_steps", [])])
        self.current_tstep_data = copy.deepcopy(self.empty_tstep_data)
        self.tstep_buffer = []
        self.main_buffer = dict([("header", {}), ("tsteps", [])])
        self.finished = False

    def init_writing(self, tsteps, num_int_steps, num_cells, cil, coa):
        self.write_file_path = self.write_file_path_template.format(tsteps,
                                                                    num_int_steps,
                                                                    num_cells,
                                                                    cil,
                                                                    coa)
        if os.path.exists(self.write_file_path):
            os.remove(self.write_file_path)
        self.finished = False

    def save_int_step(self, data):
        validate_data(self.data_labels, data)
        self.int_step_buffer.append(dict(data))
        if len(self.int_step_buffer) == self.num_int_steps:
            self.finish_int_steps()

    def curr_tpointstep_is_finished(self):
        return len(self.current_tstep_data["int_steps"]) == 0

    def finish_int_steps(self):
        self.save_cell_tstep(copy.deepcopy(self.int_step_buffer))
        if not self.curr_tpointstep_is_finished():
            err_format = "int_steps buffer is not full (only {} elements)"
            raise Exception(err_format.format(len(self.current_tstep_data[
                                                "cell_ix"])))
        else:
            self.int_step_buffer = []

    def init_tstep_save(self, cell_ix):
        if self.current_tstep_data["int_steps"] != 0:
            raise Exception(
                "CURRENT_TSTEP_DATA does not have an int_steps buffer.")
        else:
            self.current_tstep_data["cell_ix"] = cell_ix

    def save_cell_tstep(self, data):
        validate_buffer_length(self.num_int_steps, data, "INT_STEP_BUFFER")
        self.tstep_buffer.append(data)
        if len(self.tstep_buffer) == self.num_cells:
            self.finish_tstep_buffer()

    def finish_tstep_buffer(self):
        self.save_tsteps(copy.deepcopy(self.tstep_buffer))
        self.tstep_buffer = []

    def save_tsteps(self, data):
        if not self.finished:
            validate_buffer_length(self.num_cells, data, "TSTEP_BUFFER")
            self.main_buffer["tsteps"].append(data)
            if len(self.main_buffer["tsteps"]) == self.num_tsteps:
                with open(self.write_file_path, "wb") as f:
                    f.write(orjson.dumps(self.main_buffer))
                self.finished = True
        else:
            raise Exception(
                "Supposed to be finished at {} tsteps.".format(self.num_tsteps))

    def save_header(self, data):
        validate_data(self.header_labels, data)
        self.main_buffer["header"] = dict(data)
