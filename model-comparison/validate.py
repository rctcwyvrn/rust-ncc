from retrieve import *
import numpy as np
import matplotlib.pyplot as plt

init_rust_dat, rust_dat = process_data("rust")
init_py_dat, py_dat = process_data("python")

rust_poly_per_int_step_per_tstep = \
    np.array(dat_per_vert_per_int_step_per_cell_per_tstep(
        "poly", rust_dat))[:, 0, :]
py_poly_per_int_step_per_tstep = \
    np.array(dat_per_vert_per_int_step_per_cell_per_tstep(
        "poly", py_dat))[:, 0, :]

rust_rac_acts_per_int_step_per_tstep = \
    np.array(dat_per_vert_per_int_step_per_cell_per_tstep(
        "rac_acts", rust_dat))[:, 0, :]
rust_rho_acts_per_int_step_per_tstep = \
    np.array(dat_per_vert_per_int_step_per_cell_per_tstep(
        "rho_acts", rust_dat))[:, 0, :]

rust_rac_inacts_per_int_step_per_tstep = \
    np.array(dat_per_vert_per_int_step_per_cell_per_tstep(
        "rac_inacts", rust_dat))[:, 0, :]
rust_rho_inacts_per_int_step_per_tstep = \
    np.array(dat_per_vert_per_int_step_per_cell_per_tstep(
        "rho_inacts", rust_dat))[:, 0, :]

py_rac_acts_per_int_step_per_tstep = \
    np.array(dat_per_vert_per_int_step_per_cell_per_tstep(
        "rac_acts", py_dat))[:, 0, :]
py_rho_acts_per_int_step_per_tstep = \
    np.array(dat_per_vert_per_int_step_per_cell_per_tstep(
        "rho_acts", py_dat))[:, 0, :]

py_rac_inacts_per_int_step_per_tstep = \
    np.array(dat_per_vert_per_int_step_per_cell_per_tstep(
        "rac_inacts", py_dat))[:, 0, :]
py_rho_inacts_per_int_step_per_tstep = \
    np.array(dat_per_vert_per_int_step_per_cell_per_tstep(
        "rho_inacts", py_dat))[:, 0, :]


def diff(tstep, cell_ix, label, rust_dat, py_dat):
    rd = np.array(dat_per_vert_per_int_step_per_cell_per_tstep(
        label, rust_dat))[tstep, cell_ix, 0]
    pd = np.array(dat_per_vert_per_int_step_per_cell_per_tstep(
        label, py_dat))[tstep, cell_ix, 0]
    return np.linalg.norm(rd - pd)


print("poly_diff: {}".format(diff(0, 0, "poly", rust_dat, py_dat)))
print("rac_acts_diff: {}".format(diff(0, 0, "rac_acts", rust_dat, py_dat)))
print("rac_inacts: {}".format(diff(0, 0, "rac_inacts", rust_dat, py_dat)))
print("rho_acts: {}".format(diff(0, 0, "rho_inacts", rust_dat, py_dat)))
print("rho_inacts: {}".format(diff(0, 0, "rho_inacts", rust_dat, py_dat)))
print("tot_forces: {}".format(diff(0, 0, "tot_forces", rust_dat, py_dat)))
print("rgtp_forces: {}".format(diff(0, 0, "rgtp_forces", rust_dat, py_dat)))
print("edge_forces: {}".format(diff(0, 0, "edge_forces", rust_dat, py_dat)))
print("cyto_forces: {}".format(diff(0, 0, "cyto_forces", rust_dat, py_dat)))
