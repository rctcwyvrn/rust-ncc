from retrieve import *
import numpy as np
import matplotlib.pyplot as plt
import copy
import py_model.hardio as hio

NUM_TSTEPS = 18
NUM_CELLS = 2
COA = 24
CIL = 60
NUM_INT_STEPS = 10

py_out = get_py_dat(NUM_TSTEPS, NUM_INT_STEPS, NUM_CELLS, CIL, COA)
rust_out = get_rust_dat(NUM_TSTEPS, NUM_INT_STEPS, NUM_CELLS, CIL, COA)

check_header_equality(py_out, rust_out)

MAX_PLOT_TSTEP = NUM_TSTEPS

EXTRA_LABELS = ["centroid"]
EXTRA_COLORS = ["black"]

IGNORE_LABELS = ["uivs"]

py_dat = py_out["tsteps"]
rust_dat = rust_out["tsteps"]


py_data_dict_per_cell = gen_data_dict_per_cell(py_dat)
rust_data_dict_per_cell = gen_data_dict_per_cell(rust_dat)


def calc_extras(data_dict):
    global EXTRA_LABELS
    extra_dat = []
    for extra in EXTRA_LABELS:
        if extra == "centroid":
            poly_per_int_step = data_dict["poly"]
            centroids = np.average(poly_per_int_step, axis=1)
            centroid_dists = np.linalg.norm(centroids, axis=1)
            extra_dat.append(centroid_dists)

    return extra_dat


for cell_ix in range(NUM_CELLS):
    for extra_label, extra_py_dat, extra_rust_dat in zip(EXTRA_LABELS,
                                                         calc_extras(
                                                             py_data_dict_per_cell[
                                                                 cell_ix]),
                                                         calc_extras(
                                                             rust_data_dict_per_cell[
                                                                 cell_ix])):
        py_data_dict_per_cell[cell_ix][extra_label] = extra_py_dat
        rust_data_dict_per_cell[cell_ix][extra_label] = extra_rust_dat

# Norm vector data.
for cell_ix in range(NUM_CELLS):
    for data_dict in [py_data_dict_per_cell[cell_ix],
                      rust_data_dict_per_cell[cell_ix]]:
        for label in hio.DATA_LABELS:
            if label == "poly" or "forces" in label:
                orig = data_dict[label]
                normed = np.linalg.norm(data_dict[label], axis=2)
                data_dict[label] = normed

DATA_GROUPS = [["poly", "centroid"], ["kgtps_rac", "kdgtps_rac",
                                      "kgtps_rho", "kdgtps_rho"],
               ["sum_forces", "rgtp_forces", "edge_forces",
                "cyto_forces"],
               ["rac_acts", "rac_inacts", "rho_acts", "rho_inacts"],
               ["rac_act_net_fluxes"], ["conc_rac_acts"], ["x_cils"],
               ["x_coas"], ["edge_strains"], ["poly_area"]]

unmatched_labels = []
for label in hio.DATA_LABELS + EXTRA_LABELS:
    if label in IGNORE_LABELS:
        continue
    else:
        matched = False
        for group in DATA_GROUPS:
            if label in group:
                matched = True
                break
        if not matched:
            unmatched_labels.append(label)

if len(unmatched_labels) > 0:
    error = "The following labels are not assigned to a group: {}" \
        .format(unmatched_labels)
    raise Exception(error)


def group_labels():
    group_to_label = [list() for n in range(len(DATA_GROUPS))]
    label_to_group = []
    for label in ALL_LABELS:
        matched = False
        for ix, grouped_labels in enumerate(DATA_GROUPS):
            if matched:
                break
            for lbl in grouped_labels:
                if label == lbl:
                    group_to_label[ix].append(label)
                    label_to_group.append(ix)
                    matched = True
                    break
        if not matched:
            raise Exception("No match for label: {}".format(label))

    return group_to_label, label_to_group


def calculate_data_group_min_maxes(group_to_label, dict_rust_dat_per_cell,
                                   dict_py_dat_per_cell, max_int_step,
                                   focus_cell):
    global NUM_CELLS
    group_min_maxes = []
    for label_group in group_to_label:
        dat = np.empty(0)
        for label in label_group:
            for cell_ix in range(NUM_CELLS):
                if cell_ix == focus_cell or focus_cell == "all":
                    dat = np.append(dat, dict_rust_dat_per_cell[cell_ix][
                                             label][
                                         :max_int_step])
                    dat = np.append(dat, dict_py_dat_per_cell[cell_ix][label][
                                         :max_int_step])
        group_min_maxes.append((np.min(dat), np.max(dat)))
    return group_min_maxes


def calculate_label_min_maxes(dict_rust_dat_per_cell, dict_py_dat_per_cell,
                              max_int_step, focus_cell):
    global ALL_LABELS
    group_to_label, label_to_group = group_labels()
    group_min_maxes = \
        calculate_data_group_min_maxes(group_to_label, dict_rust_dat_per_cell,
                                       dict_py_dat_per_cell, max_int_step,
                                       focus_cell)
    label_min_maxes = []
    for group_ix in label_to_group:
        label_min_maxes.append(group_min_maxes[group_ix])
    return dict(zip(ALL_LABELS, label_min_maxes))


def paint(delta_vx, delta_dt, delta_mp, delta_cx):
    global fig
    global ax
    global VERTEX_PLOT_IX
    global DATA_TYPE_IX
    global MAX_PLOT_TSTEP
    global ALL_LABEL_MAXES
    global CELL_PLOT_IX
    global NUM_CELLS
    ax.cla()

    VERTEX_PLOT_IX = (VERTEX_PLOT_IX + delta_vx) % len(VERTEX_PLOT_TYPE)
    DATA_TYPE_IX = (DATA_TYPE_IX + delta_dt) % len(hio.DATA_LABELS)
    CELL_PLOT_IX = (CELL_PLOT_IX + delta_cx) % len(CELL_PLOT_TYPE)
    MAX_PLOT_TSTEP += delta_mp

    label = ALL_LABELS[DATA_TYPE_IX]
    color = ALL_COLORS[DATA_TYPE_IX]
    vert = VERTEX_PLOT_TYPE[VERTEX_PLOT_IX]
    cell = CELL_PLOT_TYPE[CELL_PLOT_IX]

    if abs(delta_mp) > 0 or abs(delta_cx) > 0:
        ALL_LABEL_MAXES = calculate_label_min_maxes(dict_rust_dat_per_cell,
                                                    dict_py_dat_per_cell,
                                                    MAX_PLOT_TSTEP,
                                                    cell)

    for m in range(NUM_CELLS):
        if m == cell or cell == "all":
            dict_py_dat = py_data_dict_per_cell[m]
            dict_rust_dat = rust_data_dict_per_cell[m]
            if len(dict_rust_dat[label].shape) == 1:
                ax.plot(
                    dict_rust_dat[label][:MAX_PLOT_TSTEP],
                    color=color)
                ax.plot(
                    dict_py_dat[label][:MAX_PLOT_TSTEP],
                    color=color,
                    linestyle="dashed")
                # ax.set_ylim(ALL_LABEL_MAXES[label])
            else:
                for n in range(16):
                    if n == vert or vert == "all":
                        ax.plot(dict_rust_dat[label][
                                :MAX_PLOT_TSTEP, n],
                                color=color)
                        ax.plot(dict_py_dat[label][
                                :MAX_PLOT_TSTEP, n], color=color,
                                linestyle="dashed")
                        # ax.set_ylim(ALL_LABEL_MAXES[label])

    ax.set_title("{}, vert: {}, cell: {}".format(label, vert, cell))


def on_press(event):
    global VERTEX_PLOT_IX
    global DATA_TYPE_IX
    global CELL_PLOT_IX
    global fig
    global MAX_PLOT_TSTEP
    print("pressed: {}".format(event.key))
    if event.key == "up":
        paint(1, 0, 0, 0)
    elif event.key == "down":
        paint(-1, 0, 0, 0)
    elif event.key == "right":
        paint(0, 1, 0, 0)
    elif event.key == "left":
        paint(0, -1, 0, 0)
    elif event.key == "z":
        paint(0, 0, -10, 0)
    elif event.key == "x":
        paint(0, 0, 10, 0)
    elif event.key == "c":
        paint(0, 0, -1, 0)
    elif event.key == "v":
        paint(0, 0, 1, 0)
    elif event.key == " ":
        paint(0, 0, 0, 1)
    elif event.key == "r":
        VERTEX_PLOT_IX = 0
        DATA_TYPE_IX = 0
        MAX_PLOT_TSTEP = 150
        CELL_PLOT_IX = 0
        paint(0, 0, 0, 0)


VERTEX_PLOT_TYPE = [n for n in range(16)] + ["all"]
CELL_PLOT_TYPE = [m for m in range(NUM_CELLS)] + ["all"]
ALL_LABELS = [label for label in hio.DATA_LABELS + EXTRA_LABELS if
              label not in IGNORE_LABELS]
ALL_COLORS = [color for (label, color) in zip(hio.DATA_LABELS + EXTRA_LABELS,
                                              hio.DATA_COLORS + EXTRA_COLORS) if
              label not in IGNORE_LABELS]
group_to_label, label_to_group = group_labels()
ALL_LABEL_MAXES = calculate_label_min_maxes(py_data_dict_per_cell,
                                            rust_data_dict_per_cell,
                                            MAX_PLOT_TSTEP, "all")
# VERTEX_PLOT_IX = 16  # set initially to plot all
# CELL_PLOT_IX = NUM_CELLS  # set initially to plot all
# DATA_TYPE_IX = 0
# fig, ax = plt.subplots()
# paint(0, 0, 0, 0)
# fig.canvas.mpl_connect('key_press_event', on_press)

poly_per_tstep = py_data_dict_per_cell[0]["poly"]
print(poly_per_tstep.shape)
plt.plot(py_data_dict_per_cell[0]["poly"][:, 1])
print(py_data_dict_per_cell[0]["poly"][:, 0])
