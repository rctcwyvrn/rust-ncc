from retrieve import *
import numpy as np
import matplotlib.pyplot as plt

NUM_TSTEPS = 18
NUM_CELLS = 2
COA = 24
CIL = 60
NUM_INT_STEPS = 10

py_out = get_py_dat(NUM_TSTEPS, NUM_INT_STEPS, NUM_CELLS, CIL, COA)
rust_out = get_rust_dat(NUM_TSTEPS, NUM_INT_STEPS, NUM_CELLS, CIL, COA)

check_header_equality(py_out, rust_out)

py_data_dict_per_cell = gen_data_dict_per_cell(py_out)
rust_data_dict_per_cell = gen_data_dict_per_cell(rust_out)

rust_poly_per_cell_per_tstep = np.array([[rust_data_dict_per_cell[c]["poly"][t]
                                          for c in range(NUM_CELLS)]
                                         for t in range(NUM_TSTEPS)])
rust_uivs_per_cell_per_tstep = np.array([[rust_data_dict_per_cell[c]["uivs"][t]
                                          for c in range(NUM_CELLS)]
                                         for t in range(NUM_TSTEPS)])
rust_uovs_per_cell_per_step = -1 * rust_uivs_per_cell_per_tstep
rust_rac_acts_per_cell_per_tstep = np.array([[rust_data_dict_per_cell[c][
                                                  "rac_acts"][t]
                                              for c in range(NUM_CELLS)]
                                             for t in range(NUM_TSTEPS)])
rust_rac_act_arrows_per_cell_per_tstep = \
    50 * rust_rac_acts_per_cell_per_tstep[:, :, :, np.newaxis] * \
    rust_uovs_per_cell_per_step
rust_rho_acts_per_cell_per_tstep = np.array([[rust_data_dict_per_cell[c][
                                                  "rho_acts"][t]
                                              for c in range(NUM_CELLS)]
                                             for t in range(NUM_TSTEPS)])
rust_rho_act_arrows_per_cell_per_tstep = \
    50 * rust_rho_acts_per_cell_per_tstep[:, :, :, np.newaxis] * \
    rust_uivs_per_cell_per_tstep

py_poly_per_cell_per_tstep = np.array([[py_data_dict_per_cell[c]["poly"][t]
                                        for c in range(NUM_CELLS)]
                                       for t in range(NUM_TSTEPS)])
py_uivs_per_cell_per_tstep = np.array([[py_data_dict_per_cell[c]["uivs"][t]
                                        for c in range(NUM_CELLS)]
                                       for t in range(NUM_TSTEPS)])
py_uovs_per_cell_per_step = -1 * py_uivs_per_cell_per_tstep
py_rac_acts_per_cell_per_tstep = np.array([[py_data_dict_per_cell[c][
                                                "rac_acts"][t]
                                            for c in range(NUM_CELLS)]
                                           for t in range(NUM_TSTEPS)])
py_rac_act_arrows_per_cell_per_tstep = \
    50 * py_rac_acts_per_cell_per_tstep[:, :, :, np.newaxis] * \
    py_uovs_per_cell_per_step
py_rho_acts_per_cell_per_tstep = np.array([[py_data_dict_per_cell[c][
                                                "rho_acts"][t]
                                            for c in range(NUM_CELLS)]
                                           for t in range(NUM_TSTEPS)])
py_rho_act_arrows_per_cell_per_tstep = \
    50 * py_rho_acts_per_cell_per_tstep[:, :, :, np.newaxis] * \
    py_uivs_per_cell_per_tstep


def paint(delta):
    global fig
    global ax
    global tstep
    global NUM_TSTEPS
    ax.cla()
    ax.set_aspect('equal')
    ax.set_xlim([-40, 200])
    ax.set_ylim([-40, 200])
    for (ci, poly) in enumerate(rust_poly_per_cell_per_tstep[tstep]):
        if ci == 0:
            poly_color = "k"
        else:
            poly_color = "g"

        for vix in range(16):
            ax.plot([poly[vix, 0], poly[(vix + 1) % 16, 0]],
                    [poly[vix, 1], poly[(vix + 1) % 16, 1]],
                    color=poly_color)
            ax.annotate(str(vix), (poly[vix, 0], poly[vix, 1]))

    for poly, rac_act_arrows in zip(
            rust_poly_per_cell_per_tstep[tstep],
            rust_rac_act_arrows_per_cell_per_tstep[tstep]
    ):
        for p, rac_arrow in zip(poly, rac_act_arrows):
            ax.arrow(p[0], p[1], 1 * rac_arrow[0], 1 * rac_arrow[1],
                     color="b",
                     length_includes_head=True, head_width=0.0)

    for poly, rho_act_arrows in zip(py_poly_per_cell_per_tstep[tstep],
                                    py_rho_act_arrows_per_cell_per_tstep[
                                        tstep]):
        for p, rho_arrow in zip(poly, rho_act_arrows):
            ax.arrow(p[0], p[1], 1 * rho_arrow[0], 1 * rho_arrow[1],
                     color="r",
                     length_includes_head=True, head_width=0.0)

    for (ci, poly) in enumerate(py_poly_per_cell_per_tstep[tstep]):
        if ci == 0:
            poly_color = "k"
        else:
            poly_color = "g"

        for vix in range(16):
            ax.plot([poly[vix, 0], poly[(vix + 1) % 16, 0]],
                    [poly[vix, 1], poly[(vix + 1) % 16, 1]],
                    color=poly_color, ls="dotted")
            # ax.annotate(str(vix), (poly[vix, 0], poly[vix, 1]))

    for poly, rac_act_arrows in zip(
            py_poly_per_cell_per_tstep[tstep],
            py_rac_act_arrows_per_cell_per_tstep[tstep]
    ):
        for p, rac_arrow in zip(poly, rac_act_arrows):
            ax.arrow(p[0], p[1], 1 * rac_arrow[0], 1 * rac_arrow[1],
                     color="b",
                     length_includes_head=True, head_width=0.0,
                     linestyle="dotted")

    for poly, rho_act_arrows in zip(py_poly_per_cell_per_tstep[tstep],
                                    py_rho_act_arrows_per_cell_per_tstep[
                                        tstep]):
        for p, rho_arrow in zip(poly, rho_act_arrows):
            ax.arrow(p[0], p[1], 1 * rho_arrow[0], 1 * rho_arrow[1],
                     color="r",
                     length_includes_head=True, head_width=0.0,
                     linestyle="dotted")

    ax.set_title("frame {}".format(tstep))
    tstep = (tstep + delta) % NUM_TSTEPS
    plt.show()


def on_press(event):
    global fig
    if event.key == 'x':
        paint(1)
    elif event.key == 'z':
        paint(-1)
    if event.key == 'c':
        paint(-5)
    elif event.key == 'v':
        paint(5)
    elif event.key == 'n':
        paint(-10)
    elif event.key == 'm':
        paint(10)
    fig.canvas.draw()


tstep = 0
fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_press)
paint(0)
