from retrieve import *
import numpy as np
import matplotlib.pyplot as plt

init_rust_dat, rust_dat = process_data("rust", 1800, 2)
init_py_dat, py_dat = process_data("python", 1800, 2)

rust_poly_per_cell_per_tstep = np.array(
    dat_per_vert_per_cell_per_tstep("poly", rust_dat))
rust_uivs_per_cell_per_tstep = np.array(
    dat_per_vert_per_cell_per_tstep("uivs", rust_dat))
rust_uovs_per_cell_per_step = -1 * rust_uivs_per_cell_per_tstep
rust_rac_acts_per_cell_per_tstep = \
    np.array(dat_per_vert_per_cell_per_tstep("rac_acts", rust_dat))
rust_rac_act_arrows_per_cell_per_tstep = \
    50 * rust_rac_acts_per_cell_per_tstep[:, :, :, np.newaxis] * \
    rust_uovs_per_cell_per_step
rust_rho_acts_per_cell_per_tstep = \
    np.array(dat_per_vert_per_cell_per_tstep("rho_acts", rust_dat))
rust_rho_act_arrows_per_cell_per_tstep = \
    50 * rust_rho_acts_per_cell_per_tstep[:, :, :, np.newaxis] * \
    rust_uivs_per_cell_per_tstep

py_poly_per_cell_per_tstep = np.array(
    dat_per_vert_per_cell_per_tstep("poly", py_dat))
py_uivs_per_cell_per_tstep = np.array(
    dat_per_vert_per_cell_per_tstep("uivs", py_dat))
py_uovs_per_cell_per_step = -1 * py_uivs_per_cell_per_tstep
py_rac_acts_per_cell_per_tstep = \
    np.array(dat_per_vert_per_cell_per_tstep("rac_acts", py_dat))
py_rac_act_arrows_per_cell_per_tstep = \
    50 * py_rac_acts_per_cell_per_tstep[:, :, :, np.newaxis] * \
    py_uovs_per_cell_per_step
py_rho_acts_per_cell_per_tstep = \
    np.array(dat_per_vert_per_cell_per_tstep("rho_acts", py_dat))
py_rho_act_arrows_per_cell_per_tstep = \
    50 * py_rho_acts_per_cell_per_tstep[:, :, :, np.newaxis] * \
    py_uivs_per_cell_per_tstep


def paint(delta):
    global fig
    global ax
    global tstep
    global num_tsteps
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
    tstep = (tstep + delta) % num_tsteps
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


num_tsteps = init_rust_dat["num_tsteps"]
tstep = 0
fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_press)
paint(0)

