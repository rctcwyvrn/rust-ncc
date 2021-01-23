import matplotlib.pyplot as plt
import numpy as np
import cbor2

snapshots = []
file_name = "../output/history_n_cells.cbor"
with open(file_name, mode='rb') as sf:
    world_history = cbor2.load(sf)
    success = True
    while success:
        try:
            snapshots += cbor2.load(sf)
        finally:
            success = False

tsteps = [s["tstep"] for s in snapshots]
state_recs = [s["cells"] for s in snapshots]
frequency = world_history["snap_freq"]


def lookup_tstep_ix(tstep):
    return int(np.floor(tstep / frequency))


def p2ds_to_numpy(p2ds):
    vs = []
    for p2d in p2ds:
        vs.append([p2d['x'], p2d['y']])
    return np.array(vs)


def extract_p2ds_from_cell_states(state_key, dat_key, state_records):
    dat_per_cell_per_tstep = []
    for rec in state_records:
        dat_per_cell = []
        for cell_rec in rec['states']:
            dat_per_cell.append(p2ds_to_numpy(cell_rec[state_key][dat_key]))
        dat_per_cell_per_tstep.append(np.array(dat_per_cell))
    return np.array(dat_per_cell_per_tstep)


def extract_p2ds_from_interactions(dat_key, state_records):
    dat_per_cell_per_tstep = []
    for rec in state_records:
        dat_per_cell = []
        for cell_rec in rec['interactions']:
            dat_per_cell.append(p2ds_to_numpy(cell_rec[dat_key]))
        dat_per_cell_per_tstep.append(np.array(dat_per_cell))
    return np.array(dat_per_cell_per_tstep)


def extract_scalars(state_key, dat_key, state_records):
    dat_per_cell_per_tstep = []
    for rec in state_records:
        dat_per_cell = []
        for cell_rec in rec['states']:
            dat_per_cell.append(np.array(cell_rec[state_key][dat_key]))
        dat_per_cell_per_tstep.append(np.array(dat_per_cell))
    return np.array(dat_per_cell_per_tstep)


poly_per_cell_per_tstep = extract_p2ds_from_cell_states('core', 'poly', state_recs)
uivs_per_cell_per_tstep = extract_p2ds_from_cell_states('geom', 'unit_inward_vecs',
                                                        state_recs)
uovs_per_cell_per_tstep = -1 * uivs_per_cell_per_tstep
rac_acts_per_cell_per_tstep = extract_scalars('core', 'rac_acts', state_recs)
rac_act_arrows_per_cell_per_tstep = 50 * rac_acts_per_cell_per_tstep[:, :, :, np.newaxis] * uovs_per_cell_per_tstep
rho_acts_per_cell_per_tstep = extract_scalars('core', 'rho_acts', state_recs)
rho_act_arrows_per_cell_per_tstep = 50 * rho_acts_per_cell_per_tstep[:, :, :, np.newaxis] * uivs_per_cell_per_tstep

adhs_per_cell_per_tstep = 5 * extract_p2ds_from_interactions('x_adhs', state_recs)

circ_vixs = np.take(np.arange(16), np.arange(17), mode='wrap')


def paint(delta):
    global fig
    global ax
    global tstep_ix
    global num_tsteps
    ax.cla()
    ax.set_aspect('equal')
    ax.set_xlim([-40, 200])
    ax.set_ylim([-40, 200])
    for (ci, poly) in enumerate(poly_per_cell_per_tstep[tstep_ix]):
        if ci == 0:
            poly_color = "k"
        else:
            poly_color = "g"

        for vix in range(16):
            ax.plot([poly[vix, 0], poly[(vix + 1) % 16, 0]],
                    [poly[vix, 1], poly[(vix + 1) % 16, 1]],
                    color=poly_color, marker=".")
            ax.annotate(str(vix), (poly[vix, 0], poly[vix, 1]))

    for poly, rac_act_arrows in zip(
            poly_per_cell_per_tstep[tstep_ix],
            rac_act_arrows_per_cell_per_tstep[tstep_ix]
    ):
        for p, rac_arrow in zip(poly, rac_act_arrows):
            ax.arrow(p[0], p[1], 1 * rac_arrow[0], 1 * rac_arrow[1], color="b",
                     length_includes_head=True, head_width=0.0)

    for poly, rho_act_arrows in zip(poly_per_cell_per_tstep[tstep_ix],
                                    rho_act_arrows_per_cell_per_tstep[tstep_ix]):
        for p, rho_arrow in zip(poly, rho_act_arrows):
            ax.arrow(p[0], p[1], 1 * rho_arrow[0], 1 * rho_arrow[1], color="r",
                     length_includes_head=True, head_width=0.0)

    for poly_ix, poly, adhs in zip(np.arange(0, len(poly_per_cell_per_tstep[0])), poly_per_cell_per_tstep[tstep_ix],
                                   adhs_per_cell_per_tstep[tstep_ix]):
        if poly_ix == 0:
            adh_arrow_color = "magenta"
        else:
            adh_arrow_color = "cyan"
        for p, adh in zip(poly, adhs):
            ax.arrow(p[0], p[1], adh[0], adh[1], color=adh_arrow_color,
                     length_includes_head=True, head_width=1.0)

    ax.set_title("frame {}".format(tsteps[tstep_ix]))
    tstep_ix = (tstep_ix + delta) % len(tsteps)
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


num_tsteps = poly_per_cell_per_tstep.shape[0]
tstep_ix = 0
fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_press)
