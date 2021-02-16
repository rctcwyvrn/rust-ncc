import matplotlib.pyplot as plt
import json
import numpy as np
import json
import cbor2
from matplotlib import animation
import os
import parse
import numpy


def lookup_tstep_ix(tstep):
    return int(np.floor(tstep / frequency))


def p2ds_to_numpy(p2ds):
    vs = []
    for p2d in p2ds:
        vs.append([p2d['x'], p2d['y']])
    return np.array(vs)


def extract_p2ds_from_cell_states(state_key, dat_key, state_recs):
    dat_per_cell_per_tstep = []
    for rec in state_recs:
        dat_per_cell = []
        for cell_rec in rec['states']:
            dat_per_cell.append(p2ds_to_numpy(cell_rec[state_key][dat_key]))
        dat_per_cell_per_tstep.append(np.array(dat_per_cell))
    return np.array(dat_per_cell_per_tstep)


def extract_p2ds_from_interactions(dat_key, state_recs):
    dat_per_cell_per_tstep = []
    for rec in state_recs:
        dat_per_cell = []
        for cell_rec in rec['interactions']:
            dat_per_cell.append(p2ds_to_numpy(cell_rec[dat_key]))
        dat_per_cell_per_tstep.append(np.array(dat_per_cell))
    return np.array(dat_per_cell_per_tstep)


def extract_scalars(state_key, dat_key, state_recs):
    dat_per_cell_per_tstep = []
    for rec in state_recs:
        dat_per_cell = []
        for cell_rec in rec['states']:
            dat_per_cell.append(np.array(cell_rec[state_key][dat_key]))
        dat_per_cell_per_tstep.append(np.array(dat_per_cell))
    return np.array(dat_per_cell_per_tstep)


poly_per_cell_per_tstep = extract_p2ds_from_cell_states('core', 'poly',
                                                        state_recs)
centroids_per_cell_per_tstep = np.array(
    [[np.average(poly, axis=0) for poly in poly_per_cell] for poly_per_cell in
     poly_per_cell_per_tstep])
uivs_per_cell_per_tstep = extract_p2ds_from_cell_states('geom',
                                                        'unit_inward_vecs',
                                                        state_recs)
uovs_per_cell_per_tstep = -1 * uivs_per_cell_per_tstep
rac_acts_per_cell_per_tstep = extract_scalars('core', 'rac_acts', state_recs)
rac_act_arrows_per_cell_per_tstep = \
    50 * rac_acts_per_cell_per_tstep[:, :, :,np.newaxis] * \
    uovs_per_cell_per_tstep
rho_acts_per_cell_per_tstep = extract_scalars('core', 'rho_acts', state_recs)
rho_act_arrows_per_cell_per_tstep = \
    50 * rho_acts_per_cell_per_tstep[:, :, :, np.newaxis] * \
    uivs_per_cell_per_tstep

adhs_per_cell_per_tstep = 5 * extract_p2ds_from_interactions('x_adhs',
                                                             state_recs)

circ_vixs = np.take(np.arange(16), np.arange(17), mode='wrap')
centroid_trails_per_cell_per_tstep = np.zeros(
    shape=(len(tsteps), len(poly_per_cell_per_tstep[0]), 2))


def paint(tstep_ix, ax):
    ax.cla()
    ax.relim()

    # bbox to control ax.relim
    centroid = np.average(centroids_per_cell_per_tstep[tstep_ix], axis=0)
    (xmin, xmax) = [centroid[0] - DEFAULT_BBOX_LIM[0] * 0.5,
                    centroid[0] + DEFAULT_BBOX_LIM[0] * 0.5]
    (ymin, ymax) = [centroid[1] - DEFAULT_BBOX_LIM[1] * 0.5,
                    centroid[1] + DEFAULT_BBOX_LIM[1] * 0.5]
    bbox = np.array(
        [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]])
    ax.plot(bbox[:, 0], bbox[:, 1], color=(0.0, 0.0, 0.0, 0.0))

    for (ci, poly) in enumerate(poly_per_cell_per_tstep[tstep_ix]):
        if ci == 0:
            poly_color = "k"
            centroid_trail_color = (140 / 255, 114 / 255, 114 / 255)
        else:
            poly_color = "g"
            centroid_trail_color = (127 / 255, 191 / 255, 63 / 255)

        # this_cell_centroids = centroids_per_cell_per_tstep[:tstep_ix, ci]
        # ax.plot(this_cell_centroids[:,0], this_cell_centroids[:,1],
        # color=centroid_trail_color)

        for vix in range(16):
            ax.plot([poly[vix, 0], poly[(vix + 1) % 16, 0]],
                    [poly[vix, 1], poly[(vix + 1) % 16, 1]],
                    color=poly_color, linewidth=0.5)
            # ax.annotate(str(vix), (poly[vix, 0], poly[vix, 1]))

    for poly, rac_act_arrows in zip(
            poly_per_cell_per_tstep[tstep_ix],
            rac_act_arrows_per_cell_per_tstep[tstep_ix]
    ):
        for p, rac_arrow in zip(poly, rac_act_arrows):
            ax.arrow(p[0], p[1], 3 * rac_arrow[0], 3 * rac_arrow[1], color="b",
                     length_includes_head=True, head_width=0.0)

    for poly, rho_act_arrows in zip(poly_per_cell_per_tstep[tstep_ix],
                                    rho_act_arrows_per_cell_per_tstep[
                                        tstep_ix]):
        for p, rho_arrow in zip(poly, rho_act_arrows):
            ax.arrow(p[0], p[1], 5 * rho_arrow[0], 5 * rho_arrow[1], color="r",
                     length_includes_head=True, head_width=0.0)

    for poly_ix, poly, adhs in zip(
            np.arange(0, len(poly_per_cell_per_tstep[0])),
            poly_per_cell_per_tstep[tstep_ix],
            adhs_per_cell_per_tstep[tstep_ix]):
        if poly_ix == 0:
            adh_arrow_color = "magenta"
        else:
            adh_arrow_color = "cyan"
        for p, adh in zip(poly, adhs):
            ax.arrow(p[0], p[1], adh[0], adh[1], color=adh_arrow_color,
                     length_includes_head=True, head_width=1.0)

    ax.set_title("frame {}".format(tsteps[tstep_ix]))
    return ax.get_children()


DEFAULT_XLIM = [-40, 200]
DEFAULT_YLIM = [-40, 200]
DEFAULT_BBOX_LIM = [DEFAULT_XLIM[1] - DEFAULT_XLIM[0],
                    DEFAULT_YLIM[1] - DEFAULT_YLIM[0]]
num_tsteps = poly_per_cell_per_tstep.shape[0]
tstep_ix = 0
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(DEFAULT_XLIM)
ax.set_ylim(DEFAULT_YLIM)
# fig.canvas.mpl_connect('key_press_event', on_press)
tstep_ixs = [n for n in range(int(len(tsteps)))]
# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

cell_ani = animation.FuncAnimation(fig, paint, frames=tstep_ixs,
                                   fargs=(ax,),
                                   interval=1, blit=True)
animation_file_name = part_template.format(MP4)
animation_file_path = "{}{}".format(OUT_DIR, animation_file_name)
cell_ani.save(animation_file_path,
              writer=writer)
