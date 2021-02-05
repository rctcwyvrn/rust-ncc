import copy
import py_model.hardio as hio
import numpy as np


def is_pos(label):
    return label == "poly"


def is_force(label):
    return np.any([(x == label) for x in ["rgtp_forces", "cyto_forces",
                                          "sum_forces", "edge_forces",
                                          "edge_forces_minus"]])


def is_other_vector(label):
    return np.any([(x in label) for x in ["uivs, edge_vecs, uevs, uevs_minus"]])


def is_vector(label):
    return is_force(label) or is_other_vector(label)


def is_extra(label):
    return is_pos_extra(label) or is_force(label) or \
           is_other_vector(label)


LABEL_COLOR_DICT = dict()
# BASICS = ["poly", "rac_acts", "rac_inacts", "rho_acts", "rho_inacts",
#           "sum_forces"]
BASIC_COLORS = ["black", "blue", "seagreen", "red", "darkviolet", "orangered"]
LABEL_COLOR_DICT.update(zip(BASIC_COLORS, BASIC_COLORS))
# GEOMETRY = ["uivs"]
GEOMETRY_COLORS = ["mediumseagreen"]
LABEL_COLOR_DICT.update(zip(GEOMETRY, GEOMETRY_COLORS))
# RAC_RATES = ["kgtps_rac", "kdgtps_rac"]
RAC_RATE_COLORS = ["blue", "seagreen"]
LABEL_COLOR_DICT.update(zip(RAC_RATES, RAC_RATE_COLORS))
# RHO_RATES = ["kgtps_rho", "kdgtps_rho"]
RHO_RATE_COLORS = ["red", "darkviolet"]
LABEL_COLOR_DICT.update(zip(RHO_RATES, RHO_RATE_COLORS))
# FORCES = ["rgtp_forces", "edge_forces", "edge_forces_minus", "cyto_forces"]
FORCE_COLORS = ["darkslategrey", "purple", "crimson", "darkgoldenrod"]
LABEL_COLOR_DICT.update(zip(FORCES, FORCE_COLORS))
# CALC_KGTPS_RAC = ["conc_rac_acts", "x_cils", "x_coas"]
CALC_KGTPS_RAC_COLORS = ["blue", "indianred", "forestgreen"]
LABEL_COLOR_DICT.update(zip(CALC_KGTPS_RAC, CALC_KGTPS_RAC_COLORS))
# DIFFUSION = ["rac_act_net_fluxes"]
DIFFUSION_COLORS = ["chocolate"]
LABEL_COLOR_DICT.update(zip(DIFFUSION, DIFFUSION_COLORS))
# OTHERS = ["edge_strains", "uevs", "poly_area", "coa_update", "cil_update"]
OTHER_COLORS = ["purple", "skyblue", "tomato", "forestgreen", "indianred"]
LABEL_COLOR_DICT.update(zip(OTHERS, OTHER_COLORS))

EXTRA_POSITIONS = \
    ["poly_x", "poly_y", "centroid", ]
EXTRA_POSITION_COLORS = ["dimgrey", "darkgrey", "rosybrown"]
LABEL_COLOR_DICT.update(zip(EXTRA_POSITIONS, EXTRA_POSITION_COLORS))

EXTRA_FORCE_LABELS = ["{}_x" for label in hio.FORCES] + \
                     ["{}_y" for label in hio.forces]
EXTRA_FORCE_COLORS = ["cadetblue", "violet", "deeppink", "gold"] + \
                     ["powderblue", "thistle", "palevioletred", "wheat"]
LABEL_COLOR_DICT.update(zip(EXTRA_FORCE_LABELS, EXTRA_FORCE_COLORS))

IGNORE_LABELS = []
ALL_LABELS = hio.DATA_LABELS


def calc_min_maxes_for_labels(labels, py_data, rust_data, max_step):
    num_cells = len(py_data)

    all_dat = np.zeros(0, dtype=np.float64)
    for label in labels:
        for ix in range(num_cells):
            pycd = py_data[ix]
            rucd = rust_data[ix]

            all_dat = np.append(all_dat, pycd[label][:max_step].flatten())
            all_dat = np.append(all_dat, rucd[label][:max_step].flatten())

    min_lim = np.min([0.0, 1.2 * np.min(all_dat)])
    max_lim = 1.2 * np.max(all_dat)
    if abs(min_lim - max_lim) < 1e-8:
        min_lim = max_lim - 0.5
        max_lim += 0.5
    return min_lim, max_lim


def calc_min_maxes_given_label_grouping(label_groups, py_data, rust_data,
                                        max_step):
    min_maxes = []
    for label_group in label_groups:
        if type(label_group) != list:
            label = label_group
            min_maxes.append((label, calc_min_maxes_for_labels([label],
                                                               py_data,
                                                               rust_data,
                                                               max_step)))
        else:
            group_min_max = calc_min_maxes_for_labels(label_group, py_data,
                                                      rust_data, max_step)
            for label in label_group:
                min_maxes.append((label, group_min_max))
    return dict(min_maxes)


def trim_data_to_step(data, max_step):
    result = copy.deepcopy(data)
    for ix in range(len(data)):
        for label in data[ix].keys():
            result[ix][label] = result[ix][label][:max_step]

    return result


class PlotDataGroup:
    labels = []
    label_groups = []
    description = ""

    def __init__(self, py_data_dict, rust_data_dict, labels, ylim_groups,
                 plot_groups, description):
        self.py_dat = [dict([(label, py_data_dict[ix][label])
                               for label in self.labels]) for ix
                         in range(len(py_data_dict))]
        self.rust_dat = [dict([(label, rust_data_dict[ix][label])
                                 for label in self.labels]) for ix
                           in range(len(rust_data_dict))]
        self.labels = labels
        self.label_groups = ylim_groups
        self.grouped_ylims_dict = dict()
        self.ylims_dict = dict()
        self.recalc_ylims(max_step)

    def recalc_ylims(self, max_step):
        self.grouped_ylims_dict = calc_min_maxes_given_label_grouping(
            self.label_groups, self.py_dat, self.rust_dat, max_step)
        self.ylims_dict = calc_min_maxes_given_label_grouping(
            self.labels, self.py_dat, self.rust_dat, max_step)



class RhoActGroup(PlotDataGroup):
    def __init__(self, py_data_dict, rust_data_dict, max_step):
        self.labels = ["rgtp_forces", "rgtp_forces_uiv_proj", "rho_acts",
                       "x_cils",
                       "kgtps_rho", "cil_update"]
        self.label_groups = [["rgtp_forces", "rgtp_forces_uiv_proj"],
                             "rho_acts", "x_cils", "kgtps_rho", "cil_update"]
        self.description = "Rho activity related data"
        super().__init__(py_data_dict, rust_data_dict, max_step)


class RhoInactGroup(PlotDataGroup):
    def __init__(self, py_data_dict, rust_data_dict, max_step):
        self.labels = ["rho_inacts", "kdgtps_rho", "rac_acts", "x_coas"]
        self.label_groups = [["rho_inacts", "rac_acts"], "kdgtps_rho", "x_coas"]
        self.description = "Rho inactivity related data"
        super().__init__(py_data_dict, rust_data_dict, max_step)


# RHO_GROUP = ("Rac related data",
#              [RHO_ACT_GROUP + RHO_INACT_GROUP])

class ForceGroup(PlotDataGroup):
    def __init__(self, py_data_dict, rust_data_dict, max_step):
        self.labels = ["sum_forces", "sum_forces_uiv_proj",
                       "sum_forces_uev_proj",
                       "sum_forces_uev_minus_proj", "rgtp_forces",
                       "rgtp_forces_uiv_proj",
                       "rgtp_forces_uev_proj",
                       "rgtp_forces_uev_proj",
                       "edge_forces",
                       "edge_forces_uiv_proj",
                       "edge_forces_uev_proj",
                       "edge_forces_uev_minus_proj", "edge_forces_minus",
                       "edge_forces_minus_uiv_proj",
                       "edge_forces_minus_uev_proj",
                       "edge_forces_minus_uev_minus_proj", "cyto_forces",
                       "cyto_forces_uiv_proj", "cyto_forces_uev_proj"]
        self.label_groups = [["sum_forces", "sum_forces_uiv_proj",
                              "sum_forces_uev_proj",
                              "sum_forces_uev_minus_proj", "rgtp_forces",
                              "rgtp_forces_uiv_proj",
                              "rgtp_forces_uev_proj",
                              "rgtp_forces_uev_proj",
                              "edge_forces",
                              "edge_forces_uiv_proj",
                              "edge_forces_uev_proj",
                              "edge_forces_uev_minus_proj", "edge_forces_minus",
                              "edge_forces_minus_uiv_proj",
                              "edge_forces_minus_uev_proj",
                              "edge_forces_minus_uev_minus_proj", "cyto_forces",
                              "cyto_forces_uiv_proj", "cyto_forces_uev_proj"]]
        self.description = "Force related data"
        super().__init__(py_data_dict, rust_data_dict, max_step)


class InteractionsGroup(PlotDataGroup):
    def __init__(self, py_data_dict, rust_data_dict, max_step):
        self.labels = ["coa_update", "x_coas", "cil_update", "x_cils"]
        self.label_groups = [["coa_update", "cil_update"], "x_coas", "x_cils"]
        self.description = "COA and CIL related data"
        super().__init__(py_data_dict, rust_data_dict, max_step)


def split_vector_dat_by_components(label, dat_per_step):
    if "_x" in label:
        return copy.deepcopy(dat_per_step[:, :, 0])
    elif "_y" in label:
        return copy.deepcopy(dat_per_step[:, :, 1])
    else:
        raise Exception("Could not extract as component: {}.".format(label))


def project_data_onto_unit_vecs(data, uvs):
    r = data * np.sum(data * uvs, axis=2)[:, :, np.newaxis]
    return np.array(r)


def calc_extras(extra_labels, data_dict):
    cells = np.arange(len(data_dict))
    labels_to_norm = []
    for ci in cells:
        data_dict[ci]["uevs_minus"] = np.roll(data_dict[ci]["uevs"], -1, axis=1)
    for extra in extra_labels:
        for ci in cells:
            if extra == "centroid":
                poly_per_step = data_dict[ci]["poly"]
                centroids = np.average(poly_per_step, axis=1)
                centroid_dists = np.linalg.norm(centroids, axis=1)
                data_dict[ci][extra] = centroid_dists
            elif "_x" in extra or "_y" in extra:
                data_dict[ci][extra] = \
                    split_vector_dat_by_components(
                        extra, data_dict[ci][extra[:-len("_x")]]
                    )
            elif "_uev_minus_proj" in extra:
                data_dict[ci][extra] = \
                    project_data_onto_unit_vecs(
                        data_dict[ci][extra[:-len("_uev_minus_proj")]],
                        data_dict[ci]["uevs_minus"],
                    )
            elif "_uev_proj" in extra:
                data_dict[ci][extra] = \
                    project_data_onto_unit_vecs(
                        data_dict[ci][extra[:-len("_uev_proj")]],
                        data_dict[ci]["uevs"],
                    )
            elif "_uiv_proj" in extra:
                data_dict[ci][extra] = \
                    project_data_onto_unit_vecs(
                        data_dict[ci][extra[:-len("_uiv_proj")]],
                        data_dict[ci]["uivs"],
                    )
            else:
                labels_to_norm.append(extra)

    labels_to_norm = list(set(labels_to_norm))
    for label in labels_to_norm:
        for ci in cells:
            data_dict[ci][label] = np.linalg.norm(data_dict[ci][label],
                                                  axis=2)

    return data_dict


def update_with_unit_edge_vecs_minus(data_dict):
    num_cells = len(data_dict)
    for ix in range(num_cells):
        data_dict[ix]["uevs_minus"] = \
            -1.0 * copy.deepcopy(np.roll(data_dict[ix]["uevs"], 1, axis=1))
    return data_dict


def update_with_extras(data_dict):
    global EXTRA_LABELS
    data_dict = calc_extras(EXTRA_LABELS, data_dict)

    return data_dict


def generate_groups(py_data_dict, rust_data_dict, max_step):
    py_data_dict = update_with_extras(py_data_dict)
    rust_data_dict = update_with_extras(rust_data_dict)
    return py_data_dict, rust_data_dict, \
           [class_init(py_data_dict, rust_data_dict, max_step)
            for class_init in [PositionGroup, RacActGroup, RacInactGroup,
                               RhoActGroup,
                               RhoInactGroup, RacActConcGroup, ForceGroup,
                               InteractionsGroup]]
