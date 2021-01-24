import numpy as np

PY_OUT = "./py-out/out_euler_T={}_NC={}.dat"
RUST_OUT = "./rust-out/out_euler_T={}_NC={}.dat"

INIT_DATA_DELIM = "******************************"
TSTEP_DELIM = "=============================="
CELL_DAT_DELIM = "++++++++++++++++++++++++++++++"
EULER_DELIM = "------------------------------"

INIT_DATA_LABELS = ["num_tsteps", "num_cells", "num_int_steps"]

basics = ["poly", "rac_acts", "rac_inacts", "rho_acts",
          "rho_inacts", "tot_forces"]
basic_colors = ["black", "blue", "slateblue", "red", "indianred", "tab:blue"]
geometry = ["uivs"]
geometry_colors = ["black"]
rac_rates = ["kgtps_rac", "kdgtps_rac"]
rac_rate_colors = ["deepskyblue", "violet"]
rho_rates = ["kgtps_rho", "kdgtps_rho"]
rho_rate_colors = ["orangered", "lightblue"]
forces = ["rgtp_forces", "edge_forces", "cyto_forces"]
force_colors = ["darkolivegreen", "steelblue", "mediumvioletred"]
calc_kgtps_rac = ["conc_rac_acts", "x_cils", "x_coas"]
calc_kgtps_rac_colors = ["blueviolet", "gold", "lightseagreen"]
diffusion = ["rac_act_net_fluxes"]
diffusion_colors = ["teal"]
others = ["edge_strains", "poly_area"]
other_colors = ["tab:orange", "green"]
DATA_LABELS = basics + geometry + forces + rac_rates + rho_rates + \
              calc_kgtps_rac + diffusion + others
DATA_COLORS = basic_colors + geometry_colors + force_colors + rac_rate_colors \
              + rho_rate_colors + calc_kgtps_rac_colors + diffusion_colors + \
              other_colors


def process_data(model_type, num_timesteps, num_cells):
    template_file_path = None
    if model_type == "python":
        template_file_path = PY_OUT
    elif model_type == "rust":
        template_file_path = RUST_OUT

    file_path = template_file_path.format(num_timesteps, num_cells)
    with open(file_path, 'r') as f:
        data = retrieve_data(f)

    return data


def retrieve_data(file):
    data = []
    init_data = retrieve_init_data(file)

    num_tsteps = init_data["num_tsteps"]
    num_cells = init_data["num_cells"]
    num_int_steps = init_data["num_int_steps"]

    for n in range(num_tsteps):
        data.append(retrieve_tstep_data(num_cells, num_int_steps, file))

    return init_data, data


def read_line(file, data_description):
    line = file.readline()
    # print(line)
    if len(line) == 0:
        raise Exception("Read_line got unexpected EOF. Wanted: {}".format(
            data_description))
    else:
        return line.strip()


def confirm_delimiter(delim, file, caller_description):
    data = read_line(file, "{}".format(delim))
    if delim != data:
        err_string = "{}. Unexpected delimiter. Wanted: {}, found: {}"
        raise Exception(err_string.format(caller_description, delim[0],
                                          data[0]))


def retrieve_init_data(file):
    global INIT_DATA_DELIM
    global INIT_DATA_LABELS
    init_data = dict([(label, -1) for label in INIT_DATA_LABELS])
    confirm_delimiter(INIT_DATA_DELIM, file, "retrieve_init_data")

    for label in INIT_DATA_LABELS:
        init_data[label] = get_labelled_dat(file, label)

    confirm_delimiter(INIT_DATA_DELIM, file, "retrieve_init_data")

    return init_data


def retrieve_tstep_data(num_cells, num_euler_tsteps, file):
    global TSTEP_DELIM
    data = dict()

    confirm_delimiter(TSTEP_DELIM, file, "retrieve_tstep_data")
    for n in range(num_cells):
        ci, cell_dat = retrieve_cell_data(num_euler_tsteps, file)
        data[ci] = cell_dat

    confirm_delimiter(TSTEP_DELIM, file, "retrieve_tstep_data")
    ixs = sorted(data.keys())
    return [data[ix] for ix in ixs]


def get_labelled_dat(file, label):
    line = read_line(file, "retrieve data for label {}".format(label))
    if line[:len(label)] == label:
        return eval(line[len(label) + 1:].strip())
    else:
        raise Exception("Unexpected label: {} (Expected: {})".format(
            line.split(":")[0].strip(), label))


def retrieve_cell_data(num_int_steps, file):
    global CELL_DAT_DELIM
    confirm_delimiter(CELL_DAT_DELIM, file, "retrieve_cell_data")
    ci = get_labelled_dat(file, "ci")
    int_steps = retrieve_euler_tstep_data(
        num_int_steps, file)
    confirm_delimiter(CELL_DAT_DELIM, file, "retrieve_cell_data")

    return ci, int_steps


def retrieve_euler_tstep_data(num_int_steps, file):
    global EULER_DELIM
    global DATA_LABELS
    euler_dat = dict([(label, []) for label in DATA_LABELS])

    for n in range(num_int_steps):
        confirm_delimiter(EULER_DELIM, file, "retrieve_euler_tstep_data")

        for label in DATA_LABELS:
            dat = np.array(get_labelled_dat(file, label))
            euler_dat[label].append(dat)

        confirm_delimiter(EULER_DELIM, file, "retrieve_euler_tstep_data")

    return euler_dat


# Input data has the form: data_per_int_step_per_cell_per_tstep
def dat_per_vert_per_int_step_per_cell_per_tstep(label, data):
    return np.array([[data_per_int_step[label] for data_per_int_step in
                      data_per_int_step_per_cell] for
                     data_per_int_step_per_cell in
                     data])


# Input data has the form: data_per_int_step_per_cell_per_tstep
def dat_per_vert_per_cell_per_int_step(label, data):
    orig = dat_per_vert_per_int_step_per_cell_per_tstep(label, data)
    new = np.swapaxes(orig, 1, 2)
    if len(new.shape) == 5:
        new = np.reshape(new, (new.shape[0] * new.shape[1], new.shape[2],
                               new.shape[3], new.shape[4]))
    else:
        new = np.reshape(new, (new.shape[0] * new.shape[1], new.shape[2],
                               new.shape[3]))
    return new


# Input data has the form: data_per_int_step_per_cell_per_tstep
def dat_per_vert_per_cell_per_tstep(label, data):
    return dat_per_vert_per_int_step_per_cell_per_tstep(label, data)[:, :, 0, :]


# Input data has the form: data_per_int_step_per_cell_per_tstep
def dat_per_vert_per_tstep(label, cell_ix, data):
    retrieved_dat = dat_per_vert_per_cell_per_tstep(label, data)
    only_cell = retrieved_dat[:, cell_ix]
    return only_cell


# Input data has the form: data_per_int_step_per_cell_per_tstep
def dat_per_vert_per_int_step(label, cell_ix, data):
    retrieved_dat = dat_per_vert_per_cell_per_int_step(label, data)
    only_cell = retrieved_dat[:, cell_ix]
    return only_cell


# Input data has the form: data_per_int_step_per_cell_per_tstep
def dat_per_tstep(label, vert_ix, cell_ix, data):
    return dat_per_vert_per_tstep(label, cell_ix, data)[:, vert_ix]
