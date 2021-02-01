import numpy as np
import py_model.hardio as hio
import orjson
import copy

OUT_PATH_TEMPLATE = "./{}-out/" + hio.WRITE_FILE_NAME_TEMPLATE


def get_py_dat(num_tsteps, num_int_steps, num_cells, cil_mag, coa_mag):
    fp = OUT_PATH_TEMPLATE.format("py", num_tsteps, num_int_steps, num_cells,
                                  cil_mag, coa_mag)
    with open(fp, "rb") as f:
        out = orjson.loads(f.read())
    return out


def get_rust_dat(num_tsteps, num_int_steps, num_cells, cil_mag, coa_mag):
    fp = OUT_PATH_TEMPLATE.format("py", num_tsteps, num_int_steps, num_cells,
                                  cil_mag, coa_mag)
    with open(fp, "rb") as f:
        out = orjson.loads(f.read())
    return out


# Input data has the form: data_per_int_step_per_cell_per_tstep
def gen_data_dict_per_cell(data):
    num_cells = data["header"]["num_cells"]
    tstep_data = data["tsteps"]
    data_dict_per_cell = [dict() for ix in range(num_cells)]
    for ix in range(num_cells):
        for label in hio.DATA_LABELS:
            data_per_tstep_for_cell = list()
            for cells_per_tstep in tstep_data:
                tstep = cells_per_tstep[ix]
                data_per_tstep_for_cell.append(copy.deepcopy(tstep[0][label]))
            data_per_tstep_for_cell = np.array(data_per_tstep_for_cell)
            data_dict_per_cell[ix][label] = copy.deepcopy(data_per_tstep_for_cell)

    return data_dict_per_cell

def get_from_header(model_desc, label, header_dat):
    if label not in header_dat.keys():
        raise Exception("{} header is missing label: {}".format(model_desc,
                                                                label,
                                                                header_dat))
    else:
        return header_dat[label]


def check_header_equality(py_dat, rust_dat):
    py_header = py_dat["header"]
    rust_header = rust_dat["header"]

    for label in hio.HEADER_LABELS:
        pd = get_from_header("py", label, py_header)
        rd = get_from_header("rust", label, rust_header)
        if pd != rd:
            raise Exception("does not match! py {}: {}, r {}: {}".format(
                label, pd, label, rd))

    print("data headers match")
