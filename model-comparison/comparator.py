import numpy as np
import cbor2
import parse
import copy

py_out = "./py-out/out_euler.dat"
rust_out = "./rust-out/out_euler.dat"

delimiter = "++++++++++++++++++++++++++++++"
stages = ["find", "ci", "vertex_coords", "rac_acts",
          "rac_inacts", "rho_acts",
          "rho_inacts", "tot_forces",
          "rgtp_forces", "edge_forces", "cyto_forces",
          "kdgtps_rac", "kgtps_rho", "kdgtps_rho", "complete"]

with open(py_out, 'r') as rf:
    lines = rf.readlines()

py_dat = [[], []]
step_dat = dict()
processed_ci = []
ci = None
stage = 0
for line in lines:
    line = line.strip()
    stage_string = stages[stage]
    if stage_string == "find" and delimiter in line:
        r = delimiter in line
        stage = (stage + 1) % len(stages)
        continue
    elif stage_string == "ci" and "ci" in line:
        r = stage_string in line
        dat_string = parse.parse("ci: {cell_index}", line).named["cell_index"]
        ci = eval(dat_string)
        processed_ci.append(ci)
        step_dat[stage_string] = ci
        stage = (stage + 1) % len(stages)
        continue
    elif stage_string == "complete" and delimiter in line:
        r = delimiter in line
        py_dat[ci].append(copy.deepcopy(step_dat))
        step_dat = dict()
        ci = None
        stage = (stage + 1) % len(stages)
        continue
    elif stage_string in line:
        r = stage_string in line
        parse_string = "{}: {{{}}}".format(stage_string, stage_string)
        dat_string = eval(parse.parse(parse_string, line).named[stage_string])
        step_dat[stage_string] = np.array(dat_string)
        stage = (stage + 1) % len(stages)
        continue

del lines

with open(rust_out, 'r') as rf:
    lines = rf.readlines()

rust_dat = [[], []]
step_dat = dict()
processed_ci = []
ci = None
stage = 0
for line in lines:
    line = line.strip()
    stage_string = stages[stage]
    if stage_string == "find" and delimiter in line:
        r = delimiter in line
        stage = (stage + 1) % len(stages)
        continue
    elif stage_string == "ci" and "ci" in line:
        r = stage_string in line
        dat_string = parse.parse("ci: {cell_index}", line).named["cell_index"]
        ci = eval(dat_string)
        processed_ci.append(ci)
        step_dat[stage_string] = ci
        stage = (stage + 1) % len(stages)
        continue
    elif stage_string == "complete" and delimiter in line:
        r = delimiter in line
        rust_dat[ci].append(copy.deepcopy(step_dat))
        step_dat = dict()
        ci = None
        stage = (stage + 1) % len(stages)
        continue
    elif stage_string in line:
        r = stage_string in line
        parse_string = "{}: {{{}}}".format(stage_string, stage_string)
        dat_string = eval(parse.parse(parse_string, line).named[stage_string])
        step_dat[stage_string] = np.array(dat_string)
        stage = (stage + 1) % len(stages)
        continue

del lines
