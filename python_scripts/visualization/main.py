CBOR = "cbor"
MP4 = "mp4"

OUT_DIR = "../output/"
EXP_PARAMS = "cil={cil}_cal={cal}_adh={adh}_coa={coa}"
RAND_INFO = "seed={seed}_{rand}"


def get_simulation_path(exp_type, cil, cal, adh, coa, seed,
                        randomization, extension):
    if randomization:
        rand_tag = "rt"
    else:
        rand_tag = "rf"

    return exp_type + "_" + \
           EXP_PARAMS.format(cil=cil, cal=cal, adh=adh, coa=coa) + "_" + \
           RAND_INFO.format(seed, rand_tag) + ".{}".format(extension)


def get_matching_simulations(out_dir, exp_type=None, cil=None, cal=None,
                             adh=None, coa=None, seed=None):

    matches = dict()
    for f in os.listdir():
        if os.is_file(os.path.join(out_dir, f)):
            r = parse.parse(F_TEMPLATE, f)
            is_match = True
            for varname, val in \
                    zip(["exp_type", "cil", "cal", "adh", "coa", "ext"],
                        [exp_type, cil, cal, adh, coa, ext]):
                if r[varname] != val:
                    is_match = False
                    break
                else:
                    continue
            if is_match:
                matches[int(r["seed"])] = f
    return matches


exp_type = "separated_pair"
cil = 60
cal = None
adh = 10
coa = 24
MATCHES = get_matching_cbor_files(OUT_DIR, exp_type, cil, cal, adh, coa, "cbor")
seeds = MATCHES.keys()
print("seeds of matching files simulation results: {}".format(sorted(seeds)))

input_file_name = part_template.format(CBOR)
input_file_path = "{}{}".format(OUT_DIR, input_file_name)

if seed not in seeds:
    raise Exception("WARNING: No simulation file: {}", FULL_TEMPLATE.format())

class SimulationData:
    def __init__(self, tsteps, state_recs, frequency):
        self.tsteps = tsteps
        self.state_recs = state_recs
        self.frequency = frequency


def load_simulation_from_file(input_path):
    snapshots = []
    with open(input_path, mode='rb') as sf:
        world_history = cbor2.load(sf)
        success = True
        while success:
            success = False
            try:
                snapshots += cbor2.load(sf)
            finally:
                success = False

    tsteps = [s["tstep"] for s in snapshots]
    state_recs = [s["cells"] for s in snapshots]
    frequency = world_history["snap_freq"]

    return SimulationData(tsteps, state_recs, frequency)


