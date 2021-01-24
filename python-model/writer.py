import os

WRITE_FILE_PATH_TEMPLATE = \
    "B:\\rust-ncc\\model-comparison\\py-out\\out_euler_T={}_NC={}.dat"
WRITE_FILE_PATH = ""


def remake_write_file(tsteps, num_cells):
    global WRITE_FILE_PATH
    WRITE_FILE_PATH = WRITE_FILE_PATH_TEMPLATE.format(tsteps, num_cells)
    if os.path.exists(WRITE_FILE_PATH):
        os.remove(WRITE_FILE_PATH)
    f = open(WRITE_FILE_PATH, "a")
    f.close()


def write(strings):
    with open(WRITE_FILE_PATH, "a") as f:
        for s in strings:
            f.write(s + "\n")
