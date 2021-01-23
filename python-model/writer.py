import os

WRITE_FILE_PATH = "../model-comparison/py-out/out_euler.dat"


def remake_write_file():
    if os.path.exists(WRITE_FILE_PATH):
        os.remove(WRITE_FILE_PATH)
    f = open(WRITE_FILE_PATH, "a")
    f.close()


def write(strings):
    with open(WRITE_FILE_PATH, "a") as f:
        for s in strings:
            f.write(s + "\n")
