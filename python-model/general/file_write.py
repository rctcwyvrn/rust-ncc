import os

WRITE_FILE_PATH = "..\\..\\model-comparison\\py-out\\"

def remove_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


