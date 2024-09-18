import subprocess
import os

def process_1Cto2A(input_paths, output_dir, sen2cor_path):
    for prod in input_paths:
        command = [
            sen2cor_path,
            prod,
            '--output_dir', output_dir
        ]
        subprocess.call(command)


def get_folders(path):
    """Return a list of folder names in the given directory using os.walk()."""
    return list(os.walk(path))[0][1]


def search_band(band, folder, file_type='jp2'):
    for file in os.listdir(folder):
        if band in file and file.endswith(file_type):
            return file
    return None


def to_tiff(input_jp2, output_tiff, resample_to=None):
    cmd = ['gdal_translate', '-of', 'GTiff', input_jp2, output_tiff]
    if resample_to:
        cmd.insert(1, '-tr')
        cmd.insert(2, str(resample_to))
        cmd.insert(3, str(resample_to))
    subprocess.call(cmd)

