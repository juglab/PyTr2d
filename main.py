from dataio import projectio
from viz import napariviz

project_folder = './data/pytr2d_projects/NewEasy_July16_MINI'

raw = projectio.load_raw(project_folder)
instances = projectio.load_instances(project_folder)

napariviz.show_instances(raw,instances)
