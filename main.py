from dataio import projectio
from viz import napariviz
from tracking import trackingsolver

project_folder = './data/pytr2d_projects/NewEasy_July16_MINI'

raw = projectio.load_raw(project_folder)
instances = projectio.load_instances(project_folder)
timepoints = projectio.load_features(instances)
tracklets = trackingsolver.ilp(timepoints)
napariviz.show_tracking(raw,instances,tracklets)