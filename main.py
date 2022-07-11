from dataio import projectio
from viz import napariviz
from tracking import trackingsolver
from tracking.cost_factory import *

project_folder = './data/pytr2d_projects/Fluo-N2DL-HeLa'

# cost_factory = cost_factory()
cost_factory = cost_factory(project_folder + '/tracking/costs.yaml')
# cost_factory.save_parameters(project_folder + '/tracking/costs.yaml')

print(cost_factory.parameters)

raw = projectio.load_raw(project_folder)
instances = projectio.load_instances(project_folder)
timepoints = projectio.load_features(instances)
tracklets = trackingsolver.ilp(timepoints, cost_factory)  # <-- Sheida: NICE! ;)
napariviz.show_tracking(raw,instances,tracklets)