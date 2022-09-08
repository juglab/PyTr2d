from dataio import projectio
from tracking.trackingsolver import save_tracking_result
from viz import napariviz
from tracking import trackingsolver
from tracking.cost_factory import cost_factory
import argparse
import numpy as np

def runTracking(project_folder):

    raw = projectio.load_raw(project_folder)
    instances = projectio.load_instances(project_folder)
    instances = instances.astype(int)
    print(instances.shape)
    timepoints = projectio.load_features(instances)
    coefficients = cost_factory(project_folder+ '/tracking/costs.yaml')
    tracklets = trackingsolver.ilp(timepoints,coefficients)  # <-- Sheida: NICE! ;)
    instances = projectio.load_updated_instances(project_folder)
    coefficients.save_parameters(project_folder + '/tracking/costs.yaml')
    save_tracking_result(project_folder)
    #tracking_GT = projectio.load_GT(project_folder)
    #napariviz.show_tracking(raw, instances, tracklets, tracking_GT)
    return raw,instances,tracklets#,tracking_GT

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--proj_folder', default='./data/pytr2d_projects/Fluo-N2DL-HeLa-1')
    args = parse.parse_args()
    runTracking(args.proj_folder)
