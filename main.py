from dataio import projectio
from viz import napariviz
from tracking import trackingsolver
from tracking.cost_factory import cost_factory
import argparse

def runTracking(project_folder):

    raw = projectio.load_raw(project_folder)
    instances = projectio.load_instances(project_folder)
    timepoints = projectio.load_features(instances)
    coefficients = cost_factory()
    tracklets = trackingsolver.ilp(timepoints,coefficients)  # <-- Sheida: NICE! ;)
    instances =projectio.load_updated_instances()
    coefficients.save_parameters(project_folder + '/tracking/costs.yaml')
    #save_tracking_result(project_folder)
    #napariviz.show_tracking(raw, instances, tracklets)
    return raw,instances,tracklets

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--proj_folder', default='./data/pytr2d_projects/NewEasy_July16_MINI')
    args = parse.parse_args()
    runTracking(args.proj_folder)

