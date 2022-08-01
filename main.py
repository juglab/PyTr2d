from dataio import projectio
from viz import napariviz
from tracking import trackingsolver
from tracking.cost_factory import cost_factory as cost_factory_fn
import argparse

def runTracking(project_folder):
    # import pdb;pdb.set_trace()
    # cost_factory = cost_factory()
    #cost_factory = cost_factory_fn(project_folder + '/tracking/costs.yaml')
    # cost_factory.save_parameters(project_folder + '/tracking/costs.yaml')

    # print(cost_factory.parameters)
    print(project_folder)
    raw = projectio.load_raw(project_folder)
    instances = projectio.load_instances(project_folder)
    timepoints = projectio.load_features(instances)
    tracklets = trackingsolver.ilp(timepoints)  # <-- Sheida: NICE! ;)
    #napariviz.show_tracking(raw, instances, tracklets)
    instances =projectio.load_updated_instances()
    return raw,instances,tracklets

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--proj_folder', default='./data/pytr2d_projects/NewEasy_July16_MINI')
    args = parse.parse_args()
    runTracking(args.proj_folder)

