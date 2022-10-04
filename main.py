from dataio import projectio
from tracking.trackingsolver import save_tracking_result
from viz import napariviz
from tracking import trackingsolver
from tracking.cost_factory import cost_factory
from tracking.random_forest import random_forest
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def runTracking(project_folder):

    raw = projectio.load_raw(project_folder)
    instances = projectio.load_instances(project_folder)
    instances = instances.astype(int)
    timepoints = projectio.load_features(instances, raw)
    #coefficients = cost_factory(project_folder+ '/tracking/costs.yaml')
    coefficients = random_forest()
    X_mov, Y_mov, norm_mov, X_div, Y_div, norm_div, X_app, Y_app, norm_app, X_dis, Y_dis, norm_dis = coefficients.load_train_data(project_folder)
    clf_mov = RandomForestClassifier(max_depth=10)
    clf_mov.fit(X_mov, Y_mov)
    clf_div = RandomForestClassifier(max_depth=10)
    clf_div.fit(X_div, Y_div)
    clf_app = RandomForestClassifier(max_depth=10)
    clf_app.fit(X_app, Y_app)
    clf_dis = RandomForestClassifier(max_depth=10)
    clf_dis.fit(X_dis, Y_dis)

    tracklets = trackingsolver.ilp(timepoints, clf_mov, norm_mov, clf_div, norm_div, clf_app, norm_app, clf_dis, norm_dis, coefficients)  # <-- Sheida: NICE! ;)
    instances = projectio.load_updated_instances(project_folder)
    #coefficients.save_parameters(project_folder + '/tracking/costs.yaml')
    save_tracking_result(project_folder)
    #tracking_GT = projectio.load_GT(project_folder)
    #napariviz.show_tracking(raw, instances, tracklets, tracking_GT)
    return raw, instances, tracklets#,tracking_GT

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--proj_folder', default='./data/Random_Forest')
    args = parse.parse_args()
    runTracking(args.proj_folder)
