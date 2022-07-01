import gurobipy as gp
from gurobipy import quicksum
from gurobipy import GRB
from scipy.spatial import distance
from itertools import combinations
from tqdm import tqdm
from scipy.spatial import KDTree

# Gurobi model
tracking_model = gp.Model('Tracking')
print("Gurobi model is created")

action = {
    'segmented': 0,
    'move': 1,
    'division': 2,
    'appearance': 3,
    'disappearance': 4
}

segments = {
    'stardist': 0,
    'embedseg': 1
}
segmentation = {}
tracking = {}
costs = {}

def ilp(timepoints):

    add_variables(timepoints)
    tracking_model.update()
    add_constraints(timepoints)
    set_objective()
    tracking_model.optimize()
    tracklets = show_result(timepoints)
    return tracklets

def add_variables(timepoints):

    for sframe in tqdm(range(len(timepoints) - 1), desc="adding variables to the model"):

        for cell_id in timepoints[sframe]['cell_ids']:
            segmentation[sframe, cell_id] = tracking_model.addVar(vtype=GRB.BINARY, name="segmentation")
            costs[sframe, cell_id, action['segmented']] = -10

        eframe = sframe + 1

        if eframe == len(timepoints)-1:
            for cell_id in timepoints[eframe]['cell_ids']:
                segmentation[eframe, cell_id] = tracking_model.addVar(vtype=GRB.BINARY, name="segmentation")
                costs[eframe, cell_id, action['segmented']] = -10

        for cell_id in timepoints[sframe]['cell_ids']:

            neighborhood = get_neighborhood(timepoints, sframe, cell_id)

            # move
            if len(neighborhood) > 0:
                for n in neighborhood:
                    tracking[sframe, eframe, cell_id, n, -1] = tracking_model.addVar(vtype=GRB.BINARY, name="move")
                    costs[sframe, cell_id, action['move']] = int(abs(
                        timepoints[sframe]['areas'][cell_id] - timepoints[eframe]['areas'][n]) + distance.euclidean(
                        timepoints[sframe]['centroids'][cell_id], timepoints[eframe]['centroids'][n]))

            # division
            if len(neighborhood) > 1:

                siblings = combinations(neighborhood,2)
                for i in siblings:
                    n1 = i[0]
                    n2 = i[1]
                    tracking[sframe, eframe, cell_id, n1, n2] = tracking_model.addVar(vtype=GRB.BINARY, name="division")
                    costs[sframe, cell_id, action['division']] = int(abs(
                        timepoints[sframe]['areas'][cell_id] - timepoints[eframe]['areas'][n1] -
                        timepoints[eframe]['areas'][n2]) + (distance.euclidean(
                        timepoints[sframe]['centroids'][cell_id],
                        timepoints[eframe]['centroids'][n1]) + distance.euclidean(
                        timepoints[sframe]['centroids'][cell_id],
                        timepoints[eframe]['centroids'][n2])) / 2 - distance.euclidean(
                        timepoints[eframe]['centroids'][n1], timepoints[eframe]['centroids'][n2]))

            # dissapearing
            tracking[sframe, -1, cell_id, -1, -1] = tracking_model.addVar(vtype=GRB.BINARY, name="diss")
            costs[sframe, cell_id, action['disappearance']] = 20

        for cell_id in timepoints[eframe]['cell_ids']:
            # appearing
            tracking[-1, eframe, -1, -1, cell_id] = tracking_model.addVar(vtype=GRB.BINARY, name="app")
            costs[eframe, cell_id, action['appearance']] = int(timepoints[eframe]['areas'][cell_id])

def add_constraints(timepoints):

    for sframe in tqdm(range(len(timepoints) - 1), desc="adding constraints to the model"):

        eframe = sframe + 1
        for cell_id in timepoints[sframe]['cell_ids']:
            # a cell in each timeframe can either move, divide or disappear (zero whe it's not segmented) so <= 1
            tracking_model.addConstr(
                quicksum(tracking[a, b, c, d, e] for (a, b, c, d, e) in tracking if a == sframe and c == cell_id) == 1)
            for (a, b, c, d, e) in tracking:
                if a == sframe and c == cell_id:
                    # if it's move/divide/disappeare then it has to be segmented
                    tracking_model.addConstr(tracking[a, b, c, d, e] <= segmentation[a, cell_id])
                    if d != -1:
                        # when it move(moved cell in next timeframe)/divide(first kid) should be segmented
                        tracking_model.addConstr(tracking[a, b, c, d, e] <= segmentation[eframe, d])
                        if e != -1:
                            # for division the second kid also needs to be segmented
                            tracking_model.addConstr(tracking[a, b, c, d, e] <= segmentation[eframe, e])

        for cell_id in timepoints[eframe]['cell_ids']:
            # a cell in each timepoint can either be a daughter of a cell/moved/appeared
            tracking_model.addConstr(
                quicksum(tracking[a, b, c, d, e] for (a, b, c, d, e) in tracking if
                         b == eframe and (d == cell_id or e == cell_id)) == 1)
            for (a, b, c, d, e) in tracking:
                if b == eframe and a == -1 and e == cell_id:
                    # for appearance it has to be segmented first
                    tracking_model.addConstr(tracking[a, b, c, d, e] <= segmentation[eframe, cell_id])

def set_objective():

    tracking_model.setObjective(quicksum(costs[frame, cell_id, action['segmented']] * segmentation[frame, cell_id] \
                                         for (frame,cell_id) in segmentation) + \
                                quicksum(costs[sframe,cell_id,action['move']] * tracking[sframe,eframe,cell_id,x,y] \
                            for (sframe,eframe,cell_id,x,y) in tracking if x != -1 and y == -1) + \
                                quicksum(costs[sframe,cell_id,action['division']] * tracking[sframe,eframe,cell_id,x,y] \
                            for (sframe,eframe,cell_id,x,y) in tracking if x != -1 and y != -1) + \
                                quicksum(costs[sframe,cell_id,action['disappearance']] * tracking[sframe,eframe,cell_id,x,y] \
                            for (sframe,eframe,cell_id,x,y) in tracking if eframe == -1 and x == -1 and y == -1) + \
                                quicksum(costs[eframe,cell_id,action['appearance']] * tracking[sframe,eframe,x,y,cell_id] \
                            for (sframe,eframe,x,y,cell_id) in tracking if sframe == -1 and x == -1 and y == -1), GRB.MINIMIZE)

def get_neighborhood(timepoints, sframe, cell_id):

    max_movement = 40
    eframe = sframe + 1
    s_kd_tree = KDTree(timepoints[sframe]['centroids'])
    e_kd_tree = KDTree(timepoints[eframe]['centroids'])

    neighborhood = s_kd_tree.query_ball_tree(e_kd_tree, max_movement)

    return neighborhood[cell_id]

def show_result(timepoints):

    number_of_ids = -1
    tracklets = []
    for t in range(len(timepoints) - 1):
        if t == 0 :
            for dot in timepoints[t]['centroids']:
                cell_id = timepoints[t]['centroids'].index(dot)
                if segmentation[t,cell_id].x == 1:
                    number_of_ids += 1
                    tracklets.insert(0,[number_of_ids,t,dot[0],dot[1]])

        for (a,b,c,d,e) in tracking:
            if a == t and b == t+1:
                if tracking[a,b,c,d,e].x == 1:
                    if e == -1 and d != -1:
                        dot = timepoints[b]['centroids'][d]
                        timepoints[b]['cell_ids'][d] = timepoints[a]['cell_ids'][c]
                        tracklets.insert(0, [timepoints[a]['cell_ids'][c], b, dot[0], dot[1]])
                    if e != -1 and d != -1:
                        dot = timepoints[b]['centroids'][d]
                        number_of_ids += 1
                        timepoints[b]['cell_ids'][d] = number_of_ids
                        tracklets.insert(0, [number_of_ids, b, dot[0], dot[1]])
                        dot = timepoints[b]['centroids'][e]
                        number_of_ids += 1
                        timepoints[b]['cell_ids'][e] = number_of_ids
                        tracklets.insert(0, [number_of_ids, b, dot[0], dot[1]])
        for dot in timepoints[t]['centroids']:
            cell_id = timepoints[t]['centroids'].index(dot)
            if segmentation[t, cell_id].x == 1 and cell_id not in range(1,number_of_ids+1):
                number_of_ids += 1
                tracklets.insert(0, [number_of_ids, t, dot[0], dot[1]])

    tracklets.sort(key=lambda x: (x[0], x[1]))

    return tracklets
