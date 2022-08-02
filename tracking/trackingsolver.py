import gurobipy as gp
from gurobipy import quicksum
from gurobipy import GRB
from matplotlib import pyplot as plt
from scipy.spatial import distance
from itertools import combinations

from skimage.measure import regionprops
from tqdm import tqdm
from scipy.spatial import KDTree
import numpy as np

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
timepoints = []
def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None
def ilp(frames):

    global timepoints
    timepoints = frames
    add_variables()
    tracking_model.update()
    add_constraints()
    set_objective()
    tracking_model.optimize()
    tracklets = show_result()
    return tracklets

def add_variables():

    global timepoints
    for sframe in tqdm(range(len(timepoints) - 1), desc="adding variables to the model"):

        for cell_id in timepoints[sframe]['cell_ids']:
            segmentation[sframe, cell_id] = tracking_model.addVar(vtype=GRB.BINARY, name="segmentation")
            costs[sframe, cell_id, action['segmented']] = -500

        eframe = sframe + 1

        if eframe == len(timepoints)-1:
            for cell_id in timepoints[eframe]['cell_ids']:
                segmentation[eframe, cell_id] = tracking_model.addVar(vtype=GRB.BINARY, name="segmentation")
                costs[eframe, cell_id, action['segmented']] = -500

        for cell_id in timepoints[sframe]['cell_ids']:

            neighborhood = get_neighborhood(timepoints, sframe, cell_id)

            # move
            if len(neighborhood) > 0:
                for n in neighborhood:
                    tracking[sframe, eframe, cell_id, n, -1] = tracking_model.addVar(vtype=GRB.BINARY, name="move")
                    costs[sframe, cell_id, action['move'],n] = int(abs(0.5*
                        timepoints[sframe]['areas'][cell_id] - timepoints[eframe]['areas'][n]) + distance.euclidean(
                        timepoints[sframe]['centroids'][cell_id], timepoints[eframe]['centroids'][n]))

            # division
            if len(neighborhood) > 1:

                siblings = combinations(neighborhood,2)
                for i in siblings:
                    n1 = i[0]
                    n2 = i[1]
                    tracking[sframe, eframe, cell_id, n1, n2] = tracking_model.addVar(vtype=GRB.BINARY, name="division")
                    costs[sframe, cell_id, action['division'],n1,n2] = int(abs(0.5*
                        timepoints[sframe]['areas'][cell_id] - timepoints[eframe]['areas'][n1] -
                        timepoints[eframe]['areas'][n2]) + (distance.euclidean(
                        timepoints[sframe]['centroids'][cell_id],
                        timepoints[eframe]['centroids'][n1]) + distance.euclidean(
                        timepoints[sframe]['centroids'][cell_id],
                        timepoints[eframe]['centroids'][n2])) / 2 - distance.euclidean(
                        timepoints[eframe]['centroids'][n1], timepoints[eframe]['centroids'][n2]))

            # disappearing
            tracking[sframe, -1, cell_id, -1, -1] = tracking_model.addVar(vtype=GRB.BINARY, name="disa")
            costs[sframe, cell_id, action['disappearance']] = 510

        for cell_id in timepoints[eframe]['cell_ids']:
            # appearing
            tracking[-1, eframe, -1, -1, cell_id] = tracking_model.addVar(vtype=GRB.BINARY, name="app")
            costs[eframe, cell_id, action['appearance']] = int(timepoints[eframe]['areas'][cell_id]) + 400

def add_constraints():

    global timepoints
    for sframe in tqdm(range(len(timepoints) - 1), desc="adding constraints to the model"):

        eframe = sframe + 1
        for cell_id in timepoints[sframe]['cell_ids']:
            # a cell in each timeframe can either move, divide or disappear (zero whe it's not segmented)
            tracking_model.addConstr(
                quicksum(tracking[a, b, c, d, e] for (a, b, c, d, e) in tracking if a == sframe and c == cell_id) == segmentation[sframe,cell_id])

        for cell_id in timepoints[eframe]['cell_ids']:
            # a cell in each timepoint can either be a daughter of a cell/moved/appeared (zero whe it's not segmented)
            tracking_model.addConstr(
                quicksum(tracking[a, b, c, d, e] for (a, b, c, d, e) in tracking if
                         b == eframe and (d == cell_id or e == cell_id)) == segmentation[eframe,cell_id])


def set_objective():

    tracking_model.setObjective(quicksum(costs[frame, cell_id, action['segmented']] * segmentation[frame, cell_id] \
                                         for (frame,cell_id) in segmentation) + \
                                quicksum(costs[sframe,cell_id,action['move'],x] * tracking[sframe,eframe,cell_id,x,y] \
                            for (sframe,eframe,cell_id,x,y) in tracking if x != -1 and y == -1) + \
                                quicksum(costs[sframe,cell_id,action['division'],x,y] * tracking[sframe,eframe,cell_id,x,y] \
                            for (sframe,eframe,cell_id,x,y) in tracking if x != -1 and y != -1) + \
                                quicksum(costs[sframe,cell_id,action['disappearance']] * tracking[sframe,eframe,cell_id,x,y] \
                            for (sframe,eframe,cell_id,x,y) in tracking if eframe == -1 and x == -1 and y == -1) + \
                                quicksum(costs[eframe,cell_id,action['appearance']] * tracking[sframe,eframe,x,y,cell_id] \
                            for (sframe,eframe,x,y,cell_id) in tracking if sframe == -1 and x == -1 and y == -1), GRB.MINIMIZE)

def get_neighborhood(timepoints, sframe, cell_id):

    max_movement = 50
    eframe = sframe + 1
    s_kd_tree = KDTree(timepoints[sframe]['centroids'])
    e_kd_tree = KDTree(timepoints[eframe]['centroids'])

    neighborhood = s_kd_tree.query_ball_tree(e_kd_tree, max_movement)

    return neighborhood[cell_id]

def show_result():

    global timepoints

    #changing the ids and generating tracklets
    new_id = 0
    tracklets = []
    for t in range(len(timepoints) - 1):
        t_plus = t+1
        #first time point
        if t == 0:
            for dot in timepoints[t]['centroids']:
                cell_id = timepoints[t]['centroids'].index(dot)
                if segmentation[t,cell_id].x == 1:
                    new_id += 1
                    timepoints[t]['cell_ids'][cell_id] = new_id
                    tracklets.insert(0,[new_id,t,dot[0],dot[1]])
                else:
                    print("NOT SEGMENTED")
                    timepoints[t]['cell_ids'][cell_id] = 0

        for (a,b,c,d,e) in tracking:
            if a == t and b == t+1:
                if tracking[a,b,c,d,e].x == 1:

                    if e == -1 and d != -1:
                        dot = timepoints[b]['centroids'][d]
                        timepoints[b]['cell_ids'][d] = timepoints[a]['cell_ids'][c]
                        tracklets.insert(0, [timepoints[a]['cell_ids'][c], b, dot[0], dot[1]])
                    if e != -1 and d != -1:
                        mother = timepoints[a]['centroids'][c]
                        daughter1 = timepoints[b]['centroids'][d]
                        new_id += 1
                        timepoints[b]['cell_ids'][d] = new_id
                        tracklets.insert(0, [new_id, a, mother[0], mother[1]])
                        tracklets.insert(0, [new_id, b, daughter1[0], daughter1[1]])
                        daughter2 = timepoints[b]['centroids'][e]
                        new_id += 1
                        timepoints[b]['cell_ids'][e] = new_id
                        tracklets.insert(0, [new_id, a, mother[0], mother[1]])
                        tracklets.insert(0, [new_id, b, daughter2[0], daughter2[1]])

        for dot in timepoints[t_plus]['centroids']:
            cell_id = timepoints[t_plus]['centroids'].index(dot)
            if tracking[-1,t_plus, -1,-1,cell_id].x == 1:
                new_id += 1
                timepoints[t_plus]['cell_ids'][cell_id] = new_id
                tracklets.insert(0, [new_id, t_plus, dot[0], dot[1]])

    #changing the labels

    #for t in range(len(timepoints)):
        #print(len(timepoints[t]['cell_ids']),timepoints[t]['cell_ids'])
    #_, axs = plt.subplots(1, 2)

    #axs[0].imshow(timepoints[4]['labels'], cmap='hot')

    for t in range(len(timepoints)):
        RP = regionprops(timepoints[t]['labels'])
        for index in range(len(RP)):
            for (i,j) in RP[index].coords:
                timepoints[t]['labels'][i][j] = timepoints[t]['cell_ids'][index]

    #axs[1].imshow(timepoints[4]['labels'],cmap='hot')
    #plt.show()

    tracklets.sort(key=lambda x: (x[0], x[1]))
    return tracklets

def returnCost(event):

    global timepoints

    frame = int(event.position[0])
    x = int(event.position[1])
    y = int(event.position[2])

    id = timepoints[frame]["labels"][x][y]
    ind = timepoints[frame]['cell_ids'].index(id)

    out = ""

    out = out + "Cost Function for cell with ID " +str(id)+ "\n\nSegmentation Cost: "

    segmentation_cost = str(costs[frame,ind,action['segmented']])

    out = out + segmentation_cost\

    if segmentation[frame, ind].x == 1:
        out = out + " ---> Segmented"

    out = out + "\n"

    out = out + "\nMoving Cost:\nto cell with ID:\n"

    for key,value in costs.items():
        if len(key) == 4:
            if key[0] == frame and key[1] == ind and key[2] == action['move']:
                out = out + str(timepoints[frame+1]['cell_ids'][key[3]])
                out = out + " is "
                out = out + str(value)
                if tracking[frame, frame+1, ind, key[3],-1].x == 1:
                    out = out + " ---> Will move"
                out = out + "\n"

    out = out + "\nDivision Cost:\nto cell with IDs:\n"

    for key, value in costs.items():
        if len(key) == 5:
            if key[0] == frame and key[1] == ind and key[2] == action['division']:
                out = out + str(timepoints[frame + 1]['cell_ids'][key[3]])
                out = out + " and "
                out = out + str(timepoints[frame + 1]['cell_ids'][key[4]])
                out = out + " is "
                out = out + str(value)
                if tracking[frame, frame+1, ind, key[3], key[4]].x == 1:
                    out = out + " ---> Will divide"
                out = out + "\n"

    if frame >=0 and frame < len(timepoints)-1:
        out = out + "\nDisappearing Cost is: "
        disappearing_cost = str(costs[frame, ind, action['disappearance']])
        out = out + str(disappearing_cost)
        if tracking[frame, -1, ind, -1, -1].x == 1:
            out = out + " ---> Will disappear"
        out = out + "\n"

    if frame >=1 and frame < len(timepoints):
        out = out + "\nAppearing Cost is: "
        appearing_cost = str(costs[frame, ind, action['appearance']])
        out = out + str(appearing_cost)
        if tracking[-1, frame, -1, -1, ind].x == 1:
            out = out + " ---> Appeared"
        out = out + "\n"

    return out

def getInstance():

    global timepoints
    instances = np.zeros((len(timepoints), len(timepoints[0]["labels"]), len(timepoints[0]["labels"][0])),dtype=int)

    for t in range(len(timepoints)):
        instances[t] = timepoints[t]["labels"]
    return instances