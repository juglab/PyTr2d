import gurobipy as gp
from gurobipy import quicksum
from gurobipy import GRB
from itertools import combinations
from enum import Enum
from skimage.measure import regionprops
from tqdm import tqdm
from scipy.spatial import KDTree
import numpy as np
from scipy.spatial import distance
from tracking.cost_factory import cost_factory
from tracking.random_forest import random_forest


# Gurobi model


tracking_model = gp.Model('Tracking')
tracking_model.Params.Threads = 1000
class Action(Enum):
    SEGMENTED = 0
    MOVE = 1
    DIVISION = 2
    APPEARANCE = 3
    DISAPPEARANCE = 4

segments = {
    'stardist': 0,
    'embedseg': 1
}

segmentation = {}
tracking = {}
costs = {}
timepoints = []
coefficients = {}

def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None

def ilp(frames,classifier1, norm1, classifier2, norm2, classifier3, norm3, classifier4, norm4, coeff):

    global timepoints, clf_mov, norm_mov, clf_div, norm_div, clf_app, norm_app, clf_dis, norm_dis, coefficients
    timepoints = frames
    coefficients = coeff
    clf_mov = classifier1
    norm_mov = norm1
    clf_div = classifier2
    norm_div = norm2
    clf_app = classifier3
    norm_app = norm3
    clf_dis = classifier4
    norm_dis = norm4
    add_variables()
    tracking_model.update()
    add_constraints()
    set_objective()

    tracking_model.optimize()
    tracklets = show_result()
    return tracklets

def add_seg_var(seg_hyp, time, cell_label):
    segmentation[seg_hyp, time, cell_label] = tracking_model.addVar(vtype=GRB.BINARY, name="segmentation")

def add_mov_var(seg_hyp1, seg_hyp2, time, cell_label, moved_cell):
    tracking[seg_hyp1, seg_hyp2, time, time+1, cell_label, moved_cell, -1] = tracking_model.addVar(vtype=GRB.BINARY, name="move")

def add_div_var(seg_hyp1, seg_hyp2, time, cell_label, daughter1, daughter2):
    tracking[seg_hyp1, seg_hyp2, time, time+1, cell_label, daughter1, daughter2] = tracking_model.addVar(vtype=GRB.BINARY, name="division")

def add_dis_var(seg_hyp, time, cell_label):
    tracking[seg_hyp, -1, time, -1, cell_label, -1, -1] = tracking_model.addVar(vtype=GRB.BINARY, name="disa")

def add_app_var(seg_hyp, time, cell_label):
    tracking[-1, seg_hyp, -1, time, -1, -1, cell_label] = tracking_model.addVar(vtype=GRB.BINARY, name="app")

def set_seg_cost(seg_hyp,time, cell_label):

    costs[seg_hyp, time, cell_label, Action.SEGMENTED.value] = -105

def set_mov_cost(seg_hyp1, seg_hyp2, time, cell_label, moved_cell):

    current_intensity = timepoints[0][time]['intensity'][cell_label]
    next_intensity = timepoints[0][time + 1]['intensity'][moved_cell]

    current_position = timepoints[0][time]['centroids'][cell_label]
    next_position = timepoints[0][time + 1]['centroids'][moved_cell]

    diff_distance = distance.euclidean(current_position, next_position)

    current_size = timepoints[0][time]['areas'][cell_label]
    next_size = timepoints[0][time + 1]['areas'][moved_cell]

    costs[seg_hyp1, seg_hyp2, time, cell_label, Action.MOVE.value, moved_cell] = \
    clf_mov.predict_proba(norm_mov.transform([[current_intensity, next_intensity, diff_distance, current_size, next_size]]))[0][0] * 100

def set_div_cost(seg_hyp1, seg_hyp2, time, cell_label, daughter1, daughter2):

    daughter1_intensity = timepoints[0][time+1]['intensity'][daughter1]
    daughter2_intensity = timepoints[0][time+1]['intensity'][daughter2]
    mother_intensity = timepoints[0][time]['intensity'][cell_label]

    daughter1_position = timepoints[0][time+1]['centroids'][daughter1]
    daughter2_position = timepoints[0][time+1]['centroids'][daughter2]
    mother_position = timepoints[0][time]['centroids'][cell_label]

    d1 = distance.euclidean(mother_position, daughter1_position)
    d2 = distance.euclidean(mother_position, daughter2_position)
    d3 = distance.euclidean(daughter1_position, daughter2_position)


    daughter1_size = timepoints[0][time+1]['areas'][daughter1]
    daughter2_size = timepoints[0][time+1]['areas'][daughter2]
    mother_size = timepoints[0][time]['areas'][cell_label]

    costs[seg_hyp1, seg_hyp2, time, cell_label, Action.DIVISION.value, daughter1, daughter2] = \
    clf_div.predict_proba(norm_div.transform([[mother_intensity, daughter1_intensity, daughter2_intensity, d1, d2, d3, mother_size, daughter1_size, daughter2_size]]))[0][0] * 100

def set_dis_cost(seg_hyp, time, cell_label):

    a = timepoints[0][time]['intensity'][cell_label]
    centroid = timepoints[seg_hyp][time]['centroids'][cell_label]
    b = min(700 - centroid[0], centroid[0] - 0, 1100 - centroid[1], centroid[1] - 0)
    c = timepoints[0][time]['areas'][cell_label]

    costs[seg_hyp, time, cell_label, Action.DISAPPEARANCE.value] = \
    clf_dis.predict_proba(norm_dis.transform([[a, b, c]]))[0][0] * 100


def set_app_cost(seg_hyp, time, cell_label):
    a = timepoints[0][time]['intensity'][cell_label]
    centroid = timepoints[seg_hyp][time]['centroids'][cell_label]
    b = min(700 - centroid[0], centroid[0] - 0, 1100 - centroid[1], centroid[1] - 0)
    c = timepoints[0][time]['areas'][cell_label]

    costs[seg_hyp, time, cell_label, Action.APPEARANCE.value] = clf_app.predict_proba(norm_app.transform([[a, b, c]]))[0][
                                                                    0] * 100

def add_variables():

    global timepoints

    for seg_hyp in range(len(timepoints)):
        for sframe in range(len(timepoints[0]) - 1):
            for cell_id in timepoints[seg_hyp][sframe]['cell_ids']:
                add_seg_var(seg_hyp, sframe, cell_id)
                set_seg_cost(seg_hyp, sframe, cell_id)

                # disappearing
                add_dis_var(seg_hyp, sframe, cell_id)
                set_dis_cost(seg_hyp, sframe, cell_id)

            eframe = sframe + 1

            if eframe == len(timepoints[0]) - 1:
                for cell_id in timepoints[seg_hyp][eframe]['cell_ids']:
                    add_seg_var(seg_hyp, eframe, cell_id)
                    set_seg_cost(seg_hyp, eframe, cell_id)

            for cell_id in timepoints[seg_hyp][eframe]['cell_ids']:
                # appearing
                add_app_var(seg_hyp, eframe, cell_id)
                set_app_cost(seg_hyp, eframe, cell_id)

    for seg_hyp1 in range(len(timepoints)):
        for seg_hyp2 in range(len(timepoints)):
            for sframe in range(len(timepoints[0]) - 1):

                for cell_id in timepoints[seg_hyp1][sframe]['cell_ids']:

                    neighborhood = get_neighborhood(seg_hyp1, seg_hyp2, sframe, cell_id)

                    # move
                    if len(neighborhood) > 0:
                        for n in neighborhood:
                            add_mov_var(seg_hyp1,seg_hyp2,sframe,cell_id,n)
                            # giving the size of cells and their centroids in each time point
                            set_mov_cost(seg_hyp1,seg_hyp2,sframe,cell_id,n)

                    # division
                    if len(neighborhood) > 1:

                        siblings = combinations(neighborhood,2)
                        for i in siblings:
                            n1 = i[0]
                            n2 = i[1]
                            add_div_var(seg_hyp1,seg_hyp2,sframe,cell_id,n1,n2)
                            set_div_cost(seg_hyp1,seg_hyp2,sframe,cell_id,n1,n2)

def add_constraints():

    global timepoints
    for hyp in range(len(timepoints)):
        for sframe in tqdm(range(len(timepoints[hyp]) - 1), desc="adding constraints to the model"):

            eframe = sframe + 1
            for cell_id in timepoints[hyp][sframe]['cell_ids']:
                # a cell in each timeframe can either move, divide or disappear (zero when it's not segmented)
                tracking_model.addConstr(
                    quicksum(tracking[h1, h2, a, b, c, d, e] for (h1, h2, a, b, c, d, e) in tracking if
                             a == sframe and c == cell_id and h1 == hyp) == segmentation[hyp, sframe, cell_id])

            for cell_id in timepoints[hyp][eframe]['cell_ids']:
                # a cell in each timepoint can either be a daughter of a cell/moved/appeared (zero when it's not segmented)
                tracking_model.addConstr(
                    quicksum(tracking[h1, h2, a, b, c, d, e] for (h1, h2, a, b, c, d, e) in tracking if
                             h2 == hyp and b == eframe and (d == cell_id or e == cell_id)) == segmentation[
                        hyp, eframe, cell_id])


    #segmentation overlapping conflicts
    for t in tqdm(range(len(timepoints[0])), desc="adding combinatorial constraints"):
        for hyp1 in range(len(timepoints)-1):
            for hyp2 in range(hyp1+1,len(timepoints)):
                for cell1 in timepoints[hyp1][t]['cell_ids']:
                    for cell2 in timepoints[hyp2][t]['cell_ids']:
                        a = timepoints[hyp1][t]['coords'][cell1]
                        b = timepoints[hyp2][t]['coords'][cell2]
                        if [x for x in a if x in b]:
                            tracking_model.addConstr(segmentation[hyp1, t, cell1] + segmentation[hyp2, t, cell2] <=1)

def set_objective():

    tracking_model.setObjective(quicksum(costs[hyp, frame, cell_id, Action.SEGMENTED.value] * segmentation[hyp, frame, cell_id] \
                                         for (hyp, frame,cell_id) in segmentation) + \
                                quicksum(costs[hyp1, hyp2, sframe,cell_id,Action.MOVE.value,x] * tracking[hyp1, hyp2, sframe,eframe,cell_id,x,y] \
                            for (hyp1, hyp2, sframe,eframe,cell_id,x,y) in tracking if x != -1 and y == -1) + \
                                quicksum(costs[hyp1, hyp2, sframe,cell_id,Action.DIVISION.value,x,y] * tracking[hyp1, hyp2, sframe,eframe,cell_id,x,y] \
                            for (hyp1, hyp2, sframe,eframe,cell_id,x,y) in tracking if x != -1 and y != -1) + \
                                quicksum(costs[hyp1, sframe,cell_id,Action.DISAPPEARANCE.value] * tracking[hyp1, hyp2, sframe,eframe,cell_id,x,y] \
                            for (hyp1, hyp2, sframe,eframe,cell_id,x,y) in tracking if eframe == -1 and x == -1 and y == -1 and hyp2 == -1) + \
                                quicksum(costs[hyp2, eframe,cell_id,Action.APPEARANCE.value] * tracking[hyp1, hyp2, sframe,eframe,x,y,cell_id] \
                            for (hyp1, hyp2, sframe,eframe,x,y,cell_id) in tracking if sframe == -1 and x == -1 and y == -1 and hyp1 == -1), GRB.MINIMIZE)

def get_neighborhood(h1, h2, sframe, cell_id):

#####################################
    max_movement = 50
#####################################
    eframe = sframe + 1
    s_kd_tree = KDTree(timepoints[h1][sframe]['centroids'])
    e_kd_tree = KDTree(timepoints[h2][eframe]['centroids'])

    neighborhood = s_kd_tree.query_ball_tree(e_kd_tree, max_movement)

    return neighborhood[cell_id]

def show_result():

    global timepoints

    #changing the ids and generating tracklets
    new_id = 0
    tracklets = []
    for t in range(len(timepoints[0]) - 1):
        t_plus = t+1
        #first time point
        if t == 0:
            for h in range(len(timepoints)):
                for dot in timepoints[h][t]['centroids']:
                    cell_id = timepoints[h][t]['centroids'].index(dot)
                    if segmentation[h,t,cell_id].x:
                        new_id += 1
                        timepoints[h][t]['cell_ids'][cell_id] = new_id
                        tracklets.insert(0,[new_id,t,dot[0],dot[1]])
                    else:
                        timepoints[h][t]['cell_ids'][cell_id] = 0

        for (h1,h2,a,b,c,d,e) in tracking:
            if a == t and b == t_plus and tracking[h1,h2,a,b,c,d,e].x:
                #move - so the id remains the same
                if e == -1 and d != -1:
                    dot = timepoints[h2][b]['centroids'][d]
                    timepoints[h2][b]['cell_ids'][d] = timepoints[h1][a]['cell_ids'][c]
                    tracklets.insert(0, [timepoints[h1][a]['cell_ids'][c], b, dot[0], dot[1]])
                elif e != -1 and d != -1:
                    mother = timepoints[h1][a]['centroids'][c]
                    daughter1 = timepoints[h2][b]['centroids'][d]
                    new_id += 1
                    timepoints[h2][b]['cell_ids'][d] = new_id
                    tracklets.insert(0, [new_id, a, mother[0], mother[1]])
                    tracklets.insert(0, [new_id, b, daughter1[0], daughter1[1]])
                    daughter2 = timepoints[h2][b]['centroids'][e]
                    new_id += 1
                    timepoints[h2][b]['cell_ids'][e] = new_id
                    tracklets.insert(0, [new_id, a, mother[0], mother[1]])
                    tracklets.insert(0, [new_id, b, daughter2[0], daughter2[1]])
        for h in range(len(timepoints)):
            for dot in timepoints[h][t_plus]['centroids']:
                cell_id = timepoints[h][t_plus]['centroids'].index(dot)
                if tracking[-1,h,-1,t_plus, -1,-1,cell_id].x:
                    new_id += 1
                    timepoints[h][t_plus]['cell_ids'][cell_id] = new_id
                    tracklets.insert(0, [new_id, t_plus, dot[0], dot[1]])
                if not segmentation[h,t_plus,cell_id].x:
                    timepoints[h][t_plus]['cell_ids'][cell_id] = 0

    #changing the labels

    #for t in range(len(timepoints)):
        #print(len(timepoints[t]['cell_ids']),timepoints[t]['cell_ids'])
    #_, axs = plt.subplots(1, 2)

    #axs[0].imshow(timepoints[4]['labels'], cmap='hot')
    for h in range(len(timepoints)):
        for t in range(len(timepoints[h])):
            RP = regionprops(timepoints[h][t]['labels'])
            for index in range(len(RP)):
                for (i,j) in RP[index].coords:
                    timepoints[h][t]['labels'][i][j] = timepoints[h][t]['cell_ids'][index]

    #axs[1].imshow(timepoints[4]['labels'],cmap='hot')
    #plt.show()

    tracklets.sort(key=lambda x: (x[0], x[1]))
    return tracklets

def return_cost(event):

    frame = int(event.position[0])
    x = int(event.position[1])
    y = int(event.position[2])

    id = timepoints[frame]["labels"][x][y]
    ind = timepoints[frame]['cell_ids'].index(id)

    out = ""

    #out = out + str(timepoints[frame]['intensity'][ind]) + "\n"

    out = out + "Cost Function for cell with ID " +str(id)+ "\n\nSegmentation Cost: "

    segmentation_cost = str(costs[frame,ind,Action.SEGMENTED.value])

    out = out + segmentation_cost\

    if segmentation[frame, ind].x:
        out = out + " ---> Segmented"

    out = out + "\n"

    out = out + "\nMoving Cost:\nto cell with ID:\n"

    for key,value in costs.items():
        if len(key) == 4:
            if key[0] == frame and key[1] == ind and key[2] == Action.MOVE.value:
                out = out + str(timepoints[frame+1]['cell_ids'][key[3]])
                out = out + " is "
                out = out + str(value)
                if tracking[frame, frame+1, ind, key[3],-1].x:
                    out = out + " ---> Will move"
                out = out + "\n"

    out = out + "\nDivision Cost:\nto cell with IDs:\n"

    for key, value in costs.items():
        if len(key) == 5:
            if key[0] == frame and key[1] == ind and key[2] == Action.DIVISION.value:
                out = out + str(timepoints[frame + 1]['cell_ids'][key[3]])
                out = out + " and "
                out = out + str(timepoints[frame + 1]['cell_ids'][key[4]])
                out = out + " is "
                out = out + str(value)
                if tracking[frame, frame+1, ind, key[3], key[4]].x:
                    out = out + " ---> Will divide"
                out = out + "\n"

    if frame >=0 and frame < len(timepoints)-1:
        out = out + "\nDisappearing Cost is: "
        disappearing_cost = str(costs[frame, ind, Action.DISAPPEARANCE.value])
        out = out + str(disappearing_cost)
        if tracking[frame, -1, ind, -1, -1].x:
            out = out + " ---> Will disappear"
        out = out + "\n"

    if frame >=1 and frame < len(timepoints):
        out = out + "\nAppearing Cost is: "
        appearing_cost = str(costs[frame, ind, Action.APPEARANCE.value])
        out = out + str(appearing_cost)
        if tracking[-1, frame, -1, -1, ind].x:
            out = out + " ---> Appeared"
        out = out + "\n"

    return out

def get_instance():

    global timepoints
    instances = np.zeros((len(timepoints[0]), len(timepoints[0][0]["labels"]), len(timepoints[0][0]["labels"][0])),dtype=int)
    print(instances.shape)
    for t in range(len(timepoints[0])):
        for h in range(len(timepoints)):
            instances[t] = instances[t] + timepoints[h][t]["labels"]
    return instances

def save_tracking_result(project_folder):

    f = open(project_folder + '/tracking/res_track.txt', 'w',encoding='utf-8')

    # saving tracking result
    # man_track.txt - A text file representing an acyclic graph for the whole video. Every line corresponds
    # to a single track that is encoded by four numbers separated by a space:
    # L B E P where
    # L - a unique label of the track (label of markers, 16-bit positive value)
    # B - a zero-based temporal index of the frame in which the track begins
    # E - a zero-based temporal index of the frame in which the track ends
    # P - label of the parent track (0 is used when no parent is defined)

    tracked = [0]
    for t in range(len(timepoints[0])):

        #first time point
        if t == 0:
            for h in range(len(timepoints)):
                for label in timepoints[h][t]['cell_ids']:
                    if label != 0:
                        L = label
                        tracked.append(L)
                        B = 0
                        next = t+1
                        line = str(L) + " " + str(B) + " "
                        if len(timepoints) == 2:
                            while label in timepoints[0][next]['cell_ids'] or label in timepoints[1][next]['cell_ids']:
                                if next == len(timepoints[0]) - 1:
                                    next += 1
                                    break
                                else:
                                    next += 1
                        else:
                            while label in timepoints[0][next]['cell_ids']:
                                if next == len(timepoints[0]) - 1:
                                    next += 1
                                    break
                                else:
                                    next += 1
                        E = next-1
                        P = 0
                        line = line + str(E) + " " + str(P) + "\n"
                        f.writelines(line)
        else:
            for h in range(len(timepoints)):
                for label in timepoints[h][t]['cell_ids']:
                    if label not in tracked:
                        L = label
                        tracked.append(L)
                        B = t
                        line = str(L) + " " + str(B) + " "
                        next = t+1
                        if next != len(timepoints[0]):
                            if len(timepoints) == 2:
                                while label in timepoints[0][next]['cell_ids'] or label in timepoints[1][next]['cell_ids']:
                                    if next == len(timepoints[0]) - 1:
                                        next += 1
                                        break
                                    else:
                                        next += 1
                            else:
                                while label in timepoints[0][next]['cell_ids']:
                                    if next == len(timepoints[0]) - 1:
                                        next += 1
                                        break
                                    else:
                                        next += 1
                        E = next - 1
                        line = line + str(E) + " "
                        for (h1,h2,a,b,c,d,e) in tracking:
                            if a == t-1 and b == t and (d == timepoints[h2][t]['cell_ids'].index(L) or e == timepoints[h2][t]['cell_ids'].index(L)) and tracking[h1,h2,a,b,c,d,e].x:
                                P = timepoints[h1][t-1]['cell_ids'][c]
                                break
                            elif a == -1 and b == t and tracking[h1,h2,a,b,c,d,e].x and e == timepoints[h2][t]['cell_ids'].index(L):
                                P = 0
                                break
                            else:
                                P = None
                        line = line + str(P) + "\n"
                        f.writelines(line)

    f.close()
