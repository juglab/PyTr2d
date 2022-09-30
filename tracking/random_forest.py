import skimage.io as skio
from glob import glob
from skimage.measure import regionprops
from scipy.spatial import distance, KDTree
from matplotlib import pyplot as plt
import numpy as np
from enum import Enum
from itertools import combinations
from sklearn.preprocessing import StandardScaler

class Action(Enum):
    SEGMENTED = 0
    MOVEMENT = 1
    DIVISION = 2
    APPEARANCE = 3
    DISAPPEARANCE = 4
    NMOVE = 5
    NDIVID = 6
    NAPP = 7
    NDIS = 8

class random_forest:

    # def __init__(self, training_directory=None):
    #     '''
    #     Constructor: if filename to parameters_yaml_file is not given, default values will be used
    #     '''
    #     if training_directory is None:
    #         return
    #     self.load_train_data(training_directory)

    def load_train_data(self, project_directory):
        """
        This function load the raw images, segmentation ground truth and tracking ground truth
        :param project_directory:
        :return: raw images, seg GT, tracking GT and lineage tree
        """
        raw_data = skio.imread(sorted(glob(project_directory + '/train/01/*.tif')), plugin='tifffile')
        seg = skio.imread(sorted(glob(project_directory + '/train/01_ST/SEG/*.tif')), plugin='tifffile')
        tracking = skio.imread(sorted(glob(project_directory + '/train/01_GT/TRA/*.tif')), plugin='tifffile')

        man_track = open(project_directory + '/train/01_GT/TRA/man_track.txt', 'r')
        LT = man_track.readlines()
        man_track.close()
        lineage_tree = []
        for line in LT:
            row = [int(x) for x in line.split(" ")]
            lineage_tree.append(row)

        def image_stdev(region, intensities):
            return np.std(intensities[region])

        feature_dict = [
            {'intensity': [props.image_stdev for props in regionprops(seg[i], raw_data[i], extra_properties=[image_stdev])],
             'centroids': [(round(props.centroid[0]), round(props.centroid[1])) for props in
                           regionprops(seg[i])],
             'areas': [props.area for props in regionprops(seg[i])],
             'cell_ids': [props.label for props in regionprops(seg[i])]} for i in
            range(len(seg))
        ]
        data = []
        move = 0
        divide = 0
        appear = 0
        disappear = 0

        for track in lineage_tree:
            label = track[0]
            track_begins = track[1]
            track_ends = track[2]
            parent = track[3]

            if track_ends > track_begins:
                move = move + track_ends - track_begins
                for t in range(track_begins, track_ends):
                    if label in feature_dict[t]['cell_ids'] and label in feature_dict[t + 1]['cell_ids']:
                        diff_intensity = self.get_diff_intensity_move(feature_dict, t, label)
                        diff_distance = self.get_diff_distance_move(feature_dict, t, label)
                        diff_size = self.get_diff_size_move(feature_dict, t, label)
                        data.append([t, diff_intensity, diff_distance, diff_size, Action.MOVEMENT.value])
                        current_index = feature_dict[t]['cell_ids'].index(label)
                        for x in self.get_neighborhood(feature_dict, t, current_index):
                            if feature_dict[t + 1]['cell_ids'][x] != label:
                                current_intensity = feature_dict[t]['intensity'][current_index]
                                next_intensity = feature_dict[t + 1]['intensity'][x]

                                diff_intensity = current_intensity - next_intensity

                                current_position = feature_dict[t]['centroids'][current_index]
                                next_position = feature_dict[t + 1]['centroids'][x]

                                diff_distance = distance.euclidean(current_position, next_position)

                                current_size = feature_dict[t]['areas'][current_index]
                                next_size = feature_dict[t + 1]['areas'][x]

                                diff_size = current_size - next_size

                                data.append([t, diff_intensity, diff_distance, diff_size, Action.NMOVE.value])
                                break

            # 8374
            if parent > 0:
                divide += 1
                current_index = feature_dict[track_begins - 1]['cell_ids'].index(parent)
                if parent in feature_dict[track_begins - 1]['cell_ids'] and label in feature_dict[track_begins]['cell_ids']:
                    for sibling_track in lineage_tree:
                        if sibling_track[3] == parent and sibling_track[0] != label and sibling_track[0] in \
                                feature_dict[track_begins]['cell_ids']:
                            diff_intensity = self.get_diff_intensity_div(feature_dict, track_begins, label, sibling_track[0],
                                                                    parent)
                            diff_distance = self.get_diff_distance_div(feature_dict, track_begins, label, sibling_track[0],
                                                                  parent)

                            diff_size = self.get_diff_size_div(feature_dict, track_begins, label, sibling_track[0], parent)
                            if [track_begins - 1, diff_intensity, diff_distance, diff_size, 2] not in data:
                                data.append(
                                    [track_begins - 1, diff_intensity, diff_distance, diff_size, Action.DIVISION.value])
                                if len(self.get_neighborhood(feature_dict, track_begins - 1, current_index)) > 2:
                                    siblings = combinations(self.get_neighborhood(feature_dict, track_begins - 1, current_index),
                                                            2)
                                    for i in siblings:
                                        if i == (feature_dict[track_begins]['cell_ids'].index(label),
                                                 feature_dict[track_begins]['cell_ids'].index(sibling_track[0])) or i == (
                                                feature_dict[track_begins]['cell_ids'].index(sibling_track[0]),
                                                feature_dict[track_begins]['cell_ids'].index(label)):
                                            continue
                                        else:

                                            diff_intensity = self.get_diff_intensity_div(feature_dict, track_begins,
                                                                                    feature_dict[track_begins]['cell_ids'][
                                                                                        i[0]],
                                                                                    feature_dict[track_begins]['cell_ids'][
                                                                                        i[1]],
                                                                                    parent)
                                            diff_distance = self.get_diff_distance_div(feature_dict, track_begins,
                                                                                  feature_dict[track_begins]['cell_ids'][
                                                                                      i[0]],
                                                                                  feature_dict[track_begins]['cell_ids'][
                                                                                      i[1]],
                                                                                  parent)

                                            diff_size = self.get_diff_size_div(feature_dict, track_begins,
                                                                          feature_dict[track_begins]['cell_ids'][i[0]],
                                                                          feature_dict[track_begins]['cell_ids'][i[1]],
                                                                          parent)
                                            data.append(
                                                [track_begins - 1, diff_intensity, diff_distance, diff_size,
                                                 Action.NDIVID.value])
                                            break
            # 94
            if track_begins > 0 and parent == 0:
                appear += 1
                if label in feature_dict[track_begins]['cell_ids']:
                    distance_from_border = self.get_distance(feature_dict, track_begins, label)
                    index = feature_dict[track_begins]['cell_ids'].index(label)
                    size = -feature_dict[track_begins]['areas'][index]
                    intensity = feature_dict[track_begins]['intensity'][index]
                    data.append([track_begins, intensity, distance_from_border, size, Action.APPEARANCE.value])
            # elif track_begins > 0:
            #     if label in feature_dict[track_begins]['cell_ids']:
            #         distance_from_border = get_distance(feature_dict, track_begins, label)
            #         index = feature_dict[track_begins]['cell_ids'].index(label)
            #         size = -feature_dict[track_begins]['areas'][index]
            #         intensity = feature_dict[track_begins]['intensity'][index]
            #         data.append([track_begins, intensity, distance_from_border, size, Action.NAPP.value])

            # 34
            if track_ends < 91:
                parent_cell = False
                for x in lineage_tree:
                    if x[3] == label:
                        parent_cell = True
                if not parent_cell:
                    disappear += 1
                    if label in feature_dict[track_ends]['cell_ids']:
                        distance_from_border = self.get_distance(feature_dict, track_ends, label)
                        index = feature_dict[track_ends]['cell_ids'].index(label)
                        size = feature_dict[track_ends]['areas'][index]
                        intensity = -feature_dict[track_ends]['intensity'][index]
                        data.append([track_ends, intensity, distance_from_border, size, Action.DISAPPEARANCE.value])
                # else:
                #     if label in feature_dict[track_ends]['cell_ids']:
                #         distance_from_border = get_distance(feature_dict, track_ends, label)
                #         index = feature_dict[track_ends]['cell_ids'].index(label)
                #         size = feature_dict[track_ends]['areas'][index]
                #         intensity = -feature_dict[track_ends]['intensity'][index]
                #         data.append([track_ends, intensity, distance_from_border, size, Action.NDIS.value])
            # 34

        divide /= 2
        divide = int(divide)
        print(move, divide, appear, disappear, len(data))

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        positions = [x[2] for x in data if x[4] == 5]
        sizes = [x[3] for x in data if x[4] == 5]
        intensity = [x[1] for x in data if x[4] == 5]

        ax.scatter(intensity, sizes, positions, c='c', label='notmovement')

        positions = [x[2] for x in data if x[4] == 1]
        sizes = [x[3] for x in data if x[4] == 1]
        intensity = [x[1] for x in data if x[4] == 1]

        ax.scatter(intensity, sizes, positions, c='r', label='movement')

        positions = [x[2] for x in data if x[4] == 6]
        sizes = [x[3] for x in data if x[4] == 6]
        intensity = [x[1] for x in data if x[4] == 6]

        ax.scatter(intensity, sizes, positions, c='violet', label='notdivision')

        positions = [x[2] for x in data if x[4] == 2]
        sizes = [x[3] for x in data if x[4] == 2]
        intensity = [x[1] for x in data if x[4] == 2]

        ax.scatter(intensity, sizes, positions, c='b', label='division')

        positions = [x[2] for x in data if x[4] == 7]
        sizes = [x[3] for x in data if x[4] == 7]
        intensity = [x[1] for x in data if x[4] == 7]

        # ax.scatter(intensity, sizes, positions, c='g', label='notappearance')

        positions = [x[2] for x in data if x[4] == 3]
        sizes = [x[3] for x in data if x[4] == 3]
        intensity = [x[1] for x in data if x[4] == 3]

        ax.scatter(intensity, sizes, positions, c='g', label='appearance')

        positions = [x[2] for x in data if x[4] == 8]
        sizes = [x[3] for x in data if x[4] == 8]
        intensity = [x[1] for x in data if x[4] == 8]

        # ax.scatter(intensity, sizes, positions, c='y', label='notdisappearance')

        positions = [x[2] for x in data if x[4] == 4]
        sizes = [x[3] for x in data if x[4] == 4]
        intensity = [x[1] for x in data if x[4] == 4]

        ax.scatter(intensity, sizes, positions, c='y', label='disappearance')

        ax.set_xlabel('intensity')
        ax.set_ylabel('size')
        ax.set_zlabel('distance')

        plt.show()
        X = []
        Y = []
        sc = StandardScaler()

        for x in data:
            X.append([x[1],x[2],x[3]])
            Y.append(x[4])
        print(X)
        X = sc.fit_transform(X)
        arr1 = np.asarray(X)
        arr2 = np.asarray(Y)
        print(X)
        return arr1, arr2

    def get_diff_intensity_move(self, features, time, label):
        current_index = features[time]['cell_ids'].index(label)
        next_index = features[time + 1]['cell_ids'].index(label)

        current_intensity = features[time]['intensity'][current_index]
        next_intensity = features[time + 1]['intensity'][next_index]

        return current_intensity - next_intensity

    def get_diff_distance_move(self, features, time, label):
        current_index = features[time]['cell_ids'].index(label)
        next_index = features[time + 1]['cell_ids'].index(label)

        current_position = features[time]['centroids'][current_index]
        next_position = features[time + 1]['centroids'][next_index]

        return distance.euclidean(current_position, next_position)

    def get_diff_size_move(self, features, time, label):
        current_index = features[time]['cell_ids'].index(label)
        next_index = features[time + 1]['cell_ids'].index(label)

        current_size = features[time]['areas'][current_index]
        next_size = features[time + 1]['areas'][next_index]

        return current_size - next_size

    def get_diff_intensity_div(self, features, time, daughter1, daughter2, mother):
        daughter1_index = features[time]['cell_ids'].index(daughter1)
        daughter2_index = features[time]['cell_ids'].index(daughter2)
        mother_index = features[time - 1]['cell_ids'].index(mother)

        daughter1_intensity = features[time]['intensity'][daughter1_index]
        daughter2_intensity = features[time]['intensity'][daughter2_index]
        mother_intensity = features[time - 1]['intensity'][mother_index]

        return -(mother_intensity - ((daughter1_intensity + daughter2_intensity) / 2))

    def get_diff_distance_div(self, features, time, daughter1, daughter2, mother):
        daughter1_index = features[time]['cell_ids'].index(daughter1)
        daughter2_index = features[time]['cell_ids'].index(daughter2)
        mother_index = features[time - 1]['cell_ids'].index(mother)

        daughter1_position = features[time]['centroids'][daughter1_index]
        daughter2_position = features[time]['centroids'][daughter2_index]
        mother_position: float = features[time - 1]['centroids'][mother_index]

        return abs((distance.euclidean(mother_position, daughter1_position) + distance.euclidean(mother_position,
                                                                                                 daughter2_position)) / 2 - distance.euclidean(
            daughter1_position, daughter2_position))

    def get_diff_size_div(self, features, time, daughter1, daughter2, mother):
        daughter1_index = features[time]['cell_ids'].index(daughter1)
        daughter2_index = features[time]['cell_ids'].index(daughter2)
        mother_index = features[time - 1]['cell_ids'].index(mother)

        daughter1_size = features[time]['areas'][daughter1_index]
        daughter2_size = features[time]['areas'][daughter2_index]
        mother_size = features[time - 1]['areas'][mother_index]

        return mother_size - (daughter1_size + daughter2_size)

    def get_distance(self, features, time, label):

        index = features[time]['cell_ids'].index(label)
        centroid = features[time]['centroids'][index]
        return min(700 - centroid[0], centroid[0] - 0, 1100 - centroid[1], centroid[1] - 0)

    def get_neighborhood(self, features, time, cell_id):
        #####################################
        max_movement = 45
        #####################################

        s_kd_tree = KDTree(features[time]['centroids'])
        e_kd_tree = KDTree(features[time + 1]['centroids'])

        neighborhood = s_kd_tree.query_ball_tree(e_kd_tree, max_movement)

        return neighborhood[cell_id]
