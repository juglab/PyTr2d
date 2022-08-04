import skimage.io as skio
from glob import glob
from skimage.measure import regionprops
from tracking import trackingsolver

def load_raw(project_folder):
    return skio.imread(sorted(glob(project_folder+'/raw/*.tif')), plugin='tifffile')

def load_instances(project_folder):
    #todo Multiple sources of segmentation hypotheses must eventually be supported!
    return skio.imread(sorted(glob(project_folder+'/seg/somesource/*.tif')), plugin='tifffile')

def load_updated_instances():
    return trackingsolver.get_instance()

def load_features(instances):

    feature_dict = [
        {'labels': instances[i],
         'coords':[props.coords for props in regionprops(instances[i])],
         'centroids': [(round(props.centroid[0]), round(props.centroid[1])) for props in
                       regionprops(instances[i])],
         'areas': [props.area for props in regionprops(instances[i])],
         'cell_ids': [labels for labels in range(len(regionprops(instances[i])))]} for i in
        range(len(instances))
    ]

    return feature_dict
