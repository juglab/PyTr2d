import skimage.io as skio
from glob import glob
from skimage.measure import regionprops
from tracking import trackingsolver
import numpy as np

def load_raw(project_folder):
    return skio.imread(sorted(glob(project_folder+'/test/02/*.tif')), plugin='tifffile')

def load_instances(project_folder):
    #todo Multiple sources of segmentation hypotheses must eventually be supported!
    instances = np.zeros((1,92,700,1100))
    #instances[1] = skio.imread(sorted(glob(project_folder+'/seg/cellpose/*.tiff')), plugin='tifffile')
    instances[0] = skio.imread(sorted(glob(project_folder+'/test/02_ST/SEG/*.tif')), plugin='tifffile')
    #for less than 10 images idk why the shape was different
    #instances = np.transpose(instances, axes=[0,3,1,2])
    return instances

def load_updated_instances(project_folder):
    save_tracking(trackingsolver.get_instance(),project_folder)
    return trackingsolver.get_instance()

def load_features(instances, raw):
    feature_dict = [None for l in range(len(instances))]
    for l in range(len(instances)):
        feature_dict[l] = [
            {'labels': instances[l][i],
             'intensity': [props.image_stdev for props in regionprops(instances[0][i], raw[i], extra_properties=[image_stdev])],
             'coords':[props.coords for props in regionprops(instances[l][i])],
             'centroids': [(round(props.centroid[0]), round(props.centroid[1])) for props in
                           regionprops(instances[l][i])],
             'areas': [props.area for props in regionprops(instances[l][i])],
             'cell_ids': [labels for labels in range(len(regionprops(instances[l][i])))]} for i in
            range(len(instances[l]))
        ]

######################################################################################################
    return feature_dict

def load_GT(project_folder):
    return skio.imread(sorted(glob(project_folder+'/tracking/GT/*.tif')), plugin='tifffile')

def save_tracking(instances,project_folder):

    # TODO: I have to write it for all datasets
    instances = instances.astype('uint16')
    for i in range(len(instances)):
        if i < 10:
            skio.imsave(project_folder + '/tracking/output/mask00' + str(i) + '.tif', instances[i], plugin='tifffile')
        elif i < 100:
            skio.imsave(project_folder + '/tracking/output/mask0' + str(i) + '.tif', instances[i], plugin='tifffile')
        # elif i < 1000:
        #     skio.imsave(project_folder + '/tracking/output/mask0' + str(i) + '.tif', instances[i], plugin='tifffile')
        # else:
        #     skio.imsave(project_folder + '/tracking/output/mask' + str(i) + '.tif', instances[i], plugin='tifffile')

def image_stdev(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.std(intensities[region])
