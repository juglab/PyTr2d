import skimage.io as skio
from glob import glob

def load_raw(project_folder):
    return skio.imread(sorted(glob(project_folder+'/raw/*.tif')), plugin='tifffile')

def load_instances(project_folder):
    #todo Multiple sources of segmentation hypotheses must eventually be supported!
    return skio.imread(sorted(glob(project_folder+'/seg/somesource/*.tif')), plugin='tifffile')
