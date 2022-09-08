import napari

def show_tracking(raw, instances, tracklets, GT):

    viewer = napari.view_image(raw, name='raw')
    viewer.add_tracks(tracklets, name='tracks')
    viewer.add_labels(instances, name='seg')
    viewer.add_labels(GT, name='GT')
    napari.run()
