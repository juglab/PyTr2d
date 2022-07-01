import napari

def show_tracking(raw, instances, tracklets):

    viewer = napari.view_image(raw, name='raw')
    viewer.add_labels(instances, name='seg')
    viewer.add_tracks(tracklets, name='tracks')
    napari.run()
