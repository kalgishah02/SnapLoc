import numpy as np
import pandas as pd
from geopy.distance import great_circle
from scipy import spatial
from sklearn.cluster import DBSCAN

# Read metadata and clean
metadata = pd.read_csv('/mnt/metadata.csv')
metadata['tags_clean'] = metadata['tags'].str.split()
metadata = metadata.replace(np.nan, '', regex=True)


# Create labels from tags in the metadata
def creating_labels(x):
    if ("nature" in str(x['tags_clean'])) or ("lake" in str(x['tags_clean'])) or ("river" in str(x['tags_clean'])) or (
            "view" in str(x['tags_clean'])) or ("beach" in str(x['tags_clean'])) or (
            "flowers" in str(x['tags_clean'])) or ("landscape" in str(x['tags_clean'])) or (
            "waterfall" in str(x['tags_clean'])) or ("sunrise" in str(x['tags_clean'])) or (
            "sunset" in str(x['tags_clean'])) or ("water" in str(x['tags_clean'])) or (
            "nationalpark" in str(x['tags_clean'])) or ("alaska" in str(x['tags_clean'])) or (
            "sky" in str(x['tags_clean'])) or ("yosemite" in str(x['tags_clean'])) or (
            "mountains" in str(x['tags_clean'])):
        return 'Natural Landscape'
    elif ("birds" in str(x['tags_clean'])) or ("wild" in str(x['tags_clean'])) or (
            "wildlife" in str(x['tags_clean'])) or ("forest" in str(x['tags_clean'])) or (
            "animals" in str(x['tags_clean'])) or ("zoo" in str(x['tags_clean'])):
        return 'Animals & Birds'
    elif ("food" in str(x['tags_clean'])) or ("brunch" in str(x['tags_clean'])) or (
            "dinner" in str(x['tags_clean'])) or ("lunch" in str(x['tags_clean'])) or (
            "bar" in str(x['tags_clean'])) or ("restaurant" in str(x['tags_clean'])) or (
            "drinking" in str(x['tags_clean'])) or ("eating" in str(x['tags_clean'])):
        return 'Food'
    elif ("urban" in str(x['tags_clean'])) or ("shop" in str(x['tags_clean'])) or (
            "market" in str(x['tags_clean'])) or ("square" in str(x['tags_clean'])) or (
            "building" in str(x['tags_clean'])) or ("citylights" in str(x['tags_clean'])) or (
            "cars" in str(x['tags_clean'])) or ("traffic" in str(x['tags_clean'])) or (
            "city" in str(x['tags_clean'])) or ("downtown" in str(x['tags_clean'])) or (
            "sanfrancisco" in str(x['tags_clean'])) or ("newyork" in str(x['tags_clean'])) or (
            "newyork" in str(x['tags_clean'])) or ("seattle" in str(x['tags_clean'])) or (
            "sandiego" in str(x['tags_clean'])) or ("washington" in str(x['tags_clean'])):
        return 'Urban Scenes'
    elif ("hotel" in str(x['tags_clean'])) or ("home" in str(x['tags_clean'])) or ("interior" in str(x['tags_clean'])):
        return 'Interiors'
    elif ("us" in str(x['tags_clean'])) or ("people" in str(x['tags_clean'])) or ("group" in str(x['tags_clean'])) or (
            "friends" in str(x['tags_clean'])):
        return 'people'
    else:
        return "Others"


metadata['labels'] = metadata.apply(creating_labels, axis=1)
metadata['labels'].value_counts()

# Spatial clusters based on the histogram
data = metadata[['latitude', 'longitude']]
db = DBSCAN(eps=0.06, min_samples=5, metric='haversine', algorithm='ball_tree')
db.fit(data)
np.unique(db.labels_, return_counts=True)

metadata['dblabel'] = db.labels_
dblabel_counts = metadata.groupby(['dblabel'])['image_id'].count().reset_index()

list_mean = metadata.groupby(['dblabel'])['latitude', 'longitude'].mean().reset_index()
list_mean = list_mean.rename(columns={'latitude': 'mean_lat', 'longitude': 'mean_long'})
metadata = metadata.merge(list_mean, left_on='dblabel', right_on='dblabel')


# Temporal bins
def temporal_bins(x):
    if (x['hod'] > 0 and x['hod'] <= 6):
        return 'dawn'
    elif (x['hod'] > 6 and x['hod'] <= 10):
        return 'morning'
    elif (x['hod'] > 10 and x['hod'] <= 14):
        return 'noon'
    elif (x['hod'] > 14 and x['hod'] <= 18):
        return 'dusk'
    elif (x['hod'] > 18 and x['hod'] <= 23):
        return 'night'


metadata['date_taken'] = pd.to_datetime(metadata['date_taken'])
metadata['hod'] = [r.hour for r in metadata.date_taken]
metadata['hour_bins'] = metadata.apply(temporal_bins, axis=1)

# No. of images based on spatial clusters, temporal binning and categories identified from tags
grouped = metadata.groupby(['labels', 'dblabel', 'hour_bins']).agg({'image_id': 'count', 'views': 'sum'}).reset_index()
grouped = pd.merge(grouped, dblabel_counts, left_on='dblabel', right_on='dblabel')
grouped = grouped.rename(columns={'image_id_x': 'num_images', 'image_id_y': 'pts_clusters'})
grouped.head()


# Get nearby categorized clusters based on a location
# input - location and time
# find the no. of cluster based on the location and time
# ouput images based on categories in that cluster
def get_filtered(lat, long, time):
    point1 = (lat, long)
    metadata['lat_long'] = metadata[['latitude', 'longitude']].apply(tuple, axis=1)
    metadata['mean_lat_long'] = metadata[['mean_lat', 'mean_long']].apply(tuple, axis=1)
    metadata['distances'] = [int(great_circle(point1, point).miles) for point in metadata['mean_lat_long']]
    filtered = metadata[(metadata['distances'] <= 20) & (metadata['hour_bins'] == time)]
    grouped = filtered.groupby(['dblabel', 'labels']).agg({'image_id': 'count', 'views': 'sum'}).reset_index()
    return filtered


get_filtered(37.7845212, -122.399388, 'morning')


def radius_pts(list_pts):
    list_diff = []
    difference = 0
    for point1 in list_pts:
        for point2 in list_pts:
            difference = abs(int(great_circle(point1, point2).miles))
            list_diff.append(difference)
    diameter = max(list_diff)
    return diameter


x = metadata.groupby(['dblabel'])['lat_long'].apply(radius_pts)

data = metadata[['latitude', 'longitude']]
kdtree = spatial.KDTree(data)
kdtree.data

pts = [40.750277, -73.987777]
kdtree.query(pts, k=5, eps=3.0, distance_upper_bound=5.0)

metadata.to_csv('/mnt/flask_data.csv')
