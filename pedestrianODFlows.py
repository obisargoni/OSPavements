import pandas as pd
import numpy as np
import os
import geopandas as gpd
import json

from shapely.geometry import Point

##################
#
# Config
#
##################
projectCRS = "epsg:27700"

with open("config.json") as f:
    config = json.load(f)

# Proportion of ITN nodes to use as vehicle ODs
prop_random_ODs = 0.1
min_distance_of_ped_od_to_road_link = 30

gis_data_dir = config['gis_data_dir']
processed_gis_dir = os.path.join(gis_data_dir, "processed_gis_data")

pavement_nodes_file = os.path.join(processed_gis_dir, config["pavement_nodes_file"])
pavement_polygons_file = os.path.join(processed_gis_dir, config["topo_pedestrian_processed_file"])
or_links_file = os.path.join(processed_gis_dir, config["openroads_link_processed_file"])

pedestrian_od_flows = os.path.join(processed_gis_dir, config['pedestrian_od_flows'])
pedestrian_od_file = os.path.join(processed_gis_dir, config['pedestrian_od_file'])

poi_file = os.path.join(gis_data_dir, config["poi_file"])
centre_poi_ref = config["centre_poi_ref"]
dist_from_centre_threshold = 50


#################
#
#
# Functions
#
#
#################
def displace_point(p, d, bearing):
    x = p.x + d*np.sin(bearing)
    y = p.y + d*np.cos(bearing)
    return Point([x,y])

def get_random_point_in_polygon(poly):
    minx, miny, maxx, maxy = poly.bounds
    while True:
        p = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if poly.contains(p):
            return p

#################
#
# Load data and select pedestrian ODs
#
#################

# Select Origin pavement nodes based on POIs
gdf_pois = gpd.read_file(poi_file)
gdfPaveNode = gpd.read_file(pavement_nodes_file)
gdfTopoPed = gpd.read_file(pavement_polygons_file)
gdfORLink = gpd.read_file(or_links_file)

centre_poi_geom = gdf_pois.loc[ gdf_pois['ref_no'] == centre_poi_ref, 'geometry'].values[0]
gdfTopoPed['dist_to_centre'] = gdfTopoPed['geometry'].distance(centre_poi_geom)
centre_pavement_geometry = gdfTopoPed.sort_values(by='dist_to_centre', ascending=True)['geometry'].values[0]

Os = []
Os.append(get_random_point_in_polygon(centre_pavement_geometry))

# Select destination nodes randomly by finding random points in polygons, after filtering out polygons that don't have pavement nodes on them.
gdfTopoPed = gpd.sjoin(gdfTopoPed, gdfPaveNode, op='intersects')
candidates = gdfTopoPed.loc[ gdfTopoPed['dist_to_centre'] > dist_from_centre_threshold, 'polyID'].values

nDs = int(prop_random_ODs * len(candidates))

# Choose random geoms, then choose random points in those geoms
Ds = []
while len(Ds)<nDs:
    ri = np.random.randint(0, gdfTopoPed.shape[0])
    pavement_geom = gdfTopoPed.iloc[ri]['geometry']
    pavement_location = get_random_point_in_polygon(pavement_geom)
    d = min(gdfORLink.distance(pavement_location))

    # filter out any locations that are too far from a road link, but only try a few times before skipping this geometry
    i = 0
    while (d>min_distance_of_ped_od_to_road_link) & (i<5):
        pavement_location = get_random_point_in_polygon(pavement_geom)
        d = min(gdfORLink.distance(pavement_location))
        i+=1

    if d<min_distance_of_ped_od_to_road_link:
        Ds.append(pavement_location)

ODs = Os+Ds
data = {'fid': ['od_{}'.format(i) for i in range(len(ODs))], 'geometry':ODs}
gdfODs = gpd.GeoDataFrame(data, geometry = 'geometry')
gdfODs.crs = projectCRS

gdfODs.to_file(pedestrian_od_file)


# Load ped ODs to get number of origins/destinations
gdfODs = gpd.read_file(pedestrian_od_file)

#################
#
# Generate Flows
#
#################

# Initialise OD flows matrix
flows = np.zeros([len(ODs), len(ODs)])
ODids = gdfODs['fid'].to_list()
dfFlows = pd.DataFrame(flows, columns = ODids, index = ODids)
Oids = ODids[:len(Os)]
Dids = ODids[len(Os):]
# Set random flows between origins and destinations
for o in Oids:
    for d in Dids:
        if (o == d):
            continue

        f_ij = np.random.rand()
        dfFlows.loc[o, d] = f_ij

        f_ji = np.random.rand()
        dfFlows.loc[d, o] = f_ji


dfFlows.to_csv(pedestrian_od_flows, index=False)