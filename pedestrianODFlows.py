import pandas as pd
import numpy as np
import os
import geopandas as gpd
import json
import fiona

from shapely.geometry import Point

##################
#
# Config
#
##################
projectCRS = "epsg:27700"

with open("config.json") as f:
    config = json.load(f)

np.random.seed(config['flows_seed'])

# Proportion of pavement polygons to locate an OD on.
prop_random_ODs = config['prop_ped_poly_ods']
min_distance_of_ped_od_to_ped_road_link = 15

gis_data_dir = config['gis_data_dir']
processed_gis_dir = os.path.join(gis_data_dir, "processed_gis_data")

pavement_nodes_file = os.path.join(processed_gis_dir, config["pavement_nodes_file"])
pavement_links_file = os.path.join(processed_gis_dir, config["pavement_links_file"])
pavement_polygons_file = os.path.join(processed_gis_dir, config["topo_pedestrian_processed_file"])

pedestrian_od_flows = os.path.join(processed_gis_dir, config['pedestrian_od_flows'])
pedestrian_od_file = os.path.join(processed_gis_dir, config['pedestrian_od_file'])

poi_file = os.path.join(gis_data_dir, config["poi_file"])
centre_poi_ref = config["centre_poi_ref"]
dist_from_centre_threshold = 50

create_ods = False


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

if create_ods:
    # Select Origin pavement nodes based on POIs
    c = fiona.open(poi_file)
    gdf_pois = gpd.GeoDataFrame.from_features(c)
    if gdf_pois.crs is None:
        gdf_pois.crs = projectCRS
    else:
        assert gdf_pois.crs.to_string().lower() == projectCRS

    gdfPaveNode = gpd.read_file(pavement_nodes_file)
    gdfPaveLink = gpd.read_file(pavement_links_file)
    gdfTopoPed = gpd.read_file(pavement_polygons_file)

    centre_poi_geom = gdf_pois.loc[ gdf_pois['ref_no'] == centre_poi_ref, 'geometry'].values[0]
    gdfTopoPed['dist_to_centre'] = gdfTopoPed['geometry'].distance(centre_poi_geom)
    centre_pavement_geometry = gdfTopoPed.sort_values(by='dist_to_centre', ascending=True)['geometry'].values[0]

    Os = []
    Os.append(get_random_point_in_polygon(centre_pavement_geometry))

    # Select destination nodes randomly by finding random points in polygons, after filtering out polygons that don't have pavement nodes on them,
    # and that don't correspond to OR road links that are listed in the config file as links to exclude (these tend to be links at the edge of the study area where the data is a bit dodgy)
    gdfTopoPed = gdfTopoPed.loc[ ~gdfTopoPed['roadLinkID'].isin(config['or_links_exclude_ped_ods'])]
    gdfTopoPed = gpd.sjoin(gdfTopoPed, gdfPaveNode, op='intersects')
    candidates = gdfTopoPed.loc[ gdfTopoPed['dist_to_centre'] > dist_from_centre_threshold, 'polyID'].values

    nDs = int(prop_random_ODs * len(candidates))

    # Choose random geoms, then choose random points in those geoms
    Ds = []
    while len(Ds)<nDs:
        ri = np.random.randint(0, gdfTopoPed.shape[0])
        pavement_geom = gdfTopoPed.iloc[ri]['geometry']
        pavement_location = get_random_point_in_polygon(pavement_geom)
        d = min(gdfPaveLink.distance(pavement_location))

        # filter out any locations that are too far from a road link, but only try a few times before skipping this geometry
        i = 0
        while (d>min_distance_of_ped_od_to_ped_road_link) & (i<5):
            pavement_location = get_random_point_in_polygon(pavement_geom)
            d = min(gdfPaveLink.distance(pavement_location))
            i+=1

        if d<min_distance_of_ped_od_to_ped_road_link:
            Ds.append(pavement_location)

    ODs = Os+Ds
    inputFlows = [460] + [0]*len(Ds)
    data = {'fid': ['od_{}'.format(i) for i in range(len(ODs))], 'inFlow':inputFlows, 'geometry':ODs}
    gdfODs = gpd.GeoDataFrame(data, geometry = 'geometry')
    gdfODs.crs = projectCRS

    gdfODs.to_file(pedestrian_od_file)

#################
#
# Generate Flows
#
#################

# Load ped ODs to get number of origins/destinations
gdfODs = gpd.read_file(pedestrian_od_file)

# Constants related to simulation run time and frequency of pedestrian addition
T = 900
v = 1/10

# Choose distribution
def get_flow(g, num_ods, total_in_flow, T=T, v = v, distribution = 'uniform'):

    if distribution == 'uniform':
        f = total_in_flow / (T * v * num_ods)
        return f 
    else:
        return 0

# Initialise OD flows matrix
nODs = gdfODs.shape[0]
flows = np.zeros([nODs, nODs])
ODids = gdfODs['fid'].to_list()
dfFlows = pd.DataFrame(flows, columns = ODids, index = ODids)

for d in ODids:


    # Calculate the flows from all other os to this d
    d_in_flow = gdfODs.loc[gdfODs['fid']==d, 'inFlow'].values[0]

    if d_in_flow == 0:
        continue

    Os = dfFlows[d].index
    nOs = len(Os) - 1
    flows = gdfODs.set_index('fid').loc[ Os, 'geometry'].map(lambda g: get_flow(g, nOs, d_in_flow, T = T, v = v, distribution = 'uniform'))

    dfFlows[d] = flows

    # No self flows
    dfFlows.loc[d, d] = 0

# Create another flows dataframe of flows to and from the central poi
dfFlowsTwoWay = dfFlows.copy()
for o in ODids:

    # Calculate the flows from all other os to this d
    o_out_flow = gdfODs.loc[gdfODs['fid']==o, 'inFlow'].values[0]

    if o_out_flow == 0:
        continue

    Ds = dfFlowsTwoWay[o].index
    nDs = len(Ds) - 1
    flows = gdfODs.set_index('fid').loc[ Ds, 'geometry'].map(lambda g: get_flow(g, nDs, o_out_flow, T = T, v = v, distribution = 'uniform'))

    dfFlowsTwoWay.loc[o] = flows

    # No self flows
    dfFlowsTwoWay.loc[o, o] = 0



dfFlows.to_csv(pedestrian_od_flows, index=False)
dfFlowsTwoWay.to_csv(os.path.splitext(pedestrian_od_flows)[0]+"Twoway.csv", index=False)