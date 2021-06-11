# Script for generating and processing crossing infrasturecture data
import sys
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import os
import itertools
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString, MultiPoint, GeometryCollection

################################
#
#
# Load data
#
#
# Road network data used to cut pedestrian and vehicle polugons.
# Pedestrian and vehicle polygons
#
################################

projectCRS = {'init' :'epsg:27700'}


with open("config.json") as f:
    config = json.load(f)


gis_data_dir = config['gis_data_dir']
output_directory = os.path.join(gis_data_dir, "processed_gis_data")


gdfORLink = gpd.read_file(os.path.join(output_directory, config["openroads_link_processed_file"]))
gdfORNode = gpd.read_file(os.path.join(output_directory, config["openroads_node_processed_file"]))
gdfORLink.crs = projectCRS
gdfORNode.crs = projectCRS

gdfPaveNode = gpd.read_file(os.path.join(output_directory, config["pavement_nodes_file"]))
gdfPaveLink = gpd.read_file(os.path.join(output_directory, config["pavement_links_file"]))
gdfPaveNode.crs = projectCRS
gdfPaveLink.crs = projectCRS

output_crossings_path = os.path.join(output_directory, 'CrossingAlternatives.shp')

# Load the Open Roads road network as a nx graph
G = nx.MultiGraph()
gdfORLink['fid_dict'] = gdfORLink.apply(lambda x: {"fid":x['fid'],'geometry':x['geometry']}, axis=1)
edges = gdfORLink.loc[:,['MNodeFID','PNodeFID', 'fid_dict']].to_records(index=False)
G.add_edges_from(edges)


#################################
#
#
# Functions
#
#
#################################
def in_angle_range(ang, a1, a2):
    if a1 < a2:
        return (ang>a1) & (ang<a2)
    else:
        b1 = (ang>a1) & (ang<2*np.pi)
        b2 = (ang>=0) & (ang<a2)
        return b1 | b2

def angle_range(a1, a2):
    if a1<a2:
        r = a2-a1
    else:
        r = (2*np.pi-a1) + a2
    return r


def unsignalised_crossing_node_from_junction_node(dfJunctionNodes):
	'''Identifies the pairs of pavement nodes whose connecting link can be treated as an
	unsignalised crossing, assuming that pedestrians have right of way when walking in a straight line.
	'''

	tolerance = 0.3
	data = {'u':[], 'v':[], 'geometry':[], 'roadLinkID':[]}
	
	is_four_way = dfJunctionNodes.shape[0]==4

	# Get pairs of nodes
	row_ids = dfJunctionNodes.index
	for i,j in itertools.combinations(row_ids, 2):
		row1 = dfJunctionNodes.loc[i]
		row2 = dfJunctionNodes.loc[j]

		# Check they are adjacent (have a shared rl id)
		rls1 = set(row1[['v1rlID', 'v2rlID']].values)
		rls2 = set(row2[['v1rlID', 'v2rlID']].values)
		shared_links = list(rls1.intersection(rls2))
		if len(shared_links) != 1:
			continue
		#print(shared_links)

		# Check total angular range between pairs. If meets threshold, consider nodes to be valid unsignalised crossing location
		range1 = angle_range(*row1[['a1','a2']])
		range2 = angle_range(*row2[['a1','a2']])

		upper_lim = np.pi/2 * (1+tolerance)
		lower_lim = np.pi/2 * (1-tolerance)

		meets_angle_criteria = (lower_lim <= range1) & (range1 <= upper_lim) & (lower_lim <= range2) & (range2 <= upper_lim)

		if (meets_angle_criteria) | (is_four_way):
			data['u'].append(row1['fid'])
			data['v'].append(row2['fid'])
			data['roadLinkID'].append(shared_links[0])
			data['geometry'].append(LineString([row1['geometry'], row2['geometry']]))

	return pd.DataFrame(data)

#################################
#
#
# Produce crossing alternatives which represent pedestrian right of way
#
# Identify pavement crossing links that cross roads at locations where pedestrians typically have right of way
# Classified here as where the pavement crossing link is the continuation of a straight pedestrian path.
#
##################################

# Select junctions - road nodes with degree > 2
dfDeg = pd.DataFrame(G.degree(), columns = ['node','degree'])
junction_nodes = dfDeg.loc[ dfDeg['degree']>2, 'node']


# Select the pavement nodes that are associated with these junction nodes
gdfPaveNode = gdfPaveNode.loc[ gdfPaveNode['juncNodeID'].isin(junction_nodes)]


dfUC = gdfPaveNode.groupby('juncNodeID').apply(unsignalised_crossing_node_from_junction_node)
gdfUC = gpd.GeoDataFrame(dfUC, geometry = 'geometry', crs = projectCRS)
gdfUC['type'] = 'unsignalised'
gdfUC['sigPhases'] = None
gdfUC['phaseDurs'] = None
gdfUC.to_file(output_crossings_path)
