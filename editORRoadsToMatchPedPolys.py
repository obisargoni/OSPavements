# Script to clean the Open Roads road network after processing the topography data
# Objective is to remove road network links that are not associated to a pedestrian
# These links are replaced by 'bridging' connections between the nodes the link connected to or are removed without replacement

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import os
from shapely.geometry import Point, LineString


#######################
#
# Read in data
#
#######################

projectCRS = {'init' :'epsg:27700'}


with open("config.json") as f:
    config = json.load(f)


gis_data_dir = config['gis_data_dir']
output_directory = os.path.join(gis_data_dir, "processed_gis_data")


gdfORLink = gpd.read_file(os.path.join(output_directory, config["openroads_link_processed_file"]))
gdfORNode = gpd.read_file(os.path.join(output_directory, config["openroads_node_processed_file"]))
gdfORLink.crs = projectCRS
gdfORNode.crs = projectCRS

gdfTopoPed = gpd.read_file(os.path.join(output_directory, config["topo_pedestrian_processed_file"]))
gdfTopoPed.crs = projectCRS


#################################
#
#
# Functions
#
#
#################################
def find_and_remove_edge_connect_ends(graph, edge_attribute, edge_value):
	edge_to_remove = None
	for u,v,d in graph.edges(data=True):
		if d[edge_attribute] == edge_value:
			edge_to_remove = (u,v)
			break
	return remove_edge_connect_ends(graph, edge_to_remove)

def remove_edge_connect_ends(graph, edge):
	'''
	'''
	edges_to_add = []
	edges_to_remove = [edge]

	u = edge[0]
	v = edge[1]

	## Find nodes linked to the 'from' node of this edge
	## Create new edges that link these directly to the 'to' node
	for node in graph[v]:
		if node != u:
			data = graph.get_edge_data(v, node)

			# overwrite this edge by removing it and adding replacement than connects to u node
			# id of new edge no longer matches geographic representation of link
			edges_to_remove.append((v,node))
			edges_to_add.append((u, node, data))
	
	# Add these edges to the graph
	graph.add_edges_from(edges_to_add)

	# Remove the edge - need to later clean up the orphan nodes
	graph.remove_edges_from(edges_to_remove)
	return graph

def remove_multiple_edges(graph, edge_attribute, edge_values):
	for edge_value in edge_values:
		graph = find_and_remove_edge_connect_ends(graph, edge_attribute, edge_value)
	return graph


######################################
#
#
# Create nx network from road link data.
#
# Edit the graph to exclude road links that do not have any pedestrian polygons associated to them.
#
# To exclude these from the network the output node of these links is replaced by the input node for all connecting edges.
#
# Then edit the gis data so that it matches this version of the road network.
#
# Then, network better represents decisions pedestrian agent must make. Need to use this version of the network in pedestrian
# strategic route.
#
#######################################
G = nx.Graph()
gdfORLink['fid_dict'] = gdfORLink['fid'].map(lambda x: {"fid":x})
edges = gdfORLink.loc[:,['MNodeFID','PNodeFID', 'fid_dict']].to_records(index=False)
G.add_edges_from(edges)

# Get fids of links which don't have any pedestrian polygons.
fids_no_ped_polys = gdfORLink.loc[~gdfORLink['fid'].isin(gdfTopoPed['roadLinkID']), 'fid'].values

# Don't remove those that are of a *significant* length - raise assertion error if this is going to happen
gdfORLink['length'] = gdfORLink['geometry'].length
assert gdfORLink.loc[gdfORLink['fid'].isin(fids_no_ped_polys) & (gdfORLink['length'] > 25)].shape[0] == 0

# Sinplify road network such that edges that don't have ped polygons associated to them are removed, and connecting nodes directly linked to source node
G_clean = remove_multiple_edges(G.copy(), 'fid', fids_no_ped_polys)

# Remove orphan nodes
G_clean.remove_nodes_from(nx.isolates(G_clean.copy()))

# Need to repreat this process on the geodataframe data so that the road links are updated to match the graph
G_clean_df = pd.DataFrame(G_clean.edges(data='fid'))
gdfORLink_clean = pd.merge(gdfORLink, G_clean_df, left_on='fid', right_on = 2, how = 'inner')
gdfORLink_clean.drop(['PNodeFID','MNodeFID'], axis=1, inplace=True)
gdfORLink_clean.rename(columns = {0:'MNodeFID', 1:'PNodeFID'}, inplace=True)

gdfORLink_clean = gdfORLink_clean.merge(gdfORNode.rename(columns = {'node_fid':'MNodeFID'}), suffixes = ('','_mnode'), on='MNodeFID')
gdfORLink_clean = gdfORLink_clean.merge(gdfORNode.rename(columns = {'node_fid':'PNodeFID'}), suffixes = ('','_pnode'), on='PNodeFID')
gdfORLink_clean.drop(['geometry',2], axis=1, inplace=True)

gdfORLink_clean['geometry'] = gdfORLink_clean.apply(lambda r: LineString([r['geometry_mnode'], r['geometry_pnode']]), axis=1)

gdfORLink_clean.drop(['geometry_pnode','geometry_mnode'], axis=1, inplace=True)
gdfORLink_clean = gdfORLink_clean.set_geometry('geometry')

gdfORNode_clean = gdfORNode.loc[ gdfORNode['node_fid'].isin(gdfORLink_clean['PNodeFID']) | gdfORNode['node_fid'].isin(gdfORLink_clean['MNodeFID']) ]

gdfORLink_clean.to_file(os.path.join(output_directory, config["openroads_link_processed_file"]))
gdfORNode_clean.to_file(os.path.join(output_directory, config["openroads_node_processed_file"]))