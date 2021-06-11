
# Load OD nodes
# Load itn nodes
# Spatial join so that I have the IDs of the OD nodes
# Load RLnodes and RRI
# Create network using these dataframes
# Query network to identify possible routes between ODs

import numpy as np
import pandas as pd
import geopandas as gpd
import os
import networkx as nx
import json

#############################
#
#
# Initialise paths to inputs and outputs
#
#
#############################
with open("config.json") as f:
    config = json.load(f)

gis_data_dir = config['gis_data_dir']
processed_gis_dir = os.path.join(gis_data_dir, "processed_gis_data")
route_info_dir = os.path.join(gis_data_dir, "itn_route_info")

itn_network_edge_data_file = os.path.join(route_info_dir, "itn_edge_list.csv")

itn_node_path = os.path.join(processed_gis_dir, config["mastermap_node_processed_file"])

output_flows_path = os.path.join(processed_gis_dir, config['vehicle_od_flows'])
OD_shapefile_path = os.path.join(processed_gis_dir, config['vehicle_od_file'])

# Proportion of ITN nodes to use as vehicle ODs
prop_random_ODs = 0.3


############################
#
#
# Create Vehicle ODs
#
# Select nodes at the edges of network and compliments with a set of randomly selected other nodes
#
############################

# Create road network
dfEdgeList = pd.read_csv(itn_network_edge_data_file)
graphRL = nx.from_pandas_edgelist(dfEdgeList, source='start_node', target = 'end_node', edge_attr = ['RoadLinkFID','weight'], create_using=nx.DiGraph())
assert graphRL.is_directed()

n_ODs = int(prop_random_ODs * len(graphRL.nodes()))

# Select nodes with degree = 2 or 1. Since directed, these are the nodes at the edge of the network
dfDegree = pd.DataFrame(graphRL.to_undirected().degree, columns = ['node','deg'])
edge_nodes = dfDegree.loc[ dfDegree['deg'] == 1, 'node'].values
n_ODs-= len(edge_nodes)

remaining_nodes = dfDegree.loc[ ~dfDegree['node'].isin(edge_nodes), 'node'].values
random_nodes = np.random.choice(remaining_nodes, n_ODs, replace=False)

vehicle_OD_nodes = np.concatenate( [edge_nodes, random_nodes])

gdfOD = gpd.read_file(itn_node_path)
gdfOD = gdfOD.loc[ gdfOD['fid'].isin(vehicle_OD_nodes), ['fid','geometry']]

# Save the vehicle OD data
gdfOD.to_file(OD_shapefile_path)

# Load the vehicle OD nodes and the road link data
#gdfOD = gpd.read_file(OD_shapefile_path)

############################
#
#
# Generate OD Flows
#
#
###########################

# Now iterate between all OD pairs and find which paths are possible and which are not
init_data = {'O':[], 'D':[], 'flowPossible':[]}
dfPossibleFlows = pd.DataFrame(init_data)
excludeFIDs = [] # Nodes that shouldn't be considered as ODs
ODfids = gdfOD['fid']
for o in ODfids:
	for d in ODfids:
		if (o == d):
			continue
		elif (o in [excludeFIDs]) | (d in excludeFIDs):
			# Record which can't be
			dfPossibleFlows = dfPossibleFlows.append({'O':o,'D':d,'flowPossible':0}, ignore_index=True)
		else:
			try:
				path = nx.shortest_path(graphRL,o,d)
				# record which OD can be traversed
				dfPossibleFlows = dfPossibleFlows.append({'O':o,'D':d,'flowPossible':1}, ignore_index=True)
			except Exception as err:
				# Record which can't be
				dfPossibleFlows = dfPossibleFlows.append({'O':o,'D':d,'flowPossible':0}, ignore_index=True)

# Use this dataframe to create flows
dfFlows = dfPossibleFlows.rename(columns = {'flowPossible':'flow'})
dfFlows['flow'] = dfFlows['flow'] * np.random.rand(dfFlows.shape[0])
dfFlows = dfFlows.set_index(['O','D']).unstack().fillna(0)

# Get rid of multiindex
dfFlows.columns = [c[1] for c in dfFlows.columns]

# Save this dataframe and use as the flows matrix
dfFlows.to_csv(output_flows_path, index=False, header=True)