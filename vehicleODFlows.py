
# Load OD nodes
# Load itn nodes
# Spatial join so that I have the IDs of the OD nodes
# Load RLnodes and RRI
# Create network using these dataframes
# Query network to identify possible routes between ODs

import numpy as np
import scipy
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

np.random.seed(config['flows_seed'])

gis_data_dir = config['gis_data_dir']
processed_gis_dir = os.path.join(gis_data_dir, "processed_gis_data")
route_info_dir = os.path.join(gis_data_dir, "itn_route_info")

itn_network_edge_data_file = os.path.join(route_info_dir, "itn_edge_list.csv")

itn_node_path = os.path.join(processed_gis_dir, config["mastermap_node_processed_file"])

output_flows_path = os.path.join(processed_gis_dir, config['vehicle_od_flows'])
output_counts_path = os.path.join(processed_gis_dir, "vehicle_OD_counts.csv")
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
graphRL = nx.from_pandas_edgelist(dfEdgeList, source='start_node', target = 'end_node', edge_attr = ['RoadLinkFID','length'], create_using=nx.DiGraph())
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
# Generate a set of trips and find out what flows these trips will produce
#
#
############################
normalization = 2/( (len(graphRL.nodes)-1) * (len(graphRL.nodes)-2) )
bcs = nx.betweenness_centrality(graphRL, normalized=False, weight='length')
dfBCs = pd.DataFrame(bcs.values(), columns = ['bc'], index=bcs.keys())

softmax_bc_param = 5
flow_magnitude_param = 5000
dfBCs['bcSftMax'] = scipy.special.softmax(softmax_bc_param * dfBCs['bc'] * normalization)


# Figuring out how to distribute trips
# Now iterate between all OD pairs and generate trips between these pairs
init_data = {'O':[], 'D':[], 'flow':[]}
dfFlows = pd.DataFrame(init_data)

flow_indicator = 'bcSftMax'

for o in vehicle_OD_nodes:
	o_out_flow = dfBCs.loc[o, flow_indicator]
	
	for d in vehicle_OD_nodes:
		if (o == d):
			continue
		else:
			try:
				path = nx.shortest_path(graphRL,o,d)
				# record which OD can be traversed

				d_in_flow = dfBCs.loc[d, flow_indicator]

				flow = o_out_flow * d_in_flow * flow_magnitude_param

				dfFlows = dfFlows.append({'O':o,'D':d,'flow':flow}, ignore_index=True)
			except Exception as err:
				# Record which can't be
				dfFlows = dfFlows.append({'O':o,'D':d,'flow':0}, ignore_index=True)

# Set index as the origins
dfFlows = dfFlows.set_index(['O','D']).unstack().fillna(0)
dfFlows.columns = [c[1] for c in dfFlows.columns]

# Now find total in and out flows at each node and use these to calibrate flow magnitudes
T = 900
v = 1/10

'''
study_nodes_bc = dfBCs[vehicle_OD_nodes, 'bc'].sum()
total_bc = dfBCs['bc'].sum()
bc_norm = study_nodes_bc / total_bc
'''

# Now use this to estimate what node counts these flows will produce
dfBCs["countIn"] = 0
dfBCs["countOut"] = 0

for OD in vehicle_OD_nodes:

	in_flows = (dfFlows[OD] * T * v).sum()
	out_flows = (dfFlows.loc[OD] * T * v).sum()

	dfBCs.loc[OD, 'countIn'] = in_flows
	dfBCs.loc[OD, 'countOut'] = out_flows

dfBCs['countTot'] = dfBCs['countIn'] + dfBCs['countOut']

dfBCs.to_csv(output_counts_path)
dfFlows.to_csv(output_flows_path, index=False, header=True)

############################
#
#
# Generate OD Flows
#
#
###########################
'''
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
'''