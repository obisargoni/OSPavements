'''
This is a script for editing OS's ITN Road Network shapefiles so that:
- the attributeds include an id for the 'to' and 'from' nodes
- line strings are duplicated along links that are bidirectional

Purpose of this script is to use the extracted orientation information from the gml data to edit the roads linesting shape file such that 
two way road are duplicated with one linestring orientated in one direction and the other orientated in another direction.
'''


import numpy as np
import pandas as pd
import geopandas as gpd
import os
import json
import networkx as nx


######################
#
#
# Initialise variables and paths to data inputs and outputs
#
#
#####################
with open("config.json") as f:
    config = json.load(f)

crs = "epsg:27700"

gis_data_dir = config['gis_data_dir']
output_directory = os.path.join(gis_data_dir, "processed_gis_data")

link_shapefile = os.path.join(output_directory, config['mastermap_link_processed_file'])
node_shapefile = os.path.join(output_directory, config['mastermap_node_processed_file'])

route_info_dir = os.path.join(gis_data_dir, "itn_route_info")
road_route_info_path = os.path.join(route_info_dir, "extracted_RRI.csv")
road_node_info_path = os.path.join(route_info_dir, "extracted_RLNodes.csv")

output_itn_edge_list = os.path.join(route_info_dir, "itn_edge_list.csv")
output_shapefile = os.path.join(output_directory, config['mastermap_itn_processed_direction_file'])


###########################
#
#
# Load Data
#
#
##########################

# Load ITN link data
gdf_itn_link = gpd.read_file(os.path.join(gis_data_dir, link_shapefile))

# Load road routing information
dfRRI = pd.read_csv(road_route_info_path)

# Read in node information
dfRLNode = pd.read_csv(road_node_info_path)

# Load ITN Nodes
gdf_itn_node = gpd.read_file(os.path.join(gis_data_dir, node_shapefile))


##########################
#
#
# Check ITN data
#
#
##########################

gdf_itn = gdf_itn_link.copy()

assert gdf_itn['geometry'].type.unique().size == 1
assert gdf_itn['fid'].duplicated().any() == False

# Join the itn road links with the node info to check that the -,+ nodes correspond to the first/last nodes of the linestring
gdf_itn = pd.merge(gdf_itn, dfRLNode, left_on="fid", right_on="RoadLinkFID", how = "left", indicator=True)

assert gdf_itn.loc[ gdf_itn['_merge'] != 'both'].shape[0] == 0
gdf_itn.drop('_merge', axis=1, inplace=True)

# Select just the node fields needed
gdf_itn_node = gdf_itn_node.reindex(columns=['fid','geometry'])

# Merge the nodes with the links
gdf_itn = gdf_itn.merge(gdf_itn_node, left_on = 'PlusNodeFID', right_on = 'fid', how = 'left', suffixes = ('','_plus_node'), indicator= True)
print(gdf_itn['_merge'].value_counts())
assert gdf_itn.loc[ gdf_itn['_merge'] != 'both'].shape[0] == 0 #- not all nodes are in the road link clipped data due to the way it was clipped. should change this
gdf_itn.rename(columns={'_merge':'_merge_plus_node'}, inplace=True)


gdf_itn = gdf_itn.merge(gdf_itn_node, left_on = 'MinusNodeFID', right_on = 'fid', how = 'left', suffixes = ('', '_minus_node'), indicator=True)
print(gdf_itn['_merge'].value_counts())
assert gdf_itn.loc[ gdf_itn['_merge'] != 'both'].shape[0] == 0 #- - not all nodes are in the road link clipped data due to the way it was clipped. should change this
gdf_itn.rename(columns={'_merge':'_merge_minus_node'}, inplace=True)

# Drop rows where plus or minus coords have not been merged in - not required since all nodes merged
#gdf_itn = gdf_itn.loc[(gdf_itn['_merge_minus_node'] == 'both') & (gdf_itn['_merge_plus_node']=='both')]

gdf_itn['line_first_coord'] = gdf_itn['geometry'].map(lambda x: x.coords[0])
gdf_itn['line_last_coord'] = gdf_itn['geometry'].map(lambda x: x.coords[-1])

# Check that the -,+ nodes match the first / last line string coords
gdf_itn['first_coords_match_minus_node'] = gdf_itn['line_first_coord'] == gdf_itn['geometry_minus_node'].map(lambda x: x.coords[0])
gdf_itn['last_coords_match_plus_node'] = gdf_itn['line_last_coord'] == gdf_itn['geometry_plus_node'].map(lambda x: x.coords[0])

assert gdf_itn['first_coords_match_minus_node'].all() == True
gdf_itn['last_coords_match_plus_node'].all() == True

gdf_itn = None

###########################
#
#
# With checks done, can now add in routing info
#
#
###########################

# Before merging itn fids should be unique.
assert gdf_itn_link['fid'].duplicated().any()==False

gdf_itn_link = pd.merge(gdf_itn_link, dfRRI, left_on='fid', right_on='DirectedLinkFID',how = 'left', indicator=True)
gdf_itn_link = pd.merge(gdf_itn_link, dfRLNode, left_on='fid', right_on='RoadLinkFID',how = 'left', indicator=False)
#assert gdf_itn_link.loc[ gdf_itn_link['_merge'] != 'both'].shape[0] ==0 # This fails, ok cos not all road links have orientation routing info

# After merging should still be unique since dfRRI only contains routing info for one way links
#assert gdf_itn_link['fid'].duplicated().any()==False # Fails since dfRRI does contian a small number of couplet entries for road links that are two way

gdf_itn_link['direction'] = gdf_itn_link['DirectedLinkOrientation']
gdf_itn_link['OneWay'] = np.nan
gdf_itn_link.loc[ gdf_itn_link['_merge'] == 'left_only', 'OneWay'] = False
gdf_itn_link.drop("_merge", axis=1, inplace=True)

# Rename the nodes fields to be less than 10 chars
gdf_itn_link.rename(columns = {"PlusNodeFID":"PNodeFID", "MinusNodeFID":"MNodeFID"}, inplace = True)

gdf_itn_link.loc[ gdf_itn_link['OneWay'] == False, 'direction'] = '-'
gdf_itn_link_two_way = gdf_itn_link.loc[ gdf_itn_link['OneWay'] == False]
gdf_itn_link_two_way['direction'] = '+'
gdf_itn_link = pd.concat([gdf_itn_link, gdf_itn_link_two_way], join = 'inner')

# Now should have some duplicated fid
assert gdf_itn_link['fid'].duplicated().any()==True

# Fix this by adding route info to id
gdf_itn_link['fid_undir'] = gdf_itn_link['fid']
gdf_itn_link['fid'] = gdf_itn_link['fid'] + "_" + gdf_itn_link['direction'].replace({'-':'minus', '+':'plus'})
assert gdf_itn_link['fid'].duplicated().any()==False

print(gdf_itn_link['OneWay'].value_counts())
print(gdf_itn_link['direction'].value_counts())
print(gdf_itn_link['DirectedLinkOrientation'].value_counts())


##################################
#
#
# Create Directed Road Network from ITN Data adn Routing Info
#
#
##################################
# Filter the nodes to be just those in the ITN Link data used for the model
dfRLNode = dfRLNode.loc[ dfRLNode["RoadLinkFID"].isin(gdf_itn_link.fid_undir)]

# Check all road links are present and that thy all have 2 nodes
assert len(dfRLNode.RoadLinkFID.unique()) == len(gdf_itn_link.fid_undir.unique())
assert dfRLNode['PlusNodeFID'].isnull().any() == False
assert dfRLNode['MinusNodeFID'].isnull().any() == False

# Merge with data on which links are directed
dfRLNode = pd.merge(dfRLNode, dfRRI, left_on = "RoadLinkFID", right_on = "DirectedLinkFID", how = "left", indicator=True)
dfRLNode.drop("_merge", axis = 1, inplace = True)

# Merge with link weight and id data
dfRLNode = pd.merge(dfRLNode, gdf_itn_link.reindex(columns = ['fid','weight']), left_on = "RoadLinkFID", right_on = "fid", how = "outer", indicator = True)

# Now build the edge list
init_data = {'start_node':[], 'end_node':[]}
dfEdgeList = pd.DataFrame(init_data)
dfRLNode.drop("_merge", axis = 1, inplace = True)

# df1 are teh links that go from minus node to plus node, orientation = +
df1 = dfRLNode.loc[dfRLNode['DirectedLinkOrientation'] == "+"].reindex(columns = ['MinusNodeFID','PlusNodeFID', "RoadLinkFID", "weight"])
# df2 are the nodes that go from plus to minus
df2 = dfRLNode.loc[dfRLNode['DirectedLinkOrientation'] == "-"].reindex(columns = ['PlusNodeFID','MinusNodeFID', "RoadLinkFID", "weight"])
# df3 are the non directed link, part 1
df3 = dfRLNode.loc[dfRLNode['DirectedLinkOrientation'].isnull()].reindex(columns = ['MinusNodeFID','PlusNodeFID', "RoadLinkFID", "weight"])
# df4 are the non directed links part 2
df4 = dfRLNode.loc[dfRLNode['DirectedLinkOrientation'].isnull()].reindex(columns = ['PlusNodeFID','MinusNodeFID', "RoadLinkFID", "weight"])

df1.rename(columns = {'MinusNodeFID':'start_node', 'PlusNodeFID':'end_node'}, inplace=True)
df2.rename(columns = {'PlusNodeFID':'start_node', 'MinusNodeFID':'end_node'}, inplace=True)
df3.rename(columns = {'PlusNodeFID':'start_node', 'MinusNodeFID':'end_node'}, inplace=True)
df4.rename(columns = {'MinusNodeFID':'start_node', 'PlusNodeFID':'end_node'}, inplace=True)

dfEdgeList = pd.concat([df1,df2,df3,df4], join = "outer")
dfEdgeList = dfEdgeList.reindex(columns=['start_node','end_node', 'RoadLinkFID','weight'])


# Now create the network
graphRL = nx.from_pandas_edgelist(dfEdgeList, source='start_node', target = 'end_node', edge_attr = ['RoadLinkFID','weight'], create_using=nx.DiGraph())
assert graphRL.is_directed()

#############################
#
#
# Save data
#
#
#############################
gdf_itn_link.to_file(output_shapefile)
dfEdgeList.to_csv(output_itn_edge_list)