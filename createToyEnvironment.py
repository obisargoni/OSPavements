# Creating toy model environments for demonstrating path finding model

import sys
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import os
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString, MultiPoint, GeometryCollection
from shapely import ops
import itertools
import re
import itertools

import createPavementNetwork as cpn

projectCRS = {'init' :'epsg:27700'}

def environment_polygon(environment_limits):
    poly_points = list(itertools.product(*environment_limits))
    poly_points.append(poly_points[0])
    order = [0,1,3,2,4]
    z = list(zip(poly_points, order))
    z.sort(key = lambda i:i[-1])
    poly_points, order = zip(*z)
    return Polygon(poly_points)

def create_grid_road_network(environment_limits, block_size, crs = projectCRS):
    '''
    envrionment_limits: tupule of min and max limit for each environment direction.
    block_size: length of road link
    '''

    ndim = len(environment_limits)
    ntiles = [ int( (lim[1]-lim[0]) / block_size) for lim in environment_limits]
    N = np.product(ntiles)

    edges_and_widths = [np.linspace(environment_limits[i][0], environment_limits[i][1],
                                    ntiles[i]+1, retstep=True)
                        for i in range(ndim)]

    edges = [ew[0] for ew in edges_and_widths]
    widths = [ew[1] for ew in edges_and_widths]

    data = {'geometry':[],
            'MNodeFID':[],
            'PNodeFID':[],
            'fid':[]}
    nodes = {   'node_fid':[],
                'geometry':[]}

    # loop over grid coords and create one horizontal and one vertical line for each point
    for i, x in enumerate(edges[0]):
        for j, y in enumerate(edges[1]):
            c1 = [x, y]
            c2 = [x+widths[0], y]
            c3 = [x, y+widths[1]]

            c1_id = "node_{}_{}".format(i,j)
            c2_id = "node_{}_{}".format(i+1,j)
            c3_id = "node_{}_{}".format(i,j+1)


            lh = LineString( [ c1, c2 ])
            lv = LineString( [ c1, c3 ])


            if i<len(edges[0])-1:
                data['geometry'].append(lh)
                data['MNodeFID'].append(c1_id)
                data['PNodeFID'].append(c2_id)
                data['fid'].append("link_{}_{}".format(c1_id.replace("node_",""), c2_id.replace("node_","")))

                for nid, nc in ( (c1_id, c1), (c2_id, c2) ):
                    if nid not in nodes['node_fid']:
                        nodes['node_fid'].append(nid)
                        nodes['geometry'].append(Point(nc))


            if j<len(edges[1])-1:
                data['geometry'].append(lv)
                data['MNodeFID'].append(c1_id)
                data['PNodeFID'].append(c3_id)
                data['fid'].append("link_{}_{}".format(c1_id.replace("node_",""), c3_id.replace("node_","")))

                for nid, nc in ( (c1_id, c1), (c3_id, c3) ):
                    if nid not in nodes['node_fid']:
                        nodes['node_fid'].append(nid)
                        nodes['geometry'].append(Point(nc))


    gdfGrid = gpd.GeoDataFrame(data, geometry='geometry', crs = crs)
    gdfGrid['pedRLID'] = gdfGrid['fid']
    gdfGrid['weight'] = gdfGrid['geometry'].length

    gdfGridNodes = gpd.GeoDataFrame(nodes, geometry='geometry', crs = crs)
    gdfGridNodes.drop_duplicates(subset = 'geometry', inplace=True)
    assert gdfGridNodes['node_fid'].duplicated().any()==False

    return gdfGrid, gdfGridNodes

def create_vehicle_road_network(gdfRoadLink, gdfRoadNode):
    '''
    '''

    assert check_road_link_direction(gdfRoadLink, gdfRoadNode)

    gdf_itn = gdfRoadLink.copy()
    gdf_itn['direction'] = '+'

    # Now create a duplicate set of links facing the other direction
    gdf_itn = gdf_itn.reindex(columns = ['fid', 'MNodeFID', 'PNodeFID', 'direction', 'geometry'])
    gdf_itn2 = gdf_itn.copy()
    gdf_itn2['direction'] = '-'
    
    # Combine the two sets of links
    gdf_itn = pd.concat([gdf_itn, gdf_itn2])

    gdf_itn['pedRLID'] = gdf_itn['fid']
    gdf_itn['fid'] = gdf_itn['fid'] + "_" + gdf_itn['direction'].replace({'+':'plus','-':'minus'})

    # get df of edge list, used in script to produce vehicle ODs and flows
    dfedges1 = gdf_itn.loc[ gdf_itn['direction']=='+'].reindex(columns = ['MNodeFID', 'PNodeFID', 'fid', 'weight']).rename(columns = {'MNodeFID':'start_node', 'PNodeFID':'end_node', 'fid':'RoadLinkFID'})
    dfedges2 = gdf_itn.loc[ gdf_itn['direction']=='-'].reindex(columns = ['MNodeFID', 'PNodeFID', 'fid', 'weight']).rename(columns = {'PNodeFID':'start_node', 'MNodeFID':'end_node', 'fid':'RoadLinkFID'})
    dfedges = pd.concat([dfedges1, dfedges2], join = "outer")

    gdf_itn['weight'] = gdf_itn['geometry'].length

    gdfITNNode = gdfRoadNode.rename(columns = {'node_fid':'fid'})
    gdfITNNode = gdfITNNode['fid'].map(lambda x: x.replace('node', 'itn_node'))

    return gdf_itn, gdfITNNode, dfedges

def check_road_link_direction(gdfRoadLink, gdfRoadNode):
    '''Check that plaus and minus nodes match up with the end and start coordinates of road link linestrings
    '''
    gdf_itn = gdfRoadLink.copy()

    # Select just the node fields needed
    gdf_itn_node = gdfRoadNode.reindex(columns=['node_fid','geometry'])

    # Check the p & m nodes match the link orientation
    assert gdf_itn['geometry'].type.unique().size == 1
    assert gdf_itn['fid'].duplicated().any() == False

    # Merge the nodes with the links
    gdf_itn = gdf_itn.merge(gdf_itn_node, left_on = 'PNodeFID', right_on = 'node_fid', how = 'left', suffixes = ('','_plus_node'), indicator= True)
    assert gdf_itn.loc[ gdf_itn['_merge'] != 'both'].shape[0] == 0
    gdf_itn.rename(columns={'_merge':'_merge_plus_node'}, inplace=True)

    gdf_itn = gdf_itn.merge(gdf_itn_node, left_on = 'MNodeFID', right_on = 'node_fid', how = 'left', suffixes = ('', '_minus_node'), indicator=True)
    assert gdf_itn.loc[ gdf_itn['_merge'] != 'both'].shape[0] == 0
    gdf_itn.rename(columns={'_merge':'_merge_minus_node'}, inplace=True)

    gdf_itn['line_first_coord'] = gdf_itn['geometry'].map(lambda x: x.coords[0])
    gdf_itn['line_last_coord'] = gdf_itn['geometry'].map(lambda x: x.coords[-1])

    # Check that the -,+ nodes match the first / last line string coords
    gdf_itn['first_coords_match_minus_node'] = gdf_itn['line_first_coord'] == gdf_itn['geometry_minus_node'].map(lambda x: x.coords[0])
    gdf_itn['last_coords_match_plus_node'] = gdf_itn['line_last_coord'] == gdf_itn['geometry_plus_node'].map(lambda x: x.coords[0])

    assert gdf_itn['first_coords_match_minus_node'].all() == True
    gdf_itn['last_coords_match_plus_node'].all() == True

    gdf_itn = None
    return True

def pavement_network_nodes(road_graph, gdfRoadNode, gdfRoadLink, angle_range = 90, disp = 5, crs=projectCRS):
    # Node metadata
    dfPedNodes = cpn.multiple_road_node_pedestrian_nodes_metadata(road_graph, gdfRoadNode)
    dfPedNodes['geometry'] = cpn.assign_boundary_coordinates_to_ped_nodes(dfPedNodes, gdfRoadLink, None, method = 'default', required_range = angle_range, default_disp = disp, crs = crs)

    n_missing_nodes = dfPedNodes.loc[ dfPedNodes['geometry'].isnull()].shape[0]
    print("Number of missing nodes: {}".format(n_missing_nodes))

    gdfPedNodes = gpd.GeoDataFrame(dfPedNodes, geometry = 'geometry', crs = crs)
    gdfPedNodes['fid'] = ["pave_node_{}".format(i) for i in range(gdfPedNodes.shape[0])]

    return gdfPedNodes

def pavement_network_links(gdfPaveNodes, gdfRoadLink, road_graph, crs = projectCRS):
    dfPaveLink = cpn.connect_ped_nodes(gdfPaveNodes, gdfRoadLink, road_graph)
    gdfPaveLink = gpd.GeoDataFrame(dfPaveLink, geometry = 'geometry', crs = crs)

    # Drop duplicated edges - don't expect any multi edges so drop duplicated fids since this implies duplicated edge between nodes
    gdfPaveLink = gdfPaveLink.drop_duplicates(subset = ['fid'])

    # Repair cases where a pavement links gets classified as a road crosisng link bc it intersects a road link
    gdfPaveLink = cpn.repair_non_crossing_links(gdfRoadLink['fid'].unique(), gdfPaveNodes, gdfPaveLink)

    return gdfPaveLink

def carriageway_geometries(gdfPaveNodes, gdfRoadNodes, gdfRoadLink, crs = projectCRS):
    '''Create carriageway polygons by grouping together pavement nodes belonging toa road link or road node and 
    calculating their convex hull
    '''

    rls = gdfRoadLink['fid'].values

    data = {'roadLinkID':[],
            'ncoords':[],
            'geometry':[]}

    for rl in rls:
        data['roadLinkID'].append(rl)
        df =  gdfPaveNodes.loc[ (gdfPaveNodes['v1rlID']==rl) | (gdfPaveNodes['v2rlID']==rl) ]

        data['ncoords'].append(df.shape[0])

        points = list(df['geometry'].values)

        # add in road node coordinates so that polygons fill intersection spaces
        ps = gdfRoadNodes.loc[ gdfRoadNodes['node_fid'].isin(df['juncNodeID']), 'geometry'].values
        points+=list(ps)


        mp = MultiPoint(points)
        data['geometry'].append(mp.convex_hull)

    gdfRoadPolys = gpd.GeoDataFrame(data, geometry='geometry', crs = crs)
    gdfRoadPolys['polyID']=['veh_poly_{}'.format(i) for i in gdfRoadPolys.index]
    gdfRoadPolys['priority']='vehicle'

    return gdfRoadPolys

def pavement_geometries(gdfRoadLink, gdfPaveLink, gdfPaveNode, pavement_width, angle_range = 90, disp = 5, crs = projectCRS):
    '''
    env_poly = environment_polygon(environment_limits)
    gdfEnv = gpd.GeoDataFrame({'geometry':[env_poly]}, geometry='geometry', crs = crs)

    # Intersection road polys and environment polygon to get pavement polygons
    gdf = gpd.overlay(gdfEnv, gdfRoadPolys, how='difference')
    polys = list(gdf['geometry'].values[0])
    gdfPavePolys = gpd.GeoDataFrame({'block_geom':polys}, crs = crs)

    # buffer these block geometres to get the builing line
    gdfPavePolys['building_geom'] = gdfPavePolys['block_geom'].map(lambda g: g.buffer(-pavement_width))

    gdfPavePolys['pavement_geometry'] = gdfPavePolys.apply(lambda row: row['block_geom'].difference(row['building_geom']),axis=1)

    gdfPavePolys = gdfPavePolys.reindex(columns = ['geometry'])
    gdfPavePolys.set_geometry('geometry', inplace=True)
    gdfPavePolys['fid']=['veh_poly_{}'.format(i) for i in gdfPavePolys.index]
    '''

    # create alternative ped node
    gdfPaveNode['alt_geom'] = cpn.assign_boundary_coordinates_to_ped_nodes(gdfPaveNode, gdfRoadLink, None, method = 'default', required_range = angle_range, default_disp = disp + pavement_width, crs = crs)

    # loop through non-crossing pavement links, get all coords corresponding to this link
    pavement_data = {   'roadLinkID':[],
                        'geometry':[]}
    for ir, row in gdfPaveLink.loc[ gdfPaveLink['linkType']=='pavement'].iterrows():
        nodes  = [row['MNodeFID'], row['PNodeFID']]
        points = gdfPaveNode.loc[gdfPaveNode['fid'].isin(nodes), ['geometry', 'alt_geom']].values.reshape(1,-1)[0]
        assert len(points)==4

        mp = MultiPoint(points)
        pavement_geom = mp.convex_hull

        # get road link id from junction nodes
        junction_nodes = gdfPaveNode.loc[ gdfPaveNode['fid'].isin(nodes), 'juncNodeID'].values
        dfRL = gdfRoadLink.loc[ (gdfRoadLink['MNodeFID'].isin(junction_nodes)) & (gdfRoadLink['PNodeFID'].isin(junction_nodes))]
        assert dfRL.shape[0]==1
        rlid = dfRL['fid'].values[0]

        pavement_data['roadLinkID'].append(rlid)
        pavement_data['geometry'].append(pavement_geom)

    gdfPavePolys = gpd.GeoDataFrame(pavement_data, geometry = 'geometry', crs = crs)
    gdfPavePolys['polyID']=['ped_poly_{}'.format(i) for i in gdfPavePolys.index]
    gdfPavePolys['priority']='pedestrian'

    gdfPaveNode.drop(['alt_geom'], axis=1, inplace=True)

    return gdfPavePolys


################################
#
#
# Set script parameters
#
#
################################
with open(os.path.join("config.json")) as f:
    config = json.load(f)


gis_data_dir = config['gis_data_dir']
output_directory = os.path.join(gis_data_dir, "processed_gis_data")

if os.path.isdir(output_directory) == False:
    os.mkdir(output_directory)

output_road_link_file = os.path.join(output_directory, config["openroads_link_processed_file"])
output_road_node_file = os.path.join(output_directory, config["openroads_node_processed_file"])

output_itn_link_file = os.path.join(output_directory, config['mastermap_itn_processed_direction_file'])
output_itn_node_file = os.path.join(output_directory, config['mastermap_node_processed_file'])

output_pave_link_file = os.path.join(output_directory, config["pavement_links_file"])
output_pave_node_file = os.path.join(output_directory, config["pavement_nodes_file"])

output_road_file = os.path.join(output_directory, config["topo_vehicle_processed_file"])
output_pave_file = os.path.join(output_directory, config["topo_pedestrian_processed_file"])

output_edgelist_file = os.path.join(gis_data_dir, "itn_route_info", "itn_edge_list.csv")

output_boundary_file = os.path.join(output_directory, config['boundary_file'])

environment_limits = ( (0,1000), (0,1000) )
block_size = 50
carriageway_width = 5
pavement_width = 3
angle_range = 90


##################################
#
#
# Produce toy environment
#
#
##################################
env_poly = environment_polygon(environment_limits)
gdfRoadLink, gdfRoadNode = create_grid_road_network(environment_limits, block_size)

# Create version of the road network vehicles travel on
gdfITNLink, gdfITNNode, dfedges = create_vehicle_road_network(gdfRoadLink, gdfRoadNode)

# Load the Open Roads road network as a nx graph
road_graph = nx.MultiGraph()
gdfRoadLink['fid_dict'] = gdfRoadLink.apply(lambda x: {"fid":x['fid'],'geometry':x['geometry']}, axis=1)
edges = gdfRoadLink.loc[:,['MNodeFID','PNodeFID', 'fid_dict']].to_records(index=False)
road_graph.add_edges_from(edges)
gdfRoadLink.drop('fid_dict', axis=1, inplace=True)

gdfPaveNode = pavement_network_nodes(road_graph, gdfRoadNode, gdfRoadLink, angle_range = angle_range, disp = carriageway_width, crs=projectCRS)
gdfPaveLink = pavement_network_links(gdfPaveNode, gdfRoadLink, road_graph, crs = projectCRS)

gdfRoadPolys = carriageway_geometries(gdfPaveNode, gdfRoadNode, gdfRoadLink, crs = projectCRS)
gdfPavePolys = pavement_geometries(gdfRoadLink, gdfPaveLink, gdfPaveNode, pavement_width, angle_range = angle_range, disp = carriageway_width, crs = projectCRS)

gdfBoundary = gpd.GeoDataFrame({'geometry':[ LineString(gdfPavePolys.geometry.unary_union.convex_hull.exterior) ], 'priority':['pedestrian_obstruction']}, geometry='geometry', crs = projectCRS)

#
# Save the data
#
gdfRoadLink.to_file(output_road_link_file)
gdfRoadNode.to_file(output_road_node_file)

gdfITNLink.to_file(output_itn_link_file)
gdfITNNode.to_file(output_itn_node_file)
dfedges.to_csv(output_edgelist_file, index=False)

gdfPaveLink.to_file(output_pave_link_file)
gdfPaveNode.to_file(output_pave_node_file)

gdfRoadPolys.to_file(output_road_file)
gdfPavePolys.to_file(output_pave_file)

gdfBoundary.to_file(output_boundary_file)