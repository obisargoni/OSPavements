# Script for producing the pedestrian network using a road network and walkable bounday
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

abs_path = os.path.abspath(__file__)
this_dir = os.path.dirname(abs_path)
with open(os.path.join(this_dir, "config.json")) as f:
    config = json.load(f)


gis_data_dir = config['gis_data_dir']
output_directory = os.path.join(gis_data_dir, "processed_gis_data")

'''
global gdfORLink
global gdfORNode
global gdfTopoPed
global gdfTopoVeh
global gdfBoundary
global G
'''

gdfORLink = None
gdfORNode = None
gdfTopoPed = None
gdfTopoVeh = None
gdfBoundary = None
G = None
output_ped_nodes_file = None
output_ped_links_file = None

# Parameters that control area searched for suitable pavement node locations
default_angle_range = 90
backup_angle_range = None
default_ray_length = 20

#################################
#
#
# Functions
#
#
#################################
def unit_vector(v):
    magV = np.linalg.norm(v)
    return v / magV

def angle_between_north_and_unit_vector(u):
    n = (0,1)

    signX = np.sign(u[0])
    signY = np.sign(u[1])

    dp = np.dot(n, u)
    a = np.arccos(dp)

    # Dot product gives angle between vectors. We need angle clockwise from north
    if (signX == 1):
        return a
    elif (signX == -1):
        return 2*np.pi - a
    else:
        return a

def angle_between_north_and_vector(v):
    unit_v = unit_vector(v)
    return angle_between_north_and_unit_vector(unit_v)
            
def ang(lineA, lineB):
    # Get nicer vector form
    lACoords = lineA.coords
    lBCoords = lineB.coords
    vA = [(lACoords[0][0]-lACoords[1][0]), (lACoords[0][1]-lACoords[1][1])]
    vB = [(lBCoords[0][0]-lBCoords[1][0]), (lBCoords[0][1]-lBCoords[1][1])]
    # Get dot prod
    dot_prod = np.dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get cosine value
    cos_ = round(dot_prod/magA/magB, 4)
    # Get angle in radians and then convert to degrees
    angle = math.arccos(cos_)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360

    if ang_deg-180>=0:
        # As in if statement
        return 360 - ang_deg
    else: 
        return ang_deg

def sample_angles(a1, a2, sample_res):
    if a1 < a2:
        sampled_angles = np.arange(a1+sample_res, a2,sample_res)
    else:
        sampled_angles = []
        ang = a1+sample_res
        while ang < 2*np.pi:
            sampled_angles.append(ang)
            ang+=sample_res
        
        ang-=2*np.pi
        
        while ang < a2:
            sampled_angles.append(ang)
            ang+=sample_res

        sampled_angles = np.array(sampled_angles)

    return sampled_angles

def angle_range(a1, a2):
    if a1<a2:
        r = a2-a1
    else:
        r = (2*np.pi-a1) + a2
    return r

def angle_range_midpoint(a1, a2):
    r = angle_range(a1, a2)
    middle = a1 + r/2
    return middle

def in_angle_range(ang, a1, a2):
    if a1 < a2:
        return (ang>a1) & (ang<a2)
    else:
        b1 = (ang>a1) & (ang<2*np.pi)
        b2 = (ang>=0) & (ang<a2)
        return b1 | b2

def filter_angle_range(a1, a2, required_range):
    if required_range is None:
        return a1, a2

    required_range = (2*np.pi) * (required_range/360.0) # convert to radians
    r = angle_range(a1, a2)

    if r < required_range:
        return a1, a2
    
    middle = angle_range_midpoint(a1, a2)
    a1 = middle - required_range / 2
    a2 = middle + required_range / 2

    if a1 > 2*np.pi:
        a1 = a1-2*np.pi
    if a2 > 2*np.pi:
        a2 = a2-2*np.pi

    return a1, a2

def rays_between_angles(a1, a2, p1, sample_res = 10, ray_length = 50):
    sample_res = (2*np.pi) * (sample_res/360.0) # convert to radians
    sampled_angles = sample_angles(a1, a2, sample_res)
    for sa in sampled_angles:
        p2 = Point([p1.x + ray_length*np.sin(sa), p1.y + ray_length*np.cos(sa)])
        l = LineString([p1,p2])
        yield l

def coord_match(c1, c2):
    x_diff = abs(c1[0]-c2[0])
    y_diff = abs(c1[1]-c2[1])

    if (x_diff<0.000001) & (y_diff<0.000001):
        return True
    else:
        return False

def linestring_bearing(l, start_point):
    if start_point.coords[0] == l.coords[0]:
        end_coord = np.array(l.coords[-1])
    elif start_point.coords[0] == l.coords[-1]:
        end_coord = np.array(l.coords[0])
    else:
        return None

    start_coord = np.array(start_point.coords[0])

    v = end_coord - start_coord

    return angle_between_north_and_vector(v)

def road_node_pedestrian_nodes_metadata(graph, road_node_geom, road_node_id):

    # Method for getting ped nodes for a single junctions
    edges = list(graph.edges(road_node_id, data=True))

    # Find neighbouring edges based on the edge geometry bearing
    edge_bearings = [linestring_bearing(e[-1]['geometry'], road_node_geom) for e in edges]

    edges_w_bearing = list(zip(edges, edge_bearings))
    edges_w_bearing.sort(key = lambda e: e[1])
    edge_pairs = zip(edges_w_bearing, edges_w_bearing[1:] + edges_w_bearing[0:1])
    
    # Iterate over pairs of road link IDs in order to find the pedestrian nodes that lie between these two road links
    dfPedNodes = pd.DataFrame()
    for (e1, bearing1), (e2, bearing2) in edge_pairs:

        # Initialise data to go into the geodataframe
        row = {"juncNodeID":road_node_id, "juncNodeX":road_node_geom.x, "juncNodeY": road_node_geom.y, "v1rlID":e1[-1]['fid'], "a1":bearing1, "v2rlID":e2[-1]['fid'], "a2":bearing2}

        dfPedNodes = dfPedNodes.append(row, ignore_index = True)
        
    # Filter dataframe to exclude empty geometries
    return dfPedNodes

def multiple_road_node_pedestrian_nodes_metadata(graph, gdfRoadNodes):
    dfPedNodes = pd.DataFrame()

    for road_node_id, road_node_geom in gdfRoadNodes.loc[:, ['node_fid', 'geometry']].values:
        #try:
        df = road_node_pedestrian_nodes_metadata(graph, road_node_geom, road_node_id)
        dfPedNodes = pd.concat([dfPedNodes, df])
        #except Exception as err:
        #    print(road_node_id, road_node_geom)
        #    print(err)

    dfPedNodes.index = np.arange(dfPedNodes.shape[0])
    return dfPedNodes

def nearest_ray_intersection_point_between_angles(a1, a2, start_point, seriesGeoms, seriesRoadLinks, required_range = None, ray_length = 20):

        si_geoms = seriesGeoms.sindex
        si_road_link = seriesRoadLinks.sindex

        min_dist = sys.maxsize
        nearest_point = None

        a1, a2 = filter_angle_range(a1, a2, required_range)

        for l in rays_between_angles(a1, a2, start_point, ray_length = ray_length):
            close = si_geoms.intersection(l.bounds)
            for geom_id in close:
                intersection = seriesGeoms[geom_id].intersection(l)

                if isinstance(intersection, (MultiPoint, MultiLineString, MultiPolygon, GeometryCollection)):
                    coords = []
                    for geom in intersection:
                        coords+=geom.coords
                else:
                    coords = intersection.coords


                p, d = nearest_point_in_coord_sequence(coords, min_dist, start_point, a1, a2, seriesRoadLinks, si_road_link)
                
                if p is not None:
                    min_dist = d
                    nearest_point = p
        
        return nearest_point

def nearest_geometry_point_between_angles(a1, a2, start_point, seriesGeoms, seriesRoadLinks, required_range = None, ray_length = 20):
        
        si_geoms = seriesGeoms.sindex
        si_road_link = seriesRoadLinks.sindex

        min_dist = sys.maxsize
        nearest_point = None

        a1, a2 = filter_angle_range(a1, a2, required_range)

        processed_boundary_geom_ids = []
        for l in rays_between_angles(a1, a2, start_point, ray_length = ray_length):
            for geom_id in si_geoms.intersection(l.bounds):
                if (seriesGeoms[geom_id].intersects(l)) & (geom_id not in processed_boundary_geom_ids):
                    
                    # Now find nearest boundary coordinate from intersecting boundaries
                    geom = seriesGeoms.loc[row_id]

                    p, d = nearest_point_in_coord_sequence(geom.exterior.coords, min_dist, start_point, a1, a2, seriesRoadLinks, si_road_link)

                    if p is not None:
                        min_dist = d
                        nearest_point = p
        
        return nearest_point

def point_located_between_angles(a1, a2, start_point, seriesRoadLinks, d = 5, d_direction = 'mid_angle'):
    si_road_link = seriesRoadLinks.sindex

    mid_angle = angle_range_midpoint(a1, a2)

    if d_direction == 'perp':
        # d is the perpendicular distance between the point ad the road link
        a = abs(a1-a2)/2
        d = d/np.sin(a)

    p = Point([start_point.x + d*np.sin(mid_angle), start_point.y + d*np.cos(mid_angle)])
    l = LineString([start_point,p])

    # Check this doesn't intersect a road link
    intersects_road_links = False
    for i in si_road_link.intersection(l.bounds):
        if seriesRoadLinks[i].intersects(l):
            intersects_road_links = True
            break

    while intersects_road_links:
        d-=0.3
        p = Point([start_point.x + d*np.sin(mid_angle), start_point.y + d*np.cos(mid_angle)])
        l = LineString([start_point,p])
    
        for i in si_road_link.intersection(l.bounds):
            if seriesRoadLinks[i].intersects(l):
                intersects_road_links = True
                break
        intersects_road_links = False

        if d<0:
            p = None
            break
    return p

def nearest_point_in_coord_sequence(coords, min_dist, start_point, a1, a2, seriesRoadLinks, si_road_link):
    chosen_point = None
    for c in coords:
        p = Point(c)
        d = start_point.distance(p)

        l = LineString([start_point, p])

        # ensure that point lies in direction between input and output angles
        a = linestring_bearing(l, start_point)

        if (d < min_dist) & (in_angle_range(a, a1, a2)):
            
            # ensure line doesn't intersect any road links
            intersects_road_links = False
            for i in si_road_link.intersection(l.bounds):
                if seriesRoadLinks[i].intersects(l):
                    intersects_road_links = True
                    break

            if intersects_road_links == False:
                min_dist = d
                chosen_point = p

    return chosen_point, min_dist

def assign_boundary_coordinates_to_ped_nodes(df_ped_nodes, gdf_road_links, series_coord_geoms, method = 'ray_intersection', required_range = None, ray_length = 20, default_disp = 5, d_direction = 'mid_angle', adjust_for_small_links = False, crs = projectCRS):
    """Identify coordinates for ped nodes based on the bounday.
    """

    # Initialise output
    index = []
    geoms = []

    # Loop through nodes, get corresponding road links and the boundary between them
    for ix, row in df_ped_nodes.iterrows():

        rlID1 = row['v1rlID']
        rlID2 = row['v2rlID']

        # exclude links connected to this road node to check that line to ped node doesn't intersect a road link
        series_road_links = gdf_road_links.loc[ (gdf_road_links['MNodeFID']!=row['juncNodeID']) & (gdf_road_links['PNodeFID']!=row['juncNodeID']), 'geometry'].copy()
        series_road_links.index = np.arange(series_road_links.shape[0])

        a1 = row['a1']
        a2 = row['a2']

        road_node = Point([row['juncNodeX'], row['juncNodeY']])

        if method == 'ray_intersection':
            ped_node_geom = nearest_ray_intersection_point_between_angles(a1, a2, road_node, series_coord_geoms, series_road_links, required_range = required_range, ray_length = ray_length)
        elif method == 'default':
            lengths = gdf_road_links.loc[ gdf_road_links['fid'].isin([rlID1,rlID2]), 'geometry'].length.values
            if (lengths[0]<30) & (lengths[1]<30) & adjust_for_small_links:
                d = default_disp/2
            else:
                d = default_disp
            ped_node_geom = point_located_between_angles(a1, a2, road_node, series_road_links, d = d, d_direction = d_direction)
        else:
            ped_node_geom = None

        index.append(ix)
        geoms.append(ped_node_geom)

    return pd.Series(geoms, index = index)

def choose_ped_node(row, pave_node_col, boundary_node_col, road_node_x_col, road_node_y_col):

    # Initialise output  
    pave_node = row[pave_node_col]
    boundary_node = row[boundary_node_col]
    road_node = Point([row[road_node_x_col], row[road_node_y_col]])

    if (pave_node is not None) & (boundary_node is None):
        return pave_node
    elif (pave_node is None) & (boundary_node is not None):
        # Displace this node slightly towards the road nodeso it does not touch the barrier
        v = np.array(road_node.coords[0]) - np.array(boundary_node.coords[0])
        a = angle_between_north_and_vector(v)
        disp = 0.5
        displaced_coord = [boundary_node.x + disp*np.sin(a), boundary_node.y + disp*np.cos(a)]
        return Point(displaced_coord)

    elif (pave_node is not None) & (boundary_node is not None):
        # return the closest to the road node
        chosen_node = None
        pave_node_dist = road_node.distance(pave_node)
        bound_node_dist = road_node.distance(boundary_node)

        if (pave_node_dist<bound_node_dist):
            chosen_node = pave_node
        else:
            # displace the boundary node
            v = np.array(road_node.coords[0]) - np.array(boundary_node.coords[0])
            a = angle_between_north_and_vector(v)
            disp = 0.5
            displaced_coord = [boundary_node.x + disp*np.sin(a), boundary_node.y + disp*np.cos(a)]
            chosen_node = Point(displaced_coord)

        return chosen_node

    else:
        return None

def neighbouring_geometries_graph(gdf_polys, id_col):
    # df to record neighbours
    df_neighbours = neighbouring_geometries_df(gdf_polys, id_col)

    g = nx.from_pandas_edgelist(df_neighbours, source=id_col, target = 'neighbourfid')

    return g

def neighbouring_geometries_df(gdf_polys, id_col):
    # df to record neighbours
    df_neighbours = pd.DataFrame()

    for index, row in gdf_polys.iterrows():
        # Touches identifies polygons with at least one point in common but interiors don't intersect. So will work as long as none of my topographic polygons intersect
        neighborFIDs = gdf_polys[gdf_polys.geometry.touches(row['geometry'])][id_col].tolist()

        # Polygons without neighbours need to be attached to themselves so they can be identified as a cluster of one
        if len(neighborFIDs) == 0:
            neighborFIDs.append(row[id_col])

        df = pd.DataFrame({'neighbourfid':neighborFIDs})
        df[id_col] = row[id_col]
        df_neighbours = pd.concat([df_neighbours, df])

    g = nx.from_pandas_edgelist(df_neighbours, source=id_col, target = 'neighbourfid')

    return df_neighbours


def connect_ped_nodes(gdfPN, gdfRoadLink, road_graph):
    """Method for connecting ped nodes.

    Connects all nodes on a road link.
    """

    ped_node_edges = []

    for rl_id in gdfRoadLink['fid'].values:

        # Get start and end node for this road link
        node_records = gdfRoadLink.loc[ gdfRoadLink['fid'] == rl_id, ['MNodeFID','PNodeFID']].to_dict(orient='records')
        assert len(node_records) == 1
        u = node_records[0]['MNodeFID']
        v = node_records[0]['PNodeFID']

        # Get small section of road network connected to this road link
        neighbour_links = [edge_data['fid'] for u, v, edge_data in road_graph.edges(nbunch = [u], data=True)]
        neighbour_links += [edge_data['fid'] for u, v, edge_data in road_graph.edges(nbunch = [v], data=True)]

        # Get the geometries for these road links
        rl_g = gdfRoadLink.loc[ gdfRoadLink['fid']==rl_id, 'geometry'].values[0]
        gdfLinkSub = gdfRoadLink.loc[ gdfRoadLink['fid'].isin(neighbour_links)]

        # Get pairs of ped nodes
        gdfPedNodesSub = gdfPN.loc[(gdfPN['v1rlID']==rl_id) | (gdfPN['v2rlID']==rl_id)]

        # Should be 4 ped nodes for each road link
        if gdfPedNodesSub.shape[0]!=4:
            continue

        ped_node_pairs = itertools.combinations(gdfPedNodesSub['fid'].values, 2)

        for ped_u, ped_v in ped_node_pairs:
            # Create linestring to join ped nodes
            g_u = gdfPedNodesSub.loc[ gdfPedNodesSub['fid'] == ped_u, 'geometry'].values[0]
            g_v = gdfPedNodesSub.loc[ gdfPedNodesSub['fid'] == ped_v, 'geometry'].values[0]

            # Get or nodes associated to each of these pavement nodes to check whether they are located around same junction
            j_u = gdfPedNodesSub.loc[ gdfPedNodesSub['fid'] == ped_u, 'juncNodeID'].values[0]
            j_v = gdfPedNodesSub.loc[ gdfPedNodesSub['fid'] == ped_v, 'juncNodeID'].values[0]

            l = LineString([g_u, g_v])
            edge_id = "pave_link_{}_{}".format(ped_u.replace("pave_node_",""), ped_v.replace("pave_node_",""))

            ped_edge = {'MNodeFID':ped_u, 'PNodeFID':ped_v, 'pedRLID':None, 'linkType':'pavement', 'pedRoadID':None, 'fid':edge_id, 'geometry':l}

            # Check whether links crosses road or not
            intersect_check = gdfLinkSub['geometry'].map(lambda g: g.intersects(l))
            if j_u==j_v:
                # Nodes belong to same junction node, must require crossing to move between them
                ped_edge['pedRLID'] = " ".join(gdfLinkSub.loc[intersect_check, 'fid'])
                ped_edge['linkType'] = 'direct_cross'

            else:
                if intersect_check.any():
                    ped_edge['pedRLID'] = " ".join(gdfLinkSub.loc[intersect_check, 'fid'])

                    # Since in this case j_u!=j_v link type is diagonal
                    ped_edge['linkType'] = 'diag_cross'

            # Could also check which ped polys edge intersects but this isn't necessary

            ped_node_edges.append(ped_edge)

        dfPedEdges = pd.DataFrame(ped_node_edges)

    return dfPedEdges

def validate_numbers_of_nodes_and_links(gdfRoadLinks, gdfPN, gdfPL):
    """Check for road links that don't have 4 ped nodes associated.
    Check for road links that don't have 2 non crossing links on either side of the road and 
    4 crossing links.
    """

    problem_links = {"missing_nodes":[], "missing_links":[]}

    for rl_id in gdfRoadLinks['fid'].values:

        # Get pairs of ped nodes
        gdfPedNodesSub = gdfPN.loc[(gdfPN['v1rlID']==rl_id) | (gdfPN['v2rlID']==rl_id)]

        # Should be 4 ped nodes for each road link
        if gdfPedNodesSub.shape[0]!=4:
            print("Road link does not have 4 ped nodes")
            print(rl_id)
            print(gdfPedNodesSub)
            print("\n")
            problem_links["missing_nodes"].append(rl_id)
            continue

        # Get edges between these nodes
        edge_ids = []
        ped_node_pairs = itertools.combinations(gdfPedNodesSub['fid'].values, 2)
        for ped_u, ped_v in ped_node_pairs:
            edge_ida = "pave_link_{}_{}".format(ped_u.replace("pave_node_",""), ped_v.replace("pave_node_",""))
            edge_idb = "pave_link_{}_{}".format(ped_v.replace("pave_node_",""), ped_u.replace("pave_node_",""))

            edge_ids.append(edge_ida)
            edge_ids.append(edge_idb)


        gdfPedEdgesSub = gdfPL.loc[gdfPL['fid'].isin(edge_ids)]
        gdfCross = gdfPedEdgesSub.loc[~gdfPedEdgesSub['pedRLID'].isnull()]
        gdfNoCross = gdfPedEdgesSub.loc[gdfPedEdgesSub['pedRLID'].isnull()]

        if gdfNoCross.shape[0]!=2:
            print("Road link does not have 2 non-crossing edges")
            print(rl_id)
            print("\n")
            problem_links["missing_links"].append(rl_id)
            continue

        nodesa = set(gdfNoCross.loc[:, ["MNodeFID", "PNodeFID"]].values[0])
        nodesb = set(gdfNoCross.loc[:, ["MNodeFID", "PNodeFID"]].values[1])
        if len(nodesa.intersection(nodesb)) > 0:
            print("Non-corssing links overlap")
            print(rl_id)
            print("\n")
            problem_links["missing_links"].append(rl_id)

        if gdfCross.shape[0] != 4:
            print("Road link does not have 4 crossing ped links")
            print(rl_id)
            print("\n")
            problem_links["missing_links"].append(rl_id)

    return problem_links

def repair_non_crossing_links(road_link_ids, gdfPN, gdfPL):

    # Loop through road link IDs, and check pavement edges for each road link
    for rl_id in road_link_ids:

        # Get pairs of ped nodes
        gdfPedNodesSub = gdfPN.loc[(gdfPN['v1rlID']==rl_id) | (gdfPN['v2rlID']==rl_id)]

        # Should be 4 ped nodes for each road link
        if gdfPedNodesSub.shape[0]!=4:
            continue

        # Get edges between these nodes
        edge_ids = []
        ped_node_pairs = itertools.combinations(gdfPedNodesSub['fid'].values, 2)
        for ped_u, ped_v in ped_node_pairs:
            edge_ida = "pave_link_{}_{}".format(ped_u.replace("pave_node_",""), ped_v.replace("pave_node_",""))
            edge_idb = "pave_link_{}_{}".format(ped_v.replace("pave_node_",""), ped_u.replace("pave_node_",""))

            edge_ids.append(edge_ida)
            edge_ids.append(edge_idb)


        gdfPedEdgesSub = gdfPL.loc[gdfPL['fid'].isin(edge_ids)]

        no_cross_edge_nodes = gdfPedEdgesSub.loc[ gdfPedEdgesSub['pedRLID'].isnull(), ['MNodeFID','PNodeFID']].values

        # If only one no cross link, find the other no cross link by finding which edge doesn't share coordinates
        if len(no_cross_edge_nodes)==1:
            no_cross_edge_nodes = set(no_cross_edge_nodes[0])
            other_edge_nodes = gdfPedEdgesSub.loc[ ~gdfPedEdgesSub['pedRLID'].isnull(), ['MNodeFID','PNodeFID']].values
            
            for nodes in other_edge_nodes:
                if len(no_cross_edge_nodes.intersection(set(nodes))) == 0:
                    # Get the id of this row and update the pedRLID field to be blank
                    ix = gdfPL.loc[ (gdfPL['MNodeFID']==nodes[0]) & (gdfPL['PNodeFID']==nodes[1]) ].index[0]
                    gdfPL.loc[ix, 'pedRLID']=None
                    gdfPL.loc[ix, 'linkType']='pavement'
                    print("Corrected no crossing edge {}".format(gdfPL.loc[ix, 'fid']))

        # Check numbers of crossing and non crossing links are as expected
        no_cross_edge_nodes = gdfPedEdgesSub.loc[ gdfPedEdgesSub['pedRLID'].isnull(), ['MNodeFID','PNodeFID']].values
        cross_edge_nodes = gdfPedEdgesSub.loc[ ~gdfPedEdgesSub['pedRLID'].isnull(), ['MNodeFID','PNodeFID']].values
        if ( (len(no_cross_edge_nodes)!=2) | (len(cross_edge_nodes)!=4)):
            print("WARNING: road Link {} does not have expected pavemetn links".format(rl_id))



    return gdfPL

def load_data():
    global gdfORLink
    global gdfORNode
    global gdfTopoPed
    global gdfTopoVeh
    global gdfBoundary
    global G
    global output_ped_nodes_file
    global output_ped_links_file

    gdfORLink = gpd.read_file(os.path.join(output_directory, config["openroads_link_processed_file"]))
    gdfORNode = gpd.read_file(os.path.join(output_directory, config["openroads_node_processed_file"]))
    gdfORLink.crs = projectCRS
    gdfORNode.crs = projectCRS

    gdfTopoVeh = gpd.read_file(os.path.join(output_directory, config["topo_vehicle_processed_file"]))
    gdfTopoPed = gpd.read_file(os.path.join(output_directory, config["topo_pedestrian_processed_file"]))
    gdfTopoVeh.crs = projectCRS
    gdfTopoPed.crs = projectCRS

    # Load boundary data - used to identify pavement nodes
    gdfBoundary = gpd.read_file(os.path.join(output_directory, config["boundary_file"]))
    gdfBoundary.crs = projectCRS

    output_ped_nodes_file = os.path.join(output_directory, config["pavement_nodes_file"])
    output_ped_links_file = os.path.join(output_directory, config["pavement_links_file"])

    # Load the Open Roads road network as a nx graph
    G = nx.MultiGraph()
    gdfORLink['fid_dict'] = gdfORLink.apply(lambda x: {"fid":x['fid'],'geometry':x['geometry']}, axis=1)
    edges = gdfORLink.loc[:,['MNodeFID','PNodeFID', 'fid_dict']].to_records(index=False)
    G.add_edges_from(edges)
    return True

def run():
    ########################################
    #
    #
    # Filter out traffic islands
    #
    # don't want pavement nodes located on traffic islands so need to filter these out.
    #
    #
    ########################################

    load_data()

    global gdfORLink
    global gdfORNode
    global gdfTopoPed
    global gdfTopoVeh
    global gdfBoundary
    global G
    global output_ped_nodes_file
    global output_ped_links_file

    # Find which ped polygons touch a barrier
    gdfTopoPedBoundary = gpd.sjoin(gdfTopoPed, gdfBoundary, op = 'intersects')
    gdfTopoPed['intersectsBounds'] = gdfTopoPed['polyID'].isin(gdfTopoPedBoundary['polyID'])

    # Find connected components of ped polygons
    G_islands = neighbouring_geometries_graph(gdfTopoPed, id_col = 'polyID')
    island_polys_to_remove = []
    island_polys_to_keep = []

    # Find connected components
    ccs = list(nx.connected_components(G_islands))
    sizes = np.array([len(cc) for cc in ccs])


    # Loop through connected components and identify islands
    for cc in ccs:

        # Calculate fraction of cc area that doesn't touch a boundary. If fraction is ~ 0 this cc is almost certainly not an island
        total_area = gdfTopoPed.loc[ (gdfTopoPed['polyID'].isin(cc)), 'geometry'].area.sum()
        area_not_touching_boundary = gdfTopoPed.loc[ (gdfTopoPed['polyID'].isin(cc)) & (gdfTopoPed['intersectsBounds']==False), 'geometry'].area.sum()
        if (total_area>500) | ( (area_not_touching_boundary / total_area) < 0.03):
            continue

        # Else, identify whether its and island
        if len(cc) == 1:
            for poly_id in cc:
                island_polys_to_remove.append(poly_id)
        else:
            # get road links associated with this component
            rls = gdfTopoPed.loc[gdfTopoPed['polyID'].isin(cc), 'roadLinkID'].to_list()

            # if these road links form a loop, not an island since surrounded by road links
            edges = gdfORLink.loc[ gdfORLink['fid'].isin(rls), ['MNodeFID','PNodeFID']].to_records(index=False)
            us, vs = list(zip(*edges))
            G_temp = nx.subgraph(G, us+vs)

            G_temp_single = nx.Graph(G_temp)
            assert len(G_temp.edges) == len(G_temp_single.edges)
            
            # If the subgraph corresponding to the road nodes around the island contains a cycle, then class island as surrounded by roads and therefore not to be excluded
            cycles = nx.cycle_basis(G_temp_single)
            if len(cycles)==0:
                island_polys_to_remove += list(cc)
            elif len(cycles)==1:
                island_polys_to_keep += list(cc)
            else:
                print(cc)
                island_polys_to_keep += list(cc)



    gdf_islands_remove = gdfTopoPed.loc[ gdfTopoPed['polyID'].isin(island_polys_to_remove)].copy()
    gdf_islands_remove['area'] = gdf_islands_remove['geometry'].area

    gdf_islands_keep = gdfTopoPed.loc[ gdfTopoPed['polyID'].isin(island_polys_to_keep)].copy()
    gdf_islands_keep['area'] = gdf_islands_keep['geometry'].area

    gdfTopoPed = gdfTopoPed.loc[~(gdfTopoPed['polyID'].isin(island_polys_to_remove))]

    gdf_islands_remove.to_file(os.path.join(output_directory, "islands_removed.shp"))

    ################################
    #
    #
    # Produce nodes metadata
    #
    # For every node in the road network, create pedestrian nodes in the regions between the links in that network
    #
    ################################

    # Node metadata
    dfPedNodes = multiple_road_node_pedestrian_nodes_metadata(G, gdfORNode)


    ##################################
    #
    #
    # Assign coordinates to nodes
    #
    # Do I need to consider a connected component of walkable surfaces?
    #
    #
    ##################################

    # Recreate index - required for spatial indexing to work
    gdfTopoPed.index = np.arange(gdfTopoPed.shape[0])
    gdfBoundary.index = np.arange(gdfBoundary.shape[0])

    # Buffer the boundary so that nodes are located slightly away from walls
    boundary_geoms = gdfBoundary['geometry']
    pavement_geoms = gdfTopoPed['geometry']
    island_geoms = gdf_islands_remove['geometry']

    boundary_geoms.index = np.arange(len(boundary_geoms))
    pavement_geoms.index = np.arange(len(pavement_geoms))
    island_geoms.index = np.arange(len(island_geoms))

    dfPedNodes['boundary_ped_node'] = assign_boundary_coordinates_to_ped_nodes(dfPedNodes, gdfORLink, boundary_geoms, method = 'ray_intersection', required_range = default_angle_range, ray_length = default_ray_length, crs = projectCRS)
    dfPedNodes['pavement_ped_node'] = assign_boundary_coordinates_to_ped_nodes(dfPedNodes, gdfORLink, pavement_geoms, method = 'ray_intersection', required_range = default_angle_range, ray_length = default_ray_length, crs = projectCRS)

    # Now choose final node
    dfPedNodes['geometry'] = dfPedNodes.apply(choose_ped_node, axis=1, pave_node_col = 'pavement_ped_node', boundary_node_col = 'boundary_ped_node', road_node_x_col = 'juncNodeX', road_node_y_col = 'juncNodeY')

    # Run checks
    n_missing_nodes = dfPedNodes.loc[ dfPedNodes['geometry'].isnull()].shape[0]
    print("Number of missing nodes: {}".format(n_missing_nodes))

    if n_missing_nodes > 0:
        print("Finding missing nodes by removing angular range constraint")
        missing_index = dfPedNodes.loc[ dfPedNodes['geometry'].isnull()].index

        dfPedNodes.loc[missing_index, 'boundary_ped_node'] = assign_boundary_coordinates_to_ped_nodes(dfPedNodes.loc[missing_index], gdfORLink, boundary_geoms, method = 'ray_intersection', required_range = backup_angle_range, ray_length = default_ray_length, crs = projectCRS)
        dfPedNodes.loc[missing_index, 'pavement_ped_node'] = assign_boundary_coordinates_to_ped_nodes(dfPedNodes.loc[missing_index], gdfORLink, pavement_geoms, method = 'ray_intersection', required_range = backup_angle_range, ray_length = default_ray_length, crs = projectCRS)
        dfPedNodes.loc[missing_index, 'geometry'] = dfPedNodes.loc[missing_index].apply(choose_ped_node, axis=1, pave_node_col = 'pavement_ped_node', boundary_node_col = 'boundary_ped_node', road_node_x_col = 'juncNodeX', road_node_y_col = 'juncNodeY')

        
        n_missing_nodes = dfPedNodes.loc[ dfPedNodes['geometry'].isnull()].shape[0]
        print("Number of missing nodes: {}".format(n_missing_nodes))

    if n_missing_nodes > 0:
        print("Finding missing nodes by including islands")
        missing_index = dfPedNodes.loc[ dfPedNodes['geometry'].isnull()].index

        dfPedNodes.loc[missing_index, 'pavement_ped_node'] = assign_boundary_coordinates_to_ped_nodes(dfPedNodes.loc[missing_index], gdfORLink, island_geoms, method = 'ray_intersection', required_range = backup_angle_range, ray_length = default_ray_length, crs = projectCRS)
        dfPedNodes.loc[missing_index, 'geometry'] = dfPedNodes.loc[missing_index].apply(choose_ped_node, axis=1, pave_node_col = 'pavement_ped_node', boundary_node_col = 'boundary_ped_node', road_node_x_col = 'juncNodeX', road_node_y_col = 'juncNodeY')

        n_missing_nodes = dfPedNodes.loc[ dfPedNodes['geometry'].isnull()].shape[0]
        print("Number of missing nodes: {}".format(n_missing_nodes))

    if n_missing_nodes > 0:
        print("Finding missing nodes by assigning default coordiantes")
        missing_index = dfPedNodes.loc[ dfPedNodes['geometry'].isnull()].index

        dfPedNodes.loc[missing_index, 'geometry'] = assign_boundary_coordinates_to_ped_nodes(dfPedNodes.loc[missing_index], gdfORLink, None, method = 'default', required_range = backup_angle_range, ray_length = default_ray_length, crs = projectCRS)

        n_missing_nodes = dfPedNodes.loc[ dfPedNodes['geometry'].isnull()].shape[0]
        print("Number of missing nodes: {}".format(n_missing_nodes))

    gdfPedNodes = gpd.GeoDataFrame(dfPedNodes, geometry = 'geometry', crs = projectCRS)
    # check for duplicates?

    gdfPedNodes['fid'] = ["pave_node_{}".format(i) for i in range(gdfPedNodes.shape[0])]
    gdfPedNodes.drop(['boundary_ped_node','pavement_ped_node'],axis=1, inplace=True)


    ###################################
    #
    #
    # Join ped nodes to create ped network
    #
    #
    ###################################

    # Previously wrote separate functions for connecting nodes at opposite ends of a link and nodes around a junction

    dfPedLinks = connect_ped_nodes(gdfPedNodes, gdfORLink, G)
    gdfPedLinks = gpd.GeoDataFrame(dfPedLinks, geometry = 'geometry', crs = projectCRS)

    # Drop duplicated edges - don't expect any multi edges so drop duplicated fids since this implies duplicated edge between nodes
    gdfPedLinks = gdfPedLinks.drop_duplicates(subset = ['fid'])

    # Repair cases where a pavement links gets classified as a road crosisng link bc it intersects a road link
    gdfPedLinks = repair_non_crossing_links(gdfORLink['fid'].unique(), gdfPedNodes, gdfPedLinks)

    ###################################
    #
    #
    # Validate and save
    #
    #
    ###################################

    # Check there are no duplicated edges by checking for duplicated node pairs
    node_pairs_a = gdfPedLinks.loc[:, ['MNodeFID', 'PNodeFID']]
    node_pairs_b = gdfPedLinks.loc[:, ['PNodeFID', 'MNodeFID']].rename(columns = {'PNodeFID':'MNodeFID', 'MNodeFID':'PNodeFID'})
    node_pairs = pd.concat([node_pairs_a, node_pairs_b])
    assert node_pairs.duplicated().any() == False


    # Check that linkType matches with pedRLID
    assert gdfPedLinks.loc[ (gdfPedLinks['linkType']!='pavement') & (gdfPedLinks['pedRLID'].isnull())].shape[0] == 0
    assert gdfPedLinks.loc[ (gdfPedLinks['linkType']=='pavement'), 'pedRLID'].isnull().all()


    problem_links = validate_numbers_of_nodes_and_links(gdfORLink, gdfPedNodes, gdfPedLinks)
    gdfPedNodes.to_file(output_ped_nodes_file)
    gdfPedLinks.to_file(output_ped_links_file)


    # Check whether all cases of links not having 4 nodes are due to being dead ends
    or_edges = gdfORLink.loc[:, ['PNodeFID','MNodeFID']].values
    G = nx.Graph()
    G.add_edges_from(or_edges)
    dfDeg = pd.DataFrame(G.degree()).rename(columns = {0:'node_fid',1:'degree'})

    dead_ends = dfDeg.loc[ dfDeg['degree']==1,'node_fid'].values

    for rl_id in problem_links['missing_nodes']:
        u,v = gdfORLink.loc[ gdfORLink['fid']==rl_id, ['PNodeFID','MNodeFID']].values[0]
        if not ( (u in dead_ends) | (v in dead_ends)):
            print("OR link not dead end and is missing nodes:{}\n".format(rl_id))