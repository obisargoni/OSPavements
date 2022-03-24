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

projectCRS = {'init' :'epsg:27700'}

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
    print(edges)
    widths = [ew[1] for ew in edges_and_widths]

    data = {'geometry':[],
            'MNodeFID':[],
            'PNodeFID':[],
            'fid':[]}

    # loop over grid coords and create one horizontal and one vertical line for each point
    for i, x in enumerate(edges[0]):
        for j, y in enumerate(edges[1]):
            lh = LineString( [ [x, y], [x+widths[0], y] ])
            lv = LineString( [ [x, y], [x, y+widths[1]] ])

            if i<len(edges[0])-1:
                data['geometry'].append(lh)
                data['MNodeFID'].append("node_{}{}".format(i,j))
                data['PNodeFID'].append("node_{}{}".format(i+1,j))
                data['fid'].append("link_{}{}_{}{}".format(i,j,i+1,j))

            if j<len(edges[1])-1:
                data['geometry'].append(lv)
                data['MNodeFID'].append("node_{}{}".format(i,j))
                data['PNodeFID'].append("node_{}{}".format(i,j+1))
                data['fid'].append("link_{}{}_{}{}".format(i,j,i,j+1))

    gdfGrid = gpd.GeoDataFrame(data, geometry='geometry', crs = crs)
    gdfGrid['pedRLID'] = gdfGrid['fid']

    return gdfGrid



with open(os.path.join("config.json")) as f:
    config = json.load(f)


gis_data_dir = config['toy_gis_dir']
output_directory = os.path.join(gis_data_dir, "processed_gis_data")

if os.path.isdir(output_directory) == False:
    os.mkdir(output_directory)

output_or_link_file = os.path.join(output_directory, config["openroads_link_processed_file"])
output_or_node_file = os.path.join(output_directory, config["openroads_node_processed_file"])

environment_limits = ( (0,1000), (0,1000))
block_size = 50

gdfORLink = create_grid_road_network(environment_limits, block_size)
gdfORLink.to_file(output_or_link_file)
#gdfORNode.to_file(output_or_node_file)