# Produce figures showing how pedestrian network is produced
import os
import sys
import json
sys.path.append("..")
import createPavementNetwork as cpn
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point

with open("figure_config.json") as f:
    fig_config = json.load(f)


#################################
#
#
# Functions
#
#
#################################
def figure_pavement_nodes_for_single_road_node(node_id, gdfPedNodes, gdfORLink, gdfTopoVeh, gdfTopoPed, config = fig_config):

    # Initialise figure
    f, ax = plt.subplots(1,1, figsize = (10,10))

    # Select the pavement nodes
    gdfPedNodesPlot = gdfPedNodes.loc[ gdfPedNodes['juncNodeID']==node_id]

    # Get corresponding road links ids
    rls = pd.concat([gdfPedNodesPlot['v1rlID'], gdfPedNodesPlot['v2rlID']]).unique()
    
    # Add pedestrian and vehicle polygons
    gdfTopoVeh = gdfTopoVeh.loc[ gdfTopoVeh['roadLinkID'].isin(rls)]
    gdfTopoPed = gdfTopoPed.loc[ gdfTopoPed['roadLinkID'].isin(rls)]

    # Get road links to plot for this node
    for i, row in gdfPedNodesPlot.head(1).iterrows():
        # Get road link angles
        a1 = row['a1']
        a2 = row['a2']

        road_node = Point([row['juncNodeX'], row['juncNodeY']])

        a1, a2 = cpn.filter_angle_range(a1, a2, cpn.default_angle_range)

        rays = cpn.rays_between_angles(a1, a2, road_node, ray_length = cpn.default_ray_length)

        gdfRays = gpd.GeoDataFrame({'geometry':rays})
        gdfRLPlot = cpn.gdfORLink.loc[ (cpn.gdfORLink['fid']==row['v1rlID']) | (cpn.gdfORLink['fid']==row['v2rlID'])]

        gdfTopoVeh.plot(ax=ax, color = config['cariageway_facecolour'])
        gdfTopoPed.plot(ax=ax, color = config['pavement_facecolour'])
        gdfRLPlot.plot(ax=ax, color='black')
        gdfRays.plot(ax=ax)

        # for some reason need to specify zorder value for this to appear on top
        gdfPedNodes.loc[ (gdfPedNodes['v1rlID'] == row['v1rlID']) & (gdfPedNodes['v2rlID'] == row['v2rlID']) ].plot(ax=ax, color = config['pavement_node_facecolour'], zorder=10)

        # Set axis limits
        boarded_dist = 5
        xmin, ymin, xmax, ymax = gdfRLPlot.total_bounds
        ax.set_xlim(xmin-5, xmax+5)
        ax.set_ylim(ymin-5, ymax+5)

    ax.set_axis_off()
    return f


#################################
#
#
# Get section of data to plot
#
#
#################################

or_link_ids = ["or_link_796","or_link_587","or_link_604","or_link_603","or_link_767","or_link_797","or_link_605"]

gdfORLinkPlot = cpn.gdfORLink.loc[ cpn.gdfORLink['fid'].isin(or_link_ids)]
gdfTopoVehPlot =cpn.gdfTopoVeh.loc[ cpn.gdfTopoVeh['roadLinkID'].isin(or_link_ids)]
gdfTopoPedPlot =cpn.gdfTopoPed.loc[ cpn.gdfTopoPed['roadLinkID'].isin(or_link_ids)]
gdfBoundaryPlot = cpn.gdfBoundary.loc[ cpn.gdfBoundary['geometry'].intersects(gdfTopoPedPlot['geometry'])]

gdfPedNodes = gpd.read_file(cpn.output_ped_nodes_file)

# Need to get nodes metadata to get angles of road links
dfPedNodes = cpn.multiple_road_node_pedestrian_nodes_metadata(cpn.G, cpn.gdfORNode)

# set which road node to illustrate getting ped nodes for
node_id = fig_config['pavenet_config']['or_node_ids'][0]

f = figure_pavement_nodes_for_single_road_node(node_id, gdfPedNodes, cpn.gdfORLink, cpn.gdfTopoVeh, cpn.gdfTopoPed, config = fig_config)