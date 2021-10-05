# Produce figures showing how pedestrian network is produced
import os
import sys
import json
sys.path.append("..")
import createPavementNetwork as cpn
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

with open("figure_config.json") as f:
    fig_config = json.load(f)

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

# Get road links to plot for this node
dfPedNodesPlot = dfPedNodes.loc[ dfPedNodes['juncNodeID']==node_id]
gdfPedNodesPlot = gdfPedNodes.loc[ gdfPedNodes['juncNodeID']==node_id]

# I think build up plot from each row
f, ax = plt.subplots(1,1, figsize = (10,10))

for i, row in gdfPedNodesPlot.head(1).iterrows():


	# Get road link angles
    a1 = row['a1']
    a2 = row['a2']

    road_node = Point([row['juncNodeX'], row['juncNodeY']])

    a1, a2 = cpn.filter_angle_range(a1, a2, cpn.default_angle_range)

    rays = cpn.rays_between_angles(a1, a2, road_node, ray_length = cpn.default_ray_length)

    gdfRays = gpd.GeoDataFrame({'geometry':rays})
    cpn.gdfORLink.loc[ (cpn.gdfORLink['fid']==row['v1rlID']) | (cpn.gdfORLink['fid']==row['v2rlID'])].plot(ax=ax, color='black')
    gdfRays.plot(ax=ax)
    gdfPedNodesPlot.loc[i:i].plot(ax=ax)
    gdfPedNodes.loc[ (gdfPedNodes['v1rlID'] == row['v1rlID']) & (gdfPedNodes['v2rlID'] == row['v2rlID']) ].plot(ax=ax)

ax.set_axis_off()
f.show()


