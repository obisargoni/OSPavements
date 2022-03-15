# Produce figures showing how pedestrian network is produced
import os
import sys
import json
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as tfrms
from shapely.geometry import Point


sys.path.append("..")
import createPavementNetwork as cpn

#################################
#
#
# Ideas for additional figures
#
#
# Dealing with isalnds
# Node placement when there isn't pavement nearby
#
#################################

#################################
#
#
# Functions
#
#
#################################
def plot_layers(ax, config, pavement = None, carriageway = None, road_link = None, road_node = None, rays = None, pavement_link = None, pavement_node = None):
    '''Keyword aarguments are geodataframes containing the shapes to be plotted
    '''
    for i, (k, v) in enumerate(locals().items()):

        # Skip no keywork arguments
        if k in ['ax', 'config']:
            continue
        if v is not None:

            if k in ['pavement','carriageway']:
                v.plot(ax=ax, color = config[k]['color'], zorder=i)
            elif k in ['road_link', 'pavement_link']:
                v.plot(ax=ax, facecolor=config[k]['color'], edgecolor = config[k]['color'], linewidth=config[k]['linewidth'], zorder=i)
            elif k in ['road_node', 'pavement_node']:
                v.plot(ax=ax, facecolor=config[k]['color'], edgecolor = config[k]['color'], linewidth=config[k]['linewidth'], zorder=i)
            elif k in ['rays']:
                v.plot(ax=ax, color=config[k]['color'], linewidth=config[k]['linewidth'], zorder=i)
            else:
                v.plot(ax=ax)

    return ax

def figure_pavement_nodes_for_single_road_node(node_id, gdfPedNodes, gdfORLink, gdfORNode, gdfTopoVeh, gdfTopoPed, config = fig_config):
    '''Figure showing how location of pavement nodes are identified
    '''

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
        gdfRNPlot = gdfORNode.loc[ gdfORNode['node_fid']==node_id]

        gdfPedNodes = gdfPedNodes.loc[ (gdfPedNodes['v1rlID'] == row['v1rlID']) & (gdfPedNodes['v2rlID'] == row['v2rlID']) ]

        plot_layers(ax, config, pavement = gdfTopoPed, carriageway = gdfTopoVeh, road_link = gdfRLPlot, road_node = gdfRNPlot, rays = gdfRays, pavement_link = None, pavement_node = gdfPedNodes)

        # Set axis limits
        xmin, ymin, xmax, ymax = gdfRLPlot.total_bounds
        ax.set_xlim(xmin-5, xmax-10)
        ax.set_ylim(ymin+20, ymax+5)

    ax.set_axis_off()
    return f

def figure_pavement_nodes_for_single_road_link(road_link_id, gdfPedNodes, gdfORLink, gdfORNode, gdfTopoVeh, gdfTopoPed, config = fig_config):
    '''Figure showing how location of pavement nodes are identified
    '''

    # Initialise figure
    f, ax = plt.subplots(1,1, figsize = (10,10))

    # Get road node ids
    node_ids = gdfORLink.loc[ gdfORLink['fid']==road_link_id, ['MNodeFID','PNodeFID']].values[0]

    # Select the pavement nodes
    gdfPedNodesPlot = gdfPedNodes.loc[ (gdfPedNodes['juncNodeID'].isin(node_ids)) ( (gdfPedNodes['v1rlID']==road_link_id) | (gdfPedNodes['v2rlID']==road_link_id) )]

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
        gdfRNPlot = gdfORNode.loc[ gdfORNode['node_fid']==node_id]

        gdfPedNodes = gdfPedNodes.loc[ (gdfPedNodes['v1rlID'] == row['v1rlID']) & (gdfPedNodes['v2rlID'] == row['v2rlID']) ]

        plot_layers(ax, config, pavement = gdfTopoPed, carriageway = gdfTopoVeh, road_link = gdfRLPlot, road_node = gdfRNPlot, rays = gdfRays, pavement_link = None, pavement_node = gdfPedNodes)

        # Set axis limits
        xmin, ymin, xmax, ymax = gdfRLPlot.total_bounds
        ax.set_xlim(xmin-5, xmax-10)
        ax.set_ylim(ymin+20, ymax+5)

    ax.set_axis_off()
    return f

def figure_connected_pavement_nodes(road_link_id, gdfTopoVeh, gdfTopoPed, gdfORNode, gdfORLink, gdfPedNodes, gdfPedLinks, config = fig_config):
    '''figure showing how pavment node are connected.
    Diagonal crossing links are shown separately to direct crossing links.
    '''
    # Initialise figure
    f, ax = plt.subplots(1,1, figsize = (10,10))

    # Select OR links and nodes - links selects include neighbouring links
    ind = gdfORLink['fid']==road_link_id
    ORNodes = pd.concat([gdfORLink.loc[ ind, 'MNodeFID'], gdfORLink.loc[ ind, 'PNodeFID']]).unique()
    gdfORLink = gdfORLink.loc[ gdfORLink['MNodeFID'].isin(ORNodes) | gdfORLink['PNodeFID'].isin(ORNodes)]
    gdfORNode = gdfORNode.loc[ gdfORNode['node_fid'].isin(ORNodes)]

    # Add pedestrian and vehicle polygons - includes neighbouring polygons
    gdfTopoVeh = gdfTopoVeh.loc[ gdfTopoVeh['roadLinkID'].isin(gdfORLink['fid'])]
    gdfTopoPed = gdfTopoPed.loc[ gdfTopoPed['roadLinkID'].isin(gdfORLink['fid'])]

    # Select the pavement links and nodes
    gdfPedNodes = gdfPedNodes.loc[ (gdfPedNodes['v1rlID']==road_link_id) | (gdfPedNodes['v2rlID']==road_link_id) ]
    gdfPedLinks = gdfPedLinks.loc[ gdfPedLinks['MNodeFID'].isin(gdfPedNodes['fid']) & gdfPedLinks['PNodeFID'].isin(gdfPedNodes['fid']) ]
    gdfPedLinksDiag = gdfPedLinks.loc[ gdfPedLinks['linkType']=='diag_cross']
    gdfPedLinksConst = gdfPedLinks.loc[ gdfPedLinks['linkType']!='diag_cross']

    # Plot these layers
    ax = plot_layers(ax, config, pavement = gdfTopoPed, carriageway = gdfTopoVeh, road_link = gdfORLink, road_node = gdfORNode, pavement_link = gdfPedLinksConst, pavement_node = gdfPedNodes)

    # plot diagonal crossing links in another colour
    gdfPedLinksDiag.plot(ax=ax, edgecolor = config['pavement_link']['color'], linewidth=config['pavement_link']['linewidth'], linestyle = '--', zorder=7)

    # Set limits
    xmin, ymin, xmax, ymax = gdfORLink.loc[ gdfORLink['fid']==road_link_id].total_bounds
    ax.set_xlim(xmin-3, xmax+3)
    ax.set_ylim(ymin-7.5, ymax+7.5)

    ax.set_axis_off()

    return f

def figure_road_network(gdfORLink, config = fig_config):
    '''Plot the road network for the study area, highlighting the residential roads.
    '''

    f, ax = plt.subplots(1,1, figsize = (10,10))

    # Plot classified links in a dark colour then unclassified (ie residential links) in a brighter colour
    gdfORLink.loc[ gdfORLink['class']!='Unclassified'].plot(ax=ax, color = fig_config["road_link"]["color"])
    gdfORLink.loc[ gdfORLink['class']=='Unclassified'].plot(ax=ax, color = fig_config["line_highlight_colour"])

    # Plot horizontal bar to show difference in number of links and total length
    gdfORLink['is_residential'] = (gdfORLink['class']=='Unclassified')

    length_sum = lambda s: s.length.sum()
    link_count = lambda s: s.unique().shape[0]
    dfTypeCount = gdfORLink.groupby('is_residential').agg(  Link_Count=pd.NamedAgg(column='fid', aggfunc=link_count),
                                                            Link_Length=pd.NamedAgg(column='geometry', aggfunc=length_sum),).reset_index()
    dfTypeCount = dfTypeCount.set_index('is_residential').T

    # create inset axis
    axins = f.add_axes([0.07, 0.65, 0.2, 0.2])
    axins2 = axins.twinx() # Create another axes that shares the same x-axis as ax.
    insets = [axins, axins2]

    bar_width = 0.4
    x = np.arange(dfTypeCount.shape[0])

    for i, xi in enumerate(x):
        b1 = insets[i].bar([xi, xi+bar_width], dfTypeCount.iloc[i], width=bar_width, color = [fig_config["line_lowlight_colour"], fig_config["line_highlight_colour"]])

        # Same thing, but offset the x.
        #b2 = axins2.bar(x + bar_width, dfTypeCount[False], width=bar_width, label='Non-residential')

    # Fix the x-axes.
    axins.set_xticks(x + + bar_width / 2)
    labels = list(dfTypeCount.index.str.replace("_", " "))
    labels[-1] = labels[-1] + "\n(m)"
    axins.set_xticklabels(labels)

    # Add bar labels
    # For each bar in the chart, add a text label.
    for a in insets:
        for bar in a.patches:
            # The text annotation for each bar should be its height.
            bar_value = int(bar.get_height())
            # Format the text with commas to separate thousands. You can do
            # any type of formatting here though.
            text = f'{bar_value:,}'
            # This will give the middle of each bar on the x-axis.
            text_x = bar.get_x() + bar.get_width() / 2
            # get_y() is where the bar starts so we add the height to it.
            text_y = bar.get_y() + bar_value
            # If we want the text to be the same color as the bar, we can
            # get the color like so:
            bar_color = bar.get_facecolor()
            # If you want a consistent color, you can just set it as a constant, e.g. #222222
            a.text(text_x, text_y, text, ha='center', va='bottom', color=bar_color, size=10)

    # Remove lines
    for a in insets:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['bottom'].set_visible(False)
        a.spines['left'].set_visible(False)
    
    axins.axes.get_yaxis().set_visible(False)
    axins2.axes.get_yaxis().set_visible(False)
    ax.set_axis_off()

    # Add hack scale bar
    trans = tfrms.blended_transform_factory( ax.transAxes, ax.transAxes )
    ax.errorbar( 531000, 172593, xerr=500, color='k', capsize=5)
    ax.text( 531000, 172590, '500m',  horizontalalignment='center', verticalalignment='top')

    f.suptitle("Residential and non-residentail roads in study area", fontsize = 20)

    return f

def figure_tactical_path(study_area_rls, origin_id, dest_id, sp, n_horizon_links, tp, gdfTopoVeh, gdfTopoPed, gdfORNode, gdfORLink, gdfPedNodes, gdfPedLinks, config = fig_config):
    '''Function for creating figures illustrating tactical path finding
    '''

    # Get study area gdfs
    gdfORLinkSA = gdfORLink.loc[ gdfORLink['fid'].isin(study_area_rls)]
    study_area_nodes = np.concat( [gdfORLinkSA['MNodeFID'].values, gdfORLinkSA['PNodeFID'].values] )
    gdfORNodeSA = gdfORNode.loc[ gdfORNode['fid'].isin(study_area_nodes)]
    gdfTopoPedSA = gdfTopoPed.loc[gdfTopoPed['roadLinkID'].isin(study_area_rls)]
    gdfTopoVehSA = gdfTopoVeh.loc[gdfTopoVeh['roadLinkID'].isin(study_area_rls)]
    gdfPedNodesSA = gdfPedNodes.loc[ gdfPedNodes['juncNodeID'].isin(study_area_nodes)]
    gdfPedLinksSA = gdfPedLinks.loc[ (gdfPedLinks['MNodeFID'].isin(gdfPedNodesSA['fid']) & gdfPedLinks['PNodeFID'].isin(gdfPedNodesSA['fid']) )]


    # Plot study area layers
    ax = plot_layers(ax, config, pavement = gdfTopoPed, carriageway = gdfTopoVeh, road_link = gdfORLink, road_node = gdfORNode, pavement_link = gdfPedLinksConst, pavement_node = gdfPedNodes)

    # Plot route layers

    # plot diagonal crossing links in another colour
    gdfPedLinksDiag.plot(ax=ax, edgecolor = config['pavement_link']['color'], linewidth=config['pavement_link']['linewidth'], linestyle = '--', zorder=7)

    # Set limits
    xmin, ymin, xmax, ymax = gdfORLink.loc[ gdfORLink['fid']==road_link_id].total_bounds
    ax.set_xlim(xmin-3, xmax+3)
    ax.set_ylim(ymin-7.5, ymax+7.5)

    ax.set_axis_off()


    return f

#################################
#
#
# Globals
#
#
#################################

with open("figure_config.json") as f:
    fig_config = json.load(f)

img_dir = "."
output_rays_node_fig_path = os.path.join(img_dir, "pavement_node_rays.png")
output_pavement_nodes_fig_path = os.path.join(img_dir, "pavement_nodes.png")
output_pavement_links_fig_path = os.path.join(img_dir, "pavement_links.png")
output_residential_network_fig_path = os.path.join(img_dir, "road_network_residential.png")

# class_rename_dict is taken from the 'pavenet_centrality.py' script
class_rename_dict = {   'Unknown':'Unclassified',
                        'Not Classified': 'Unclassified',
                        'Unclassified_Unknown': 'Unclassified',
                        'Unknown_Unclassified': 'Unclassified',
                        'Unclassified_Not Classified': 'Unclassified',
                        'Not Classified_Unclassified': 'Unclassified',
                        'Not Classified_Unknown': 'Unclassified',
                        'Unknown_Not Classified': 'Unknown',
                        'Unknown_A Road': 'A Road',
                        'Unclassified_A Road':'A Road',
                        'Unclassified_B Road':'B Road',
                        'B Road_Unclassified': 'B Road',
                        'Unknown_Classified Unnumbered': 'Classified Unnumbered',
                        'Unknown_Unclassified_Classified Unnumbered': 'Classified Unnumbered',
                        'Unclassified_Classified Unnumbered':'Classified Unnumbered',
                        'Not Classified_Classified Unnumbered': 'Classified Unnumbered',
                        'Classified Unnumbered_A Road': 'A Road',
                        'Classified Unnumbered_Unclassified': 'Classified Unnumbered',
                        'B Road_A Road': 'A Road',
                        'A Road_Not Classified':'A Road',
                        'Not Classified_A Road': 'A Road',
                        'Unclassified_B Road_A Road':'A Road',
                        'B Road_Unclassified_A Road': 'A Road',
                        'Classified Unnumbered_Unknown': 'Classified Unnumbered',
                        'B Road_Unknown': 'B Road',
                        'Not Classified_B Road': 'B Road',
                        'B Road_Classified Unnumbered': 'B Road',
                        'Unclassified_Classified Unnumbered_Unknown': 'Classified Unnumbered',
                        'Unclassified_Unknown_A Road': 'A Road',
                        'Unknown_Unclassified_A Road': 'A Road',
                        'Not Classified_Unclassified_A Road': 'A Road',
                        'Classified Unnumbered_B Road': 'B Road',
                        'B Road_Not Classified': 'B Road',
                        'Classified Unnumbered_Not Classified': 'Classified Unnumbered',
                        'Unclassified_Not Classified_A Road': 'A Road'
                    }

cpn.gdfORLink['class'] = cpn.gdfORLink['class'].replace(class_rename_dict)
assert cpn.gdfORLink.loc[ ~cpn.gdfORLink['class'].isin(['Unclassified','A Road','B Road', 'Classified Unnumbered'])].shape[0] == 0

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
gdfPedLinks = gpd.read_file(cpn.output_ped_links_file)

# set which road node to illustrate getting ped nodes for
node_id = fig_config['pavenet_config']['or_node_ids'][0]
road_link_id = gdfPedNodes.loc[ gdfPedNodes['juncNodeID']==node_id, 'v1rlID'].values[0]

road_link_id = "9745D155-3C95-4CCD-BC65-0908D57FA83A_0"

#
# Figures showing pavement network creation
#
f = figure_pavement_nodes_for_single_road_node(node_id, gdfPedNodes, cpn.gdfORLink, cpn.gdfORNode, cpn.gdfTopoVeh, cpn.gdfTopoPed, config = fig_config)
f.show()

f_road_pave_nodes = figure_pavement_nodes_for_single_road_link(road_link_id, gdfPedNodes, cpn.gdfORLink, cpn.gdfORNode, cpn.gdfTopoVeh, cpn.gdfTopoPed, config = fig_config)
f_road_pave_nodes.savefig(output_pavement_nodes_fig_path)

f_node_joined = figure_connected_pavement_nodes(road_link_id, cpn.gdfTopoVeh, cpn.gdfTopoPed, cpn.gdfORNode, cpn.gdfORLink, gdfPedNodes, gdfPedLinks, config = fig_config)
f_node_joined.savefig(output_pavement_links_fig_path)

#
# Figures showing effect of planning horizon on routing
#
f_tactical1_ph20_mincross = figure_tactical_path(origin_id, dest_id, sp, n_horizon_links, tp, cpn.gdfTopoVeh, cpn.gdfTopoPed, cpn.gdfORNode, cpn.gdfORLink, gdfPedNodes, gdfPedLinks, config = fig_config)

f_road = figure_road_network(cpn.gdfORLink)
f_road.show()


f.savefig(output_rays_node_fig_path)
f_road.savefig(output_residential_network_fig_path)