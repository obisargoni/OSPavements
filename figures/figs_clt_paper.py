# Produce figures showing how pedestrian network is produced
import os
import sys
import json
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as tfrms
from shapely.geometry import Point, LineString


sys.path.append("..")
import createPavementNetwork as cpn

#plt.style.use('dark_background')

with open("figure_config.json") as f:
    fig_config = json.load(f)

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
def plot_layers(ax, config, pavement = None, carriageway = None, road_link = None, road_node = None, rays = None, pavement_link = None, pavement_node = None, pavement_polys_exclude = []):
    '''Keyword aarguments are geodataframes containing the shapes to be plotted
    '''
    pavement  = pavement.loc[ ~pavement['polyID'].isin(pavement_polys_exclude)]
    for i, (k, v) in enumerate(locals().items()):

        # Skip no keywork arguments
        if k in ['ax', 'config']:
            continue
        if v is not None:

            if k in ['pavement','carriageway']:
                ec = config[k]['edgecolor']
                fc = config[k]['color']
                if ec=="":
                    ec=None
                if fc=="":
                    fc=None
                v.plot(ax=ax, facecolor = fc, edgecolor = ec, zorder=i)
            elif k in ['road_link', 'pavement_link']:
                v.plot(ax=ax, facecolor=config[k]['color'], edgecolor = config[k]['color'], linewidth=config[k]['linewidth'], zorder=i)
            elif k in ['road_node', 'pavement_node']:
                v.plot(ax=ax, facecolor=config[k]['color'], edgecolor = config[k]['color'], linewidth=config[k]['linewidth'], zorder=i)
            elif k in ['rays']:
                v.plot(ax=ax, color=config[k]['color'], linewidth=config[k]['linewidth'], zorder=i)
            elif k in ['pavement_polys_exclude']:
                pass
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

def figure_pavement_nodes_for_single_road_link(road_link_id, gdfTopoVeh, gdfTopoPed, gdfORNode, gdfORLink, gdfPedNodes, gdfPedLinks, config = fig_config):
    '''Figure showing how location of pavement nodes are identified
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

    # Plot these layers
    ax = plot_layers(ax, config, pavement = gdfTopoPed, carriageway = gdfTopoVeh, road_link = gdfORLink, road_node = gdfORNode, pavement_link = None, pavement_node = None)

    # plot pavement links in the highligh colour
    gdfPedNodes.plot(ax=ax, edgecolor = config['pavement_node']['path_color'], facecolor = config['pavement_node']['path_color'], linewidth=config['pavement_node']['linewidth'], zorder=8)

    # Set limits
    xmin, ymin, xmax, ymax = gdfORLink.loc[ gdfORLink['fid']==road_link_id].total_bounds
    ax.set_xlim(xmin-3, xmax+3)
    ax.set_ylim(ymin-6.5, ymax+6.5)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
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

    # Plot these layers
    ax = plot_layers(ax, config, pavement = gdfTopoPed, carriageway = gdfTopoVeh, road_link = gdfORLink, road_node = gdfORNode, pavement_link = None, pavement_node = None)

    # plot pavement links in the highligh colour
    gdfPedLinks.plot(ax=ax, edgecolor = config['pavement_link']['path_color'], linewidth=config['pavement_link']['linewidth'], linestyle = '-', zorder=8)
    gdfPedNodes.plot(ax=ax, edgecolor = config['pavement_node']['path_color'], facecolor = config['pavement_node']['path_color'], linewidth=config['pavement_node']['linewidth'], zorder=8)

    # Set limits
    xmin, ymin, xmax, ymax = gdfORLink.loc[ gdfORLink['fid']==road_link_id].total_bounds
    ax.set_xlim(xmin-3, xmax+3)
    ax.set_ylim(ymin-6.5, ymax+6.5)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
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

def figure_tactical_path(study_area_rls, origin_id, dest_id, sp, n_horizon_links, tp, gdfTopoVeh, gdfTopoPed, gdfORNode, gdfORLink, gdfPedNodes, gdfPedLinks, gdfPedODs, ped_links_exclude, od_offsets, config = fig_config, xlims= (-2, 2), ylims = (-3, 3)):
    '''Function for creating figures illustrating tactical path finding
    '''

    # Initialise figure
    f, ax = plt.subplots(1,1, figsize = (10,10))

    # Get study area gdfs
    gdfORLinkSA = gdfORLink.loc[ gdfORLink['fid'].isin(study_area_rls)]
    study_area_nodes = np.concatenate( [gdfORLinkSA['MNodeFID'].values, gdfORLinkSA['PNodeFID'].values] )
    gdfORNodeSA = gdfORNode.loc[ gdfORNode['node_fid'].isin(study_area_nodes)]
    gdfTopoPedSA = gdfTopoPed.loc[gdfTopoPed['roadLinkID'].isin(study_area_rls)]
    gdfTopoVehSA = gdfTopoVeh.loc[gdfTopoVeh['roadLinkID'].isin(study_area_rls)]
    
    gdfPedNodesSA = gdfPedNodes.loc[ (gdfPedNodes['v1rlID'].isin(study_area_rls) | gdfPedNodes['v2rlID'].isin(study_area_rls) )]
    gdfPedLinksSA = gdfPedLinks.loc[ (gdfPedLinks['MNodeFID'].isin(gdfPedNodesSA['fid']) & gdfPedLinks['PNodeFID'].isin(gdfPedNodesSA['fid']) & ~gdfPedLinks['fid'].isin(ped_links_exclude))]

    # Plot study area layers
    ax = plot_layers(ax, config, pavement = gdfTopoPedSA, carriageway = gdfTopoVehSA, road_link = None, road_node = None, pavement_link = None, pavement_node = None)

    # Select route layers
    gdfspph = gdfORLink.loc[ gdfORLink['fid'].isin( sp[:n_horizon_links] )]
    gdfspph_ = gdfORLink.loc[ gdfORLink['fid'].isin( sp[n_horizon_links:] )]

    gdftp = gdfPedLinks.loc[ gdfPedLinks['fid'].isin(tp)]
    gdftp_nodes = gdfPedNodes.loc[ (gdfPedNodes['fid'].isin(gdftp['MNodeFID']) | (gdfPedNodes['fid'].isin(gdftp['PNodeFID'])))]

    gdfods = gdfPedODs.loc[ gdfPedODs['fid'].isin( [origin_id, dest_id])]

    # plot these additional layers
    gdfspph.plot(ax=ax, edgecolor = config['road_link']['color'], linewidth=config['road_link']['pathwidth'], linestyle = '-', zorder=7)
    gdfspph_.plot(ax=ax, edgecolor = config['road_link']['color'], linewidth=config['road_link']['pathwidth'], linestyle = ':', zorder=7)

    gdftp.plot(ax=ax, edgecolor = config['pavement_link']['path_color'], linewidth=config['pavement_link']['pathwidth'], linestyle = '-', zorder=8)
    gdftp_nodes.plot(ax=ax, edgecolor = config['pavement_node']['path_color'], facecolor = config['pavement_node']['path_color'], linewidth=config['pavement_node']['pathwidth'], zorder=8)

    gdfods.plot(ax=ax, edgecolor = config['od']['color'], facecolor = config['od']['color'], linewidth=config['od']['linewidth'], zorder=9)

    # add annotations
    o_coords = gdfPedODs.loc[ gdfPedODs['fid']==origin_id, 'geometry'].values[0].coords[0]
    d_coords = gdfPedODs.loc[ gdfPedODs['fid']==dest_id, 'geometry'].values[0].coords[0]
    ax.annotate("O", xy=o_coords, xycoords='data', xytext=(o_coords[0]+od_offsets[0][0], o_coords[1]+od_offsets[0][1]), textcoords='data', color=config['od']['color'], fontsize=config['annotate_fontsize'], fontweight = 'bold')
    ax.annotate("D", xy=d_coords, xycoords='data', xytext=(d_coords[0]+od_offsets[1][0], d_coords[1]+od_offsets[1][1]), textcoords='data', color=config['od']['color'], fontsize=config['annotate_fontsize'], fontweight = 'bold')

    # Set limits
    xmin, ymin, xmax, ymax = gdfTopoPedSA.total_bounds
    ax.set_xlim(xmin+xlims[0], xmax+xlims[1])
    ax.set_ylim(ymin+ylims[0], ymax+ylims[1])
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ax.set_axis_off()

    return f

def figure_operational_waypoints(study_area_rls, origin_id, dest_id, tp, gdfTopoVeh, gdfTopoPed, gdfORNode, gdfORLink, gdfPedNodes, gdfPedLinks, gdfPedODs, offsets, annotation_color, config = fig_config):
    '''Function for creating figures illustrating tactical path finding
    '''

    # Initialise figure
    f, ax = plt.subplots(1,1, figsize = (10,10))

    # Get study area gdfs
    gdfORLinkSA = gdfORLink.loc[ gdfORLink['fid'].isin(study_area_rls)]
    study_area_nodes = np.concatenate( [gdfORLinkSA['MNodeFID'].values, gdfORLinkSA['PNodeFID'].values] )
    gdfTopoPedSA = gdfTopoPed.loc[gdfTopoPed['roadLinkID'].isin(study_area_rls)]
    gdfTopoVehSA = gdfTopoVeh.loc[gdfTopoVeh['roadLinkID'].isin(study_area_rls)]
    

    # Plot study area layers
    ax = plot_layers(ax, config, pavement = gdfTopoPedSA, carriageway = gdfTopoVehSA, road_link = None, road_node = None, pavement_link = None, pavement_node = None)

    # Select route layers

    gdftp = gdfPedLinks.loc[ gdfPedLinks['fid'].isin(tp)]
    gdftp_nodes = gdfPedNodes.loc[ (gdfPedNodes['fid'].isin(gdftp['MNodeFID']) | (gdfPedNodes['fid'].isin(gdftp['PNodeFID'])))]

    gdfods = gdfPedODs.loc[ gdfPedODs['fid'].isin( [origin_id, dest_id])]

    # plot these additional layers
    gdftp.plot(ax=ax, edgecolor = config['pavement_link']['path_color'], linewidth=config['pavement_link']['pathwidth'], linestyle = '-', zorder=8)
    gdftp_nodes.plot(ax=ax, edgecolor = config['pavement_node']['path_color'], facecolor = config['pavement_node']['path_color'], linewidth=config['pavement_node']['pathwidth'], zorder=8)

    gdfods.plot(ax=ax, edgecolor = config['od']['color'], facecolor = config['od']['color'], linewidth=config['od']['linewidth'], zorder=9)

    # annotate the location of tactical path junctions
    gdftp_nodes['y'] = gdftp_nodes.geometry.map(lambda g: g.coords[0][1])
    gdftp_nodes.sort_values(by='y', ascending=True, inplace=True)
    for i,g in enumerate(gdftp_nodes['geometry'].values):
        x,y = g.coords[0]
        ax.annotate(i+1, xy=(x, y), xycoords='data', xytext=(x+offsets[i][0], y+offsets[i][1]), textcoords='data', color=annotation_color, fontsize = config['annotate_fontsize'], fontweight='bold')

    # Set limits
    xmin, ymin, xmax, ymax = gdfTopoPedSA.total_bounds
    ax.set_xlim(xmin-3, xmax+3)
    ax.set_ylim(ymin-2, ymax+2)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ax.set_axis_off()

    return f

def figure_operational_path(study_area_rls, origin_id, dest_id, tp, gdfTopoVeh, gdfTopoPed, gdfORNode, gdfORLink, gdfPedNodes, gdfPedLinks, gdfPedODs, offsets, annotation_color, crossing_coords, crossing_coords_tp_index, crossing_offsets, config = fig_config):
    '''Function for creating figures illustrating tactical path finding
    '''

    # Initialise figure
    f, ax = plt.subplots(1,1, figsize = (10,10))

    # Get study area gdfs
    gdfORLinkSA = gdfORLink.loc[ gdfORLink['fid'].isin(study_area_rls)]
    study_area_nodes = np.concatenate( [gdfORLinkSA['MNodeFID'].values, gdfORLinkSA['PNodeFID'].values] )
    gdfTopoPedSA = gdfTopoPed.loc[gdfTopoPed['roadLinkID'].isin(study_area_rls)]
    gdfTopoVehSA = gdfTopoVeh.loc[gdfTopoVeh['roadLinkID'].isin(study_area_rls)]
    

    # Plot study area layers
    ax = plot_layers(ax, config, pavement = gdfTopoPedSA, carriageway = gdfTopoVehSA, road_link = None, road_node = None, pavement_link = None, pavement_node = None)

    # Select route layers

    gdftp = gdfPedLinks.loc[ gdfPedLinks['fid'].isin(tp)]
    gdftp_nodes = gdfPedNodes.loc[ (gdfPedNodes['fid'].isin(gdftp['MNodeFID']) | (gdfPedNodes['fid'].isin(gdftp['PNodeFID'])))]

    gdfods = gdfPedODs.loc[ gdfPedODs['fid'].isin( [origin_id, dest_id])]
    gdfods.plot(ax=ax, edgecolor = config['od']['color'], facecolor = config['od']['color'], linewidth=config['od']['linewidth'], zorder=9)

    # plot these additional layers
    gdftp_nodes.plot(ax=ax, edgecolor = config['pavement_node']['path_color'], facecolor = config['pavement_node']['path_color'], linewidth=config['pavement_node']['pathwidth'], zorder=8)

    gdftp_nodes['y'] = gdftp_nodes.geometry.map(lambda g: g.coords[0][1])
    gdftp_nodes.sort_values(by='y', ascending=True, inplace=True)
    
    # get path linestraings    

    tp_coords = [g.coords[0] for g in gdftp_nodes.geometry.values[1:]]
    path_coords = []
    path_coords.append(gdfods.geometry.values[0].coords[0])
    for i in range(len(tp_coords)):
        if i==crossing_coords_tp_index:
            path_coords.append(crossing_coords[0])
            path_coords.append(crossing_coords[1])
        path_coords.append(tp_coords[i])

    lines = [LineString(u) for u in zip(path_coords[:-1], path_coords[1:])]


    #lines = [LineString([c1, c2]), LineString([c2, c3]), LineString([c3, c4]), LineString([c4, c5])]
    gdfPathLines = gpd.GeoDataFrame({'geometry':lines})

    gdfPathLines.plot(ax=ax, edgecolor = config['pavement_link']['path_color'], linewidth=config['pavement_link']['pathwidth'], linestyle = '--', zorder=8)

    # annotate the location of tactical path junctions
    for i,g in enumerate(gdftp_nodes['geometry'].values):
        x,y = g.coords[0]
        ax.annotate(i+1, xy=(x, y), xycoords='data', xytext=(x+offsets[i][0], y+offsets[i][1]), textcoords='data', color=annotation_color, fontsize = config['annotate_fontsize'], fontweight='bold')

    for i,c in enumerate(crossing_coords):
        x,y = c
        ax.annotate("C{}".format(i+1), xy=(x, y), xycoords='data', xytext=(x+crossing_offsets[i][0], y+crossing_offsets[i][1]), textcoords='data', color=config['od']['color'], fontsize = config['annotate_fontsize'], fontweight='bold')

    # Set limits
    xmin, ymin, xmax, ymax = gdfTopoPedSA.total_bounds
    ax.set_xlim(xmin-3, xmax+3)
    ax.set_ylim(ymin-2, ymax+2)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ax.set_axis_off()

    return f

def figure_experiments(study_area_rls, origin_id, dest_id, tp, gdfTopoVeh, gdfTopoPed, gdfORNode, gdfORLink, gdfPedNodes, gdfPedLinks, gdfPedODs, ca_files, ca_colors, road_links_annotate, rl_annotate_positions, od_offsets, bounds_offsets, pavement_polys_exclude = [], config = fig_config, figsize = (10,6)):
    '''Function for creating figures illustrating parameter setting experiments
    '''

    # Initialise figure
    f, ax = plt.subplots(1,1, figsize = figsize)

    # Get study area gdfs
    gdfORLinkSA = gdfORLink.loc[ gdfORLink['fid'].isin(study_area_rls)]
    study_area_nodes = np.concatenate( [gdfORLinkSA['MNodeFID'].values, gdfORLinkSA['PNodeFID'].values] )
    gdfTopoPedSA = gdfTopoPed.loc[gdfTopoPed['roadLinkID'].isin(study_area_rls)]
    gdfTopoVehSA = gdfTopoVeh.loc[gdfTopoVeh['roadLinkID'].isin(study_area_rls)]
    

    # Plot study area layers
    ax = plot_layers(ax, config, pavement = gdfTopoPedSA, carriageway = gdfTopoVehSA, road_link = gdfORLinkSA, road_node = None, pavement_link = None, pavement_node = None, pavement_polys_exclude = pavement_polys_exclude)

    # Select route layers

    gdftp = gdfPedLinks.loc[ gdfPedLinks['fid'].isin(tp)]
    gdftp_nodes = gdfPedNodes.loc[ (gdfPedNodes['fid'].isin(gdftp['MNodeFID']) | (gdfPedNodes['fid'].isin(gdftp['PNodeFID'])))]

    gdfods = gdfPedODs.loc[ gdfPedODs['fid'].isin( [origin_id, dest_id])]
    gdfods.plot(ax=ax, edgecolor = config['od']['color'], facecolor = config['od']['color'], linewidth=config['od']['linewidth'], zorder=9)

    # plot these additional layers
    gdftp.plot(ax=ax, edgecolor = config['pavement_link']['path_color'], linewidth=config['pavement_link']['pathwidth'], linestyle = '-', zorder=8)
    gdftp_nodes.plot(ax=ax, edgecolor = config['pavement_node']['path_color'], facecolor = config['pavement_node']['path_color'], linewidth=config['pavement_node']['pathwidth'], zorder=8)

    #gdftp_nodes['y'] = gdftp_nodes.geometry.map(lambda g: g.coords[0][1])
    #gdftp_nodes.sort_values(by='y', ascending=True, inplace=True)

    
    for i, path in enumerate(ca_files):
        gdf = gpd.read_file(path)
        gdf.plot(ax=ax, edgecolor = ca_colors[i], linewidth=config['pavement_link']['pathwidth'], linestyle = '-', zorder=8+i)
        

    # add annotations
    o_coords = gdfPedODs.loc[ gdfPedODs['fid']==origin_id, 'geometry'].values[0].coords[0]
    d_coords = gdfPedODs.loc[ gdfPedODs['fid']==dest_id, 'geometry'].values[0].coords[0]
    ax.annotate("O", xy=o_coords, xycoords='data', xytext=(o_coords[0]+od_offsets[0][0], o_coords[1]+od_offsets[0][1]), textcoords='data', color=config['od']['color'], fontsize=config['annotate_fontsize'], fontweight = 'bold')
    ax.annotate("D", xy=d_coords, xycoords='data', xytext=(d_coords[0]+od_offsets[1][0], d_coords[1]+od_offsets[1][1]), textcoords='data', color=config['od']['color'], fontsize=config['annotate_fontsize'], fontweight = 'bold')
    
    for i, rl in enumerate(road_links_annotate):
        rl_cent = gdfORLink.loc[ gdfORLink['fid']==rl, 'geometry'].values[0].centroid.coords[0]
        ax.text(rl_annotate_positions[i][0], rl_annotate_positions[i][1], "RL{}".format(i+1), color=config['od']['color'], fontsize = config['annotate_fontsize'], fontweight='bold')

    # Set limits
    xmin, ymin, xmax, ymax = gdfTopoPedSA.total_bounds
    ax.set_xlim(xmin+bounds_offsets[0][0], xmax+bounds_offsets[0][1])
    ax.set_ylim(ymin+bounds_offsets[1][0], ymax+bounds_offsets[1][1])
    plt.subplots_adjust(left=0.05, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ax.set_axis_off()

    return f

#################################
#
#
# Globals
#
#
#################################

img_dir = "./clt_images/new"
output_rays_node_fig_path = os.path.join(img_dir, "pavement_node_rays.png")
output_pavement_nodes_fig_path = os.path.join(img_dir, "pavement_nodes.png")
output_pavement_links_fig_path = os.path.join(img_dir, "pavement_links.png")

output_tp1_ph20_mincross = os.path.join(img_dir, "route_tactical_1_ph20_mincross_trip2.png")
output_tp2_ph20_mincross = os.path.join(img_dir, "route_tactical_2_ph20_mincross_trip2.png")
output_tp1_ph90_mincross = os.path.join(img_dir, "route_tactical_1_ph90_mincross_trip2.png")

output_tp2_md = os.path.join(img_dir, "route_tactical_1_ph100_mindist.png")
output_tp2_mc = os.path.join(img_dir, "route_tactical_1_ph100_mincross.png")

output_operational1_links = os.path.join(img_dir, "route_operational_trip2_ph20_mincross_primary_a.png")
output_operational1_path = os.path.join(img_dir, "route_operational_trip2_ph20_mincross_primary_b.png")
output_operational2_links = os.path.join(img_dir, "route_operational_trip2_ph20_mincross_secondary_a.png")
output_operational2_path = os.path.join(img_dir, "route_operational_trip2_ph20_mincross_secondary_b.png")

output_experiment1_path = os.path.join(img_dir, "postpone_crossing_configuration.png")
output_experiment2_path = os.path.join(img_dir, "sigspatial_replicate_configuration.png")


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

#
# Load data
#
cpn.load_data()
cpn.gdfORLink['class'] = cpn.gdfORLink['class'].replace(class_rename_dict)
assert cpn.gdfORLink.loc[ ~cpn.gdfORLink['class'].isin(['Unclassified','A Road','B Road', 'Classified Unnumbered'])].shape[0] == 0

#################################
#
#
# Get section of data to plot
#
#
#################################
def run():
    plt.tight_layout()

    or_link_ids = ["or_link_796","or_link_587","or_link_604","or_link_603","or_link_767","or_link_797","or_link_605"]

    gdfORLinkPlot = cpn.gdfORLink.loc[ cpn.gdfORLink['fid'].isin(or_link_ids)]
    gdfTopoVehPlot =cpn.gdfTopoVeh.loc[ cpn.gdfTopoVeh['roadLinkID'].isin(or_link_ids)]
    gdfTopoPedPlot =cpn.gdfTopoPed.loc[ cpn.gdfTopoPed['roadLinkID'].isin(or_link_ids)]
    gdfBoundaryPlot = cpn.gdfBoundary.loc[ cpn.gdfBoundary['geometry'].intersects(gdfTopoPedPlot['geometry'])]

    gdfPedNodes = gpd.read_file(cpn.output_ped_nodes_file)
    gdfPedLinks = gpd.read_file(cpn.output_ped_links_file)

    gdfPedODs = gpd.read_file( os.path.join(cpn.output_directory, cpn.config['pedestrian_od_file']) )

    # set which road node to illustrate getting ped nodes for
    node_id = fig_config['tactical1_config']['or_node_ids'][0]
    road_link_id = gdfPedNodes.loc[ gdfPedNodes['juncNodeID']==node_id, 'v1rlID'].values[0]

    road_link_id = fig_config['tactical1_config']['road_link_id']

    #
    # Figures showing pavement network creation
    #
    #
    f_node_rays = figure_pavement_nodes_for_single_road_node(node_id, gdfPedNodes, cpn.gdfORLink, cpn.gdfORNode, cpn.gdfTopoVeh, cpn.gdfTopoPed, config = fig_config)
    f_node_rays.savefig(output_rays_node_fig_path)

    f_road_pave_nodes = figure_pavement_nodes_for_single_road_link(road_link_id,cpn.gdfTopoVeh, cpn.gdfTopoPed, cpn.gdfORNode, cpn.gdfORLink, gdfPedNodes, gdfPedLinks, config = fig_config)
    f_road_pave_nodes.savefig(output_pavement_nodes_fig_path)

    f_node_joined = figure_connected_pavement_nodes(road_link_id, cpn.gdfTopoVeh, cpn.gdfTopoPed, cpn.gdfORNode, cpn.gdfORLink, gdfPedNodes, gdfPedLinks, config = fig_config)
    f_node_joined.savefig(output_pavement_links_fig_path)

    #
    # Figures showing effect of planning horizon on routing
    #
    study_area_rls = fig_config['tactical1_config']['study_area_ids']
    origin_id, dest_id = fig_config['tactical1_config']['od']
    ped_links_exclude = fig_config['tactical1_config']['pave_links_exclude']
    sp = fig_config['tactical1_config']['sp']

    n_horizon_links = fig_config['tactical1_config']['ntl20']
    tp = fig_config['tactical1_config']['tp20_1']
    od_offsets = fig_config['tactical1_config']['od_offsets']

    f_tactical1_ph20_mincross = figure_tactical_path(study_area_rls, origin_id, dest_id, sp, n_horizon_links, tp, cpn.gdfTopoVeh, cpn.gdfTopoPed, cpn.gdfORNode, cpn.gdfORLink, gdfPedNodes, gdfPedLinks, gdfPedODs, ped_links_exclude, od_offsets, config = fig_config, xlims = (35, -10), ylims = (-1, +1))
    f_tactical1_ph20_mincross.tight_layout()
    f_tactical1_ph20_mincross.savefig(output_tp1_ph20_mincross)

    tp = fig_config['tactical1_config']['tp20_2']

    f_tactical2_ph20_mincross = figure_tactical_path(study_area_rls, origin_id, dest_id, sp[-1:], 1, tp, cpn.gdfTopoVeh, cpn.gdfTopoPed, cpn.gdfORNode, cpn.gdfORLink, gdfPedNodes, gdfPedLinks, gdfPedODs, ped_links_exclude, od_offsets, config = fig_config, xlims = (35, -10), ylims = (-1, +1))
    f_tactical2_ph20_mincross.tight_layout()
    f_tactical2_ph20_mincross.savefig(output_tp2_ph20_mincross)

    n_horizon_links = fig_config['tactical1_config']['ntl90']
    tp = fig_config['tactical1_config']['tp90_1']

    f_tactical1_ph90_mincross = figure_tactical_path(study_area_rls, origin_id, dest_id, sp, n_horizon_links, tp, cpn.gdfTopoVeh, cpn.gdfTopoPed, cpn.gdfORNode, cpn.gdfORLink, gdfPedNodes, gdfPedLinks, gdfPedODs, ped_links_exclude, od_offsets, config = fig_config, xlims = (35, -10), ylims = (-1, +1))
    f_tactical1_ph90_mincross.tight_layout()
    f_tactical1_ph90_mincross.savefig(output_tp1_ph90_mincross)


    #
    # Figures showing effect of heuristics
    #
    study_area_rls = fig_config['tactical2_config']['study_area_ids']
    origin_id, dest_id = fig_config['tactical2_config']['od']
    ped_links_exclude = fig_config['tactical2_config']['pave_links_exclude']

    sp = fig_config['tactical2_config']['sp']
    n_horizon_links = fig_config['tactical2_config']['ntl']

    tpmd = fig_config['tactical2_config']['tpmd']
    od_offsets = fig_config['tactical2_config']['od_offsets']

    f_tactical2_md = figure_tactical_path(study_area_rls, origin_id, dest_id, sp, n_horizon_links, tpmd, cpn.gdfTopoVeh, cpn.gdfTopoPed, cpn.gdfORNode, cpn.gdfORLink, gdfPedNodes, gdfPedLinks, gdfPedODs, ped_links_exclude, od_offsets, config = fig_config)
    #f_tactical2_md.tight_layout()
    f_tactical2_md.savefig(output_tp2_md)

    tpmc = fig_config['tactical2_config']['tpmc']

    f_tactical2_mc = figure_tactical_path(study_area_rls, origin_id, dest_id, sp, n_horizon_links, tpmc, cpn.gdfTopoVeh, cpn.gdfTopoPed, cpn.gdfORNode, cpn.gdfORLink, gdfPedNodes, gdfPedLinks, gdfPedODs, ped_links_exclude, od_offsets, config = fig_config)
    #f_tactical2_mc.tight_layout()
    f_tactical2_mc.savefig(output_tp2_mc)


    #
    # Illustrating operational route choice waypoints
    #
    study_area_rls = fig_config['operational1_config']['study_area_ids']
    origin_id = fig_config['operational1_config']['od'][0]
    ped_links_exclude = fig_config['operational1_config']['pave_links_exclude']
    offsets = fig_config['operational1_config']['offsets']
    annotation_color = fig_config['operational1_config']['annotation_color']


    tp = fig_config['operational1_config']['tp']

    f_operational1_links = figure_operational_waypoints(study_area_rls, origin_id, None, tp, cpn.gdfTopoVeh, cpn.gdfTopoPed, cpn.gdfORNode, cpn.gdfORLink, gdfPedNodes, gdfPedLinks, gdfPedODs, offsets, annotation_color, config = fig_config)
    #f_operational1_links.tight_layout()
    f_operational1_links.savefig(output_operational1_links)

    crossing_coords = fig_config['operational1_config']['crossing_coords']
    crossing_coords_tp_index = fig_config['operational1_config']['crossing_coords_tp_index']
    crossing_offsets = fig_config['operational1_config']['crossing_offsets']

    f_operational1_path = figure_operational_path(study_area_rls, origin_id, None, tp, cpn.gdfTopoVeh, cpn.gdfTopoPed, cpn.gdfORNode, cpn.gdfORLink, gdfPedNodes, gdfPedLinks, gdfPedODs, offsets, annotation_color, crossing_coords, crossing_coords_tp_index, crossing_offsets, config = fig_config)
    #f_operational1_path.tight_layout()
    f_operational1_path.savefig(output_operational1_path)


    offsets = fig_config['operational2_config']['offsets']
    annotation_color = fig_config['operational2_config']['annotation_color']

    tp = fig_config['operational2_config']['tp']

    f_operational2_links = figure_operational_waypoints(study_area_rls, origin_id, None, tp, cpn.gdfTopoVeh, cpn.gdfTopoPed, cpn.gdfORNode, cpn.gdfORLink, gdfPedNodes, gdfPedLinks, gdfPedODs, offsets, annotation_color, config = fig_config)
    #f_operational2_links.tight_layout()
    f_operational2_links.savefig(output_operational2_links)

    crossing_coords = fig_config['operational2_config']['crossing_coords']
    crossing_coords_tp_index = fig_config['operational2_config']['crossing_coords_tp_index']
    crossing_offsets = fig_config['operational2_config']['crossing_offsets']

    f_operational2_path = figure_operational_path(study_area_rls, origin_id, None, tp, cpn.gdfTopoVeh, cpn.gdfTopoPed, cpn.gdfORNode, cpn.gdfORLink, gdfPedNodes, gdfPedLinks, gdfPedODs, offsets, annotation_color, crossing_coords, crossing_coords_tp_index, crossing_offsets, config = fig_config)
    #f_operational2_path.tight_layout()
    f_operational2_path.savefig(output_operational2_path)



    #
    # Figures showing environments used to set parameter ranges
    #
    # Need to load different sets of data
    gis_data_dir = fig_config['experiment1_config']['gis_data_dir']
    experiment_dir = os.path.join( gis_data_dir, fig_config['experiment1_config']['experiment_folder'])

    cas_files = [os.path.join(experiment_dir, i) for i in fig_config['experiment1_config']['ca_files']]
    gdfod = gpd.read_file(os.path.join(experiment_dir, "OD_pedestrian_nodes.shp"))

    gdfTopoPed = gpd.read_file( os.path.join(gis_data_dir, "processed_gis_data", "topographicAreaPedestrian.shp") )
    gdfTopoVeh = gpd.read_file( os.path.join(gis_data_dir, "processed_gis_data", "topographicAreaVehicle.shp") )
    gdfORNode = gpd.read_file( os.path.join(gis_data_dir, "processed_gis_data", "open-roads RoadNode Intersect Within simplify angles.shp") )
    gdfORLink = gpd.read_file( os.path.join(gis_data_dir, "processed_gis_data", "open-roads RoadLink Intersect Within simplify angles.shp") )
    gdfPedLinks = gpd.read_file( os.path.join(gis_data_dir, "processed_gis_data", "pedNetworkLinks.shp") )
    gdfPedNodes = gpd.read_file( os.path.join(gis_data_dir, "processed_gis_data", "pedNetworkNodes.shp") )

    study_area_rls = fig_config['experiment1_config']['study_area_ids']
    origin_id, dest_id = fig_config['experiment1_config']['od']
    tp = fig_config['experiment1_config']['tp']
    ca_colors = fig_config['experiment1_config']['ca_colors']
    road_links_annotate = fig_config['experiment1_config']['road_links_annotate']
    rl_annotate_positions = fig_config['experiment1_config']['annotate_positions']
    od_offsets = fig_config['experiment1_config']['od_offsets']
    bounds_offsets = fig_config['experiment1_config']['bounds_offsets']
    ppe = fig_config['experiment1_config']['pavement_polys_exclude']

    f_experiment1 = figure_experiments(study_area_rls, origin_id, dest_id, tp, gdfTopoVeh, gdfTopoPed, gdfORNode, gdfORLink, gdfPedNodes, gdfPedLinks, gdfod, cas_files, ca_colors, road_links_annotate, rl_annotate_positions, od_offsets, bounds_offsets, pavement_polys_exclude = ppe, config = fig_config)
    #f_experiment1.tight_layout()
    f_experiment1.savefig(output_experiment1_path)





    experiment_dir = os.path.join( gis_data_dir, fig_config['experiment2_config']['experiment_folder'])

    cas_files = [os.path.join(experiment_dir, i) for i in fig_config['experiment2_config']['ca_files']]
    gdfod = gpd.read_file(os.path.join(experiment_dir, "OD_pedestrian_nodes.shp"))

    study_area_rls = fig_config['experiment2_config']['study_area_ids']
    origin_id, dest_id = fig_config['experiment2_config']['od']
    tp = fig_config['experiment2_config']['tp']
    ca_colors = fig_config['experiment2_config']['ca_colors']
    road_links_annotate = fig_config['experiment2_config']['road_links_annotate']
    rl_annotate_positions = fig_config['experiment2_config']['annotate_positions']
    od_offsets = fig_config['experiment2_config']['od_offsets']
    bounds_offsets = fig_config['experiment2_config']['bounds_offsets']
    ppe = fig_config['experiment2_config']['pavement_polys_exclude']

    f_experiment2 = figure_experiments(study_area_rls, origin_id, dest_id, tp, gdfTopoVeh, gdfTopoPed, gdfORNode, gdfORLink, gdfPedNodes, gdfPedLinks, gdfod, cas_files, ca_colors, road_links_annotate, rl_annotate_positions, od_offsets, bounds_offsets, pavement_polys_exclude = ppe, config = fig_config)
    #f_experiment2.tight_layout()
    f_experiment2.savefig(output_experiment2_path)