# Figures showing different road networks pedestrian routes are simulated on
import os
import sys
import json
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as tfrms
from shapely.geometry import Point


#sys.path.append("..")
#import createPavementNetwork as cpn


#################################
#
#
# Functions
#
#
#################################
def subplot_road_network(f, ax, gdfORLink, class_col, title, inset_rec, config):
    # Plot classified links in a dark colour then unclassified (ie residential links) in a brighter colour
    gdfORLink.loc[ gdfORLink[class_col]=='true'].plot(ax=ax, color = config["road_link"]["color"])
    gdfORLink.loc[ gdfORLink[class_col]=='false'].plot(ax=ax, color = config["line_highlight_color"])

    # Plot horizontal bar to show difference in number of links and total length
    gdfORLink['is_restricted'] = (gdfORLink[class_col]=='false')

    length_sum = lambda s: s.length.sum()
    link_count = lambda s: s.unique().shape[0]
    dfTypeCount = gdfORLink.groupby('is_restricted').agg(  Link_Count=pd.NamedAgg(column='fid', aggfunc=link_count),
                                                            Link_Length=pd.NamedAgg(column='geometry', aggfunc=length_sum),).reset_index()
    dfTypeCount = dfTypeCount.set_index('is_restricted').T

    # create inset axis
    axins = f.add_axes(inset_rec)
    axins2 = axins.twinx() # Create another axes that shares the same x-axis as ax.
    insets = [axins, axins2]

    bar_width = 0.4
    x = np.arange(dfTypeCount.shape[0])

    for i, xi in enumerate(x):
        b1 = insets[i].bar([xi, xi+bar_width], dfTypeCount.iloc[i], width=bar_width, color = [config["road_link"]["color"], config["line_highlight_color"]])

        # Same thing, but offset the x.
        #b2 = axins2.bar(x + bar_width, dfTypeCount[False], width=bar_width, label='Non-residential')

    # Fix the x-axes.
    axins.tick_params(axis=u'both', which=u'both',length=0, pad=20)
    axins.set_xticks(x + + bar_width / 2)
    labels = list(dfTypeCount.index.str.replace("_", " "))
    labels[-1] = labels[-1] + " /m"
    axins.set_xticklabels(labels, fontdict = {'fontsize':15})

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
            a.text(text_x, text_y, text, ha='center', va='bottom', color=bar_color, size=15)

    # Remove lines
    for a in insets:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['bottom'].set_visible(False)
        a.spines['left'].set_visible(False)
    
    axins.axes.get_yaxis().set_visible(False)
    axins2.axes.get_yaxis().set_visible(False)
    ax.set_axis_off()

    # set x and y lims
    xmin, ymin, xmax, ymax = gdfORLink.total_bounds
    ax.set_xlim(xmin = xmin-10, xmax=xmax)
    ax.set_ylim(ymin = ymin-70, ymax=ymax)

    # Add hack scale bar
    trans = tfrms.blended_transform_factory( ax.transAxes, ax.transAxes )
    ax.errorbar( xmin+250, ymin-60, xerr=250, color='white', capsize=5)
    ax.text( xmin+250, ymin-60-10, '500m',  horizontalalignment='center', verticalalignment='top', fontsize=15)
    ax.set_title(title, y=-0.2, fontdict={'fontsize':25})

    return ax

def figure_road_network(gdfs, class_col, titles, inset_rects, config):
    '''Plot the road network for the study area, highlighting the residential roads.
    '''
    nplots = len(titles)
    f, axs = plt.subplots(1,nplots, figsize = (10*nplots,12))
    for i, gdfORLink in enumerate(gdfs):
        ax = subplot_road_network(f, axs[i], gdfORLink, class_col, titles[i], inset_rects[i], config)
    f.suptitle("Restricted and non-restricted roads in model environments", fontsize = 30)
    #plt.tight_layout()
    return f
    
#################################
#
#
# Globals
#
#
#################################
plt.style.use('dark_background')
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

with open("figure_config.json") as f:
    fig_config = json.load(f)

img_dir = "./thesis_images/"
output_road_network_fig_path = os.path.join(img_dir, "road_networks.png")

with open("../configs/config_toygrid_block_169nodes.json") as f:
    uniform_config = json.load(f)

with open("../configs/config_quadgrid_block100.json") as f:
    quad_config = json.load(f)

with open("../configs/config_claphamcommon.json") as f:
    cc_config = json.load(f)

uniform_road_link_file = os.path.join(uniform_config['gis_data_dir'], "processed_gis_data", uniform_config["openroads_link_processed_file"])
gdfORLinkUniform = gpd.read_file(uniform_road_link_file)

quad_road_link_file = os.path.join(quad_config['gis_data_dir'], "processed_gis_data", quad_config["openroads_link_processed_file"])
gdfORLinkQuad = gpd.read_file(quad_road_link_file)

cc_road_link_file = os.path.join(cc_config['gis_data_dir'], "processed_gis_data", cc_config["openroads_link_processed_file"])
gdfORLinkCC = gpd.read_file(cc_road_link_file)

#cpn.gdfORLink['class'] = cpn.gdfORLink['class'].replace(class_rename_dict)
#assert cpn.gdfORLink.loc[ ~cpn.gdfORLink['class'].isin(['Unclassified','A Road','B Road', 'Classified Unnumbered'])].shape[0] == 0

#################################
#
#
# Get section of data to plot
#
#
#################################

class_col = 'infCross'

gdfs = [gdfORLinkUniform, gdfORLinkQuad, gdfORLinkCC]
titles = ['Uniform Grid', 'Quad Grid', 'Clapham Common']
error_bar_locs = [(500,-20), (500,-20),  (528680, 174545)]
inset_rects = [ [0.05, 0.65, 0.1, 0.15], [0.35, 0.65, 0.1, 0.15], [0.65, 0.65, 0.1, 0.15] ]

'''
gdfs = [gdfORLinkUniform, gdfORLinkCC]
titles = ['Uniform Grid', 'Clapham Common']
error_bar_locs = [(500,-20),  (528680, 174545)]
inset_rects = [ [0.0, 0.65, 0.2, 0.1], [0.5, 0.65, 0.2, 0.1]]
'''

f_road = figure_road_network(gdfs, class_col, titles, inset_rects, fig_config)
f_road.savefig(output_road_network_fig_path)