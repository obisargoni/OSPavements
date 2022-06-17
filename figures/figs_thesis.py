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

import figs_clt_paper as fcp

#################################
#
#
# Globals
#
#
#################################

with open("figure_config.json") as f:
    fig_config = json.load(f)

img_dir = "./thesis_images"

output_thesis_low_level_exp_path = os.path.join(img_dir, "sigspatial_replicate_configuration.png")


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
plt.tight_layout()

or_link_ids = ["or_link_796","or_link_587","or_link_604","or_link_603","or_link_767","or_link_797","or_link_605"]

gdfORLinkPlot = cpn.gdfORLink.loc[ cpn.gdfORLink['fid'].isin(or_link_ids)]
gdfTopoVehPlot =cpn.gdfTopoVeh.loc[ cpn.gdfTopoVeh['roadLinkID'].isin(or_link_ids)]
gdfTopoPedPlot =cpn.gdfTopoPed.loc[ cpn.gdfTopoPed['roadLinkID'].isin(or_link_ids)]
gdfBoundaryPlot = cpn.gdfBoundary.loc[ cpn.gdfBoundary['geometry'].intersects(gdfTopoPedPlot['geometry'])]

gdfPedNodes = gpd.read_file(cpn.output_ped_nodes_file)
gdfPedLinks = gpd.read_file(cpn.output_ped_links_file)

gdfPedODs = gpd.read_file( os.path.join(cpn.output_directory, cpn.config['pedestrian_od_file']) )


#
# Version of the low level environment figure for the thesis
#
gis_data_dir = fig_config['experiment1_config']['gis_data_dir']
experiment_dir = os.path.join( gis_data_dir, fig_config['low_level_exp_config']['experiment_folder'])

cas_files = [os.path.join(experiment_dir, i) for i in fig_config['low_level_exp_config']['ca_files']]
gdfod = gpd.read_file(os.path.join(experiment_dir, "OD_pedestrian_nodes.shp"))

study_area_rls = fig_config['low_level_exp_config']['study_area_ids']
origin_id, dest_id = fig_config['low_level_exp_config']['od']
tp = []
ca_colors = fig_config['low_level_exp_config']['ca_colors']
road_links_annotate = fig_config['low_level_exp_config']['road_links_annotate']
rl_annotate_positions = fig_config['low_level_exp_config']['annotate_positions']
od_offsets = fig_config['low_level_exp_config']['od_offsets']
bounds_offsets = fig_config['low_level_exp_config']['bounds_offsets']
ppe = fig_config['low_level_exp_config']['pavement_polys_exclude']

f_experiment2 = fcp.figure_experiments(study_area_rls, origin_id, dest_id, tp, cpn.gdfTopoVeh, cpn.gdfTopoPed, cpn.gdfORNode, cpn.gdfORLink, gdfPedNodes, gdfPedLinks, gdfod, cas_files, ca_colors, road_links_annotate, rl_annotate_positions, od_offsets, bounds_offsets, pavement_polys_exclude = ppe, config = fig_config, figsize = (10,7))
f_experiment2.tight_layout()
f_experiment2.savefig(output_thesis_low_level_exp_path)