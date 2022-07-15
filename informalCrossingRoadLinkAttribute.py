# Simple script to add categorisation of OR road links as permitting informal crossing or not
import numpy as np
import geopandas as gpd
import os
import json


######################
#
#
# Initialise variables and paths to data inputs and outputs
#
#
#####################
projectCRS = "epsg:27700"

with open("config.json") as f:
    config = json.load(f)

gis_data_dir = config['gis_data_dir']
output_directory = os.path.join(gis_data_dir, "processed_gis_data")

output_or_link_file = os.path.join(output_directory, config["openroads_link_processed_file"])



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
                        'Unclassified_Not Classified_A Road': 'A Road',
                        'A Road_B Road':'A Road',
                        'A Road_Not Classified_Unknown':'B Road',
                        'A Road_Unknown':'Unclassified'

                    }


#
# Load the data
#
gdfORLink = gpd.read_file(output_or_link_file)

###########################################
#
#
# Add in informal crossing classification to or road links
#
#
###########################################

no_informal_crossing_links = None

if 'class' in gdfORLink:
	gdfORLink['class'] = gdfORLink['class'].replace(class_rename_dict)
	assert gdfORLink.loc[ ~gdfORLink['class'].isin(['Unclassified','A Road','B Road', 'Classified Unnumbered'])].shape[0] == 0

	no_informal_crossing_links = gdfORLink.loc[ gdfORLink['class'].isin(['A Road']), 'fid'].tolist()
	no_informal_crossing_links+= config['no_informal_crossing_links']
else:
	no_informal_crossing_links = config['no_informal_crossing_links']


gdfORLink['infCross'] = 'true'
gdfORLink.loc[ gdfORLink['fid'].isin(no_informal_crossing_links), 'infCross'] = 'false'


gdfORLink.to_file(output_or_link_file)