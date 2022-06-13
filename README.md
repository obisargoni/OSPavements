# OSPavements
A python library for extracting pavement and carriageway polygons and networks from Ordnance Survey data

Scripts in this directory are used to extract carriadgeway and pavement polygons from Ordnance Survey (OS) Mastermap data, as well as road networks from OS Open Roads and OS ITN data sets.

Currently, to produce the required data the scripts need to be run in a certain order as they each do a different step of the processing. This file explains this process.

1. extractITNRouting.py

This gets the directoinality information for road links and road nodes for all of the ITN road link data and saves it

2. processRoadNetworkData.py

Selects the road network data that lies in the study area. Cleans the Open Roads data to simplify the lines to have minimal angular deviation along a line.

3. processOSTopographicData.py

Given a polygon covering the study area, this extracts the road links and nodes, pedestrian and vehicles topographic space, and lines obstructing pedestrian movement that should be included to model that study area.

4. makeITNdirectional.py

This uses the direction information extracted with the first script and edits the portion of the road network selected in the section so that it represents a directed road network.

Needs to happen after processOSTopographicData because that scripts adds in the lookup to OR road link ID.


It also processes and cleans this data, for example linking road links (both open road and ITN) with vehicle and pedestrian space

5. createPavementNetwork.py

Creates a network approximating the paths available to a pedestrian in that separate side of the road have different nodes with links to connect them across junctions

6. vehicleODFlows.py

NOTE: Requires manually placing shape files of vehicle OD nodes in the processed data directory.

Given the road nodes to use as vehicle ODs this procudes random flows between ODs, checking that a route is possible.

7. pedestrianODFlows.py

Very simple script to generate random flows between pedestrian ODs

8. processCrossingAlternatives.py

Create teh crossing infrastructure


## How to create a toy environment

1. createToyEnvironment.py

This creates the toy road networks, pavement network adn pavement and carriageway polygons

2. Manually create a POI file

This is used to set the pedestrian trip destination

3. vehicleODFlows.py

Creates vehicle ODs and the flows between them

4. pedestrianODFlows.py

Creates the pedestrian ODs and the flows between them

5. processCrossingAlternatives.py

Create teh crossing infrastructure

6. informalCrossingRoadLinkAttribute.py

Sets certain links to restrict informal crossing