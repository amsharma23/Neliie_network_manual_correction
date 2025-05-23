#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 16:58:26 2025

@author: amansharma
"""

"""
State management for the Nellie Napari plugin.
"""

class AppState:
    def __init__(self):
        self.loaded_folder = None
        self.folder_type = "Single TIFF"
        self.current_extracted_file = ""
        self.nellie_output_path = None
        self.raw_layer = None
        self.skeleton_layer = None
        self.points_layer = None
        self.highlighted_layer = None
        self.node_path = None
        self.node_dataframe = None
        self.slider_images = []
        self.current_image_index = 0  # Current image index; default is 0
        self.image_sets_keys = []
        self.image_sets = {}
        self.selected_node_position = []
        self.editable_node_positions = []
        self.graph_image_path = ""

app_state = AppState()