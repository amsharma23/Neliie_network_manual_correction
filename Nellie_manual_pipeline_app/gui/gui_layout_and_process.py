#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 17:47:10 2025

@author: amansharma
"""
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, 
QLabel, QPushButton, QSpinBox, QTextEdit, 
QVBoxLayout, QHBoxLayout, QWidget, QFileDialog)
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QScrollArea
from natsort import natsorted
from app_state import app_state
import os
from utils.adjacency_reader import adjacency_to_extracted
from .update_display import update_image
from .status import log
from .browse import browse_folder
from .process_image import process_clicked
from .view_images import view_clicked
from .visualize_graph import make_multigraph_image
from .network_gen import network_click
from .visualize_graph import load_graph_on_viewer

class FileLoaderWidget(QWidget):
    """Widget for loading image files and setting processing options."""
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.setup_ui()
        
    def setup_ui(self):
        """Create the user interface."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Title
        self.title_label = QLabel("Nellie Network Analysis")  # Store as class attribute
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.title_label)
        
        # File selection section
        self.file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        self.file_group.setLayout(file_layout)
        
        # File type selection
        type_layout = QHBoxLayout()
        self.type_label = QLabel("File Type:")  # Store as class attribute
        type_layout.addWidget(self.type_label)
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Single TIFF", "Time Series"])
        type_layout.addWidget(self.type_combo)
        file_layout.addLayout(type_layout)
        
        # File path display and browse button
        path_layout = QHBoxLayout()
        self.path_label = QLabel("No file selected")
        path_layout.addWidget(self.path_label)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.on_browse_clicked)
        path_layout.addWidget(self.browse_btn)
        file_layout.addLayout(path_layout)

        layout.addWidget(self.file_group)
        
        # Processing options section
        self.proc_group = QGroupBox("Processing Options")
        proc_layout = QFormLayout()
        self.proc_group.setLayout(proc_layout)
        
        # Channel selection
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(0, 10)
        self.channel_spin.setValue(0)
        proc_layout.addRow("Channel Number:", self.channel_spin)
        
        # Remove edges option
        self.remove_edges_check = QCheckBox()
        self.remove_edges_check.setChecked(False)
        proc_layout.addRow("Remove Edge Artifacts:", self.remove_edges_check)
        
        layout.addWidget(self.proc_group)
        
        # Buttons section
        button_layout = QHBoxLayout()
        
        self.process_btn = QPushButton("Run Nellie Processing")
        self.process_btn.clicked.connect(self.on_process_clicked)
        self.process_btn.setEnabled(False)
        button_layout.addWidget(self.process_btn)
        
        self.view_btn = QPushButton("View Results")
        self.view_btn.clicked.connect(self.on_view_clicked)
        self.view_btn.setEnabled(False)
        button_layout.addWidget(self.view_btn)
        
        layout.addLayout(button_layout)
        
        # Image slider section
        self.slider_group = QGroupBox("Image Navigation")
        slider_layout = QVBoxLayout()
        self.slider_group.setLayout(slider_layout)
        
        # Slider control
        slider_control_layout = QHBoxLayout()
        self.image_label = QLabel("Current Image: 1/1")
        slider_control_layout.addWidget(self.image_label)
        
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.on_prev_clicked)
        self.prev_btn.setEnabled(False)
        slider_control_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.on_next_clicked)
        self.next_btn.setEnabled(False)
        slider_control_layout.addWidget(self.next_btn)
        
        slider_layout.addLayout(slider_control_layout)
        
        # Slider widget
        slider_widget_layout = QHBoxLayout()
        self.slider_text_label = QLabel("Image:")  # Store as class attribute
        slider_widget_layout.addWidget(self.slider_text_label)
        self.image_slider = QSpinBox()
        self.image_slider.setMinimum(1)
        self.image_slider.setMaximum(1)
        self.image_slider.setValue(1)
        self.image_slider.valueChanged.connect(self.on_slider_changed)
        slider_widget_layout.addWidget(self.image_slider)
        
        slider_layout.addLayout(slider_widget_layout)
        layout.addWidget(self.slider_group)
        
        # Network section - reorganized
        self.network_group = QGroupBox("Network Analysis")
        network_layout = QVBoxLayout()
        self.network_group.setLayout(network_layout)
        
        # Network buttons
        network_btn_layout = QHBoxLayout()
        
        self.network_btn = QPushButton("Generate Network")
        self.network_btn.clicked.connect(self.on_network_clicked)
        self.network_btn.setEnabled(False)
        network_btn_layout.addWidget(self.network_btn)
        
        self.graph_btn = QPushButton("Visualize Graph")
        self.graph_btn.clicked.connect(self.on_view_graph_clicked)
        self.graph_btn.setEnabled(False)
        network_btn_layout.addWidget(self.graph_btn)
        
        self.open_graph_btn = QPushButton("Open Graph in New Window")
        # Connect the signal when method is implemented
        # self.open_graph_btn.clicked.connect(self.on_open_graph_window_clicked)
        self.open_graph_btn.setEnabled(False)
        network_btn_layout.addWidget(self.open_graph_btn)
        
        network_layout.addLayout(network_btn_layout)
        
        # Graph visualization section
        self.graph_scroll = QScrollArea()
        self.graph_scroll.setWidgetResizable(True)
        self.graph_scroll.setMinimumHeight(150)  # Increased height for better visibility
        
        # Container widget for the graph image
        self.graph_container = QWidget()
        self.graph_container_layout = QVBoxLayout()
        self.graph_container.setLayout(self.graph_container_layout)
        
        # Label to display the graph image
        self.graph_image_label = QLabel("No graph generated yet")
        self.graph_image_label.setAlignment(Qt.AlignCenter)
        self.graph_container_layout.addWidget(self.graph_image_label)
        
        self.graph_scroll.setWidget(self.graph_container)
        network_layout.addWidget(self.graph_scroll)
        
        layout.addWidget(self.network_group)
        
        # Status section
        self.status_label = QLabel("Status:")  # Store as class attribute
        layout.addWidget(self.status_label)
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)
        layout.addWidget(self.status_text)
        
    def log_status(self, message):
        """Add a message to the status log."""
        log(self.status_text, message)

    def on_browse_clicked(self):
         """Handle browse button click to select input file or folder."""
         file_path = QFileDialog.getExistingDirectory(self, "Select Folder")
         browse_folder(self, self.path_label, self.process_btn, self.view_btn, self.network_btn, self.graph_btn, self.type_combo,file_path)
    
    def on_process_clicked(self):
        """Handle process button click to run Nellie processing."""
        process_clicked(self)
            
    def on_view_clicked(self):
        """Handle view button click to display processing results."""
        app_state.folder_type = self.type_combo.currentText()
        view_clicked(self,self.viewer,self.next_btn,self.prev_btn,self.image_slider,self.image_label,self.network_btn)
             
    def on_network_clicked(self):
        """Handle network button click to generate network representation."""
        if not app_state.nellie_output_path:
            self.log_status("No data to analyze. Please run processing and view results first.")
            return
        else:
            network_click(self)   
    
    def on_prev_clicked(self):
        """Handle previous button click to show previous image."""
        current = self.image_slider.value()
        if current > 1:
            self.next_btn.setEnabled(True)
            self.image_slider.setValue(current-1)
        elif (current) == 0:
            self.prev_btn.setEnabled(False)        
            self.log_status('Reached End of Time Series')

    def on_next_clicked(self):
        """Handle next button click to show next image."""
        current = self.image_slider.value()
        if current < self.image_slider.maximum():
            self.prev_btn.setEnabled(True)
            self.image_slider.setValue(current+1)
        elif (current) == self.image_slider.maximum():
            self.next_btn.setEnabled(False)            
            self.log_status('Reached End of Time Series')
            
    def on_slider_changed(self, value):
        """Handle slider value change to update displayed image."""
        self.image_label.setText(f"Current Image: {value}/{self.image_slider.maximum()}")
        self.update_displayed_image(value - 1)  # Convert to 0-based index
        
    def update_displayed_image(self, index):
        """Update the displayed image based on slider index."""
        current = self.image_slider.value()
        viewer = self.viewer
        update_image(self,viewer,current,index)

    def on_view_graph_clicked(self):

        """Handle graph button click to visualize the network graph."""
        if app_state.folder_type == 'Single TIFF':
                
            tif_files = os.listdir(app_state.nellie_output_path)
            multigraph_im = [f for f in tif_files if (f.endswith('multigraph.png') or f.endswith('multigraph.pdf'))]
            
            if not multigraph_im:
                pixel_class_path = [f for f in tif_files if (f.endswith('im_pixel_class.ome.tif') or f.endswith('im_pixel_class.ome.tiff'))][0]
                pixel_base_name = os.path.basename(pixel_class_path).split(".")[0]

                adjacency_path = [fls for fls in os.listdir(app_state.nellie_output_path) if fls.endswith('_adjacency_list.csv')][0]
                adjacency_path = os.path.join(app_state.nellie_output_path, adjacency_path)

                extracted_data_path = [fls for fls in os.listdir(app_state.nellie_output_path) if fls.endswith('_extracted.csv')][0]
                extracted_data_path = os.path.join(app_state.nellie_output_path, extracted_data_path)
                
                if not os.path.exists(extracted_data_path) and os.path.exists(adjacency_path):
                    adjacency_to_extracted(extracted_data_path,adjacency_path)
                    self.log_status("Error: Extracted data not found.")
                    return

                success = make_multigraph_image(self,extracted_data_path,pixel_base_name)

                if success:
                    load_graph_on_viewer(self)
                else:
                    self.log_status(f"Error: Making Graph visual failed for {extracted_data_path}.")
            
            elif multigraph_im:
                    app_state.graph_image_path = os.path.join(nellie_path, multigraph_im[0])
                    self.log_status(f"Graph already generated for {current_subdir}.")
                    load_graph_on_viewer(self)

        elif app_state.folder_type == 'Time Series':

            subdirs = [d for d in os.listdir(app_state.loaded_folder) 
                      if os.path.isdir(os.path.join(app_state.loaded_folder, d))]
            subdirs = natsorted(subdirs)

            if subdirs:
                
                app_state.current_image_index = self.image_slider.value() - 1
                current_subdir = subdirs[app_state.current_image_index]
                print(f"Current subdir: {current_subdir}")

                subdir_path = os.path.join(app_state.loaded_folder, current_subdir)
                nellie_path = os.path.join(subdir_path,'nellie_output')
                tif_files = os.listdir(nellie_path)
                multigraph_im = [f for f in tif_files if (f.endswith('multigraph.png') or f.endswith('multigraph.pdf'))]
                
                if not multigraph_im:
                    
                    pixel_class_path = [f for f in tif_files if (f.endswith('im_pixel_class.ome.tif') or f.endswith('im_pixel_class.ome.tiff'))][0]
                    pixel_base_name = os.path.basename(pixel_class_path).split(".")[0]

                    adjacency_path = [fls for fls in os.listdir(nellie_path) if fls.endswith('_adjacency_list.csv')][0]
                    adjacency_path = os.path.join(nellie_path, adjacency_path)

                    extracted_data_path = [fls for fls in os.listdir(nellie_path) if (fls == pixel_base_name+'_extracted.csv')][0]
                    extracted_data_path = os.path.join(nellie_path, extracted_data_path)
                    
                    if not os.path.exists(extracted_data_path) and os.path.exists(adjacency_path):
                        adjacency_to_extracted(extracted_data_path,adjacency_path)
                        self.log_status("Error: Extracted data not found.")
                        return
                    
                    self.log_status(f"Skeleton base name: {pixel_base_name}")
                    success = make_multigraph_image(self,extracted_data_path,pixel_base_name)
                
                    if success:
                        load_graph_on_viewer(self)
                    else:
                        self.log_status(f"Error: Graph visualization failed for {current_subdir}.")
                    
                elif multigraph_im:
                    app_state.graph_image_path = os.path.join(nellie_path, multigraph_im[0])
                    self.log_status(f"Graph already generated for {current_subdir}.")
                    load_graph_on_viewer(self)
            
            elif not subdirs:
                self.log_status("No time series data found.")
                return