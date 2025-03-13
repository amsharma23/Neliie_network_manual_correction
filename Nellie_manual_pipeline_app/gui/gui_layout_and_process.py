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
from app_state import app_state
import os
from tifffile import imread
from natsort import natsorted
from utils.layer_loader import load_image_and_skeleton
#from processing.network_generator import get_network
#from modifying_topology.add_tip import load_tip 
#from modifying_topology.add_junction import load_junction
from modifying_topology.edit_node import highlight
from modifying_topology.add_edge import join
from modifying_topology.remove_edge import remove
from .update_display import update_image
from .status import log
from .browse import browse_folder
from .process_image import process_clicked
from .view_images import view_clicked

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
        title = QLabel("Nellie Network Analysis")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        # File selection section
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)
        
        # File type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("File Type:")
        type_layout.addWidget(type_label)
        
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

        layout.addWidget(file_group)
        
        # Processing options section
        proc_group = QGroupBox("Processing Options")
        proc_layout = QFormLayout()
        proc_group.setLayout(proc_layout)
        
        # Channel selection
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(0, 10)
        self.channel_spin.setValue(0)
        proc_layout.addRow("Channel Number:", self.channel_spin)
        
        # Remove edges option
        self.remove_edges_check = QCheckBox()
        self.remove_edges_check.setChecked(False)
        proc_layout.addRow("Remove Edge Artifacts:", self.remove_edges_check)
        
        layout.addWidget(proc_group)
        
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
        slider_group = QGroupBox("Image Navigation")
        slider_layout = QVBoxLayout()
        slider_group.setLayout(slider_layout)
        
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
        slider_widget_layout.addWidget(QLabel("Image:"))
        self.image_slider = QSpinBox()
        self.image_slider.setMinimum(1)
        self.image_slider.setMaximum(1)
        self.image_slider.setValue(1)
        self.image_slider.valueChanged.connect(self.on_slider_changed)
        slider_widget_layout.addWidget(self.image_slider)
        
        slider_layout.addLayout(slider_widget_layout)
        layout.addWidget(slider_group)
    
        # Status section
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(300)
        layout.addWidget(QLabel("Status:"))
        layout.addWidget(self.status_text)
        
        # Network analysis button
        self.network_btn = QPushButton("Generate Network")
        self.network_btn.clicked.connect(self.on_network_clicked)
        self.network_btn.setEnabled(False)
        layout.addWidget(self.network_btn)
        
    def log_status(self, message):
        """Add a message to the status log."""
        log(self.status_text, message)

    def on_browse_clicked(self):
         """Handle browse button click to select input file or folder."""
         file_path = QFileDialog.getExistingDirectory(self, "Select Folder")
         browse_folder(self, self.path_label, self.process_btn, self.view_btn, self.type_combo,file_path)
    
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
            
        
            
    def on_prev_clicked(self):
        """Handle previous button click to show previous image."""
        current = self.image_slider.value()
        if current > 1:
            self.next_btn.setEnabled(True)
            self.image_slider.setValue(current - 1)
        elif (current) == 0:
            self.prev_btn.setEnabled(False)            
            self.log_status('Reached End of Time Series')

    def on_next_clicked(self):
        """Handle next button click to show next image."""
        current = self.image_slider.value()
        if current < self.image_slider.maximum():
            self.prev_btn.setEnabled(True)
            self.image_slider.setValue(current + 1)
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