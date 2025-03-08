#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Napari plugin for analyzing 3D skeletal structures using the Nellie library.
This plugin allows for loading, processing, and visualization of 3D microscopy data.
"""
#%%Libraries 
import re
import numpy as np
import pandas as pd
from tifffile import imread
import os
from scipy.ndimage import label as labell
import networkx as nx
import csv
import warnings

try:
    from nellie.im_info.im_info import ImInfo
    from nellie.segmentation.filtering import Filter
    from nellie.segmentation.labelling import Label
    from nellie.segmentation.networking import Network
    NELLIE_AVAILABLE = True
except ImportError:
    NELLIE_AVAILABLE = False
    warnings.warn("Nellie library not found. Some functionality will be limited.")

import napari
from napari.utils.notifications import show_info, show_warning, show_error
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QCheckBox, QSpinBox, QFormLayout,
    QFileDialog, QComboBox, QGroupBox, QTextEdit
)
from magicgui import magic_factory, magicgui
from qtpy.QtCore import Qt

#% Global variables for tracking state and functions to act on the app class
class AppState:
    def __init__(self):
        self.loaded_folder = None
        self.folder_type = "Single TIFF"
        self.nellie_output_path = None
        self.raw_layer = None
        self.skeleton_layer = None
        self.points_layer = None
        self.node_path = None
        self.node_dataframe = None

# Initialize global state
app_state = AppState()

# ============= UTILITY FUNCTIONS =============

def get_float_pos(st):
    """Parse string representation of position to get coordinates.
    
    Args:
        st (str): String containing position coordinates
        
    Returns:
        list: List of integer coordinates
    """
    st = re.split(r'[ \[\]]', st)
    pos = [int(element) for element in st if element != '']
    return pos

def get_float_pos_comma(st):
    """Parse string representation of position to get coordinates.
    
    Args:
        st (str): String containing position coordinates
        
    Returns:
        list: List of integer coordinates
    """
    st = re.split(r'[ \[\,\]]', st)
    pos = [int(element) for element in st if element != '']
    return pos


def run_nellie_processing(im_path, num_t=None, remove_edges=False, ch=0):
    
    """Run the complete Nellie processing pipeline on an image.
    
    Args:
        im_path (str): Path to input image file
        num_t (int, optional): Number of time points
        remove_edges (bool): Whether to remove edge artifacts
        ch (int): Channel number to process
        
    Returns:
        ImInfo: Object containing processed data
    """
    if not NELLIE_AVAILABLE:
        show_error("Nellie library is required for processing. Please install it first.")
        return None
    
    try:
        
        # Initialize ImInfo with the image
        im_info = ImInfo(im_path, ch=ch)
        
        # Set dimension sizes (adjust these values based on your imaging parameters)
        im_info.dim_sizes = {'Z': 0.30, 'Y': 0.17, 'X': 0.17, 'T': 0}
        show_info(f"Dimension sizes set: {im_info.dim_sizes}")
        
        # Filtering step
        preprocessing = Filter(im_info, num_t, remove_edges=remove_edges)
        preprocessing.run()
        show_info("Filtering complete")
        
        # Segmentation step
        segmenting = Label(im_info, num_t)
        segmenting.run()
        show_info("Segmentation complete")
        
        # Network analysis
        networking = Network(im_info, num_t)
        networking.run()
        show_info("Networking complete")
        
        return im_info
    
    except Exception as e:
        show_error(f"Error in Nellie processing: {str(e)}")
        return None

def get_network(pixel_class_path):
    
    """Generate network representation from a skeleton image.
    
    Args:
        pixel_class_path (str): Path to pixel classification image
        
    Returns:
        tuple: (save_path, edge_path) - Paths to generated CSV and edge list files
    """
    try:
        # Define output file paths
        base_name = os.path.basename(pixel_class_path).split(".")[0]
        save_name = f"{base_name}_adjacency_list.csv"
        save_path = os.path.join(os.path.dirname(pixel_class_path), save_name)
        
        edge_name = f"{base_name}_edge_list.txt"
        edge_path = os.path.join(os.path.dirname(pixel_class_path), edge_name)
        
        # Load the skeleton image
        skeleton = imread(pixel_class_path)
        skeleton = np.transpose(skeleton)
        show_info(f"Skeleton shape: {np.shape(skeleton)}")
        print(np.shape(skeleton))
        # Define 3D connectivity structure
        struct = np.ones((3, 3, 3))
        
        # Extract tree structures
        trees, num_trees = labell(skeleton > 0, structure=struct)
        show_info(f"Found {num_trees} tree structures")
        
        # Convert tips and lone-tips to nodes (all nodes will have value 4)
        skeleton[skeleton == 2] = 4  # Tips
        skeleton[skeleton == 1] = 4  # Lone-tips
        
        # Extract edges (all voxels except nodes)
        no_nodes = np.where(skeleton == 4, 0, skeleton)
        edges, num_edges = labell(no_nodes > 0, structure=struct)
        show_info(f"Found {num_edges} edges")
        
        # Extract nodes
        nodes = np.where(skeleton == 4, 4, 0)
        node_labels, num_nodes = labell(nodes > 0, structure=struct)
        show_info(f"Found {num_nodes} nodes")
        
        # Map nodes to their connected edges
        node_edges = {}
        node_positions = {}
        
        # For each node, find connected edges
        for j_id in range(1, num_nodes + 1):
            # Get coordinates of all voxels in this node
            j_coords = np.argwhere(node_labels == j_id)
            
            # Track edges connected to this node
            connected_edges = set()
            
            if len(j_coords) > 0:
                # Take the first voxel's coordinates
                x, y, z = j_coords[0]
                node_positions[j_id] = (x, y, z)
            else:
                # Fallback if node has no voxels (shouldn't happen)
                node_positions[j_id] = (0, 0, 0)
            
            
            # Check 3x3x3 neighborhood around each node voxel
            for (x, y, z) in j_coords:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            # Skip the center voxel
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                                
                            # Neighbor coordinates
                            xx, yy, zz = x + dx, y + dy, z + dz
                            
                            # Check bounds
                            if (0 <= xx < skeleton.shape[0] and
                                0 <= yy < skeleton.shape[1] and
                                0 <= zz < skeleton.shape[2]):
                                
                                # If neighbor is part of an edge, add to connected edges
                                edge_label = edges[xx, yy, zz]
                                if edge_label != 0:
                                    connected_edges.add(edge_label)
            
            # Store edges connected to this node
            node_edges[j_id] = connected_edges
        
        # Map edges to connected nodes
        edge_nodes = {}
        for n_id, e_set in node_edges.items():
            for e_id in e_set:
                if e_id not in edge_nodes:
                    edge_nodes[e_id] = set()
                edge_nodes[e_id].add(n_id)
        
        # Create network graph
        G = nx.MultiGraph()
        
        # Add all nodes to graph
        for j_id in range(1, num_nodes + 1):
            x, y, z = node_positions[j_id]
            G.add_node(j_id, pos_x=x, pos_y=y, pos_z=z)

        
        # Add edges between nodes
        for e_id, connected_nodes in edge_nodes.items():
            cn = list(connected_nodes)
            
            if len(cn) == 2:
                # Standard edge between two nodes
                n1, n2 = cn
                G.add_edge(n1, n2, edge_id=e_id)
            elif len(cn) == 1:
                # Self-loop (edge connects to same node)
                (n1,) = cn
                G.add_edge(n1, n1, edge_id=e_id)
            elif len(cn) > 2:
                # Edge connects multiple nodes - add edges between all pairs
                for i in range(len(cn)):
                    for j in range(i + 1, len(cn)):
                        G.add_edge(cn[i], cn[j], edge_id=e_id)
        
        # Find connected components (separate trees)
        components = list(nx.connected_components(G))
        show_info(f"Found {len(components)} connected components")
        
        # Write adjacency list to CSV
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Header row
            writer.writerow(["component_num", "node", "pos_x", "pos_y", "pos_z", "adjacencies"])
            
            # Write each component
            for comp_num, comp in enumerate(components, start=1):
                # Create subgraph for this component
                subG = G.subgraph(comp).copy()
                
                # For each node, write its adjacencies
                for node in sorted(subG.nodes()):
                    
                    # Get node attributes (positions)
                    pos_x = subG.nodes[node]['pos_x']
                    pos_y = subG.nodes[node]['pos_y']
                    pos_z = subG.nodes[node]['pos_z']
                    
                    adjacencies = sorted(list(subG[node]))
                    writer.writerow([comp_num, node, pos_x, pos_y, pos_z, adjacencies])
            
        
        # Write edge list
        nx.write_edgelist(G, edge_path)
        
        show_info(f"Network analysis complete. Files saved to:\n- {save_path}\n- {edge_path}")
        return save_path, edge_path
        
    except Exception as e:
        show_error(f"Error generating network: {str(e)}")
        return None, None

def adjacency_to_extracted(extracted_csv_path,adjacency_path):
    
    adj_df = pd.read_csv(adjacency_path)
    if os.path.exists(extracted_csv_path):
        ext_df = pd.read_csv(extracted_csv_path)
    else:
        ext_df={}
        
    adjs_list = adj_df['adjacencies'].tolist()
    deg_nd_i = []
    deg_nd = []
    
    for el in adjs_list:
        elf = get_float_pos_comma(el)
        deg_nd_i.append(len(elf))
        if (len(elf)>0):
            deg_nd.append(len(elf))
        
    pos_x = adj_df['pos_x'].tolist()
    pos_y = adj_df['pos_y'].tolist()
    pos_z = adj_df['pos_z'].tolist()

    pos_zxy = [[pos_z[i_n],pos_y[i_n],pos_x[i_n]] for i_n,i in enumerate(deg_nd) if i>0]    
    
    ext_df['Degree of Node'] = deg_nd
    ext_df['Position(ZXY)'] = pos_zxy
    
    ext_df = pd.DataFrame.from_dict(ext_df)
    
    print(ext_df)
    
    ext_df.to_csv(extracted_csv_path,index=False)    
        
    
def load_image_and_skeleton(nellie_output_path):
    """Load raw image and skeleton from Nellie output directory.
    
    Args:
        nellie_output_path (str): Path to Nellie output directory
        
    Returns:
        tuple: (raw_image, skeleton_image, face_colors, positions, colors)
    """
    try:
        # Find relevant files in the output directory
        tif_files = os.listdir(nellie_output_path)
        
        # Find raw image file (channel 0)
        raw_files = [f for f in tif_files if f.endswith('-ch0-ome.ome.tif')]
        if not raw_files:
            show_error("No raw image file found in the output directory")
            return None, None, [], [], []
            
        raw_file = raw_files[0]
        basename = raw_file.split(".")[0]
        print('Basename is: '+basename)
        
        # Find skeleton image file
        skel_files = [f for f in tif_files if f.endswith('-ch0-im_pixel_class.ome.tif')]
        if not skel_files:
            show_error("No skeleton file found in the output directory")
            return None, None, [], [], []
            
        skel_file = skel_files[0]
        
        # Get full paths
        raw_im_path = os.path.join(nellie_output_path, raw_file)
        skel_im_path = os.path.join(nellie_output_path, skel_file)
        
        # Check for node data file
        node_path_extracted = os.path.join(nellie_output_path, f"{basename}_extracted.csv")
        adjacency_path = os.path.join(nellie_output_path, f"{basename}_adjacency_list.csv")
        app_state.node_path = node_path_extracted
        
        # Load images
        raw_im = imread(raw_im_path)
        skel_im = imread(skel_im_path)
        skel_im = np.transpose(np.nonzero(skel_im))
        
        # Default all points to red
        face_color_arr = ['red' for _ in range(len(skel_im))]
        
        #Check if an adjaceny list exists and convert to extracted csv if so
        if os.path.exists(adjacency_path) and not os.path.exists(node_path_extracted):
            adjacency_to_extracted(node_path_extracted,adjacency_path)
        
        if os.path.exists(adjacency_path) and os.path.exists(node_path_extracted):
            node_df = pd.read_csv(node_path_extracted)
            app_state.node_dataframe = node_df            
            if node_df.empty or pd.isna(node_df.index.max()):
                adjacency_to_extracted(node_path_extracted,adjacency_path)
        
        # Process extracted nodes if available
        if os.path.exists(node_path_extracted):
            node_df = pd.read_csv(node_path_extracted)
            app_state.node_dataframe = node_df
            
            if not node_df.empty and not pd.isna(node_df.index.max()):
                # Extract node positions and degrees
                pos_extracted = node_df['Position(ZXY)'].values
                show_info(f"Extracted positions: {pos_extracted}")
                
                deg_extracted = node_df['Degree of Node'].values.astype(int)
                positions = [get_float_pos_comma(el) for el in pos_extracted]
                print(positions)
                # Generate colors based on node degree
                colors = []
                for i, degree in enumerate(deg_extracted):
                    if degree == 1:
                        colors.append('blue')  # Endpoint nodes
                    else:
                        colors.append('green')  # Junction nodes
                        
                return raw_im, skel_im, face_color_arr, positions, colors
                
            else:
                # Create empty dataframe if no data
                app_state.node_dataframe = pd.DataFrame(columns=['Degree of Node', 'Position(ZXY)'])
                app_state.node_dataframe.to_csv(node_path_extracted, index=False)
                return raw_im, skel_im, face_color_arr, [], []
        else:
            # Create new node file if none exists
            app_state.node_dataframe = pd.DataFrame(columns=['Degree of Node', 'Position(ZXY)'])
            app_state.node_dataframe.to_csv(node_path_extracted, index=False)
            return raw_im, skel_im, face_color_arr, [], []
            
    except Exception as e:
        show_error(f"Error loading image and skeleton: {str(e)}")
        return None, None, [], [], []

# ============= GUI WIDGETS =============

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
        
        # File path display and browse button
        path_layout = QHBoxLayout()
        self.path_label = QLabel("No file selected")
        path_layout.addWidget(self.path_label)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.on_browse_clicked)
        path_layout.addWidget(self.browse_btn)
        file_layout.addLayout(path_layout)
        
        # File type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("File Type:")
        type_layout.addWidget(type_label)
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Single TIFF", "Time Series"])
        type_layout.addWidget(self.type_combo)
        file_layout.addLayout(type_layout)
        
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
        
        # Status section
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)
        layout.addWidget(QLabel("Status:"))
        layout.addWidget(self.status_text)
        
        # Network analysis button
        self.network_btn = QPushButton("Generate Network")
        self.network_btn.clicked.connect(self.on_network_clicked)
        self.network_btn.setEnabled(False)
        layout.addWidget(self.network_btn)
        
    def log_status(self, message):
        """Add a message to the status log."""
        current_text = self.status_text.toPlainText()
        self.status_text.setPlainText(f"{current_text}\n{message}" if current_text else message)
        self.status_text.verticalScrollBar().setValue(self.status_text.verticalScrollBar().maximum())
        
    def on_browse_clicked(self):
        """Handle browse button click to select input file or folder."""
        file_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        
        if file_path:
            app_state.loaded_folder = file_path
            self.path_label.setText(os.path.basename(file_path))
            self.process_btn.setEnabled(True)
            self.log_status(f"Selected folder: {file_path}")
            
            #check if there's an output folder already
            directory_list = [item for item in os.listdir(file_path) if (os.path.isdir(os.path.join(file_path,item)))]
            if 'nellie_output' in directory_list:
                app_state.nellie_output_path = os.path.join(file_path, 'nellie_output')
                self.view_btn.setEnabled(True)
                self.log_status(f'{file_path} has a processed output already!')
                
            
    def on_process_clicked(self):
        """Handle process button click to run Nellie processing."""
        if not app_state.loaded_folder:
            self.log_status("No folder selected. Please select a folder first.")
            return
            
        app_state.folder_type = self.type_combo.currentText()
        
        try:
            # Find TIFF files in the directory
            tif_files = [f for f in os.listdir(app_state.loaded_folder) if f.endswith('.ome.tif')]
            
            if not tif_files:
                self.log_status("No .ome.tif files found in the selected folder.")
                return
                
            # Use the first TIFF file found
            input_file = tif_files[0]
            im_path = os.path.join(app_state.loaded_folder, input_file)
            
            self.log_status(f"Processing {im_path}...")
            
            # Run Nellie processing
            im_info = run_nellie_processing(
                im_path, 
                remove_edges=self.remove_edges_check.isChecked(),
                ch=self.channel_spin.value()
            )
            
            if im_info:
                # Set output path
                app_state.nellie_output_path = os.path.join(app_state.loaded_folder, 'nellie_output')
                self.log_status("Processing complete!")
                self.view_btn.setEnabled(True)
                
        except Exception as e:
            self.log_status(f"Error during processing: {str(e)}")
            
    def on_view_clicked(self):
        """Handle view button click to display processing results."""
        if not app_state.nellie_output_path or not os.path.exists(app_state.nellie_output_path):
            self.log_status("No results to view. Please run processing first.")
            return
            
        try:
            # Clear existing layers
            self.viewer.layers.clear()
            
            # Load images
            raw_im, skel_im, face_colors, positions, colors = load_image_and_skeleton(app_state.nellie_output_path)
            
            if raw_im is not None and skel_im is not None:
                # Add layers to viewer
                app_state.raw_layer = self.viewer.add_image(
                    raw_im, 
                    scale=[1.765, 1, 1],  # Z, Y, X scaling
                    name='Raw Image'
                )
                
                app_state.skeleton_layer = self.viewer.add_points(
                    skel_im,
                    size=3,
                    face_color=face_colors,
                    scale=[1.765, 1, 1],
                    name='Skeleton'
                )
                
                # Add extracted points if available
                if positions and colors:
                    app_state.points_layer = self.viewer.add_points(
                        positions,
                        size=5,
                        face_color=colors,
                        scale=[1.765, 1, 1],
                        name='Extracted Nodes'
                    )
                    
                self.log_status("Visualization loaded successfully")
                self.network_btn.setEnabled(True)
                
        except Exception as e:
            self.log_status(f"Error viewing results: {str(e)}")
            
    def on_network_clicked(self):
        """Handle network button click to generate network representation."""
        if not app_state.nellie_output_path:
            self.log_status("No data to analyze. Please run processing and view results first.")
            return
            
        try:
            # Find pixel classification file
            tif_files = os.listdir(app_state.nellie_output_path)
            pixel_class_files = [f for f in tif_files if f.endswith('-ch0-im_pixel_class.ome.tif')]
            
            if not pixel_class_files:
                self.log_status("No pixel classification file found.")
                return
                
            pixel_class_path = os.path.join(app_state.nellie_output_path, pixel_class_files[0])
            
            # Generate network
            self.log_status("Generating network representation...")
            adjacency_path, edge_path = get_network(pixel_class_path)
            
            if adjacency_path and edge_path:
                self.log_status(f"Network analysis complete. Files saved to:\n- {adjacency_path}\n- {edge_path}")
                
        except Exception as e:
            self.log_status(f"Error generating network: {str(e)}")
#%%
# ============= MAIN APP =============

def main():
    """Main function to start the application."""
    viewer = napari.Viewer(title="Nellie Network Analysis")
    
    # Add main widget to viewer
    file_loader = FileLoaderWidget(viewer)
    viewer.window.add_dock_widget(file_loader, area='right', name="Nellie Controls")
    
    # Check for Nellie library
    if not NELLIE_AVAILABLE:
        show_warning("Nellie library not found. Please install it for full functionality.")
    
    return viewer

if __name__ == "__main__":
    viewer = main()
    napari.run()