from app_state import app_state
from processing.network_generator import get_network
import os
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, 
QLabel, QPushButton, QSpinBox, QTextEdit, 
QVBoxLayout, QHBoxLayout, QWidget, QFileDialog)


def network_click(widget):
    try:
            
            # Find pixel classification file
            tif_files = os.listdir(app_state.nellie_output_path)
            pixel_class_files = [f for f in tif_files if f.endswith('-ch0-im_pixel_class.ome.tif')]
            
            if not pixel_class_files:
                widget.log_status("No pixel classification file found.")
                return
                
            pixel_class_path = os.path.join(app_state.nellie_output_path, pixel_class_files[0])
            
            # Generate network
            widget.log_status("Generating network representation...")
            adjacency_path, edge_path = get_network(pixel_class_path)
            
            if adjacency_path and edge_path:
                widget.log_status(f"Network analysis complete. Files saved to:\n- {adjacency_path}\n- {edge_path}")
                
    except Exception as e:
        widget.log_status(f"Error generating network: {str(e)}")