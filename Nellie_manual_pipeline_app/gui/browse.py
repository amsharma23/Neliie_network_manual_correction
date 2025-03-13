import os
from natsort import natsorted
from app_state import app_state
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, 
QLabel, QPushButton, QSpinBox, QTextEdit, 
QVBoxLayout, QHBoxLayout, QWidget, QFileDialog)



def browse_folder(widget, path_label, process_btn, view_btn, type_combo,file_path):
    """Handle browse button click to select input file or folder.
    
    Args:
        widget: The parent widget for QFileDialog
        path_label: The QLabel to update with folder name
        process_btn: The process button to enable
        view_btn: The view button to potentially enable
        type_combo: ComboBox with folder type selection
    """
    
    app_state.folder_type = type_combo.currentText()
    
    if file_path:
        app_state.loaded_folder = file_path
        path_label.setText(os.path.basename(file_path))
        process_btn.setEnabled(True)
        widget.log_status(f"Selected folder: {file_path}")
        
        #check if there's an output folder already
        if app_state.folder_type == 'Single TIFF':
            directory_list = [item for item in os.listdir(file_path) if (os.path.isdir(os.path.join(file_path,item)))]
            if 'nellie_output' in directory_list:
                app_state.nellie_output_path = os.path.join(file_path, 'nellie_output')
                view_btn.setEnabled(True)
                widget.log_status( f'{file_path} has a processed output already!')
        
        elif app_state.folder_type == 'Time Series':
            subdirs = [d for d in os.listdir(app_state.loaded_folder) 
                      if os.path.isdir(os.path.join(app_state.loaded_folder, d))]
            
            if subdirs:
                # Process each subfolder as a time point
                widget.log_status(f"Found {len(subdirs)} time point folders")
                subdirs = natsorted(subdirs)
                
                for subdir in subdirs:
                    subdir_path = os.path.join(app_state.loaded_folder, subdir)
                    check_nellie_path = os.path.exists(os.path.join(subdir_path,'nellie_output'))
                
                    if check_nellie_path:
                        view_btn.setEnabled(True)
                        widget.log_status(f"Results to view for {subdir_path} are already available!")