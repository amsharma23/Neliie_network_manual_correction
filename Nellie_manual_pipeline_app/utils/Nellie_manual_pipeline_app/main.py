#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 16:54:53 2025

@author: amansharma
"""
import napari
from napari.utils.notifications import show_warning

# Import from local modules
from utils.layer_loader import load_image_and_skeleton
from app_state import app_state
from gui.gui_layout_and_process import FileLoaderWidget
from gui.viewer import load_viewer

# Check if Nellie is available
try:
    from nellie.im_info.im_info import ImInfo
    NELLIE_AVAILABLE = True
except ImportError:
    NELLIE_AVAILABLE = False


def main():
    """Main function to initialize the GUI."""    
    viewer =  load_viewer()

    # Check for Nellie library
    if not NELLIE_AVAILABLE:
        show_warning("Nellie library not found. Please install it for full functionality.")
    
    return viewer

if __name__ == "__main__":
    viewer = main()
    napari.run()