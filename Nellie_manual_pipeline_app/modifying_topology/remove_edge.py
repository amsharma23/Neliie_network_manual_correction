import pandas as pd
import numpy as np
from app_state import app_state
from utils.parsing import get_float_pos_comma


def remove(viewer)->bool:
    flag = False

    #Extracted nodes dataframe and path
    nd_pdf = app_state.node_dataframe
    node_path = app_state.node_path

    #indices of selected nodes and their positions
    if (len(list(viewer.layers[1].selected_data))!=2):
        flag = True
        return flag
    ind_0 = list(viewer.layers[1].selected_data)[0]
    ind_1 = list(viewer.layers[1].selected_data)[1]
    pos_0 =list((viewer.layers[1].data[ind_0]))
    pos_1 =list((viewer.layers[1].data[ind_1]))
    #Find connected nodes if any
    node_ids = nd_pdf['Node ID'].tolist()
    node_positions = nd_pdf['Position(ZXY)'].tolist()
    node_positions_fl = [get_float_pos_comma(st) for st in node_positions]
    
    check_ind_0 = False
    check_ind_1 = False
    for posts in node_positions_fl:
        check_ind_0 = np.all(pos_0 == posts) or check_ind_0
        check_ind_1 = np.all(pos_1 == posts) or check_ind_1
        
    if check_ind_0 and check_ind_1:
        
        node_index_0 = node_positions_fl.index(pos_0)
        node_index_1 = node_positions_fl.index(pos_1)

        nd_pdf.drop(index=node_index_0,inplace=True)        
        nd_pdf.drop(index=node_index_1,inplace=True)    
        nd_pdf.to_csv(node_path,index=False)
        return flag

    else:
        flag = True
        return flag