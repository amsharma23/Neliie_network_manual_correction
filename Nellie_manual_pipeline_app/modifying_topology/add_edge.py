import pandas as pd
import numpy as np
from app_state import app_state
from utils.parsing import get_float_pos_comma


def join(viewer):
    
    #Extracted nodes dataframe and path
    nd_pdf = app_state.node_dataframe
    node_path = app_state.node_path

    #indices of selected nodes and their positions
    if (len(list(viewer.layers[1].selected_data))!=2):
        return
    ind_0 = list(viewer.layers[1].selected_data)[0]
    ind_1 = list(viewer.layers[1].selected_data)[1]
    pos_0 =(viewer.layers[1].data[ind_0])
    pos_1 =(viewer.layers[1].data[ind_1])
    
    #Find connected nodes if any
    node_ids = nd_pdf['Node ID'].tolist()
    node_positions = nd_pdf['Position(ZXY)'].tolist()
    node_positions_fl = [get_float_pos_comma(st) for st in node_positions]
    print(node_positions_fl)
    print(pos_0,pos_1)
    check_ind_0 = np.any(np.all(pos_0== node_positions_fl))
    check_ind_1 = np.any(np.all(pos_1== node_positions_fl))

    if check_ind_0 and check_ind_1:
        
        node_id_0 = node_ids[node_positions_fl.index(pos_0)]
        connected_nodes_0 = get_float_pos_comma(nd_pdf.loc[node_id_0,'Neighbour ID'])
    
        node_id_1 = node_ids[node_positions_fl.index(pos_1)]
        connected_nodes_1 = get_float_pos_comma(nd_pdf.loc[node_id_1,'Neighbour ID'])

        connected_nodes_0.append(node_id_1)
        connected_nodes_1.append(node_id_0)

        nd_pdf.loc[node_id_0,'Neighbour ID'] = connected_nodes_0
        nd_pdf.loc[node_id_0,'Degree of Node'] = len(connected_nodes_0)

        nd_pdf.loc[node_id_1,'Neighbour ID'] = connected_nodes_1
        nd_pdf.loc[node_id_1,'Degree of Node'] = len(connected_nodes_1)
        nd_pdf.to_csv(node_path,index=False)
        return

    nodes_extracted = nd_pdf['Node ID'].tolist()
    node_ids = [int(st) for st in nodes_extracted]
    max_node_id = max(node_ids)

    if (not check_ind_0) and check_ind_1:

        node_id_1 = node_ids[node_positions_fl.index(pos_1)]
        connected_nodes_1 = get_float_pos_comma(nd_pdf.loc[node_id_1,'Neighbour ID'])

        insert_loc = nd_pdf.index.max()
        if pd.isna(insert_loc):
            insert_loc = 0    
        else:
            insert_loc = insert_loc+1
        
        nd_pdf.loc[insert_loc,'Node ID'] = max_node_id+1
        nd_pdf.loc[insert_loc,'Degree of Node'] = 1
        nd_pdf.loc[insert_loc,'Position(ZXY)'] = str(pos_0)
        nd_pdf.loc[insert_loc,'Neighbour ID'] = [node_id_1]

        connected_nodes_1.append(max_node_id+1)
        nd_pdf.loc[node_id_1,'Neighbour ID'] = connected_nodes_1
        nd_pdf.loc[node_id_1,'Degree of Node'] = len(connected_nodes_1)

        nd_pdf.to_csv(node_path,index=False)
        return
    


    if (not check_ind_1) and check_ind_0:

        node_id_0 = node_ids[node_positions_fl.index(pos_0)]
        connected_nodes_0 = get_float_pos_comma(nd_pdf.loc[node_id_0,'Neighbour ID'])

        insert_loc = nd_pdf.index.max()
        if pd.isna(insert_loc):
            insert_loc = 0    
        else:
            insert_loc = insert_loc+1
        
        nd_pdf.loc[insert_loc,'Node ID'] = max_node_id+1
        nd_pdf.loc[insert_loc,'Degree of Node'] = 1
        nd_pdf.loc[insert_loc,'Position(ZXY)'] = str(pos_1)
        nd_pdf.loc[insert_loc,'Neighbour ID'] = [node_id_0]

        connected_nodes_0.append(max_node_id+1)
        nd_pdf.loc[node_id_0,'Neighbour ID'] = connected_nodes_0
        nd_pdf.loc[node_id_0,'Degree of Node'] = len(connected_nodes_0)

        nd_pdf.to_csv(node_path,index=False)
        return
    
    if (not check_ind_0) and (not check_ind_1):
        
        insert_loc = nd_pdf.index.max()
        if pd.isna(insert_loc):
            insert_loc = 0    
        else:
            insert_loc = insert_loc+1
        
        nd_pdf.loc[insert_loc,'Node ID'] = max_node_id+1
        nd_pdf.loc[insert_loc,'Degree of Node'] = 1
        nd_pdf.loc[insert_loc,'Position(ZXY)'] = str(pos_0)
        nd_pdf.loc[insert_loc,'Neighbour ID'] = [max_node_id+2]

        nd_pdf.loc[insert_loc+1,'Node ID'] = max_node_id+2
        nd_pdf.loc[insert_loc+1,'Degree of Node'] = 1
        nd_pdf.loc[insert_loc+1,'Position(ZXY)'] = str(pos_1)
        nd_pdf.loc[insert_loc+1,'Neighbour ID'] = [max_node_id+1]

        nd_pdf.to_csv(node_path,index=False)
        return