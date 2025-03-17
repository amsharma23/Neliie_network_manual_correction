import pandas as pd
import ast
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
# Read the CSV file
# If you have the file locally, use:
# df = pd.read_csv('t_3_extracted.csv')
fld_place = '/Users/amansharma/Desktop/Manual_annotation_testing/2/'
op_path = '/Users/amansharma/Desktop/Manual_annotation_testing/2/edge_lists/'
for i in range(3,40,3):
    df = pd.read_csv(fld_place + str(i)+'/nellie_output/t_' + str(i) + '_extracted.csv')
    print(fld_place + str(i)+'/nellie_output/t_' + str(i) + '_extracted.csv')
    # Convert Node ID to integer
    # Convert Node ID to integer
    df['Node ID'] = df['Node ID'].astype(int)
    srcs = []
    # Convert Neighbour ID from string to list
    df['Neighbour ID'] = df['Neighbour ID'].apply(ast.literal_eval)

    # Create edge list preserving multiple edges to the same neighbor
    edge_list = []
    for _, row in df.iterrows():
        source = int(row['Node ID'])
        srcs.append(source)        
        neighbors = row['Neighbour ID']
        
        # Count occurrences of each neighbor
        neighbor_counts = Counter(neighbors)
        
        # Create edges based on the count
        for neighbor, count in neighbor_counts.items():
            for _ in range(count):
                if (neighbor not in srcs) or (neighbor == source):
                    edge_list.append((source, int(neighbor)))
        
    print("Edge list (preserving multiple edges to same neighbor):")
    
    # Create a MultiGraph
    G = nx.MultiGraph()
    G.add_edges_from(edge_list)
    nx.write_edgelist(G, op_path + 'edge_list_' + str(i) + '.txt', data=False)