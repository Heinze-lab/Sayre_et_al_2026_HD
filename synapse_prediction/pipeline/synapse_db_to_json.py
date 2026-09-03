import numpy as np
from pymongo import MongoClient, ASCENDING, TEXT
import json
import sys 
from tqdm import tqdm
# Script to export synapses from mongodb in json file for viewing in neuroglancer


def to_jsoner(param):

    client = MongoClient(param['db_host'])
    db = client[param['db_name']]
    nodes_collection = db['syn.nodes']
    edges_collection = db['syn.edges']

    syn_json_list = []

    cursor = edges_collection.find(no_cursor_timeout=True)
    data = [d for d in cursor]

    for edge in tqdm(data): #in edges_collection.find({}, no_cursor_timeout=True):
        source_node = nodes_collection.find_one({'id': edge['target']})
        target_node = nodes_collection.find_one({'id': edge['source']})
        if source_node and target_node:
            # my neuroglancer viewer script inverts zyx, so I keep it zyx here
            # might need to make values int()
            syn_dic = {
                'id': source_node['id'],
                'location_pre': [source_node['z'], source_node['y'], source_node['x']],
                'location_post': [target_node['z'], target_node['y'], target_node['x']],
                'score': source_node['score'],
                'size': target_node['size']
            }
            syn_json_list.append(syn_dic)
    #edge.close()

    #print(syn_json_list)  

    # Define the output file path
    output_file = param['output_name']

    # Write the dictionary list to a JSON file
    with open(output_file, 'w') as f:
        json.dump(syn_json_list, f)

if __name__ == '__main__':
    param = sys.argv[1]
    with open(param, 'r') as f:
        param = json.load(f)
    to_jsoner(param)
