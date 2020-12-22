import itertools
import networkx as nx
import matplotlib.pyplot as plt
from collections import OrderedDict
from networkx.drawing.nx_pydot import graphviz_layout


class impl:
    def __init__(self, device, edge, cloud):
        self.device = device
        self.edge = edge
        self.cloud = cloud

class trans:
    def __init__(self, d2e, e2c,d2c):
        self.d2e = d2e
        self.e2c = e2c
        self.d2c = d2c

def build_graph():
    G = nx.DiGraph()
    node_list = list(range(8))
    G.add_nodes_from(node_list)
    G.add_edges_from([(0,1), (0,2), (1,3), (2,4), (3,5), (3,6), (2,6), (5,7), (6,7), (4,7)])

    return G

def longest_path(G):
    nodes = list(nx.topological_sort(G))
    source = nodes[0]

    def helper(node):
        if node == source:
            return 0
        preds = list(G.predecessors(node))
        dist = max([helper(i) + 1 for i in preds])
        return dist

    path_dict = OrderedDict()
    for node in nodes:
        path_dict[node] = helper(node)

    return path_dict


def get_layer(G):
    path_dict = longest_path(G)
    max_len = path_dict[max(path_dict, key=path_dict.get)]
    layer_dict = OrderedDict()
    for layer in range(max_len + 1):
        layer_item = []
        for k, v in path_dict.items():
            if v == layer:
                layer_item.append(k)
        layer_dict[layer] = layer_item

    return layer_dict



def assign_nodes_to_layers(G, layer_dict):
    nodes = list(nx.topological_sort(G))
    source = nodes[0]

    def get_subset_input_sibling(node, v):
        subset = set()
        siblings = []
        pred_list = list(G.predecessors(node))
        for i in range(1, len(pred_list) + 1):
            data = itertools.combinations(pred_list, i)
            subset.add(tuple(data))

        for j in v:
            if j != node:
                if tuple(G.predecessors(j)) in subset:
                    siblings.append(j)

        return siblings


    # k: layer index, v: list of nodes which belongs to layer k
    for k, v in layer_dict.items():
        print("Start partition in layer ", k)
        for node in v:
            # if G.nodes[node].get('location') == 'None':
            pred_list = list(G.predecessors(node))
            pred_location = []

            for pred in pred_list:
                pred_location.append(G.nodes[pred].get('location'))

            if 'cloud' in pred_location:
                last_location = 'cloud'
            elif 'edge' in pred_location:
                last_location = 'edge'
            else:
                last_location = 'device'

            time_device = 0
            time_edge = 0
            time_cloud = 0
            print('the pred location list is', pred_location)
            print('the last location is', last_location)
            if last_location == 'device':

                # put node on device
                time_device = 0 + G.nodes[node].get('attr').device
                # put node on edge
                for pred in pred_list:
                    time_edge = time_edge + G.edges[(pred, node)].get('attr').d2e + G.nodes[node].get('attr').edge
                # put node on cloud
                for pred in pred_list:
                    time_cloud = time_cloud + G.edges[(pred, node)].get('attr').d2c + G.nodes[node].get('attr').edge

                time_list = list([time_device, time_edge, time_cloud])
                time_min = min(time_list)

                if time_min == time_device:
                    node_location = 'device'
                elif time_min == time_edge:
                    node_location = 'edge'
                else:
                    node_location = 'cloud'

            elif last_location == 'edge':
                # put node on edge
                for pred in pred_list:
                    if G.nodes[pred].get('location') == 'device':
                        time_edge = time_edge + G.edges[(pred, node)].get('attr').d2e + G.nodes[node].get('attr').edge
                        time_cloud = time_cloud + G.edges[(pred, node)].get('attr').d2c + G.nodes[node].get('attr').cloud
                    else:
                        time_edge = time_edge + 0 + G.nodes[node].get('attr').edge
                        time_cloud = time_cloud + G.edges[(pred, node)].get('attr').e2c + G.nodes[node].get('attr').cloud

                time_list = list([time_edge, time_cloud])
                time_min = min(time_list)

                if time_min == time_edge:
                    node_location = 'edge'
                else:
                    node_location = 'cloud'
            else:
                # for pred in pred_list:
                #     if G.nodes[pred].get('location') == 'device':
                #         time_cloud = time_cloud + G.edges[(pred, node)].get('attr').d2c + G.nodes[node].get('attr').cloud
                #     elif G.nodes[pred].get('location') == 'edge':
                #         time_cloud = time_cloud + G.edges[(pred, node)].get('attr').e2c + G.nodes[node].get('attr').cloud
                #     else:
                #         time_cloud = time_cloud + 0 + G.nodes[node].get('attr').cloud
                node_location = 'cloud'

            G.nodes[node]['location'] = node_location

            # update subset siblings
            location_dict = {'device':0, 'edge':1, 'cloud':2}
            siblings = get_subset_input_sibling(node, v)
            for sibling in siblings:
                if G.nodes[sibling].get('location') == None:
                    G.nodes[sibling]['location'] = node_location
                else:
                    if location_dict[G.nodes[sibling].get('location')] < location_dict[node_location]:
                        G.nodes[sibling]['location'] = node_location

    return G


if __name__ == '__main__':
    G = build_graph()

    layer_dict = get_layer(G)
    print(layer_dict)

    G.add_node('input')
    G.add_node('output')
    G.add_edge('input', 0)
    G.add_edge(7, 'output')

    for node in G.nodes:
        G.nodes[node]['attr'] = impl(3,2,1)
    for edge in G.edges:
        G.edges[edge]['attr'] = trans(0.1, 0.2, 0.3)

    G.nodes['input']['location'] = 'device'
    G.edges[('input', 0)]['attr'] = trans(4, 8, 12)
    G.edges[(7, 'output')]['attr'] = trans(4, 8, 12)
    print(G.edges(data=True))

    G = assign_nodes_to_layers(G, layer_dict)
    for node in G.nodes:
        print('Node %s is at %s' % (str(node), G.nodes[node].get('location')))

    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'location')
    nx.draw_networkx_nodes(G, pos=pos)
    nx.draw_networkx_labels(G, pos=pos, labels=labels)
    nx.draw_networkx_edges(G, pos=pos, arrows=True)
    plt.show()