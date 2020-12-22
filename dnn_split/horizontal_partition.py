import itertools
import networkx as nx
import matplotlib.pyplot as plt
from collections import OrderedDict


# class impl:
#     def __init__(self, device, edge, cloud):
#         self.device = device
#         self.edge = edge
#         self.cloud = cloud
#
# class trans:
#     def __init__(self, d2e, e2c,d2c):
#         self.d2e = d2e
#         self.e2c = e2c
#         self.d2c = d2c

# def build_graph():
#     G = nx.DiGraph()
#     node_list = list(range(11))
#     G.add_nodes_from(node_list)
#     for i in range(10):
#         G.add_edge(i, i+1)
#
#     return G

# def build_alex_graph():
#     G = nx.DiGraph()
#     node_list = list(range(11))
#     G.add_nodes_from(node_list)
#     for i in range(10):
#         G.add_edge(i, i+1)
#
#     G.add_node('input')
#     G.add_node('output')
#     G.add_edge('input', 0)
#     G.add_edge(10, 'output')
#     G.nodes[0]['attr'] = impl(0.0387906074523925, 0.00161535739898681, 0.000158524800837039)
#     G.nodes[1]['attr'] = impl(0.00837891101837158, 0.000992012023925781, 0.000104918397590518)
#     G.nodes[2]['attr'] = impl(0.0532735347747802, 0.000886416435241699, 0.000335372802615165)
#     G.nodes[3]['attr'] = impl(0.00678062438964843, 0.000534486770629882, 0.0000306591999717056)
#     G.nodes[4]['attr'] = impl(0.025732421875, 0.00070655345916748, 0.000196169601380825)
#     G.nodes[5]['attr'] = impl(0.0373493671417236, 0.000864291191101074, 0.000201350398361682)
#     G.nodes[6]['attr'] = impl(0.0266769647598266, 0.00054333209991455, 0.000142950402200222)
#     G.nodes[7]['attr'] = impl(0.00391316413879394, 0.000162029266357421, 0.0000364096000790596)
#     G.nodes[8]['attr'] = impl(0.0905773401260376, 0.00514111518859863, 0.000916819202899932)
#     G.nodes[9]['attr'] = impl(0.05386643409729, 0.00220365524291992, 0.000424079996347427)
#     G.nodes[10]['attr'] = impl(0.0138338804244995, 0.000534629821777343, 0.00014015680104494)
#
#     d2e = 64.95
#     e2c = 31.53
#     d2c = 29.78
#
#     output_size_01 = 5.908203125
#     G.edges[(0, 1)]['attr'] = trans(output_size_01 / d2e, output_size_01 / e2c, output_size_01 / d2c)
#     output_size_12 = 1.423828125
#     G.edges[(1, 2)]['attr'] = trans(output_size_12 / d2e, output_size_12 / e2c, output_size_12 / d2c)
#     output_size_23 = 4.271484375
#     G.edges[(2, 3)]['attr'] = trans(output_size_23 / d2e, output_size_23 / e2c, output_size_23 / d2c)
#     output_size_34 = 0.990234375
#     G.edges[(3, 4)]['attr'] = trans(output_size_34 / d2e, output_size_34 / e2c, output_size_34 / d2c)
#     output_size_45 = 1.98046875
#     G.edges[(4, 5)]['attr'] = trans(output_size_45 / d2e, output_size_45 / e2c, output_size_45 / d2c)
#     output_size_56 = 1.3203125
#     G.edges[(5, 6)]['attr'] = trans(output_size_56 / d2e, output_size_56 / e2c, output_size_56 / d2c)
#     output_size_67 = 1.3203125
#     G.edges[(6, 7)]['attr'] = trans(output_size_67 / d2e, output_size_67 / e2c, output_size_67 / d2c)
#     output_size_78 = 0.28125
#     G.edges[(7, 8)]['attr'] = trans(output_size_78 / d2e, output_size_78 / e2c, output_size_78 / d2c)
#     output_size_89 = 0.125
#     G.edges[(8, 9)]['attr'] = trans(output_size_89 / d2e, output_size_89 / e2c, output_size_89 / d2c)
#     output_size_910 = 0.125
#     G.edges[(9, 10)]['attr'] = trans(output_size_910 / d2e, output_size_910 / e2c, output_size_910 / d2c)
#
#     G.nodes['input']['attr'] = 'device'
#     input_size = 4.59375
#     G.edges[('input', 0)]['attr'] = trans(input_size / d2e, input_size / e2c, input_size/d2c)
#     G.edges[(10, 'output')]['attr'] = trans(10000, 10000, 10000)
#
#     return G

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
                print('pred is', pred)
                print('node is ', node)
                time_device = 0 + G.nodes[node].get('attr').device
                # put node on edge
                for pred in pred_list:
                    print('edge trans', G.edges[(pred, node)].get('attr').d2e)
                    time_edge = time_edge + G.edges[(pred, node)].get('attr').d2e + G.nodes[node].get('attr').edge
                # put node on cloud
                for pred in pred_list:
                    print('cloud trans', G.edges[(pred, node)].get('attr').d2c)
                    time_cloud = time_cloud + G.edges[(pred, node)].get('attr').d2c + G.nodes[node].get('attr').cloud

                time_list = list([time_device, time_edge, time_cloud])
                print(time_list)
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
    None
    # G = build_graph()
    #
    # layer_dict = get_layer(G)
    # print(layer_dict)
    #
    # G.add_node('input')
    # G.add_node('output')
    # G.add_edge('input', 0)
    # G.add_edge(10, 'output')
    # G.nodes[0]['attr'] = impl(0.03941894, 0.00148439, 0.000245695993)
    # G.nodes[1]['attr'] = impl(0.00977516, 0.00098157, 0.0000405439995)
    # G.nodes[2]['attr'] = impl(0.05405164, 0.00077534, 0.000319615990)
    # G.nodes[3]['attr'] = impl(0.00755548, 0.00052238, 0.0000353920013)
    # G.nodes[4]['attr'] = impl(0.02662373, 0.00069451, 0.000292640001)
    # G.nodes[5]['attr'] = impl(0.03826237, 0.00082541, 0.000202368006)
    # G.nodes[6]['attr'] = impl(0.02726197, 0.00062299, 0.000149023995)
    # G.nodes[7]['attr'] = impl(0.00411129, 0.00015831, 0.0000356799997)
    # G.nodes[8]['attr'] = impl(0.09115124, 0.00550461, 0.000915455997)
    # G.nodes[9]['attr'] = impl(0.05428672, 0.00237727, 0.000427552015)
    # G.nodes[10]['attr'] = impl(0.01386261, 0.00059962, 0.000106495999)
    #
    # output_size_01 = 64 * 55 * 55 * 4 / (1024 * 1024)
    # G.edges[(0, 1)]['attr'] = trans(output_size_01/10, output_size_01/8, output_size_01/7)
    # output_size_12 = 64 * 27 * 27 * 4 / (1024 * 1024)
    # G.edges[(1, 2)]['attr'] = trans(output_size_12/10, output_size_12/8, output_size_12/7)
    # output_size_23 = 192 * 27 * 27 * 4 / (1024 * 1024)
    # G.edges[(2, 3)]['attr'] = trans(output_size_23/10, output_size_23/8, output_size_23/7)
    # output_size_34 = 192 * 13 * 13 * 4 / (1024 * 1024)
    # G.edges[(3, 4)]['attr'] = trans(output_size_34/10, output_size_34/8, output_size_34/7)
    # output_size_45 = 384 * 13 * 13 * 4 / (1024 * 1024)
    # G.edges[(4, 5)]['attr'] = trans(output_size_45 / 10, output_size_45 /8, output_size_45/7)
    # output_size_56 = 256 * 13 * 13 * 4 / (1024 * 1024)
    # G.edges[(5, 6)]['attr'] = trans(output_size_56 / 10, output_size_56 /8, output_size_56/7)
    # output_size_67 = 256 * 13 * 13 * 4 / (1024 * 1024)
    # G.edges[(6, 7)]['attr'] = trans(output_size_67 / 10, output_size_67 / 8, output_size_67/7)
    # output_size_78 = 9216 * 1 * 1 * 4 / (1024 * 1024)
    # G.edges[(7, 8)]['attr'] = trans(output_size_78 / 10, output_size_78 /8, output_size_78/7)
    # output_size_89 = 4096 * 1 * 1 * 4 / (1024 * 1024)
    # G.edges[(8, 9)]['attr'] = trans(output_size_89 / 10, output_size_89 / 8, output_size_89/7)
    # output_size_910 = 4096 * 1 * 1 * 4 / (1024 * 1024)
    # G.edges[(9, 10)]['attr'] = trans(output_size_910 / 10, output_size_910 / 8, output_size_910/7)
    #
    # G.nodes['input']['attr'] = 'device'
    # input_size = 3 * 224 * 224 * 4 / (1024 * 1024)
    # G.edges[('input', 0)]['attr'] = trans(input_size / 6, input_size / 3, input_size)
    # G.edges[(10, 'output')]['attr'] = trans(100, 100, 100)
    # print(G.edges(data=True))
    #
    # G = assign_nodes_to_layers(G, layer_dict)
    # for node in G.nodes:
    #     print('Node %s is at %s' % (str(node), G.nodes[node].get('location')))
    #
    # pos = nx.spring_layout(G)
    # labels = nx.get_node_attributes(G, 'location')
    # nx.draw_networkx_nodes(G, pos=pos)
    # nx.draw_networkx_labels(G, pos=pos, labels=labels)
    # nx.draw_networkx_edges(G, pos=pos, arrows=True)
    # plt.show()