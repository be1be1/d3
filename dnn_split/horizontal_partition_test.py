import itertools
import networkx as nx
import matplotlib.pyplot as plt
from collections import OrderedDict
from dnn_split.horizontal_partition import *

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

def build_alex_graph(d2e, e2c, d2c):
    G = nx.DiGraph()
    node_list = list(range(11))
    G.add_nodes_from(node_list)
    for i in range(10):
        G.add_edge(i, i+1)

    layer_dict = get_layer(G)

    G.add_node('input')
    G.add_node('output')
    G.add_edge('input', 0)
    G.add_edge(10, 'output')
    G.nodes['input']['attr'] = impl(0, 0, 0)
    G.nodes[0]['attr'] = impl(0.0387906074523925, 0.00161535739898681, 0.000158524800837039)
    G.nodes[1]['attr'] = impl(0.00837891101837158, 0.000992012023925781, 0.000104918397590518)
    G.nodes[2]['attr'] = impl(0.0532735347747802, 0.000886416435241699, 0.000335372802615165)
    G.nodes[3]['attr'] = impl(0.00678062438964843, 0.000534486770629882, 0.0000306591999717056)
    G.nodes[4]['attr'] = impl(0.025732421875, 0.00070655345916748, 0.000196169601380825)
    G.nodes[5]['attr'] = impl(0.0373493671417236, 0.000864291191101074, 0.000201350398361682)
    G.nodes[6]['attr'] = impl(0.0266769647598266, 0.00054333209991455, 0.000142950402200222)
    G.nodes[7]['attr'] = impl(0.00391316413879394, 0.000162029266357421, 0.0000364096000790596)
    G.nodes[8]['attr'] = impl(0.0905773401260376, 0.00514111518859863, 0.000916819202899932)
    G.nodes[9]['attr'] = impl(0.05386643409729, 0.00220365524291992, 0.000424079996347427)
    G.nodes[10]['attr'] = impl(0.0138338804244995, 0.000534629821777343, 0.00014015680104494)
    G.nodes['output']['attr'] = impl(0, 0, 0)

    d2e = d2c
    e2c = e2c
    d2c = d2c

    output_size_01 = 5.908203125
    G.edges[(0, 1)]['attr'] = trans(output_size_01 / d2e, output_size_01 / e2c, output_size_01 / d2c)
    output_size_12 = 1.423828125
    G.edges[(1, 2)]['attr'] = trans(output_size_12 / d2e, output_size_12 / e2c, output_size_12 / d2c)
    output_size_23 = 4.271484375
    G.edges[(2, 3)]['attr'] = trans(output_size_23 / d2e, output_size_23 / e2c, output_size_23 / d2c)
    output_size_34 = 0.990234375
    G.edges[(3, 4)]['attr'] = trans(output_size_34 / d2e, output_size_34 / e2c, output_size_34 / d2c)
    output_size_45 = 1.98046875
    G.edges[(4, 5)]['attr'] = trans(output_size_45 / d2e, output_size_45 / e2c, output_size_45 / d2c)
    output_size_56 = 1.3203125
    G.edges[(5, 6)]['attr'] = trans(output_size_56 / d2e, output_size_56 / e2c, output_size_56 / d2c)
    output_size_67 = 1.3203125
    G.edges[(6, 7)]['attr'] = trans(output_size_67 / d2e, output_size_67 / e2c, output_size_67 / d2c)
    output_size_78 = 0.28125
    G.edges[(7, 8)]['attr'] = trans(output_size_78 / d2e, output_size_78 / e2c, output_size_78 / d2c)
    output_size_89 = 0.125
    G.edges[(8, 9)]['attr'] = trans(output_size_89 / d2e, output_size_89 / e2c, output_size_89 / d2c)
    output_size_910 = 0.125
    G.edges[(9, 10)]['attr'] = trans(output_size_910 / d2e, output_size_910 / e2c, output_size_910 / d2c)

    G.nodes['input']['location'] = 'device'
    input_size = 4.59375
    G.edges[('input', 0)]['attr'] = trans(input_size / d2e, input_size / e2c, input_size/d2c)
    G.edges[(10, 'output')]['attr'] = trans(10000, 10000, 10000)

    return G, layer_dict


def build_vgg_graph(d2e, e2c, d2c):
    G = nx.DiGraph()
    node_list = list(range(21))
    G.add_nodes_from(node_list)
    for i in range(20):
        G.add_edge(i, i+1)

    layer_dict = get_layer(G)

    G.add_node('input')
    G.add_node('output')
    G.add_edge('input', 0)
    G.add_edge(20, 'output')
    G.nodes['input']['attr'] = impl(0, 0, 0)
    G.nodes[0]['attr'] = impl(0.0609938383102417, 0.00338995456695556, 0.0000605106353759765)
    G.nodes[1]['attr'] = impl(0.592135882377624, 0.00845353603363037, 0.0000553369522094726)
    G.nodes[2]['attr'] = impl(0.0820929288864135, 0.00880284309387207, 0.0000228643417358398)
    G.nodes[3]['attr'] = impl(0.226624870300292, 0.00373132228851318, 0.0000540971755981445)
    G.nodes[4]['attr'] = impl(0.448662877082824, 0.00653448104858398, 0.0000515937805175781)
    G.nodes[5]['attr'] = impl(0.0405383110046386, 0.00466926097869873, 0.0000332355499267578)
    G.nodes[6]['attr'] = impl(0.176293754577636, 0.00315544605255126, 0.0000571727752685546)
    G.nodes[7]['attr'] = impl(0.340849637985229, 0.00554869174957275, 0.0000574827194213867)
    G.nodes[8]['attr'] = impl(0.334842944145202, 0.00576448440551757, 0.0000536680221557617)
    G.nodes[9]['attr'] = impl(0.0198351860046386, 0.0023181676864624, 0.0000220775604248046)
    G.nodes[10]['attr'] = impl(0.150825953483581, 0.0032393455505371, 0.0000629425048828125)
    G.nodes[11]['attr'] = impl(0.295534491539001, 0.00640218257904052, 0.0000623226165771484)
    G.nodes[12]['attr'] = impl(0.295758509635925, 0.00628831386566162, 0.0000530242919921875)
    G.nodes[13]['attr'] = impl(0.0102149486541748, 0.00123481750488281, 0.0000325918197631835)
    G.nodes[14]['attr'] = impl(0.0862767219543457, 0.00252645015716552, 0.0000523328781127929)
    G.nodes[15]['attr'] = impl(0.0842627763748169, 0.00222053527832031, 0.0000505447387695312)
    G.nodes[16]['attr'] = impl(0.0855239391326904, 0.00214335918426513, 0.0000607967376708984)
    G.nodes[17]['attr'] = impl(0.00350527763366699, 0.000349688529968261, 0.0000220775604248046)
    G.nodes[18]['attr'] = impl(0.330885338783264, 0.0138950824737548, 0.0000518321990966796)
    G.nodes[19]['attr'] = impl(0.0538443565368652, 0.00220961570739746, 0.0000421762466430664)
    G.nodes[20]['attr'] = impl(0.0138441324234008, 0.000549554824829101, 0.000037240982055664)
    G.nodes['output']['attr'] = impl(0, 0, 0)

    d2e = d2e
    e2c = e2c
    d2c = d2c

    output_size_01 = 98
    G.edges[(0, 1)]['attr'] = trans(output_size_01 / d2e, output_size_01 / e2c, output_size_01 / d2c)
    output_size_12 = 98
    G.edges[(1, 2)]['attr'] = trans(output_size_12 / d2e, output_size_12 / e2c, output_size_12 / d2c)
    output_size_23 = 24.5
    G.edges[(2, 3)]['attr'] = trans(output_size_23 / d2e, output_size_23 / e2c, output_size_23 / d2c)
    output_size_34 = 49
    G.edges[(3, 4)]['attr'] = trans(output_size_34 / d2e, output_size_34 / e2c, output_size_34 / d2c)
    output_size_45 = 49
    G.edges[(4, 5)]['attr'] = trans(output_size_45 / d2e, output_size_45 / e2c, output_size_45 / d2c)
    output_size_56 = 12.25
    G.edges[(5, 6)]['attr'] = trans(output_size_56 / d2e, output_size_56 / e2c, output_size_56 / d2c)
    output_size_67 = 24.5
    G.edges[(6, 7)]['attr'] = trans(output_size_67 / d2e, output_size_67 / e2c, output_size_67 / d2c)
    output_size_78 = 24.5
    G.edges[(7, 8)]['attr'] = trans(output_size_78 / d2e, output_size_78 / e2c, output_size_78 / d2c)
    output_size_89 = 24.5
    G.edges[(8, 9)]['attr'] = trans(output_size_89 / d2e, output_size_89 / e2c, output_size_89 / d2c)
    output_size_910 = 6.125
    G.edges[(9, 10)]['attr'] = trans(output_size_910 / d2e, output_size_910 / e2c, output_size_910 / d2c)
    output_size_1011 = 12.25
    G.edges[(10, 11)]['attr'] = trans(output_size_1011 / d2e, output_size_1011 / e2c, output_size_1011 / d2c)
    output_size_1112 = 12.25
    G.edges[(11, 12)]['attr'] = trans(output_size_1112 / d2e, output_size_1112 / e2c, output_size_1112 / d2c)
    output_size_1213 = 12.25
    G.edges[(12, 13)]['attr'] = trans(output_size_1213 / d2e, output_size_1213 / e2c, output_size_1213 / d2c)
    output_size_1314 = 3.0625
    G.edges[(13, 14)]['attr'] = trans(output_size_1314 / d2e, output_size_1314 / e2c, output_size_1314 / d2c)
    output_size_1415 = 3.0625
    G.edges[(14, 15)]['attr'] = trans(output_size_1415 / d2e, output_size_1415 / e2c, output_size_1415 / d2c)
    output_size_1516 = 3.0625
    G.edges[(15, 16)]['attr'] = trans(output_size_1516 / d2e, output_size_1516 / e2c, output_size_1516 / d2c)
    output_size_1617 = 3.0625
    G.edges[(16, 17)]['attr'] = trans(output_size_1617 / d2e, output_size_1617 / e2c, output_size_1617 / d2c)
    output_size_1718 = 0.765625
    G.edges[(17, 18)]['attr'] = trans(output_size_1718 / d2e, output_size_1718 / e2c, output_size_1718 / d2c)
    output_size_1819 = 0.125
    G.edges[(18, 19)]['attr'] = trans(output_size_1819 / d2e, output_size_1819 / e2c, output_size_1819 / d2c)
    output_size_1920 = 0.125
    G.edges[(19, 20)]['attr'] = trans(output_size_1920 / d2e, output_size_1920 / e2c, output_size_1920 / d2c)

    G.nodes['input']['location'] = 'device'
    input_size = 4.59375
    G.edges[('input', 0)]['attr'] = trans(input_size / d2e, input_size / e2c, input_size/d2c)
    G.edges[(20, 'output')]['attr'] = trans(10000, 10000, 10000)

    return G, layer_dict


def build_inception_graph(d2e, e2c, d2c):
    G = nx.DiGraph()
    node_list = list(range(24))
    G.add_nodes_from(node_list)
    for i in range(23):
        G.add_edge(i, i+1)

    layer_dict = get_layer(G)

    G.add_node('input')
    G.add_node('output')
    G.add_edge('input', 0)
    G.add_edge(23, 'output')
    G.nodes['input']['attr'] = impl(0, 0, 0)
    G.nodes[0]['attr'] = impl(0.01985724, 0.000497532, 0.000126123)
    G.nodes[1]['attr'] = impl(0.042263031, 0.000740719, 0.000103807)
    G.nodes[2]['attr'] = impl(0.073670936, 0.001400018, 9.36E-05)
    G.nodes[3]['attr'] = impl(0.078483748, 0.004977775, 0.000189853)
    G.nodes[4]['attr'] = impl(0.206167006, 0.003538108, 0.002597237)
    G.nodes[5]['attr'] = impl(0.067958188, 0.003599906, 0.000158358)
    G.nodes[6]['attr'] = impl(0.124842119, 0.002338004, 0.005271482)
    G.nodes[7]['attr'] = impl(0.138248348, 0.002236056, 0.003134561)
    G.nodes[8]['attr'] = impl(0.13957026, 0.002309346, 0.006592107)
    G.nodes[9]['attr'] = impl(0.138262939, 0.002335835, 0.003136873)
    G.nodes[10]['attr'] = impl(0.177802181, 0.004854488, 0.003598857)
    G.nodes[11]['attr'] = impl(0.247489667, 0.003648567, 0.009092951)
    G.nodes[12]['attr'] = impl(0.273446012, 0.003607512, 0.009114122)
    G.nodes[13]['attr'] = impl(0.267560506, 0.003571081, 0.008089685)
    G.nodes[14]['attr'] = impl(0.218984056, 0.003588629, 0.001395845)
    G.nodes[15]['attr'] = impl(0.300414777, 0.003548193, 0.001451564)
    G.nodes[16]['attr'] = impl(0.320718908, 0.003593445, 0.001513076)
    G.nodes[17]['attr'] = impl(0.237069464, 0.003683615, 0.001054978)
    G.nodes[18]['attr'] = impl(0.108347154, 0.002796745, 0.000617337)
    G.nodes[19]['attr'] = impl(0.087156343, 0.002951241, 0.001044345)
    G.nodes[20]['attr'] = impl(0.085878754, 0.002769732, 0.003273726)
    G.nodes[21]['attr'] = impl(0.083295107, 0.002769542, 0.005679941)
    G.nodes[22]['attr'] = impl(0.003227234, 1.73E-05, 1.52E-05)
    G.nodes[23]['attr'] = impl(0.038788509, 0.006604266, 0.006083632)
    G.nodes['output']['attr'] = impl(0, 0, 0)

    d2e = d2e
    e2c = e2c
    d2c = d2c

    output_size_01 = 12.03222656
    G.edges[(0, 1)]['attr'] = trans(output_size_01 / d2e, output_size_01 / e2c, output_size_01 / d2c)
    output_size_12 = 11.60253906
    G.edges[(1, 2)]['attr'] = trans(output_size_12 / d2e, output_size_12 / e2c, output_size_12 / d2c)
    output_size_23 = 23.20507813
    G.edges[(2, 3)]['attr'] = trans(output_size_23 / d2e, output_size_23 / e2c, output_size_23 / d2c)
    output_size_34 = 14.23828125
    G.edges[(3, 4)]['attr'] = trans(output_size_34 / d2e, output_size_34 / e2c, output_size_34 / d2c)
    output_size_45 = 15.84375
    G.edges[(4, 5)]['attr'] = trans(output_size_45 / d2e, output_size_45 / e2c, output_size_45 / d2c)
    output_size_56 = 7.32421875
    G.edges[(5, 6)]['attr'] = trans(output_size_56 / d2e, output_size_56 / e2c, output_size_56 / d2c)
    output_size_67 = 7.32421875
    G.edges[(6, 7)]['attr'] = trans(output_size_67 / d2e, output_size_67 / e2c, output_size_67 / d2c)
    output_size_78 = 7.32421875
    G.edges[(7, 8)]['attr'] = trans(output_size_78 / d2e, output_size_78 / e2c, output_size_78 / d2c)
    output_size_89 = 7.32421875
    G.edges[(8, 9)]['attr'] = trans(output_size_89 / d2e, output_size_89 / e2c, output_size_89 / d2c)
    output_size_910 = 7.32421875
    G.edges[(9, 10)]['attr'] = trans(output_size_910 / d2e, output_size_910 / e2c, output_size_910 / d2c)
    output_size_1011 = 4.5
    G.edges[(10, 11)]['attr'] = trans(output_size_1011 / d2e, output_size_1011 / e2c, output_size_1011 / d2c)
    output_size_1112 = 4.5
    G.edges[(11, 12)]['attr'] = trans(output_size_1112 / d2e, output_size_1112 / e2c, output_size_1112 / d2c)
    output_size_1213 = 4.5
    G.edges[(12, 13)]['attr'] = trans(output_size_1213 / d2e, output_size_1213 / e2c, output_size_1213 / d2c)
    output_size_1314 = 4.5
    G.edges[(13, 14)]['attr'] = trans(output_size_1314 / d2e, output_size_1314 / e2c, output_size_1314 / d2c)
    output_size_1415 = 4.5
    G.edges[(14, 15)]['attr'] = trans(output_size_1415 / d2e, output_size_1415 / e2c, output_size_1415 / d2c)
    output_size_1516 = 4.5
    G.edges[(15, 16)]['attr'] = trans(output_size_1516 / d2e, output_size_1516 / e2c, output_size_1516 / d2c)
    output_size_1617 = 4.5
    G.edges[(16, 17)]['attr'] = trans(output_size_1617 / d2e, output_size_1617 / e2c, output_size_1617 / d2c)
    output_size_1718 = 4.5
    G.edges[(17, 18)]['attr'] = trans(output_size_1718 / d2e, output_size_1718 / e2c, output_size_1718 / d2c)
    output_size_1819 = 1.171875
    G.edges[(18, 19)]['attr'] = trans(output_size_1819 / d2e, output_size_1819 / e2c, output_size_1819 / d2c)
    output_size_1920 = 1.171875
    G.edges[(19, 20)]['attr'] = trans(output_size_1920 / d2e, output_size_1920 / e2c, output_size_1920 / d2c)
    output_size_2021 = 1.171875
    G.edges[(20, 21)]['attr'] = trans(output_size_2021 / d2e, output_size_2021 / e2c, output_size_2021 / d2c)
    output_size_2122 = 1.171875
    G.edges[(21, 22)]['attr'] = trans(output_size_2122 / d2e, output_size_2122 / e2c, output_size_2122 / d2c)
    output_size_2223 = 0.046875
    G.edges[(22, 23)]['attr'] = trans(output_size_2223 / d2e, output_size_2223 / e2c, output_size_2223 / d2c)

    G.nodes['input']['location'] = 'device'
    input_size = 4.59375
    G.edges[('input', 0)]['attr'] = trans(input_size / d2e, input_size / e2c, input_size/d2c)
    G.edges[(23, 'output')]['attr'] = trans(10000, 10000, 10000)

    return G, layer_dict

def calc_latency(G_assigned, d2e, e2c, d2c):
    pred = 0
    latency = 0
    for node in G_assigned.nodes:
        cur = G_assigned.nodes[node].get('location')
        if cur != pred:
            if pred == 'device' and cur == 'edge':
                latency = latency + G_assigned.edges[(pred, node)].get('attr').d2e / d2e
            if pred == 'edge' and cur == 'cloud':
                latency = latency + G_assigned.edges[(pred, node)].get('attr').e2c / e2c
            if pred == 'device' and cur == 'cloud':
                latency = latency + G_assigned.edges[(pred, node)].get('attr').d2c / d2c

        if cur == 'device':
            latency = latency + G_assigned.nodes[node].get('attr').device
        if cur == 'edge':
            latency = latency + G_assigned.nodes[node].get('attr').edge
        if cur == 'cloud':
            latency = latency + G_assigned.nodes[node].get('attr').cloud

    return latency

if __name__ == '__main__':
    d2e=64.95
    e2c=31.53
    d2c=29.78
    alex_G, alex_layer_dict = build_alex_graph(d2e, e2c, d2c)
    print(alex_layer_dict)
    # print(Alex_G.nodes[0]['attr'].device)
    # print(alex_G.nodes(data=True))
    Alex_G_assigned = assign_nodes_to_layers(alex_G, alex_layer_dict)
    for node in Alex_G_assigned.nodes:
        print('Node %s is at %s' % (str(node), Alex_G_assigned.nodes[node].get('location')))
    print('Latency is ', calc_latency(Alex_G_assigned, d2e, e2c, d2c))
    print('------------------------------------')

    vgg_G, vgg_layer_dict = build_vgg_graph(d2e, e2c, d2c)
    print(vgg_layer_dict)
    # print(Alex_G.nodes[0]['attr'].device)
    # print(alex_G.nodes(data=True))
    VGG_G_assigned = assign_nodes_to_layers(vgg_G, vgg_layer_dict)
    for node in VGG_G_assigned.nodes:
        print('Node %s is at %s' % (str(node), VGG_G_assigned.nodes[node].get('location')))
    print('Latency is ', calc_latency(VGG_G_assigned, d2e, e2c, d2c))
    print('------------------------------------')

    inception_G, inception_layer_dict = build_inception_graph(d2e, e2c, d2c)
    print(inception_layer_dict)
    # print(Alex_G.nodes[0]['attr'].device)
    # print(alex_G.nodes(data=True))
    inception_G_assigned = assign_nodes_to_layers(inception_G, inception_layer_dict)
    for node in inception_G_assigned.nodes:
        print('Node %s is at %s' % (str(node), inception_G_assigned.nodes[node].get('location')))
    print('Latency is ', calc_latency(inception_G_assigned, d2e, e2c, d2c))
    print('------------------------------------')