import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from graphviz import Digraph
from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
from dnn_split.model_util import get_all_layers
from dnn_models.mynet import MyNet
from networkx.drawing.nx_pydot import graphviz_layout
import time


def make_graph(var, params):
    param_map = {id(v): k for k, v in params.items()}
    print(param_map)
    id_counter = 0
    param_list = []

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')

    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    G = nx.DiGraph()
    G_compute = nx.DiGraph()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    output_nodes = (var.grad_fn,) if not isinstance(var, tuple) else tuple(v.grad_fn for v in var)

    def add_nodes(var):
        nonlocal id_counter
        nonlocal param_list
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
                G_compute.add_node(str(id(var)), name=param_map.get(id(var)), attr=size_to_str(var.size()))

            elif hasattr(var, 'variable'):
                u = var.variable
                print("variable1 ", var)
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')

                G_compute.add_node(str(id(var)), name=param_map.get(id(u)), attr=size_to_str(u.size()))
                param_list.append(str(id(var)))
            else:
                dot.node(str(id(var)), str(type(var).__name__))
                print(str(var))
                if str(type(var).__name__) != "TBackward" and str(type(var).__name__) != "ExpandBackward" and str(type(var).__name__) != "ViewBackward":
                    G.add_node(str(id(var)), id=id_counter, name=str(type(var).__name__))
                    G_compute.add_node(str(id(var)), id=id_counter, name=str(type(var).__name__))
                    id_counter = id_counter + 1

                else:
                    G_compute.add_node(str(id(var)), name=str(type(var).__name__))

            seen.add(var)


            if hasattr(var, 'next_functions'):
                # print(var.next_functions)
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        if str(type(u[0]).__name__) != "AccumulateGrad" and str(type(u[0]).__name__) != "TBackward" and str(type(u[0]).__name__) != "ExpandBackward":
                            G.add_edge(str(id(u[0])), str(id(var)))
                        G_compute.add_edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])

            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    G_compute.add_edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot, G, G_compute, param_list


def remove_empty_nodes(G):
    node_removal = []
    for node in G.nodes():
        if G.nodes[node] == {}:
            node_removal.append(node)
    for node in node_removal:
        parent = next(G.predecessors(node))
        G = nx.contracted_nodes(G, parent, node)

    return G


def make_summray(model):
    summary = OrderedDict()
    for layer in model.named_children():
        layer_name = layer[0]
        layer_func = layer[1]
        summary[layer_name] = layer_func

    return summary


def assign_func(G, G_compute, summary, param_dict):
    # Traverse the graph
    # roots = [n for n, d in G.in_degree() if d == 0]
    # tree = nx.bfs_tree(G, source=roots[0], reverse=False)
    # nodes = [roots[0]] + [v for u, v in tree.edges()]
    nodes = list(nx.topological_sort(G))

    for node in nodes:
        node_name = G.nodes[node].get('name')
        if node_name == 'MkldnnConvolutionBackward':
            type = 'conv'
            pred_id = list(G_compute.predecessors(node))
            print("pred ",pred_id)
            for id in pred_id:
                if id in param_dict:
                    type = G_compute.nodes[id].get('name').split(".")[0].split("'")[0]
                    print("type is", type)
                    break
            func = summary.get(type)
            G.nodes[node]['func'] = func
        elif node_name == 'MaxPool2DWithIndicesBackward':
            type = 'maxpool2d'
            pred_id = list(G_compute.predecessors(node))
            print("pred ", pred_id)
            for id in pred_id:
                if id in param_dict:
                    type = G_compute.nodes[id].get('name').split(".")[0].split("'")[0]
                    print("type is", type)
                    break
            func = summary.get(type)
            G.nodes[node]['func'] = func
        elif node_name == 'ReluBackward1':
            type = 'relu'
            func = summary.get(type)
            G.nodes[node]['func'] = func
        elif node_name == 'AddBackward0':
            type = 'sum'
            func = np.sum
            G.nodes[node]['func'] = func
        elif node_name == 'MulBackward0':
            type = 'product'
            func = np.prod
            G.nodes[node]['func'] = func
        elif node_name == 'DivBackward0':
            type = 'division'
            func = np.divide
            G.nodes[node]['func'] = func

    return G


def make_forward(G, G_compute, x):
    # Traverse the graph
    roots = [n for n, d in G.in_degree() if d == 0]
    tree = nx.bfs_tree(G, source=roots[0], reverse=False)
    nodes = [roots[0]] + [v for u, v in tree.edges()]

    for node in nodes:
        func = G.nodes[node].get('func')
        if func != None:
            pred_id = list(G.predecessors(node))
            if len(pred_id) == 0:
                res = func(x)
                G.nodes[node]['output'] = res
            elif len(pred_id) == 1:
                pred_res = G.nodes[pred_id[0]].get('output')
                res = func(pred_res)
                G.nodes[node]['output'] = res
            else:
                pred_res = []
                for id in pred_id:
                    pred_res.append(G.nodes[id].get('output'))
                res = func(pred_res)
                G.nodes[node]['output'] = res

    return G.nodes[nodes[len(nodes)-1]].get('output')


if __name__ == "__main__":
    inputs = torch.randn(1, 3, 224, 224)
    net = MyNet()
    net.eval()
    for i in range(10):
        start = time.time()
        res1 = net(inputs)
        end = time.time()
        print("time original is ", end - start)
    print("original result is ", res1)
    # print(get_all_layers(resnet18))
    y = net(Variable(inputs))
    dot, G, G_compute, param_list = make_graph(y, params=dict(net.named_parameters()))
    dot.view(filename="mynet", directory="../models/")
    G = remove_empty_nodes(G)

    summary = make_summray(net)

    G = assign_func(G, G_compute, summary, param_list)
    print("Graph is ", G.nodes(data=True))
    for i in range(10):
        start = time.time()
        res2 = make_forward(G, G_compute, inputs)
        end = time.time()
        print("time static graph is ", end - start)
    print("static graph result is ", res2)

    labels = nx.get_node_attributes(G, 'name')
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos=pos)
    nx.draw_networkx_labels(G, pos=pos, labels=labels)
    nx.draw_networkx_edges(G, pos=pos, arrows=True)
    plt.show()