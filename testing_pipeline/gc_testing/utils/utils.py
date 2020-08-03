import numpy as np
import networkx as nx
from Gcf import fstat

def construct_graph_gcf(rmse_ur, rmse_r):
    rmse_keys = list(rmse_ur.keys())
    #graph_dict = collections.defaultdict(dict)
    graph_dict = {k:[] for k in rmse_ur}
    print(graph_dict)
    for k in rmse_keys:
        cand = rmse_ur[k]
        #graph_dict[k] = []
        for c in cand:
            fs = fstat(rmse_ur[k][c], rmse_r[k][c])
            if fs > 0.05:
                graph_dict[c].append(k)
        #if graph_dict[c] == []:
        #    graph_dict.pop(k, None)
    graph_dict = {k:v for (k,v) in graph_dict.items() if v != []}
    G = nx.DiGraph(graph_dict, directed=True)
    return G, graph_dict

"""
G, g_dict = construct_graph(model_ur_rmse,model_r_rmse)

    options = {
    'node_size': 1000,
    'width': 1,
    'arrowstyle': '-|>',
    'arrowsize': 10,
    }

    pos = nx.shell_layout(G)
    nx.draw(G, pos, with_labels=True, **options)
"""