import networkx as nx
import sys
import matplotlib.pyplot as plt

sys.path.append(".")

def generate_graph():
    DG = nx.DiGraph()
    DG.add_edge(2, 1)   # adds the nodes in order 2, 1
    DG.add_edge(1, 3)
    DG.add_edge(2, 4)
    DG.add_edge(1, 2)
    assert list(DG.successors(2)) == [1, 4]
    assert list(DG.edges) == [(2, 1), (2, 4), (1, 3), (1, 2)]
    
    G = nx.petersen_graph()
    subax1 = plt.subplot(121)
    nx.draw(G, with_labels=True, font_weight='bold')
    subax2 = plt.subplot(122)
    nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')

    plt.show()

generate_graph()