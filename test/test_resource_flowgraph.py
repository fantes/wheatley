import sys

sys.path.append(".")

#import pytest
from utils.resource_flowgraph import ResourceFlowGraph

def test_resource_flowgraph():
    rg = ResourceFlowGraph(4, True)
    a0 = rg.availability(3)
    assert a0 == 0
    assert rg.frontier == [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
    assert rg.find_max_pos(0)==3
    rg.consume(1,3,0,4)
    assert rg.frontier == [[0, 0, 1], [4, 1, 1], [4, 1, 1], [4, 1, 1]]

    a1 = rg.availability(1)
    assert a1 == 0
    assert rg.find_max_pos(0)==0
    rg.consume(2,1,0,6)
    assert rg.frontier == [[4, 1, 1], [4, 1, 1], [4, 1, 1], [6, 2, 1]]

    a2 = rg.availability(2)
    assert a2 == 4
    assert rg.find_max_pos(4)==2
    rg.consume(3,2,4,9)
    assert rg.frontier== [[4, 1, 1], [6, 2, 1], [9, 3, 1], [9, 3, 1]]

    a3 = rg.availability(4)
    assert a3 == 9
    assert rg.find_max_pos(11)==3
    rg.consume(4,4,11,17)
    assert rg.frontier==[[17, 4, 1], [17, 4, 1], [17, 4, 1], [17, 4, 1]]
