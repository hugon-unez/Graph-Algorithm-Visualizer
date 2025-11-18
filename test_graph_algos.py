from graph import AlgorithmGraph

def test_tree_graph():
    """Test all algorithms on a simple tree - all should give same distances"""
    
    # Create a simple tree:
    #       0
    #      / \
    #     1   2
    #    / \   \
    #   3   4   5
    
    # Unweighted tree
    adj_list = {
        0: [1, 2],
        1: [0, 3, 4],
        2: [0, 5],
        3: [1],
        4: [1],
        5: [2]
    }
    
    G = AlgorithmGraph(adjacency_list=adj_list, layout="tree", root_vertex=0)
    
    print("="*50)
    print("Testing BFS on unweighted tree")
    print("="*50)
    bfs_distances = {}
    for event in G.bfs(start=0):
        if event[0] == 'visit':
            vertex = event[1]
            state = event[2]
            bfs_distances[vertex] = state['distance'][vertex]
            print(f"Visit {vertex}, distance: {state['distance'][vertex]}")
    
    print("\nFinal BFS distances:", bfs_distances)
    
    print("\n" + "="*50)
    print("Testing DFS on unweighted tree")
    print("="*50)
    dfs_parent = {}
    for event in G.dfs(start=0):
        if event[0] == 'visit':
            vertex = event[1]
            state = event[2]
            dfs_parent[vertex] = state['parent'][vertex]
            print(f"Visit {vertex}, parent: {state['parent'][vertex]}")
    
    print("\nFinal DFS parents:", dfs_parent)
    
    # Expected distances for tree rooted at 0
    expected_distances = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2}
    
    print("\n" + "="*50)
    print("Verification")
    print("="*50)
    print(f"Expected distances: {expected_distances}")
    print(f"BFS distances match: {bfs_distances == expected_distances}")
    
    assert bfs_distances == expected_distances, "BFS distances incorrect!"
    print("BFS passed!")


def test_weighted_tree():
    """Test weighted algorithms on a weighted tree"""
    
    # Weighted tree:
    #       0
    #      /5 \3
    #     1    2
    #    /2 \7  \4
    #   3    4   5
    
    adj_list = {
        0: {1: 5, 2: 3},
        1: {0: 5, 3: 2, 4: 7},
        2: {0: 3, 5: 4},
        3: {1: 2},
        4: {1: 7},
        5: {2: 4}
    }
    
    G = AlgorithmGraph(adjacency_list=adj_list, weighted=True, layout="tree", root_vertex=0)
    
    print("\n" + "="*50)
    print("Testing Dijkstra on weighted tree")
    print("="*50)
    dijkstra_distances = {}
    for event in G.dijkstra(start=0):
        if event[0] == 'visit':
            vertex = event[1]
            state = event[2]
            dijkstra_distances[vertex] = state['distances'][vertex]
            print(f"Visit {vertex}, distance: {state['distances'][vertex]}")
        elif event[0] == 'update':
            neighbor, current, new_dist = event[1], event[2], event[3]
            print(f"  Update {neighbor} via {current}, new distance: {new_dist}")
    
    print("\n" + "="*50)
    print("Testing Bellman-Ford on weighted tree")
    print("="*50)
    bf_distances = {0: 0}  # Initialize with start vertex
    for event in G.bellman_ford(start=0):
        if event[0] == 'iteration_start':
            iteration = event[1]
            print(f"\nIteration {iteration}")
        elif event[0] == 'init':
            vertex, distance = event[1], event[2]
            print(f"Initialize {vertex}, distance: {distance}")
        elif event[0] == 'relax':
            vertex, parent, distance = event[1], event[2], event[3]
            bf_distances[vertex] = distance
            print(f"  Relax {vertex} via {parent}, distance: {distance}")
    
    # Expected distances from vertex 0
    expected_weighted_distances = {0: 0, 1: 5, 2: 3, 3: 7, 4: 12, 5: 7}
    
    print("\n" + "="*50)
    print("Verification")
    print("="*50)
    print(f"Expected distances: {expected_weighted_distances}")
    print(f"Dijkstra distances: {dijkstra_distances}")
    print(f"Bellman-Ford distances: {bf_distances}")
    print(f"Dijkstra correct: {dijkstra_distances == expected_weighted_distances}")
    print(f"Bellman-Ford correct: {bf_distances == expected_weighted_distances}")
    
    assert dijkstra_distances == expected_weighted_distances, "Dijkstra distances incorrect!"
    assert bf_distances == expected_weighted_distances, "Bellman-Ford distances incorrect!"
    print("Dijkstra passed!")
    print("Bellman-Ford passed!")


def test_simple_path():
    """Test on a simple linear path: 0-1-2-3"""
    adj_list = {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2]
    }
    
    # Provide explicit positions for linear layout
    layout = {0: [-3, 0, 0], 1: [-1, 0, 0], 2: [1, 0, 0], 3: [3, 0, 0]}
    
    G = AlgorithmGraph(
        adjacency_list=adj_list, 
        layout=layout
    )
    
    print("\n" + "="*50)
    print("Testing BFS on linear path")
    print("="*50)
    
    for event in G.bfs(start=0):
        print(event)
    
    print("Simple path test completed!")


if __name__ == "__main__":
    test_tree_graph()
    test_weighted_tree()
    test_simple_path()
    print("\n" + "="*50)
    print("ALL TESTS PASSED! ✓")
    print("="*50)