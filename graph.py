from manim import Graph
import heapq

class AlgorithmGraph(Graph):
    def __init__(self, adjacency_list, directed=False, weighted=False, vertex_labels=None, **kwargs):
        """Initialize a graph from an adjacency list for algorithm visualization."""
        # Generate labels if not provided
        if vertex_labels is None:
            vertex_labels = self._generate_vertex_labels(list(adjacency_list.keys()))
        
        self.vertex_labels = vertex_labels
        
        vertices, edges, weights = self._parse_adjacency_list(
            adjacency_list, directed
        )
        
        super().__init__(vertices, edges, **kwargs)
        
        self.directed = directed
        self.weighted = weighted
        self.weights = weights
        
    @staticmethod
    def _generate_vertex_labels(vertices):
        """Generate A, B, C, ..., Z, AA, AB, ... labels for vertices."""
        labels = {}
        for i, v in enumerate(vertices):
            if i < 26:
                labels[v] = chr(65 + i)
            else:
                # AA, AB, AC, ... for vertices beyond Z
                first = (i // 26) - 1
                second = i % 26
                labels[v] = chr(65 + first) + chr(65 + second)
        return labels
    
    def _parse_adjacency_list(self, adj_list, directed):
        """Parse NetworkX-style adjacency list into vertices, edges, and weights."""
        vertices = list(adj_list.keys())
        edges = []
        weights = {}
        self.adj = {v: [] for v in vertices}
        
        for u, neighbors in adj_list.items():
            if isinstance(neighbors, dict):
                # Weighted: {v: weight}
                for v, weight in neighbors.items():
                    if directed or (v, u) not in edges:
                        edges.append((u, v))
                        weights[(u, v)] = weight
                        if not directed:
                            weights[(v, u)] = weight
                    self.adj[u].append(v)
            else:
                # Unweighted: [v, ...]
                for v in neighbors:
                    if directed or (v, u) not in edges:
                        edges.append((u, v))
                    self.adj[u].append(v)
        
        return vertices, edges, weights
    
    def bfs(self, start):
        """Breadth-First Search generator that yields events for animation."""
        state = {
            'visited': set(),
            'queue': [start],
            'current': None,
            'parent': {start: None},
            'distance': {start: 0}
        }
        
        while state['queue']:
            state['current'] = state['queue'].pop(0)
            yield ('visit', state['current'], state)
            state['visited'].add(state['current'])
            
            for neighbor in self.adj[state['current']]:
                if neighbor not in state['visited'] and neighbor not in state['queue']:
                    state['queue'].append(neighbor)
                    state['parent'][neighbor] = state['current']
                    state['distance'][neighbor] = state['distance'][state['current']] + 1
                    yield ('discover', neighbor, state['current'], state)
                    
    def dfs(self, start):
        """Depth-First Search generator that yields events for animation."""
        state = {
            'visited': set(),
            'stack': [start],
            'current': None,
            'parent': {start: None}
        }
        
        while state['stack']:
            state['current'] = state['stack'].pop()
            
            if state['current'] in state['visited']:
                continue
                
            yield ('visit', state['current'], state)
            state['visited'].add(state['current'])
            
            for neighbor in self.adj[state['current']]:
                if neighbor not in state['visited']:
                    state['stack'].append(neighbor)
                    if neighbor not in state['parent']:
                        state['parent'][neighbor] = state['current']
                        yield ('discover', neighbor, state['current'], state)
    
    def dijkstra(self, start):
        """Dijkstra's shortest path algorithm generator that yields events for animation."""
        state = {
            'visited': set(),
            'distances': {v: float('inf') for v in self.adj},
            'parent': {start: None},
            'current': None,
            'pq': [(0, start)]
        }
        state['distances'][start] = 0
        
        while state['pq']:
            current_dist, state['current'] = heapq.heappop(state['pq'])
            
            if state['current'] in state['visited']:
                continue
            
            yield ('visit', state['current'], state)
            state['visited'].add(state['current'])
            
            for neighbor in self.adj[state['current']]:
                if neighbor not in state['visited']:
                    edge_weight = self.weights.get((state['current'], neighbor), 1)
                    new_dist = state['distances'][state['current']] + edge_weight
                    
                    # First time discovering this neighbor
                    if state['distances'][neighbor] == float('inf'):
                        yield ('discover', neighbor, state['current'], state)
                    
                    # Check if we found a better path
                    if new_dist < state['distances'][neighbor]:
                        state['distances'][neighbor] = new_dist
                        state['parent'][neighbor] = state['current']
                        heapq.heappush(state['pq'], (new_dist, neighbor))
                        yield ('update', neighbor, state['current'], new_dist, state)
    
    def bellman_ford(self, start):
        """Bellman-Ford algorithm generator that yields events for animation."""
        state = {
            'distances': {v: float('inf') for v in self.adj},
            'parent': {start: None},
            'iteration': 0
        }
        state['distances'][start] = 0
        
        yield ('init', start, 0, state)
        
        # Get all edges
        edges = []
        for u in self.adj:
            for v in self.adj[u]:
                weight = self.weights.get((u, v), 1)
                edges.append((u, v, weight))
        
        # Relax edges |V| - 1 times
        for i in range(len(self.adj) - 1):
            state['iteration'] = i + 1
            yield ('iteration_start', i + 1, state)
            
            for u, v, weight in edges:
                if state['distances'][u] != float('inf') and state['distances'][u] + weight < state['distances'][v]:
                    state['distances'][v] = state['distances'][u] + weight
                    state['parent'][v] = u
                    yield ('relax', v, u, state['distances'][v], state)
        
        # Check for negative cycles
        for u, v, weight in edges:
            if state['distances'][u] != float('inf') and state['distances'][u] + weight < state['distances'][v]:
                yield ('negative_cycle', (u, v), state)