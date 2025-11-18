from manim import *
from graph import AlgorithmGraph
import numpy as np
from manim import config as manim_config
from manim import tempconfig


class AnimationConfig:
    """Configuration for animation colors and timing"""
    # Dijkstra colors
    DIJKSTRA_UNDISCOVERED = BLACK   # Hollow (fill_opacity=1, fill_color=BLACK)
    DIJKSTRA_IN_PQ = YELLOW_C   # In priority queue
    DIJKSTRA_UPDATED = GREEN_C  # Just updated this step
    DIJKSTRA_FINALIZED = BLUE_C # Popped from PQ, shortest path confirmed
    
    # Bellman-Ford colors  
    BF_UNREACHED = BLACK    # Distance = infinity
    BF_REACHED = YELLOW_C   # Has finite distance
    BF_UPDATED = GREEN_C    # Just relaxed this iteration
    BF_FINALIZED = BLUE_C   # All iterations complete
    
    # BFS/DFS colors
    START_COLOR = RED
    VISITED_COLOR = BLUE_C
    DISCOVERED_COLOR = YELLOW_C
    
    # Edge colors
    TREE_COLOR = GREEN_C
    EDGE_FLASH_COLOR = YELLOW
    ANIMATION_SPEED = 0.75


class GraphAlgorithmAnimator:
    def __init__(self, scene, config=None):
        self.scene = scene
        self.config = config or AnimationConfig()
        self.start_node = None
        self.tree_edges = set()
        
        self.dist_table = None  # the Table mobject
        self.dist_entries = {}  # (vertex, "dist"/"prev") -> MathTex
        self.row_for_vertex = {}    # vertex -> row index
        
        self.updated_this_step = set() # for djikstra bellman ford


    # STATIC/UTILITY METHODS

    @staticmethod
    def _is_tree(adjacency_list, directed):
        """Check if graph is a tree (connected, acyclic-ish heuristic)"""
        num_vertices = len(adjacency_list)
        num_edges = sum(len(neighbors) for neighbors in adjacency_list.values())
        if not directed:
            num_edges //= 2
        # Tree has exactly n-1 edges (this doesn't check connectivity fully)
        return num_edges == num_vertices - 1

    @staticmethod
    def _determine_best_layout(adjacency_list, directed):
        """Intelligently choose a layout based on graph properties"""
        num_vertices = len(adjacency_list) 

        if GraphAlgorithmAnimator._is_tree(adjacency_list, directed):
            return 'tree'
        if num_vertices <= 10:
            return 'circular'
        if num_vertices < 100:
            return 'kamada_kawai'
        if num_vertices < 500:
            return 'spring'
        return 'spring'
    
    def _tree_key(self, u, v, directed):
        """Generate a key for tree edge storage."""
        if directed:
            return (u, v)
        return (min(u, v), max(u, v))
    
    # GRAPH SETUP & INITIALIZATION

    def _add_vertex_labels(self, graph):
        """Add letter labels inside vertex circles"""
        for vertex_id in graph.vertices:
            v_mob = graph[vertex_id]
            label_text = graph.vertex_labels[vertex_id]
            
            # Create white text label
            label = Text(
                label_text, 
                font_size=20,
                color=WHITE,
                weight=NORMAL,
                font="Sans"
            )
            label.move_to(v_mob.get_center())
            
            self.scene.add(label)
            
            # Store label for later reference if needed
            if not hasattr(graph, 'vertex_label_mobs'):
                graph.vertex_label_mobs = {}
            graph.vertex_label_mobs[vertex_id] = label

    def _add_weight_labels(self, graph):
        """Add static weight labels aligned with their edges."""
        graph.weight_labels = {}

        for (u, v), w in graph.weights.items():
            # For undirected graphs, only label one direction
            if not graph.directed and u > v:
                continue

            e_mob = self._get_edge(graph, u, v)
            if e_mob is None:
                continue

            start = e_mob.get_start()
            end = e_mob.get_end()

            # Put label at midpoint of edge
            midpoint = (start + end) / 2

            # Direction vector along edge
            direction = end - start
            norm = np.linalg.norm(direction)
            
            if norm == 0:
                # Degenerate edge; just nudge upward a bit
                offset = np.array([0.0, 0.2, 0.0])
                angle = 0.0
            else:
                direction /= norm

                # Perpendicular (rotate 90° in the plane) for offset
                perpendicular = np.array([-direction[1], direction[0], 0.0])

                # Always push label "above" the edge (positive y-direction preferred)
                if perpendicular[1] < 0:
                    perpendicular *= -1

                # Distance to push the label away from the edge
                offset = 0.125 * perpendicular

                # Angle of the edge in the plane
                raw_angle = np.arctan2(direction[1], direction[0])

                # Normalize so text is never upside-down
                angle = raw_angle
                if angle > np.pi / 2:
                    angle -= np.pi
                elif angle < -np.pi / 2:
                    angle += np.pi

            label = MathTex(str(w)).scale(0.5)
            label.move_to(midpoint + offset)
            label.rotate(angle, axis=OUT)

            self.scene.add(label)
            graph.weight_labels[(u, v)] = label

    def _get_vertex(self, graph, v):
        """Get the Manim mobject for a vertex."""
        return graph[v]

    def _get_edge(self, graph, u, v):
        """Get the Manim mobject for an edge, handling directed/undirected cases."""
        
        # Try directed key first
        if (u, v) in graph.edges:
            return graph.edges[(u, v)]
        if (v, u) in graph.edges:
            return graph.edges[(v, u)]
        # Fallback for implementations that use frozenset for undirected
        fs = frozenset({u, v})
        if fs in graph.edges:
            return graph.edges[fs]
        return None

    def _highlight_start_node(self, graph):
        """Animate a halo effect around the starting vertex."""
        if self.start_node is None:
            return

        v_mob = self._get_vertex(graph, self.start_node)
        halo = Circle(color=self.config.START_COLOR).surround(v_mob).scale(1.1)

        # Color the start vertex with its own color and add a halo
        self.scene.play(
            v_mob.animate.set_fill(self.config.START_COLOR, opacity=1),
            Create(halo),
            run_time=self.config.ANIMATION_SPEED
        )
        # Keep the fill but fade out the halo
        self.scene.play(
            FadeOut(halo),
            run_time=self.config.ANIMATION_SPEED / 2
        )


    # TABLE MANAGEMENT (Dijkstra/Bellman-Ford)

    def _create_distance_table(self, graph, algorithm_name: str):
        """Create the distance/prev table for Dijkstra/Bellman-Ford."""
        vertices = sorted(graph.vertices)

        # Use empty strings for data cells
        data = [[r"\text{Vertex}", r"\text{Dist}", r"\text{Prev}"]]
        for v in vertices:
            label = graph.vertex_labels[v]
            data.append([label, "", ""])

        table = Table(
            data,
            include_outer_lines=True,
            h_buff=0.6,
            v_buff=0.3,
            element_to_mobject=MathTex,
        ).scale(0.5)

        table.to_edge(RIGHT).shift(LEFT * 0.5)

        # Reset bookkeeping
        self.dist_table = table
        self.dist_entries = {}
        self.row_for_vertex = {}

        # Add table + title to scene
        title_text = "Dijkstra" if algorithm_name == "dijkstra" else "Bellman–Ford"
        title = Text(title_text, font_size=24).next_to(table, UP)
        self.scene.play(FadeIn(table), FadeIn(title))

        # Now create our own dist/prev labels (these are NOT children of the table)
        for i, v in enumerate(vertices, start=2):  # rows are 1-indexed, header=1
            self.row_for_vertex[v] = i

            # Positions for the dist and prev cells
            dist_cell = table.get_cell((i, 2))
            prev_cell = table.get_cell((i, 3))

            # Initial values: infinity and "-"
            dist_tex = MathTex(r"\infty").scale(0.5).move_to(dist_cell.get_center())
            prev_tex = MathTex("-").scale(0.5).move_to(prev_cell.get_center())

            self.scene.add(dist_tex, prev_tex)

            self.dist_entries[(v, "dist")] = dist_tex
            self.dist_entries[(v, "prev")] = prev_tex
        
    def _update_distance_row(self, graph, vertex, state):
        """Update distance and parent values in the table for a vertex."""
        
        if self.dist_table is None:
            return

        dist = state["distances"][vertex]
        parent = state["parent"].get(vertex)

        dist_str = r"\infty" if dist == float("inf") else str(dist)
        parent_str = "-" if parent is None else graph.vertex_labels[parent]

        old_dist = self.dist_entries[(vertex, "dist")]
        old_prev = self.dist_entries[(vertex, "prev")]

        new_dist = MathTex(dist_str).scale(0.5).move_to(old_dist.get_center())
        new_prev = MathTex(parent_str).scale(0.5).move_to(old_prev.get_center())

        self.scene.play(
            ReplacementTransform(old_dist, new_dist),
            ReplacementTransform(old_prev, new_prev),
            run_time=self.config.ANIMATION_SPEED / 2,
        )

        # Update stored references
        self.dist_entries[(vertex, "dist")] = new_dist
        self.dist_entries[(vertex, "prev")] = new_prev

        # Highlight the row with background color
        row = self.row_for_vertex[vertex]
        row_group = VGroup(
            self.dist_table.get_cell((row, 1)),
            self.dist_table.get_cell((row, 2)),
            self.dist_table.get_cell((row, 3)),
        )
        
        # Store original colors
        original_colors = [cell.get_fill_color() for cell in row_group]
        
        # Highlight with yellow background, then fade back
        self.scene.play(
            *[cell.animate.set_fill(YELLOW, opacity=.9) for cell in row_group],
            run_time=self.config.ANIMATION_SPEED
        )
        self.scene.play(
            *[cell.animate.set_fill(original_colors[i], opacity=0) for i, cell in enumerate(row_group)],
            run_time=self.config.ANIMATION_SPEED
        )

    # ANIMATION ORCHESTRATION
        
    def animate(self, graph_or_adj, algorithm, start, **kwargs):
        """Main entry point to animate a graph algorithm from start vertex."""

        self.start_node = start
        self.tree_edges.clear()
        
        # Add title
        algo_title_map = {
            "bfs": "BFS",
            "dfs": "DFS",
            "dijkstra": "Dijkstra",
            "bellman_ford": "Bellman–Ford",
        }
        title_text = algo_title_map.get(algorithm, algorithm)
        self.scene_title = Text(title_text, font_size=48, color=WHITE)
        self.scene_title.to_edge(UP)
        self.scene.play(FadeIn(self.scene_title))

        # If adjacency list provided, create the graph
        if isinstance(graph_or_adj, dict):
            layout = kwargs.get(
                "layout",
                self._determine_best_layout(
                    graph_or_adj,
                    kwargs.get("directed", False),
                ),
            )

            # Set root_vertex when using tree layout
            extra_graph_kwargs = {}
            if layout == "tree":
                extra_graph_kwargs["root_vertex"] = start

            edge_config = {
            "stroke_width": 4,
            "buff": 0.25,      # > vertex radius (0.25) so arrows stop outside
            "tip_length": 0.15,
            }

            graph = AlgorithmGraph(
                adjacency_list=graph_or_adj,
                layout=layout,
                directed=kwargs.get("directed", False),
                weighted=kwargs.get("weighted", False),
                vertex_config={
                    "fill_color": BLACK,
                    "fill_opacity": 1,
                    "stroke_color": WHITE,
                    "stroke_width": 3,
                    "radius": 0.25,
                },
                edge_config=edge_config,
                edge_type=Arrow,
                **extra_graph_kwargs,
            )
            
            self.scene.play(Create(graph))
            self._add_vertex_labels(graph)
            
            if graph.weighted:
                self._add_weight_labels(graph)
            self._highlight_start_node(graph)
            
            if algorithm in ("dijkstra", "bellman_ford"):
                self._create_distance_table(graph, algorithm)
            self.scene.wait()
        else:
            graph = graph_or_adj
            if getattr(graph, "weighted", False):
                self._add_weight_labels(graph)
            self._highlight_start_node(graph)


        # Get the algorithm generator
        algo_method = getattr(graph, algorithm)
        algo_generator = algo_method(start)

        # Animate the algorithm
        self._animate_algorithm(graph, algo_generator)

    def _animate_algorithm(self, graph, generator):
        """Generic animation handler for any algorithm"""
        for event in generator:
            self._handle_event(graph, event)
        
        #self._highlight_tree(graph)
    
    def _handle_event(self, graph, event):
        """Map events emitted by algorithms to animations."""
        event_type = event[0]

        if event_type == 'visit':
            # BFS / DFS / Dijkstra
            node = event[1]
            state = event[2]
            self._animate_visit(graph, node, state)

        elif event_type == 'discover':
            # BFS / DFS / Dijkstra
            node = event[1]
            parent = event[2]
            state = event[3]
            self._animate_discover(graph, parent, node, state)

        elif event_type == 'update':
            # Dijkstra improved distance
            node = event[1]
            parent = event[2]
            new_dist = event[3]
            state = event[4]
            self._animate_update(graph, parent, node, new_dist, state)

        elif event_type == 'init':
            # Bellman–Ford initialization
            start = event[1]
            dist = event[2]
            state = event[3]
            self._animate_init(graph, start, dist, state)

        elif event_type == 'iteration_start':
            iteration = event[1]
            state = event[2]
            self._animate_iteration_start(graph, iteration, state) 
            
        elif event_type == "finish":
            node = event[1]
            state = event[2]
            self._animate_finish(graph, node, state)


        elif event_type == 'relax':
            v = event[1]
            u = event[2]
            new_dist = event[3]
            state = event[4]
            self._animate_relax(graph, u, v, new_dist, state)

        elif event_type == 'negative_cycle':
            (u, v) = event[1]
            state = event[2]
            self._animate_negative_cycle(graph, u, v, state)

        else:
            # Unknown event
            pass


    # EVENT HANDLERS - BFS/DFS/Dijkstra

    def _animate_visit(self, graph, node, state):
        """Animate visiting/finalizing a node (turns blue)."""

        # First: reset all "updated this step" nodes from green to yellow
        reset_anims = []
        for updated_node in self.updated_this_step:
            if updated_node != node:  # Don't reset the node we're about to visit
                updated_mob = self._get_vertex(graph, updated_node)
                reset_anims.append(
                    updated_mob.animate.set_fill(self.config.DIJKSTRA_IN_PQ, opacity=1)
                )
        
        if reset_anims:
            self.scene.play(*reset_anims, run_time=self.config.ANIMATION_SPEED / 2)
        
        # Clear the updated set
        self.updated_this_step.clear()
        
        # Now visit the current node, it turns blue 
        v_mob = self._get_vertex(graph, node)
        self.scene.play(
            v_mob.animate.set_fill(self.config.DIJKSTRA_FINALIZED, opacity=1),
            run_time=self.config.ANIMATION_SPEED
        )
        
    def _animate_finish(self, graph, node, state):
        """DFS post-order: node is completely processed, turn it blue."""
        v_mob = self._get_vertex(graph, node)
        self.scene.play(
            v_mob.animate.set_fill(self.config.VISITED_COLOR, opacity=1),
            run_time=self.config.ANIMATION_SPEED,
        )



    def _animate_discover(self, graph, parent, node, state):
        """Animate discovering a new node via an edge (turns yellow)."""

        # Remember this as a tree edge
        self.tree_edges.add(self._tree_key(parent, node, graph.directed))

        child_mob = self._get_vertex(graph, node)
        e_mob = self._get_edge(graph, parent, node)

        # Flash edge to show we're considering it
        anims = [
            child_mob.animate.set_fill(self.config.DIJKSTRA_IN_PQ, opacity=1),
        ]
        if e_mob is not None:
            anims.append(e_mob.animate.set_stroke(self.config.EDGE_FLASH_COLOR, width=8))

        self.scene.play(*anims, run_time=self.config.ANIMATION_SPEED)

        # Settle: edge becomes part of the tree
        if e_mob is not None:
            self.scene.play(
                e_mob.animate.set_stroke(self.config.TREE_COLOR, width=4),
                run_time=self.config.ANIMATION_SPEED / 2,
        )

    def _animate_update(self, graph, parent, node, new_dist, state):
        """Animate updating a node's distance via a better path (turns green)."""
        
        # update tree edge for this node
        self.tree_edges.add(self._tree_key(parent, node, graph.directed))

        v_mob = self._get_vertex(graph, node)
        e_mob = self._get_edge(graph, parent, node)

        # Flash edge
        anims = []
        if e_mob is not None:
            anims.append(e_mob.animate.set_stroke(self.config.EDGE_FLASH_COLOR, width=8))

        if anims:
            self.scene.play(*anims, run_time=self.config.ANIMATION_SPEED / 2)
        
        # Update distance table
        if getattr(self, "dist_table", None) is not None:
            self._update_distance_row(graph, node, state)

        # Node turns green
        self.updated_this_step.add(node)
        
        edge_anims = [v_mob.animate.set_fill(self.config.DIJKSTRA_UPDATED, opacity=1)]
        if e_mob is not None:
            edge_anims.append(e_mob.animate.set_stroke(self.config.TREE_COLOR, width=4))

        self.scene.play(
            *edge_anims,
            run_time=self.config.ANIMATION_SPEED / 2,
        )

    # EVENT HANDLERS - Bellman-Ford

    def _animate_init(self, graph, start, dist, state):
        """Animate Bellman-Ford initialization with start node."""

        v_mob = self._get_vertex(graph, start)
        # Start node gets reached (yellow)
        self.scene.play(
            v_mob.animate.set_fill(self.config.BF_REACHED, opacity=1),
            run_time=self.config.ANIMATION_SPEED
        )
        
        # table: show initial distances
        if self.dist_table is not None:
            for v in state["distances"]:
                self._update_distance_row(graph, v, state)

    def _animate_iteration_start(self, graph, iteration, state):
        """Animate the boundary between Bellman-Ford iterations."""

        # Reset all "updated last iteration" nodes from green to yellow
        reset_anims = []
        for updated_node in self.updated_this_step:
            updated_mob = self._get_vertex(graph, updated_node)
            reset_anims.append(
                updated_mob.animate.set_fill(self.config.BF_REACHED, opacity=1)
            )
        
        if reset_anims:
            self.scene.play(*reset_anims, run_time=self.config.ANIMATION_SPEED / 2)
        
        # Clear the updated set
        self.updated_this_step.clear()
        
        # Brief pause to show iteration boundary
        self.scene.wait(self.config.ANIMATION_SPEED / 2)

    def _animate_relax(self, graph, u, v, new_dist, state):
        """Animate relaxing an edge in Bellman-Ford (node turns green)."""

        # record relaxed edge as part of current best tree
        self.tree_edges.add(self._tree_key(u, v, graph.directed))

        v_mob = self._get_vertex(graph, v)
        e_mob = self._get_edge(graph, u, v)

        # Flash edge
        anims = []
        if e_mob is not None:
            anims.append(e_mob.animate.set_stroke(self.config.EDGE_FLASH_COLOR, width=8))

        if anims:
            self.scene.play(*anims, run_time=self.config.ANIMATION_SPEED / 2)
        
        # Update distance table
        self._update_distance_row(graph, v, state)

        # Node turns green
        self.updated_this_step.add(v)
        
        edge_anims = [v_mob.animate.set_fill(self.config.BF_UPDATED, opacity=1)]
        if e_mob is not None:
            edge_anims.append(e_mob.animate.set_stroke(self.config.TREE_COLOR, width=4))

        self.scene.play(
            *edge_anims,
            run_time=self.config.ANIMATION_SPEED / 2,
        )

    def _animate_negative_cycle(self, graph, u, v, state):
        """Animate detection of a negative cycle edge (turns red)."""
        
        e_mob = self._get_edge(graph, u, v)
        if e_mob:
            self.scene.play(
                e_mob.animate.set_stroke(RED, width=10),
                run_time=self.config.ANIMATION_SPEED
            )


    # ============================================================================
    # POST-PROCESSING
    # ============================================================================

    def _highlight_tree(self, graph):
        """Color all tree edges with TREE_COLOR to show the final structure."""
        anims = []
        for (u, v) in self.tree_edges:
            e_mob = self._get_edge(graph, u, v)
            if e_mob is not None:
                anims.append(e_mob.animate.set_stroke(self.config.TREE_COLOR, width=6))

        if anims:
            self.scene.play(*anims, run_time=self.config.ANIMATION_SPEED)


# ==============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTION
# ==============================================================================

def animate_algorithm(adjacency_list, algorithm, start, **kwargs):    
    class _AlgorithmScene(Scene):
        def construct(self):
            animator = GraphAlgorithmAnimator(self)
            animator.animate(adjacency_list, algorithm, start, **kwargs)
    
    # Configure and render with PNG export enabled
    with tempconfig({
        "save_pngs": True,           # Export individual frames
        "write_to_movie": True,      
        "disable_caching": True,
        "pixel_height": 1080,        # Higher quality
        "pixel_width": 1920,
        "frame_rate":60,            # Standard frame rate
    }):
        scene = _AlgorithmScene()
        scene.render()