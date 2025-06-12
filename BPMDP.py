import networkx as nx
import numpy as np
from math import sqrt



class BoundedMDP:
    def __init__(self):
        self.history = []

    def intervals_intersect(self, interval1, interval2):
        """
        Check if two intervals intersect.

        Args:
        interval1: Tuple representing the first interval as (lower_bound, upper_bound).
        interval2: Tuple representing the second interval as (lower_bound, upper_bound).

        Returns:
        True if the intervals intersect, False otherwise.
        """
        lower_bound1, upper_bound1 = interval1
        lower_bound2, upper_bound2 = interval2

        # Check if intervals intersect
        if max(lower_bound1, lower_bound2) <= min(upper_bound1, upper_bound2):
            return True
        else:
            return False

    def is_within_twice(self, a, b):
        # 计算两个数的最大值和最小值
        max_val = max(a, b)
        min_val = min(a, b)

        # 判断最大值是否不超过最小值的两倍
        return max_val <= 2 * min_val

    def check_consistent(self,s, t, a, G, S, P, epsilon=0.1):
        """
        Input:
            s, t: states
            a: action
            G: graph with edges E
            S: list of states
            P: function P(s, a) -> dict of next state probs
            epsilon: allowed error
        Output:
            True if transition distributions are close on all components
        """
        components = list(nx.connected_components(G))

        total_diff = 0.0
        for C in components:
            ps = sum(P[s][a][s_next] for s_next in C)
            pt = sum(P[t][a][s_next] for s_next in C)
            if (not ps == 1) or  (not pt == 1):
                print("fds")
            total_diff += abs(ps - pt)

        return total_diff <= epsilon

    def compute_aggregation(self, maze, history=0, n_actions=0):
        # Build the graph G
        num_states = maze.nb_states
        num_actions = maze.nb_actions
        G = self.build_graph(num_states, num_actions, maze)
        #actions = range(n_actions)
        states = range(num_states)
        actions = list(range(num_actions))
        # history_indices = {h.idx: h for h in history}

        # Repeat until no edge is deleted

        deleted = True
        while deleted:
            deleted = False
            edges_to_remove = []
            for (s, t) in list(G.edges()):
                for a in range(num_actions):
                    if not self.check_consistent(s, t, a, G, range(num_states), maze.transition_function, epsilon=0.1):
                        edges_to_remove.append((s, t))
                        deleted = True
                        break
            G.remove_edges_from(edges_to_remove)

        # Step 3: connected components = equivalence classes
        components = list(nx.connected_components(G))

        # Step 4: check diameter
        max_diameter = max(
            nx.diameter(G.subgraph(c)) if len(c) > 1 else 0
            for c in components
        )

        if max_diameter >= np.sqrt(num_states):
            return [set(range(num_states))]
        else:
            return components


    def build_graph(self, num_states, num_actions, maze):
        G = nx.Graph()
        states = range(num_states)
        G.add_nodes_from(states)
        for i in range(num_states):
            for j in range(i + 1, num_states):
                for a in range(num_actions):
                    #if history[i].reward_R[a] == history[j].reward_R[a]:  #如果奖励不相等，即交集为空，则不连边
                    if maze.reward_function[i] == maze.reward_function[j]:
                        G.add_edge(i, j)       # 暂时设定奖励相等
        return G