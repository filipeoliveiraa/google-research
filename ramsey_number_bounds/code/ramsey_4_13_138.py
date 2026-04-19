# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Finds a Ramsey number lower bound."""

import random
import time
from typing import FrozenSet, List, Optional, Set, Tuple
import igraph as ig
import numpy as np

# Define constants for clarity
K_CLIQUE = 4
K_ANTICLIQUE = 13
POPULATION_SIZE = 5  # A small, elite population
TABU_BURST_ITERS = (
    250  # Number of local search iterations per individual per cycle
)


class GraphSymbiote:
  """A smaller, parasitic graph to explore local topology on a host graph."""

  def __init__(
      self, host_state, infected_nodes, size = 20
  ):
    self.host_state = host_state
    self.size = min(len(infected_nodes), size)
    # Infect the most problematic nodes
    node_vils = np.zeros(host_state.n)
    for cset in host_state.clique4_sets:
      for node in cset:
        node_vils[node] += 1
    for aset in host_state.anticlique13_sets:
      for node in aset:
        node_vils[node] += 5  # anticliques are harder

    # Get top nodes based on violations, plus some random context neighbors
    infected_vils = {node: node_vils[node] for node in infected_nodes}
    sorted_nodes = sorted(infected_vils, key=infected_vils.get, reverse=True)

    core_size = int(self.size * 0.8)
    context_size = self.size - core_size

    self.nodes = sorted_nodes[:core_size]

    # Add some random neighbors of the core nodes to provide context
    potential_context = set()
    for u in self.nodes:
      neighbors = np.where(self.host_state.g[u] == 1)[0]
      potential_context.update(neighbors)

    potential_context.difference_update(self.nodes)
    if potential_context:
      context_list = list(potential_context)
      random.shuffle(context_list)
      self.nodes.extend(context_list[:context_size])

    # Fill up if we are short (e.g. disconnected components or small graph)
    if len(self.nodes) < self.size:
      remaining = [n for n in range(self.host_state.n) if n not in self.nodes]
      if remaining:
        self.nodes.extend(
            random.sample(
                remaining, min(len(remaining), self.size - len(self.nodes))
            )
        )

    self.nodes = sorted(
        list(set(self.nodes))
    )  # Ensure unique and sorted for indexing

    # The symbiote's "genome" is the subgraph induced by these nodes
    self.g = host_state.g[np.ix_(self.nodes, self.nodes)].copy()

  def get_local_score(self):
    """Score is based on violations *fully contained* within the symbiote."""
    local_cliques = 0
    g_ig = ig.Graph.Adjacency(self.g.tolist(), mode=ig.ADJ_UPPER)
    for _ in g_ig.cliques(min=K_CLIQUE, max=K_CLIQUE):
      local_cliques += 1

    local_anticliques = 0
    if self.size >= K_ANTICLIQUE:
      # Heuristic for anticliques: count number of non-edges (edges in
      # complement graph)
      # The more non-edges, the more likely for anticliques.
      # This is a fast approximation for local search within the symbiote.
      complement_graph = 1 - self.g - np.eye(self.size)
      num_non_edges = np.sum(complement_graph) // 2  # Sum of upper triangle
      local_anticliques = (
          num_non_edges  # Use non-edges count directly as a proxy
      )

    return local_cliques + local_anticliques

  def evolve(self, iterations):
    """Run a fast, simple local search on the symbiote's internal edges."""
    for _ in range(iterations):
      if self.size < 2:
        continue
      u_idx, v_idx = sorted(random.sample(range(self.size), 2))

      current_score = self.get_local_score()

      self.g[u_idx, v_idx] = 1 - self.g[u_idx, v_idx]
      self.g[v_idx, u_idx] = self.g[u_idx, v_idx]

      new_score = self.get_local_score()

      # Simple hill-climbing: revert if not better
      if new_score >= current_score:
        self.g[u_idx, v_idx] = 1 - self.g[u_idx, v_idx]
        self.g[v_idx, u_idx] = self.g[u_idx, v_idx]

  def graft_back_to_host(self):
    """Applies the symbiote's evolved subgraph back to the host GraphState."""
    # Create a copy of the host graph to apply changes
    new_g = self.host_state.g.copy()
    new_g[np.ix_(self.nodes, self.nodes)] = self.g

    # Return the new full graph matrix. The main loop will create a new
    # state from it.
    return new_g


class GraphState:
  """Manages the graph, its cliques, and anticliques for efficient updates."""

  def __init__(self, g):
    self.g = g
    self.n = g.shape[0]
    self.clique4_sets: Set[FrozenSet[int]] = set()
    self.anticlique13_sets: Set[FrozenSet[int]] = set()
    self.edge_flip_counts = np.zeros((self.n, self.n), dtype=int)
    # Tabu list is now managed by the search function, not the state object
    self._initialize_violations()

  def _initialize_violations(self):
    """Initializes all 4-cliques and 13-anticliques from scratch."""
    if self.n < K_CLIQUE:  # No K-cliques possible
      self.clique4_sets.clear()
    else:
      g_ig = ig.Graph.Adjacency(self.g.tolist(), mode=ig.ADJ_UPPER)
      for clique_nodes in g_ig.cliques(min=K_CLIQUE, max=K_CLIQUE):
        self.clique4_sets.add(frozenset(clique_nodes))

    if self.n < K_ANTICLIQUE:  # No K-anticliques possible
      self.anticlique13_sets.clear()
    else:
      complement = 1 - self.g - np.eye(self.n)
      g_comp_ig = ig.Graph.Adjacency(complement.tolist(), mode=ig.ADJ_UPPER)
      for anticlique_nodes in g_comp_ig.cliques(
          min=K_ANTICLIQUE, max=K_ANTICLIQUE
      ):
        self.anticlique13_sets.add(frozenset(anticlique_nodes))

  def get_score(self):
    """Returns the total number of 4-cliques and 13-anticliques."""
    return len(self.clique4_sets) + len(self.anticlique13_sets)

  def get_problematic_edges(self):
    """Identifies edges involved in 4-cliques or 13-anticliques using tracked sets."""
    edges = []
    for clique_fset in self.clique4_sets:
      clique = list(clique_fset)
      for i, rename_me in enumerate(clique):
        for j in range(i + 1, len(clique)):
          u, v = sorted((rename_me, clique[j]))
          edges.append((u, v))

    # Increased weight slightly as anticliques are generally harder to resolve
    anticlique_edge_weight = 4

    for anticlique_fset in self.anticlique13_sets:
      anticlique = list(anticlique_fset)
      for i, rename_me in enumerate(anticlique):
        for j in range(i + 1, len(anticlique)):
          u, v = sorted((rename_me, anticlique[j]))
          # Weight anticlique edges higher because they are harder to break
          edges.extend([(u, v)] * anticlique_edge_weight)

    return edges  # Return list with duplicates to bias sampling

  def get_delta_score_and_updates(self, u, v):
    """Calculates the change in score if edge (u,v) is flipped.

    Identifies specific cliques/anticliques added/removed.

    Args:
      u: First vertex index.
      v: Second vertex index.

    Returns:
      A tuple containing:
        - delta_score: Change in total score.
        - added_cliques: List of new 4-cliques.
        - removed_cliques: List of removed 4-cliques.
        - added_anticliques: List of new 13-anticliques.
        - removed_anticliques: List of removed 13-anticliques.
    """
    current_edge_val = self.g[u, v]
    delta_cliques = 0
    delta_anticliques = 0
    added_cliques = []
    removed_cliques = []
    added_anticliques = []
    removed_anticliques = []

    # --- Calculate delta for 4-cliques ---
    # Case 1: Edge (u,v) exists (current_edge_val == 1), we are removing it
    if current_edge_val == 1:
      for clique_fset in self.clique4_sets:
        if u in clique_fset and v in clique_fset:
          delta_cliques -= 1
          removed_cliques.append(clique_fset)

    # Case 2: Edge (u,v) does not exist (current_edge_val == 0), we are adding
    # it
    else:  # current_edge_val == 0
      # Common neighbors for forming {u,v,x,y} 4-cliques (x,y are neighbors of
      # u AND v)
      common_neighbors_u = np.where(self.g[u] == 1)[0]
      common_neighbors_v = np.where(self.g[v] == 1)[0]

      potential_xy_nodes = np.intersect1d(
          common_neighbors_u, common_neighbors_v
      )

      if len(potential_xy_nodes) >= K_CLIQUE - 2:  # >= 2
        for i, rename_me in enumerate(potential_xy_nodes):
          for j in range(i + 1, len(potential_xy_nodes)):
            x = rename_me
            y = potential_xy_nodes[j]
            if self.g[x, y] == 1:  # If x,y are connected in current graph
              new_clique = frozenset({u, v, x, y})
              if new_clique not in self.clique4_sets:  # Ensure it's truly new
                delta_cliques += 1
                added_cliques.append(new_clique)

    # --- Calculate delta for 13-anticliques ---
    # An anticlique is a clique in the complement graph.

    # Case 1: Edge (u,v) exists (current_edge_val == 1), we are removing it
    # from g.
    # This means edge (u,v) is ADDED to g_complement.
    if current_edge_val == 1:
      # Find new 13-cliques in g_complement that include edge (u,v)
      # These are formed by u, v and an 11-clique in their common neighbors in
      # g_complement.
      # The common neighbors are nodes 'w' such that g[u,w]==0 and g[v,w]==0

      mask_u_non_adj = self.g[u] == 0
      mask_v_non_adj = self.g[v] == 0

      common_non_neighbors_in_g = np.where(mask_u_non_adj & mask_v_non_adj)[0]

      # Remove u, v from this list if they are accidentally included
      common_non_neighbors_in_g = common_non_neighbors_in_g[
          common_non_neighbors_in_g != u
      ]
      common_non_neighbors_in_g = common_non_neighbors_in_g[
          common_non_neighbors_in_g != v
      ]

      if len(common_non_neighbors_in_g) >= K_ANTICLIQUE - 2:  # >= 11
        subg_nodes_comp = common_non_neighbors_in_g

        # To find 11-cliques among these, we need to check their connections in
        # the complement of 'g'.
        # This means we want 11 nodes from `subg_nodes_comp` that are all
        # non-adjacent to each other in `g`.
        subg_adj = self.g[np.ix_(subg_nodes_comp, subg_nodes_comp)]
        subg_comp_adj = (
            1 - subg_adj - np.eye(len(subg_nodes_comp))
        )  # Complement of this subgraph

        g_temp_comp_ig = ig.Graph.Adjacency(
            subg_comp_adj.tolist(), mode=ig.ADJ_UPPER
        )
        for k_minus_2_clique_nodes_idx in g_temp_comp_ig.cliques(
            min=K_ANTICLIQUE - 2, max=K_ANTICLIQUE - 2
        ):
          nodes_in_k_minus_2 = {
              subg_nodes_comp[i] for i in k_minus_2_clique_nodes_idx
          }
          new_anticlique = frozenset(nodes_in_k_minus_2.union({u, v}))
          if new_anticlique not in self.anticlique13_sets:
            delta_anticliques += 1
            added_anticliques.append(new_anticlique)

    # Case 2: Edge (u,v) does not exist (current_edge_val == 0), we are adding
    # it to g.
    # This means edge (u,v) is REMOVED from g_complement.
    else:  # current_edge_val == 0
      for anticlique_fset in self.anticlique13_sets:
        if u in anticlique_fset and v in anticlique_fset:
          delta_anticliques -= 1
          removed_anticliques.append(anticlique_fset)

    total_delta = delta_cliques + delta_anticliques
    return (
        total_delta,
        added_cliques,
        removed_cliques,
        added_anticliques,
        removed_anticliques,
    )

  def apply_move(
      self,
      u,
      v,
      added_cliques,
      removed_cliques,
      added_anticliques,
      removed_anticliques,
  ):
    """Applies the edge flip and updates the clique/anticlique sets."""
    self.g[u, v] = 1 - self.g[u, v]
    self.g[v, u] = self.g[u, v]
    self.edge_flip_counts[u, v] += 1
    self.edge_flip_counts[v, u] += 1

    for ac in added_cliques:
      self.clique4_sets.add(ac)
    for rc in removed_cliques:
      self.clique4_sets.remove(rc)

    for aa in added_anticliques:
      self.anticlique13_sets.add(aa)
    for ra in removed_anticliques:
      self.anticlique13_sets.remove(ra)

  def extend_graph(self, new_node_vector):
    """Extends the graph by one vertex and updates state incrementally."""
    old_n = self.n
    new_vertex_idx = self.n
    self.n += 1

    new_g_matrix = np.zeros((self.n, self.n), dtype=int)
    new_g_matrix[:old_n, :old_n] = self.g
    new_g_matrix[new_vertex_idx, :old_n] = new_node_vector
    new_g_matrix[:old_n, new_vertex_idx] = new_node_vector
    self.g = new_g_matrix

    # Resize the flip counts matrix
    new_counts = np.zeros((self.n, self.n), dtype=int)
    new_counts[:old_n, :old_n] = self.edge_flip_counts
    self.edge_flip_counts = new_counts

    # Update violations incrementally instead of full re-scan
    # 1. Check for new 4-cliques involving the new node
    neighbors = np.where(new_node_vector == 1)[0]
    if len(neighbors) >= K_CLIQUE - 1:
      # Check for (K_CLIQUE-1)-cliques in the subgraph of neighbors
      subg_adj = self.g[np.ix_(neighbors, neighbors)]
      g_sub = ig.Graph.Adjacency(subg_adj.tolist(), mode=ig.ADJ_UPPER)

      for clique_m1 in g_sub.cliques(min=K_CLIQUE - 1, max=K_CLIQUE - 1):
        # Map back to original indices
        original_indices = [neighbors[i] for i in clique_m1]
        new_clique = frozenset(original_indices + [new_vertex_idx])
        self.clique4_sets.add(new_clique)

    # 2. Check for new 13-anticliques involving the new node
    non_neighbors = np.where(new_node_vector == 0)[0]
    if len(non_neighbors) >= K_ANTICLIQUE - 1:
      # Check for (K_ANTICLIQUE-1)-anticliques in the subgraph of non-neighbors
      # i.e., (K_ANTICLIQUE-1)-cliques in the complement of the subgraph
      if len(non_neighbors) > 0:  # Avoid empty slicing
        subg_adj_non = self.g[np.ix_(non_neighbors, non_neighbors)]
        subg_comp_adj = 1 - subg_adj_non - np.eye(len(non_neighbors))
        g_sub_comp = ig.Graph.Adjacency(
            subg_comp_adj.tolist(), mode=ig.ADJ_UPPER
        )

        for anticlique_m1 in g_sub_comp.cliques(
            min=K_ANTICLIQUE - 1, max=K_ANTICLIQUE - 1
        ):
          original_indices = [non_neighbors[i] for i in anticlique_m1]
          new_anticlique = frozenset(original_indices + [new_vertex_idx])
          self.anticlique13_sets.add(new_anticlique)

  def get_delta_score_for_extension(self, new_node_vector):
    """Calculates the change in score if a new node is added.

    Args:
      new_node_vector: Connection vector for the new node.

    Returns:
      Change in score (cliques + anticliques).
    """
    # The current graph has score 0 when this function is called (before
    # extension)
    # So we are calculating the total number of new violations if this vector
    # is added.

    delta_cliques = 0
    delta_anticliques = 0

    # Check for new 4-cliques involving the new node
    neighbors = np.where(new_node_vector == 1)[0]
    if (
        len(neighbors) >= K_CLIQUE - 1
    ):  # We need at least 3 neighbors to form a K4 with the new node
      # Find (K_CLIQUE-1)-cliques (3-cliques) among these neighbors in the
      # *current* graph
      if len(neighbors) > 0:
        subg_adj = self.g[np.ix_(neighbors, neighbors)]
        g_sub = ig.Graph.Adjacency(subg_adj.tolist(), mode=ig.ADJ_UPPER)
        for _ in g_sub.cliques(min=K_CLIQUE - 1, max=K_CLIQUE - 1):
          delta_cliques += 1

    # Check for new 13-anticliques involving the new node
    non_neighbors = np.where(new_node_vector == 0)[0]
    if (
        len(non_neighbors) >= K_ANTICLIQUE - 1
    ):  # We need at least 12 non-neighbors to form a K13-anticlique
      # Find (K_ANTICLIQUE-1)-anticliques (12-anticliques) among these
      # non-neighbors in the *current* graph
      # This means finding (K_ANTICLIQUE-1)-cliques in the complement of the
      # subgraph formed by non_neighbors
      if len(non_neighbors) > 0:
        subg_adj_non = self.g[np.ix_(non_neighbors, non_neighbors)]
        subg_comp_adj = 1 - subg_adj_non - np.eye(len(non_neighbors))
        g_sub_comp = ig.Graph.Adjacency(
            subg_comp_adj.tolist(), mode=ig.ADJ_UPPER
        )
        for _ in g_sub_comp.cliques(min=K_ANTICLIQUE - 1, max=K_ANTICLIQUE - 1):
          delta_anticliques += 1

    return delta_cliques + delta_anticliques


def find_graph():
  """Finds a graph with no 4-cliques and no 13-anti-cliques using Tabu Search."""
  start_time = time.time()

  # Start with the known R(4,13) lower bound graph of size 127.
  # This is the Cayley graph on the finite field F_127, with edges defined
  # by cubic residues. This graph is known to be free of 4-cliques and
  # 13-anticliques, providing a vastly superior starting point over random
  # graphs.
  p = 127
  # An element 'a' is a cubic residue if a^((p-1)/3) = 1 (mod p).
  exponent = (p - 1) // 3
  cubic_residues = {i for i in range(1, p) if pow(i, exponent, p) == 1}

  g_initial = np.zeros((p, p), dtype=int)
  for i in range(p):
    for j in range(i + 1, p):
      # The graph is undirected because -1 is a cubic residue in F_127
      # (since (-1)^42 = 1), so if (i-j) is a CR, so is (j-i).
      if (i - j) % p in cubic_residues:
        g_initial[i, j] = 1
        g_initial[j, i] = 1

  graph_state = GraphState(g_initial)

  best_g: Optional[np.ndarray] = None

  # Store candidates for g2 (size N1+1 to N1+4)
  stats = {
      "iterations": 0,
      "size_extensions": 0,
      "symbiote_grafts": 0,
  }

  tabu_list = {}
  iter_count = 0

  current_score = graph_state.get_score()
  best_score_in_phase = current_score

  no_improve_iters = 0

  while time.time() - start_time < 2350:
    iter_count += 1
    stats["iterations"] += 1

    # Clean up tabu list
    expired = [k for k, v in tabu_list.items() if v <= iter_count]
    for k in expired:
      del tabu_list[k]

    if current_score == 0:

      best_g = graph_state.g.copy()  # This is our n1 graph

      # Clear g2 candidates, as a new, better g1 has been found.

      # Reset for the next phase of search at a larger size
      tabu_list = {}
      iter_count = 0
      no_improve_iters = 0

      # Try to extend
      old_n = graph_state.n

      # Generate multiple candidates for the new node's connection vector
      candidates = []

      # Since we start from 127, old_n will always be >= 2.
      # Strategy A: Symmetry-Aware Candidates (Cyclic Shifts of Existing Rows)
      # Try cyclic shifts of the last few rows to maintain or explore
      # circulant properties
      for offset in range(
          1, min(old_n, 5)
      ):  # Check shifts from the last 1-4 rows
        base_row = graph_state.g[old_n - offset, :].copy()
        candidates.append(np.roll(base_row, 1))
        candidates.append(np.roll(base_row, -1))

      # Strategy B: Perturbations of Existing Rows (Density-targeted noise)
      # Select a few existing rows and apply random noise to them
      indices_to_perturb = np.random.choice(
          old_n, size=min(old_n, 10), replace=False
      )
      for idx in indices_to_perturb:
        cand = graph_state.g[idx, :].copy()
        # Introduce a controlled amount of noise
        noise_count = random.randint(
            1, max(1, old_n // 15)
        )  # Smaller noise burst to stay closer to original
        noise_indices = np.random.choice(old_n, size=noise_count, replace=False)
        cand[noise_indices] = 1 - cand[noise_indices]
        candidates.append(cand)

      # Strategy C: Completely Random with target density
      # Generate a few fully random candidates, aiming for average density
      avg_degree = np.sum(graph_state.g) / old_n if old_n > 0 else 0
      for _ in range(
          3
      ):  # Fewer purely random candidates to save evaluation time
        target_density = np.clip(
            avg_degree / old_n + random.uniform(-0.1, 0.1), 0.1, 0.9
        )
        cand = (np.random.random(old_n) < target_density).astype(int)
        candidates.append(cand)

      # Select the best candidate based on its initial violation score using
      # the new fast delta calculation
      best_candidate_vec = candidates[0]
      # The initial graph_state score is 0 before extension, so candidate
      # score is just the delta.
      best_candidate_total_violations = (
          graph_state.get_delta_score_for_extension(best_candidate_vec)
      )

      for i in range(1, len(candidates)):
        cand = candidates[i]
        delta = graph_state.get_delta_score_for_extension(cand)
        if delta < best_candidate_total_violations:
          best_candidate_total_violations = delta
          best_candidate_vec = cand

      # Use the best candidate to extend the graph
      graph_state.extend_graph(best_candidate_vec)
      # The current_score needs to be re-evaluated fully after extension to be
      # accurate.
      # This is done by graph_state.get_score() which sums the tracked
      # violations.
      current_score = graph_state.get_score()
      best_score_in_phase = current_score  # Reset best score for this new phase

      stats["size_extensions"] += 1

      # The main loop will now optimize this new, larger graph
      continue

    # --- Main Tabu Search Step (if not extending) ---

    # Identify candidates for flipping using GraphState method
    candidates = graph_state.get_problematic_edges()

    # If no problematic edges are identified (e.g., score is 0, or just very
    # few issues),
    # select some random edges to keep the search moving and explore new
    # states.
    if not candidates and graph_state.get_score() > 0:
      num_random_edges_to_sample = max(1, graph_state.n // 5)
      for _ in range(num_random_edges_to_sample):
        if graph_state.n < 2:
          continue  # Ensure enough vertices to pick 2
        u, v = sorted(random.sample(range(graph_state.n), 2))
        candidates.append((u, v))
    # If candidates is still empty and score is 0, it means we are in the
    # extension phase where the graph is perfect, so no random moves are
    # needed at this point.

    # Two-stage move evaluation
    sample_size = min(len(candidates), 300)  # Increased sample size

    moves_with_partial_scores = []
    seen_moves = set()

    # For partial score, use an estimate of 4-clique delta (cheaper) +
    # heuristic for 13-anticlique risk
    # candidates list contains duplicates for weighting; we sample from it then
    # deduplicate
    sampled_candidates = random.sample(candidates, sample_size)

    for u, v in sampled_candidates:
      if (u, v) in seen_moves:
        continue
      seen_moves.add((u, v))

      # Calculate exact delta for 4-cliques (fast)
      common_neighbors_uv = np.where(
          (graph_state.g[u] == 1) & (graph_state.g[v] == 1)
      )[0]
      delta_k4 = 0

      # Calculate heuristic risk for 13-anticliques
      # Common non-neighbors count is a proxy for risk of creating
      # 13-anticlique when removing an edge
      common_non_neighbors_uv = np.where(
          (graph_state.g[u] == 0) & (graph_state.g[v] == 0)
      )[0]
      num_cnn = len(common_non_neighbors_uv)

      heuristic_delta_k13 = 0.0

      if (
          graph_state.g[u, v] == 1
      ):  # If removing edge (u,v) -> creating non-edge
        # K4: Removing edge breaks 4-cliques containing (u,v).
        if len(common_neighbors_uv) >= K_CLIQUE - 2:
          subg_adj = graph_state.g[
              np.ix_(common_neighbors_uv, common_neighbors_uv)
          ]
          delta_k4 = (
              -np.sum(subg_adj) // 2
          )  # Num edges in subgraph of common neighbors

        # K13: Removing edge might form 13-anticliques.
        # Risk increases with number of common non-neighbors (num_cnn).
        if num_cnn >= K_ANTICLIQUE - 2:
          # Heuristic penalty: proportional to how many nodes are "available"
          heuristic_delta_k13 = (num_cnn - (K_ANTICLIQUE - 3)) * 0.2

      else:  # If adding edge (u,v) -> removing non-edge
        # K4: Adding edge might form 4-cliques.
        if len(common_neighbors_uv) >= K_CLIQUE - 2:
          subg_adj = graph_state.g[
              np.ix_(common_neighbors_uv, common_neighbors_uv)
          ]
          delta_k4 = np.sum(subg_adj) // 2

        # K13: Adding edge breaks 13-anticliques containing non-edge (u,v).
        # Benefit increases with number of common non-neighbors (num_cnn).
        if num_cnn >= K_ANTICLIQUE - 2:
          # Heuristic benefit: Proportional to how many potential anticliques
          # we might break.
          heuristic_delta_k13 = -(num_cnn - (K_ANTICLIQUE - 3)) * 0.1

      moves_with_partial_scores.append(((u, v), delta_k4 + heuristic_delta_k13))

    moves_with_partial_scores.sort(
        key=lambda x: x[1]
    )  # Prioritize moves that reduce 4-cliques (or increase least)

    # Stage 2: Perform full, expensive evaluation for the top candidates.
    best_move_tuple = None
    best_move_delta = float("inf")
    best_move_updates = None

    # Dynamic adjustment of num_full_evals based on graph size
    num_full_evals = max(5, min(50, graph_state.n // 5))

    # Stochastic selection for full evaluation to avoid local optima driven by
    # imperfect heuristics
    # Take top 50% strictly, sample remaining 50% from the next best chunk
    num_strict = num_full_evals // 2
    num_stochastic = num_full_evals - num_strict

    candidates_for_full_eval = [
        move_tuple for (move_tuple, _) in
        moves_with_partial_scores[:num_strict]
    ]

    # Pool for stochastic selection (next 3 * num_stochastic candidates)
    pool_start = num_strict
    pool_end = min(
        len(moves_with_partial_scores), num_strict + 3 * num_stochastic
    )

    if pool_end > pool_start:
      stochastic_pool = [
          move_tuple
          for (move_tuple, _) in moves_with_partial_scores[pool_start:pool_end]
      ]
      # Sample without replacement
      sampled = random.sample(
          stochastic_pool, min(len(stochastic_pool), num_stochastic)
      )
      candidates_for_full_eval.extend(sampled)

    if not candidates_for_full_eval and candidates:
      candidates_for_full_eval = random.sample(
          candidates, min(num_full_evals, len(candidates))
      )

    for u, v in candidates_for_full_eval:
      is_tabu = (u, v) in tabu_list

      delta, ac, rc, aa, ra = graph_state.get_delta_score_and_updates(u, v)
      new_score_potential = current_score + delta

      # Aspiration criteria: accept tabu move if it leads to a new best score
      # in this phase.
      is_aspirated = new_score_potential < best_score_in_phase

      if (not is_tabu) or is_aspirated:
        if delta < best_move_delta:
          best_move_delta = delta
          best_move_tuple = (u, v)
          best_move_updates = (ac, rc, aa, ra)

    if best_move_tuple:
      u, v = best_move_tuple
      graph_state.apply_move(u, v, *best_move_updates)

      current_score += best_move_delta

      if current_score < best_score_in_phase:
        best_score_in_phase = current_score
        no_improve_iters = 0
      else:
        no_improve_iters += 1

      # Stubbornness-weighted tenure: edges that flip often stay tabu longer
      stubbornness = graph_state.edge_flip_counts[u, v]
      # Increase base tenure slightly to force broader exploration
      base_tenure = int(np.sqrt(graph_state.n)) + 6 + min(stubbornness, 25)

      if no_improve_iters > 100:
        base_tenure += int(np.sqrt(graph_state.n))

      tabu_list[(u, v)] = iter_count + base_tenure + random.randint(-2, 4)

      if iter_count % 1000 == 0:
        real_score = graph_state.get_score()
        if real_score != current_score:
          current_score = real_score
    else:
      # If no beneficial non-tabu move found, try a random non-tabu move
      possible_moves = []
      for _ in range(min(graph_state.n // 2, 50)):  # Sample a few random edges
        u, v = sorted(random.sample(range(graph_state.n), 2))
        if u == v:
          continue
        if (u, v) not in tabu_list:
          possible_moves.append((u, v))

      if possible_moves:
        ku, kv = random.choice(possible_moves)
        delta, ac, rc, aa, ra = graph_state.get_delta_score_and_updates(ku, kv)
        graph_state.apply_move(ku, kv, ac, rc, aa, ra)
        current_score += delta
        tabu_list[(ku, kv)] = (
            iter_count + int(np.sqrt(graph_state.n)) // 2 + 1
        )  # Shorter tabu for random moves
      else:  # Fallback if all sampled random moves are tabu or no moves sampled
        ku, kv = sorted(random.sample(range(graph_state.n), 2))
        delta, ac, rc, aa, ra = graph_state.get_delta_score_and_updates(ku, kv)
        graph_state.apply_move(ku, kv, ac, rc, aa, ra)
        current_score += delta

      no_improve_iters += 1

    # Stagnation handling: "Symbiotic Intervention"
    if no_improve_iters > 500 and graph_state.n > 20:
      stats["symbiote_grafts"] += 1

      # 1. Infection: Identify problematic nodes
      all_problem_nodes = set()
      for cset in graph_state.clique4_sets:
        for node in cset:
          all_problem_nodes.add(node)
      for aset in graph_state.anticlique13_sets:
        for node in aset:
          all_problem_nodes.add(node)

      if not all_problem_nodes:  # If no violations, nothing to infect
        no_improve_iters = 0
        continue

      # 2. Create and Evolve Symbiote
      symbiote = GraphSymbiote(
          graph_state, list(all_problem_nodes), size=max(25, graph_state.n // 5)
      )
      symbiote.evolve(iterations=500)  # Fast local evolution

      # 3. Graft Back and Re-evaluate
      new_g_matrix = symbiote.graft_back_to_host()

      # Create a new state, as many violations will have changed. Full
      # re-evaluation is safest.
      graph_state = GraphState(new_g_matrix)

      # Reset search state after this radical change
      current_score = graph_state.get_score()

      # If symbiote made things worse (rare but possible), revert?
      # No, we accept it to escape the local optimum, even if score is
      # temporarily higher.

      best_score_in_phase = current_score
      no_improve_iters = 0
      tabu_list = {}  # Clear tabu list after grafting

  g1 = best_g
  if g1 is None:
    g1 = np.array([[0]])

  return g1
