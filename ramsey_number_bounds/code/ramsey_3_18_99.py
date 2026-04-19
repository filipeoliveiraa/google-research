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
from typing import Optional
import igraph as ig
import numpy as np

R3_K = 18
K_SIZE = R3_K - 1  # We need alpha <= 17, i.e., no (R3_K)-anticlique.


def try_extend_graph(
    adj, existing_anticliques = None
):
  """Attempts to extend the graph by one vertex maintaining Ramsey properties.

  Args:
    adj: The adjacency matrix of the current graph.
    existing_anticliques: Optional list of pre-computed anticliques of size 17.

  Returns:
    A tuple (new_adj, anticliques_17) where new_adj is the extended adjacency
    matrix and anticliques_17 is the updated list of anticliques of size 17.
  """
  n = adj.shape[0]

  # 1. Identify "problematic" independent sets of size 17 (alpha = 17)
  if existing_anticliques is None:
    g_bar_a = np.ones((n, n), dtype=np.int8) - adj - np.eye(n, dtype=np.int8)
    g_bar = ig.Graph.Adjacency(g_bar_a.tolist(), mode=ig.ADJ_UNDIRECTED)
    try:
      # Find all 17-cliques in the complement graph (17-anticliques in G)
      anticliques_17 = g_bar.cliques(min=K_SIZE, max=K_SIZE)
    except (ValueError, RuntimeError):
      # If igraph fails, it could be due to memory or other issues.
      # In this case, we cannot proceed with the extension.
      return None, None
  else:
    anticliques_17 = existing_anticliques

  if not anticliques_17:
    # If there are no 17-anticliques, any maximal independent set can be used
    # to extend.
    # This indicates the graph's alpha is likely < 17, making extension easier.
    # We find one such set and extend.

    # Randomized Greedy MIS to find a maximal independent set I
    i_indices = []
    candidates = list(range(n))
    random.shuffle(candidates)

    is_candidate = np.ones(n, dtype=bool)
    for u in candidates:
      if is_candidate[u]:
        i_indices.append(u)
        # Mark neighbors as not candidates
        neighbors = np.where(adj[u] == 1)[0]
        is_candidate[neighbors] = False

    i_set = set(i_indices)
    remaining_nodes = [x for x in range(n) if x not in i_set]
    sub_adj = adj[np.ix_(remaining_nodes, remaining_nodes)]

    # The subgraph G[V \ I] must have alpha < 17.
    # Since I is a *maximal* IS, V \ I is a vertex cover.
    # Any IS in G[V \ I] is also an IS in G.
    # A key property of maximal IS is that every node not in I is adjacent to
    # at least one node in I.
    # If alpha(G) < 17, then alpha(G[V \ I]) must also be < 17.
    # Given no 17-anticliques were found, we can proceed.

    new_n = n + 1
    new_adj = np.zeros((new_n, new_n), dtype=np.int8)
    new_adj[:n, :n] = adj
    for u in i_indices:
      new_adj[n, u] = 1
      new_adj[u, n] = 1
    return new_adj, None

  # --- If 17-anticliques exist, we need a more careful approach ---
  # We need to find an Independent Set I in G that is a "hitting set" for all
  # 17-anticliques.
  # And alpha(G[V \ I]) < 17.

  anticlique_sets = [set(ac) for ac in anticliques_17]

  # 1. Generate several large Maximal Independent Sets (MIS)
  # We use many trials (500) to get diverse and large candidates for I.
  mis_candidates = find_large_independent_set_fast(
      adj, target_size=0, num_trials=500, return_sets=True
  )

  # 2. Test the candidates
  for hitting_set_i_frozen in mis_candidates:
    hitting_set_i = set(hitting_set_i_frozen)

    # Check Condition 1: Is I a hitting set for all 17-anticliques?
    is_hitting_set = True
    for ac_set in anticlique_sets:
      if not ac_set.intersection(hitting_set_i):
        is_hitting_set = False
        break

    if not is_hitting_set:
      continue

    # Condition 1 satisfied. Now check Condition 2: alpha(G[V \ I]) < 17.
    remaining_nodes = list(set(range(n)) - hitting_set_i)

    if not remaining_nodes:  # I is the whole graph. Subgraph is empty, alpha=0.
      # Success!
      new_n = n + 1
      new_adj = np.zeros((new_n, new_n), dtype=np.int8)
      new_adj[:n, :n] = adj
      for u in hitting_set_i:
        new_adj[n, u] = 1
        new_adj[u, n] = 1
      return new_adj, None

    sub_adj = adj[np.ix_(remaining_nodes, remaining_nodes)]

    # Heuristic check first
    # Increase trials for the sub-graph check to reduce reliance on the full
    # igraph call
    if not find_large_independent_set_fast(
        sub_adj, target_size=K_SIZE, num_trials=500
    ):
      # More rigorous check (necessary because find_large_independent_set_fast
      # is heuristic)
      sub_n = len(remaining_nodes)
      sub_g_bar_a = (
          np.ones((sub_n, sub_n), dtype=np.int8)
          - sub_adj
          - np.eye(sub_n, dtype=np.int8)
      )
      sub_g_bar = ig.Graph.Adjacency(
          sub_g_bar_a.tolist(), mode=ig.ADJ_UNDIRECTED
      )
      if not any(sub_g_bar.cliques(min=K_SIZE, max=K_SIZE)):
        # Success! Found a valid extension.
        new_n = n + 1
        new_adj = np.zeros((new_n, new_n), dtype=np.int8)
        new_adj[:n, :n] = adj
        for u in hitting_set_i:
          new_adj[n, u] = 1
          new_adj[u, n] = 1
        return new_adj, None

  # If all attempts fail, return failure, but pass back the computed
  # anticliques.
  return None, anticliques_17


# --- Original helper functions (moved and potentially slightly optimized) ---


def check_graph(adj_matrix):
  """Checks if a graph represented by adj_matrix is (3, R3_K)-Ramsey-good.

  Args:
    adj_matrix: The adjacency matrix of the graph.

  Returns:
    A tuple (is_good, max_clique_size, max_independent_set_size). A graph G is
    (3, R3_K)-Ramsey-good if omega(G) < 3 and alpha(G) < R3_K.
  """
  n = adj_matrix.shape[0]
  if n == 0:
    return True, 0, 0

  # 1. Check for 3-clique (omega(G) < 3)
  # The GA mutations ensure triangle-freeness, so this should ideally be 0.
  adj = adj_matrix
  # Standard way to count 3-cliques
  num_3_cliques = np.trace(adj @ adj @ adj) // 6

  is_triangle_free = num_3_cliques == 0

  max_clique_size = (
      3 if num_3_cliques > 0 else (2 if n > 1 and np.any(adj) else min(n, 1))
  )

  # 2. Check for R3_K-anticlique (alpha(G) < R3_K)
  g_bar_a = (
      np.ones((n, n), dtype=np.int8) - adj_matrix - np.eye(n, dtype=np.int8)
  )
  g_bar = ig.Graph.Adjacency(g_bar_a.tolist(), mode=ig.ADJ_UNDIRECTED)

  try:
    # Use igraph for large clique finding. `any()` stops on first match.
    has_r3_k_anticlique = any(g_bar.cliques(min=R3_K, max=R3_K))
  except (ValueError, RuntimeError):
    has_r3_k_anticlique = True  # Assume failure means violation

  if has_r3_k_anticlique:
    max_independent_set_size = R3_K
  else:
    max_independent_set_size = R3_K - 1

  is_good = is_triangle_free and (not has_r3_k_anticlique)

  return is_good, max_clique_size, max_independent_set_size


def find_large_independent_set_fast(
    adj,
    target_size,
    num_trials = 10,
    return_sets = False,
):
  """Attempts to find a large independent set using a randomized MIS approach.

  Args:
    adj: The adjacency matrix of the graph.
    target_size: The target size of the independent set.
    num_trials: The number of random trials.
    return_sets: If True, returns the sets found instead of a boolean.

  Returns:
    True if an IS of size target_size is found, False otherwise.
    If return_sets is True, returns a list of found MISs, sorted by size
    descending.
  """
  n = adj.shape[0]
  if n < target_size and not return_sets:
    return False
  if n == 0:
    return False if not return_sets else []

  degrees = np.sum(adj, axis=1)
  found_sets = []

  for _ in range(num_trials):
    independent_set = []
    available_nodes_set = set(range(n))

    while available_nodes_set:

      # Heuristic selection: Choose a node from `available_nodes_set`
      # Prioritize low-degree nodes.

      current_nodes_list = tuple(available_nodes_set)
      sample_size = min(len(current_nodes_list), 40)
      sampled_nodes_candidates = random.sample(current_nodes_list, sample_size)

      v = sampled_nodes_candidates[0]
      min_d = degrees[v]
      for node in sampled_nodes_candidates[1:]:
        d = degrees[node]
        if d < min_d:
          min_d = d
          v = node

      independent_set.append(v)
      available_nodes_set.remove(v)

      neighbors_of_v = np.where(adj[v] == 1)[0]
      available_nodes_set.difference_update(neighbors_of_v)

      if not return_sets and len(independent_set) >= target_size:
        return True

      if (
          not return_sets
          and len(independent_set) + len(available_nodes_set) < target_size
      ):
        break  # Pruning for simple boolean check

    if return_sets:
      found_sets.append(frozenset(independent_set))

  if not return_sets:
    return False
  else:
    # Filter unique sets and sort by size descending
    unique_sets = list(set(found_sets))
    unique_sets.sort(key=len, reverse=True)
    return unique_sets


def calculate_badness_score(adj_matrix):
  """Calculates the number of 3-cliques and checks for R3_K-anticliques.

  Args:
    adj_matrix: The adjacency matrix of the graph.

  Returns:
    A tuple (num_3_cliques, num_r3_k_anticliques).
  """
  n = adj_matrix.shape[0]
  if n == 0:
    return 0, 0

  # 1. Number of 3-cliques
  adj = adj_matrix
  adj_2 = adj @ adj
  adj_3 = adj_2 @ adj
  num_3_cliques = np.trace(adj_3) // 6

  # 2. Check for R3_K-anticlique (alpha(G) >= R3_K)
  # We only need to know if alpha is >= R3_K, not its exact value.
  # Use the fast heuristic `find_large_independent_set_fast` for this.
  # For g2, we need a reasonably reliable estimate, but not as high confidence
  # as for g1's filter.
  # Increase trials from 10 to 50 for a better trade-off.
  # Use the fast heuristic `find_large_independent_set_fast` for this.
  # For g2, increase trials for better confidence during SA optimization.
  has_r3_k_anticlique = find_large_independent_set_fast(
      adj_matrix, target_size=R3_K, num_trials=200
  )
  num_r3_k_anticliques = 1 if has_r3_k_anticlique else 0

  return num_3_cliques, num_r3_k_anticliques


def estimate_alpha(adj):
  """Estimates alpha by checking if there is an IS of size 17."""
  # Instead of returning True/False, find the actual size encountered

  # Note: We reuse find_large_independent_set_fast but modify it slightly
  # to return the max size found, although we need to be careful not to
  # introduce new functions outside the search/replace blocks.

  # Let's rely on the existing check logic, focusing on target_size=K_SIZE (17).
  # We will use the fast heuristic to approximate the current "goodness"

  # Check if alpha >= 17 is the only thing we can reliably check fast.
  # Let's use the heuristic to determine if alpha is likely 17 or 16.
  # Since the input adj_matrix is guaranteed to have alpha <= 17,
  # we primarily seek to reduce the density of 17-IS, which we proxy by
  # trying to break ISs of size 17.

  # Heuristic Objective: 1 if alpha >= 17, 0 otherwise (using 500 trials for SA)
  # We prefer 0. 1000 trials might be too slow for frequent SA steps.
  is_17 = find_large_independent_set_fast(
      adj, target_size=K_SIZE, num_trials=500
  )
  return 1 if is_17 else 0


def sa_refine_g1(
    adj_matrix,
    time_limit = 50,
    problematic_anticliques = None,
):
  """Performs Simulated Annealing on the existing G1 graph.

  Finds a structurally better, equally sized, valid graph (triangle-free,
  alpha<=17). The objective is to minimize the maximum independent set size
  (approximated).

  Args:
    adj_matrix: The adjacency matrix of the graph.
    time_limit: The time limit in seconds.
    problematic_anticliques: A list of anticliques that are problematic.

  Returns:
    A tuple (refined_adj, improved) where improved is True if refined_adj is
    structurally different from input, False otherwise.
  """
  start_time = time.time()
  n = adj_matrix.shape[0]

  current_adj = adj_matrix.copy()

  # Initial state evaluation
  # Since current_adj is a valid G1, estimate_alpha should return <= 1
  # (1 means alpha is likely 17, 0 means it's likely 16 or less).
  current_alpha_score = estimate_alpha(current_adj)

  # SA Parameters
  initial_temperature = 5.0
  cooling_rate = 0.9999  # Very slow cooling for structural changes
  temperature = initial_temperature

  best_adj = current_adj.copy()
  best_alpha_score = current_alpha_score
  improved = False

  # Run for fixed time or steps
  steps = 0
  max_steps = 100000  # Upper limit

  while time.time() - start_time < time_limit and steps < max_steps:
    steps += 1
    temperature *= cooling_rate

    # Biased edge selection if problematic_anticliques are provided
    if problematic_anticliques and n > 0:
      # Calculate node importance: how many problematic anticliques each node
      # belongs to
      node_importance = np.zeros(n, dtype=int)
      for ac in problematic_anticliques:
        for node_idx in ac:
          node_importance[node_idx] += 1

      # Normalize importance to probabilities for np.random.choice.
      # Add a small constant to each node's importance to ensure all nodes
      # still have a
      # non-zero chance of being selected, but problematic nodes have a higher
      # chance.
      total_importance = np.sum(node_importance)
      if total_importance > 0:
        probabilities = (node_importance + 1) / (
            total_importance + n
        )  # Add 1 to each to ensure non-zero prob
        r = np.random.choice(n, p=probabilities)
      else:
        # If no problematic anticliques, or all nodes equally problematic, fall
        # back to uniform
        r = random.randrange(n)

      # For c, we can keep it random to encourage broader exploration around
      # the hot spot `r`.
      c = random.randrange(n)
      while r == c:
        c = random.randrange(n)
    else:
      # Fallback to original random selection if no specific anticliques are
      # provided
      r, c = random.randrange(n), random.randrange(n)
      while r == c:
        r, c = random.randrange(n), random.randrange(n)

    temp_adj = current_adj.copy()

    is_add_op = temp_adj[r, c] == 0
    temp_adj[r, c] = 1 - temp_adj[r, c]
    temp_adj[c, r] = 1 - temp_adj[c, r]

    # 1. Constraint Check: Must remain Triangle Free (omega < 3)
    is_triangle_free = True
    if is_add_op:
      # Check if adding edge (r, c) creates a triangle
      # Triangle if there exists a node k such that (r, k) and (k, c) are edges

      # Find common neighbors
      common_neighbors = np.where(current_adj[r] & current_adj[c])[0]
      if len(common_neighbors) > 0:
        is_triangle_free = False

    if not is_triangle_free:
      # Infinite penalty: Reject move immediately if it creates a triangle.
      continue

    # 2. Evaluation: Check alpha constraint (approximate)
    new_alpha_score = estimate_alpha(temp_adj)

    delta_score = new_alpha_score - current_alpha_score

    # Acceptance Criterion
    if (
        delta_score < 0
    ):  # Improvement (Lower alpha score, i.e., less likely to have alpha=17)
      accept = True
    else:  # Worse or equal (Note: delta_score can only be 0 or 1 here)
      if temperature > 0:
        # We are optimizing for structure, so accept non-improving moves
        # occasionally
        # to escape local minima, even if they maintain the score.
        accept_probability = np.exp(-delta_score / temperature)
        accept = random.random() < accept_probability
      else:
        accept = False

    if accept:
      current_adj = temp_adj
      current_alpha_score = new_alpha_score

      if current_alpha_score < best_alpha_score:
        best_alpha_score = current_alpha_score
        best_adj = current_adj.copy()
        improved = True
      elif (best_adj != current_adj).any():
        # If we accept a move that maintained the alpha score,
        # and the graph structure changed, it might be a better starting
        # point for extension.
        best_adj = current_adj.copy()
        improved = True

  # Final check: Ensure the best graph found is truly valid using the exact
  # check
  is_good, _, _ = check_graph(best_adj)
  if not is_good:
    # This should rarely happen if constraints are maintained, but can happen
    # if the
    # heuristic estimate of alpha was wrong. If so, return the original.
    return adj_matrix, False

  return best_adj, improved


def find_graph():
  """Finds a Ramsey number lower bound graph."""
  # 1. Initialize with K_17, 17 (n=34) as a safe fallback
  # Initialize with a known good graph for R(3,18) with n=35.
  # The circulant graph C_35 with difference set D = {1, 5, 7, 9, 13, 14, 16}
  # is triangle-free and has an independent set number (alpha) of 17.
  n_init = 35
  initial_g1_adj = np.zeros((n_init, n_init), dtype=np.int8)
  diff_set = {1, 5, 7, 9, 13, 14, 16}
  for g in diff_set:
    for i in range(n_init):
      initial_g1_adj[i, (i + g) % n_init] = 1
      initial_g1_adj[i, (i - g + n_init) % n_init] = 1  # Ensure symmetry

  # Validate the hardcoded graph to be absolutely sure.
  is_good_init, _, _ = check_graph(initial_g1_adj)

  if not is_good_init:
    # Fallback to K_17,17 if the hardcoded graph is unexpectedly bad.
    # This shouldn't happen if the construction is correct.
    m = 17
    n_init = 2 * m
    j_m = np.ones((m, m), dtype=np.int8)
    o_m = np.zeros((m, m), dtype=np.int8)
    best_g1_adj = np.block([[o_m, j_m], [j_m, o_m]])
  else:
    best_g1_adj = initial_g1_adj

  best_g1_n = n_init
  stats = {
      "cyclic_iterations": 0,
      "extension_successes": 0,
      "sa_refinements": 0,
  }

  # 2. Iterative search for larger cyclic graphs (for g1)
  # Goal: maximize n such that graph is triangle-free and alpha <= 17.

  start_time = time.time()
  time_limit = 1750  # Max seconds allowed for primary search + g2 optimization

  current_n = best_g1_n + 1  # Start search from n+1 of the initial best_g1.

  # Parameters for transitioning
  max_consecutive_n_failures_cyclic = 15
  max_consecutive_n_failures_extension = 10
  consecutive_n_failures = 0
  phase = "cyclic"

  # Cache for extension phase
  cached_anticliques = None

  while time.time() - start_time < time_limit:
    if phase == "cyclic":
      trials = 0
      found_for_n = False
      max_trials_per_n = 100000

      while trials < max_trials_per_n:
        if time.time() - start_time > time_limit:
          break
        trials += 1
        stats["cyclic_iterations"] += 1

        # Randomized Greedy Construction of Sum-Free Set for Cyclic Graphs
        candidates = list(range(1, current_n // 2 + 1))
        random.shuffle(candidates)
        s_sym = set()
        gens = []
        for g in candidates:
          if (3 * g) % current_n == 0:
            continue
          if (2 * g) % current_n in s_sym:
            continue
          is_valid = True
          for s in s_sym:
            if (-g - s) % current_n in s_sym:
              is_valid = False
              break
          if not is_valid:
            continue
          gens.append(g)
          s_sym.add(g)
          s_sym.add((current_n - g) % current_n)

        adj = np.zeros((current_n, current_n), dtype=int)
        for g in gens:
          idx = np.arange(current_n)
          nb = (idx + g) % current_n
          adj[idx, nb] = 1
          adj[nb, idx] = 1

        # Heuristic check
        has_large_is = find_large_independent_set_fast(
            adj, target_size=R3_K, num_trials=300
        )

        if not has_large_is:
          is_good, _, _ = check_graph(adj)
          if is_good:
            best_g1_adj = adj
            best_g1_n = current_n
            found_for_n = True
            consecutive_n_failures = 0
            break

      if found_for_n:
        current_n += 1
      else:
        consecutive_n_failures += 1
        current_n += 1
        if consecutive_n_failures >= max_consecutive_n_failures_cyclic:
          phase = "extension"
          consecutive_n_failures = 0

    elif phase == "extension":
      # Reserve time for G2
      if time_limit - (time.time() - start_time) < 100 and best_g1_n > 35:
        break

      # try_extend_graph now returns (new_adj, cached_anticliques)
      # If successful, cached_anticliques is None (since graph changed).
      # If failed, it returns the computed anticliques so we can reuse them.
      extended_adj, returned_cache = try_extend_graph(
          best_g1_adj, existing_anticliques=cached_anticliques
      )

      if extended_adj is not None:
        best_g1_adj = extended_adj
        best_g1_n = best_g1_adj.shape[0]
        cached_anticliques = None  # Invalidate cache
        stats["extension_successes"] += 1
        consecutive_n_failures = 0  # Reset failure counter on success
      else:
        cached_anticliques = returned_cache
        consecutive_n_failures += 1
        if consecutive_n_failures >= max_consecutive_n_failures_extension:
          phase = "SA_refinement"
          consecutive_n_failures = 0

    elif phase == "SA_refinement":
      # Reserve time for G2
      if time_limit - (time.time() - start_time) < 100 and best_g1_n > 35:
        break

      stats["sa_refinements"] += 1
      refined_adj, improved = sa_refine_g1(
          best_g1_adj,
          time_limit=50,
          problematic_anticliques=cached_anticliques,
      )  # Spend limited time refining

      if refined_adj is not None:
        if improved:
          # A better structure was found, potentially one that is more
          # extendable.
          best_g1_adj = refined_adj
        # Regardless of whether structural improvement occurred, try extension
        # again.
        # We always return to extension after SA refinement, hoping the local
        # perturbation
        # created a window for growth.
        phase = "extension"
        cached_anticliques = None  # Reset cache
      else:
        # SA failed or timed out. If we hit SA multiple times without growth,
        # stop trying.
        break

  return best_g1_adj
