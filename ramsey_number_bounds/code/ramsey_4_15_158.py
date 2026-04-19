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

import collections
import random
import time
from typing import Optional
import igraph as ig
import numpy as np

# Harmonic Genetic Memory: Stores success of "orbits" (edge distances) and
# specific edges.
MAX_N_FOR_GENE_POOL = 250
_EDGE_GENE_POOL = np.zeros(
    (MAX_N_FOR_GENE_POOL, MAX_N_FOR_GENE_POOL), dtype=int
)
_ORBIT_GENE_POOL = np.zeros(MAX_N_FOR_GENE_POOL // 2 + 1, dtype=int)
_GENE_POOL_HIT_COUNT = np.zeros(MAX_N_FOR_GENE_POOL, dtype=int)
_ORBIT_HIT_COUNT = np.zeros(MAX_N_FOR_GENE_POOL // 2 + 1, dtype=int)


def get_orbit(u, v, n):
  """Returns the circulant distance (orbit) between two nodes."""
  dist = abs(u - v)
  return min(dist, n - dist)


def get_complement_graph(adj):
  """Computes the complement graph."""
  return ig.Graph.Adjacency(
      (1 - adj - np.eye(adj.shape[0])).tolist(), mode="undirected"
  )


def count_cliques(g, k):
  """Counts k-cliques."""
  return len(g.cliques(min=k, max=k))


def find_violating_independent_set(
    adj, k_target, max_trials = 100
):
  """Heuristically tries to find an independent set of size at least k_target.

  Uses adjacency matrix directly and optimized set operations with sampling.

  Args:
    adj: The adjacency matrix of the graph.
    k_target: The target size of the independent set.
    max_trials: The maximum number of random trials.

  Returns:
    A set of vertices forming an independent set of size at least k_target,
    or None if not found.
  """
  n = adj.shape[0]
  # Precompute neighbor sets for fast intersection
  neighbors = [set(np.where(adj[i] == 1)[0]) for i in range(n)]
  nodes = list(range(n))

  for _ in range(max_trials):
    independent_set = set()
    current_nodes = set(nodes)

    # Iteratively build independent set
    while current_nodes:
      if len(independent_set) + len(current_nodes) < k_target:
        break

      # Sampling-based greedy selection to avoid O(N^2) degree calculation
      sample_size = min(len(current_nodes), 15)
      candidates_sample = random.sample(list(current_nodes), sample_size)

      best_v = candidates_sample[0]
      best_deg = len(neighbors[best_v].intersection(current_nodes))

      for v in candidates_sample[1:]:
        deg = len(neighbors[v].intersection(current_nodes))
        if deg < best_deg:
          best_deg = deg
          best_v = v
          if deg == 0:
            break

      independent_set.add(best_v)
      if len(independent_set) >= k_target:
        return independent_set

      # Remove chosen node and its neighbors from candidates
      current_nodes.difference_update(neighbors[best_v])
      current_nodes.discard(best_v)

  return None


def _get_common_neighbors(adj, u, v):
  """Returns a list of common neighbors of u and v."""
  return np.where((adj[u] == 1) & (adj[v] == 1))[0].tolist()


def _update_k4s_add_edge(
    adj,
    u,
    v,
    current_k4s,
    k_target,
):
  """Adds new k-cliques formed by adding edge (u,v). Only for k=4."""
  if k_target != 4:
    return  # This optimization is for K4 specifically

  cn = _get_common_neighbors(adj, u, v)

  # Find edges in the subgraph induced by common neighbors.
  # These edges will form a K4 with u and v.
  for i, rename_me in enumerate(cn):
    for j in range(i + 1, len(cn)):
      x, y = rename_me, cn[j]
      if adj[x, y] == 1:
        current_k4s.add(tuple(sorted((u, v, x, y))))


def _calculate_potential_new_k4s(adj, u, v):
  """Calculates new k-cliques (k=4) formed by adding edge (u,v)."""
  # This assumes k_target is 4.

  # Common neighbors of u and v (not including u or v themselves)
  cn = _get_common_neighbors(adj, u, v)

  new_k4_count = 0
  cn_len = len(cn)
  if cn_len < 2:
    return 0  # Cannot form K4 if less than 2 common neighbors

  # Find edges in the subgraph induced by common neighbors.
  # These edges will form a K4 with u and v.
  for i in range(cn_len):
    for j in range(i + 1, cn_len):
      x, y = cn[i], cn[j]
      if adj[x, y] == 1:
        new_k4_count += 1
  return new_k4_count


def _update_k4s_remove_edge(
    u, v, current_k4s
):
  """Removes k-cliques that contained the edge (u,v)."""
  # Create a temporary list because we can't modify a set while iterating it.
  k4s_to_remove = []
  for clique in current_k4s:
    if u in clique and v in clique:
      k4s_to_remove.append(clique)
  for clique in k4s_to_remove:
    current_k4s.remove(clique)


def create_paley_like_graph(n, power):
  """Creates a generalized Paley graph.

  Vertices are 0 to n-1. Edge (i, j) exists if (i-j) is a k-th power residue
  mod n. 'n' should be a prime number.

  Args:
    n: The number of vertices (should be prime).
    power: The power to use for residues.

  Returns:
    The adjacency matrix of the generated graph.
  """
  if n <= 1:
    return np.zeros((n, n), dtype=int)

  residues = {pow(i, power, n) for i in range(1, n)}
  adj = np.zeros((n, n), dtype=int)
  for i in range(n):
    for j in range(i + 1, n):
      if (i - j) % n in residues:
        adj[i, j] = 1
        adj[j, i] = 1
  return adj


_PRIMES_CACHE: dict[int, Optional[int]] = {}


def find_closest_prime(num):
  """Finds the largest prime less than or equal to num."""
  if num in _PRIMES_CACHE:
    return _PRIMES_CACHE[num]
  i = num
  while i > 1:
    if all(i % j != 0 for j in range(2, int(i**0.5) + 1)):
      _PRIMES_CACHE[num] = i
      return i
    i -= 1
  return None


def find_graph():
  """Finds a Ramsey number lower bound graph.

  Returns:
    The adjacency matrix of the found graph.
  """
  start_time = time.time()
  time_limit = 3550

  best_g1 = np.array([[0]], dtype=int)
  stats = {
      "attempts": 0,
      "simulated_annealing_steps": 0,
      "extensions": 0,
  }

  k_target, alpha_target = 4, 15

  initial_n_start_search = (
      128  # Start search from N=128, a common benchmark for R(4,x)
  )
  n = initial_n_start_search

  while time.time() - start_time < time_limit:
    stats["attempts"] += 1

    current_n1 = best_g1.shape[0]
    if current_n1 > 1:
      n = current_n1 + 1  # Always try to extend the best graph found
    else:
      # If no good graph found yet, try increasing initial_n_start_search
      n = max(n, initial_n_start_search)
      initial_n_start_search = (
          n + 1
      )  # Increment for next attempt if current fails

    # Heuristic: If current N is already past gene pool effective range, it
    # might be better to cap the N increase or be more random, but the
    # individual gene pool checks `if u < MAX_N_FOR_GENE_POOL` handle this.

    # 1. Initial Graph Construction
    adj = np.zeros((n, n), dtype=int)

    if current_n1 == n - 1 and current_n1 > 1:
      adj[: best_g1.shape[0], : best_g1.shape[0]] = best_g1.copy()
      stats["extensions"] += 1

      # Smart connection for new vertex: copy an existing vertex's connections
      # and perturb
      if n > 1:
        # When extending, try to use gene pool to connect new node (n-1)
        # Bias connections to nodes that have appeared in successful graphs
        available_nodes_for_copy = [
            i
            for i, count in enumerate(_GENE_POOL_HIT_COUNT[: n - 1])
            if count > 0
        ]
        if available_nodes_for_copy:
          v_to_copy = random.choice(available_nodes_for_copy)
        else:
          v_to_copy = random.randrange(n - 1)

        for i in range(n - 1):
          orbit = get_orbit(n - 1, i, n)
          # Harmonic Gene Pool: Combine edge memory with distance (orbit) memory
          gene_pool_prob = 0
          if (
              v_to_copy < MAX_N_FOR_GENE_POOL
              and i < MAX_N_FOR_GENE_POOL
              and _GENE_POOL_HIT_COUNT[v_to_copy] > 0
          ):
            gene_pool_prob = (
                _EDGE_GENE_POOL[v_to_copy, i] / _GENE_POOL_HIT_COUNT[v_to_copy]
            )

          orbit_prob = 0.5
          if orbit < len(_ORBIT_GENE_POOL) and _ORBIT_HIT_COUNT[orbit] > 0:
            orbit_prob = _ORBIT_GENE_POOL[orbit] / _ORBIT_HIT_COUNT[orbit]

          target_density = 0.5
          # Blend edge probability (50%), orbit probability (30%), and target
          # density (20%)
          prob_of_edge = (
              gene_pool_prob * 0.5 + orbit_prob * 0.3 + target_density * 0.2
          )

          if random.random() < prob_of_edge:
            adj[n - 1, i] = 1
            adj[i, n - 1] = 1

        # After initial gene-pool based connection, perturb a small fraction
        # for the new vertex
        # This ensures exploration even with strong gene pool guidance
        for i in range(n - 1):
          if (
              random.random() < 0.02
          ):  # Smaller perturbation rate if gene pool is already guiding
            adj[n - 1, i] = 1 - adj[n - 1, i]
            adj[i, n - 1] = adj[n - 1, i]
    else:
      p = find_closest_prime(n)
      if p is None:
        p = n  # Use n if no smaller prime found, for non-prime constructions

      # Randomly choose between Paley-like and another construction
      if random.random() < 0.5:
        power = 2 if p % 4 == 1 else 3
        init_adj = create_paley_like_graph(p, power=power)
        adj[:p, :p] = init_adj
      else:
        # Decide whether to use gene pool guidance or Paley/QR
        use_gene_pool_init = False
        if n <= MAX_N_FOR_GENE_POOL:
          # Check if enough nodes in gene pool have been 'hit' to form a basis
          num_active_nodes_in_gene_pool = sum(
              1 for x in _GENE_POOL_HIT_COUNT[:n] if x > 0
          )
          if (
              num_active_nodes_in_gene_pool >= n * 0.5
          ):  # At least 50% of nodes have gene pool history
            use_gene_pool_init = True

        if use_gene_pool_init:
          for i in range(n):
            for j in range(i + 1, n):
              gene_pool_prob_sum = 0
              # Sum gene pool influence from both ends of the potential edge
              if _GENE_POOL_HIT_COUNT[i] > 0:
                gene_pool_prob_sum += (
                    _EDGE_GENE_POOL[i, j] / _GENE_POOL_HIT_COUNT[i]
                )
              if _GENE_POOL_HIT_COUNT[j] > 0:
                gene_pool_prob_sum += (
                    _EDGE_GENE_POOL[j, i] / _GENE_POOL_HIT_COUNT[j]
                )

              # Average probability, and blend with target density
              if _GENE_POOL_HIT_COUNT[i] > 0 or _GENE_POOL_HIT_COUNT[j] > 0:
                avg_gene_pool_prob = gene_pool_prob_sum / 2
              else:
                avg_gene_pool_prob = 0.5

              # Blend gene pool influence with target density and some
              # randomness
              target_density = 0.5
              prob_of_edge = (
                  avg_gene_pool_prob * 0.7 + target_density * 0.3
              )  # Stronger gene pool bias

              if random.random() < prob_of_edge:
                adj[i, j] = 1
                adj[j, i] = 1
          # Add a small perturbation to avoid getting stuck in local
          # optima from gene pool
          for i in range(n):
            for j in range(i + 1, n):
              if random.random() < 0.01:  # 1% chance to flip edge
                adj[i, j] = 1 - adj[i, j]
                adj[j, i] = adj[i, j]
        else:  # Fallback to Paley-like or QR if no strong gene pool history
          p = find_closest_prime(n)
          if p is None:
            # Use n if no smaller prime found, for non-prime constructions
            p = n

          # Randomly choose between Paley-like and another construction
          if random.random() < 0.5 and p > 1:  # Ensure p > 1 for Paley graph
            power = 2 if p % 4 == 1 else 3
            init_adj = create_paley_like_graph(p, power=power)
            adj[:p, :p] = init_adj
          elif p > 2:  # Quadratic residue needs p > 2
            # Quadratic residue graph construction
            residues = {pow(i, 2, p) for i in range(1, p // 2 + 1)}
            init_adj = np.zeros((p, p), dtype=int)
            for i in range(p):
              for j in range(i + 1, p):
                if (i - j) % p in residues:
                  init_adj[i, j] = 1
                  init_adj[j, i] = 1
            adj[:p, :p] = init_adj
          # Fill in the rest of the matrix for n > p, or if no Paley/QR was used
          for i in range(n):
            for j in range(i + 1, n):
              if (
                  adj[i, j] == 0 and random.random() < 0.5
              ):  # Fill empty spots with 0.5 density
                adj[i, j] = 1
                adj[j, i] = 1

    # 2. Optimized Local Search (Conflict-based)
    max_steps = 500000  # Allow many more steps due to speed
    tabu_size = 50
    tabu_list = collections.deque(maxlen=tabu_size)

    # Simulated Annealing parameters
    initial_temp = 1.0
    base_cooling_rate = 0.99995  # Slower cooling allows for more exploration
    temperature = initial_temp

    # Adaptive cooling parameters
    last_k4_count_check = len({
        tuple(sorted(c))
        for c in ig.Graph.Adjacency(adj.tolist(), mode="undirected").cliques(
            min=k_target, max=k_target
        )
    })  # Initialize with actual K4s
    steps_since_k4_change = 0

    best_valid_adj = (
        None  # Stores graph if it satisfies K4=0 and heuristic IS=0
    )

    # Initial K4s calculation for the starting graph, or re-use if existing adj
    g_current = ig.Graph.Adjacency(adj.tolist(), mode="undirected")
    current_k4s = {
        tuple(sorted(c)) for c in g_current.cliques(min=k_target, max=k_target)
    }

    step = 0
    for step in range(max_steps):
      if time.time() - start_time > time_limit - 5:
        break

      # Adaptive cooling rate
      current_cooling_rate = base_cooling_rate

      # Adjust cooling based on K4 count
      num_k4 = len(current_k4s)
      if num_k4 == 0:
        current_cooling_rate = (
            0.999999  # Very slow cooling when K4-free, focus on IS
        )
      elif num_k4 < 5:
        current_cooling_rate = 0.99999  # Slower cooling when close to K4-free

      # Track K4 count stagnation to potentially speed up cooling
      steps_since_k4_change += 1
      if num_k4 != last_k4_count_check:
        last_k4_count_check = num_k4
        steps_since_k4_change = 0
      elif (
          steps_since_k4_change > 10000 and num_k4 > 0
      ):  # If stagnant and not K4-free
        current_cooling_rate = max(
            0.9995, current_cooling_rate * 0.99
        )  # Speed up cooling slightly
        steps_since_k4_change = 0  # Reset after adjustment

      # Update temperature for SA
      temperature *= current_cooling_rate

      if step % 5000 == 0 and step > 0:

        # Periodically resync for robustness against missed K4s or if initial
        # K4s calculation was skipped
        g_current_sync = ig.Graph.Adjacency(adj.tolist(), mode="undirected")
        actual_k4s = {
            tuple(sorted(c))
            for c in g_current_sync.cliques(min=k_target, max=k_target)
        }
        if len(actual_k4s) != num_k4:
          current_k4s = actual_k4s
          num_k4 = len(current_k4s)

      elif step % 1000 == 0 and step > 0:

        # "Harmonic Tunnelling": If stuck with K4s, identify the most
        # 'toxic' distance orbit and flip it
        if num_k4 > 5:
          orbit_counts = {}
          for clique in random.sample(
              list(current_k4s), min(len(current_k4s), 30)
          ):
            for i in range(4):
              for j in range(i + 1, 4):
                orb = get_orbit(clique[i], clique[j], n)
                orbit_counts[orb] = orbit_counts.get(orb, 0) + 1
          if orbit_counts:
            toxic_orbit = max(orbit_counts, key=orbit_counts.get)
            # Flip all edges in the toxic orbit with high probability
            for i in range(n):
              j = (i + toxic_orbit) % n
              if random.random() < 0.3:  # Probability of tunnelling
                if adj[i, j] == 1:
                  adj[i, j] = 0
                  adj[j, i] = 0
                else:
                  adj[i, j] = 1
                  adj[j, i] = 1
            # Re-sync after massive change
            g_current = ig.Graph.Adjacency(adj.tolist(), mode="undirected")
            current_k4s = {
                tuple(sorted(c))
                for c in g_current.cliques(min=k_target, max=k_target)
            }
            num_k4 = len(current_k4s)

      if num_k4 == 0:
        # K4 constraint satisfied. Check Independent Set.
        # When K4 count is zero, prioritize thorough IS check
        violating_is = find_violating_independent_set(
            adj, alpha_target, max_trials=500
        )

        if violating_is is None:
          # Heuristically valid!

          try:

            g_comp = get_complement_graph(adj)

            # Check if a clique of size alpha_target exists in the complement
            # graph.
            # This is equivalent to checking if an independent set of size
            # alpha_target exists in the original graph.
            # Using g.cliques(min=k, max=k) is much faster than
            # g.clique_number()
            # as it stops after finding the first such clique.
            violating_is_candidates = g_comp.cliques(
                min=alpha_target, max=alpha_target
            )

            if (
                not violating_is_candidates
            ):  # If no independent set of size alpha_target was found
              best_valid_adj = adj.copy()
              break  # Break search loop to save result
            else:
              # An independent set of size alpha_target was found.
              violating_is = set(violating_is_candidates[0])
          except Exception:  # pylint: disable=broad-exception-caught

            # If exact check failed, and heuristic also failed to find an IS,
            # it implies alpha is likely ok, so consider it valid.
            if violating_is is None:
              best_valid_adj = adj.copy()
              break  # Break search loop to save result

        if violating_is is not None:
          # We have an IS violation. Break it by ADDING an edge.
          is_nodes = list(violating_is)

          # Try to add an edge between two nodes in the violating IS
          # Prioritize adding edges that create the fewest new K4s.
          best_move = (None, float("inf"))

          # Sample pairs from the IS to consider adding an edge
          # Limit the number of pairs to check for performance
          if len(is_nodes) > 2:
            sample_size = min(len(is_nodes) * (len(is_nodes) - 1) // 2, 100)
            pairs = random.sample(
                list(zip(*np.triu_indices(len(is_nodes), 1))), sample_size
            )
          else:
            pairs = [(0, 1)]

          for i, j in pairs:
            u, v = is_nodes[i], is_nodes[j]
            if adj[u, v] == 0:
              potential_k4s = _calculate_potential_new_k4s(adj, u, v)
              if potential_k4s < best_move[1]:
                best_move = ((u, v), potential_k4s)
                if potential_k4s == 0:
                  break  # Found a zero-cost move

          if best_move[0] is not None:
            u, v = best_move[0]
            adj[u, v] = 1
            adj[v, u] = 1
            _update_k4s_add_edge(adj, u, v, current_k4s, k_target)
            tabu_list.append(tuple(sorted((u, v))))
          else:
            # Fallback if no edge could be added (should be rare)
            if len(is_nodes) >= 2:
              u, v = random.sample(is_nodes, 2)
              if adj[u, v] == 0:
                adj[u, v] = 1
                adj[v, u] = 1
                _update_k4s_add_edge(adj, u, v, current_k4s, k_target)
                tabu_list.append(tuple(sorted((u, v))))

      else:
        # We have K4 violations. Break them by REMOVING edges.

        # Sample a few K4s from current_k4s to find a good edge to remove
        k4_samples_list = list(current_k4s)

        if (
            not k4_samples_list
        ):  # Should not happen if num_k4 > 0, but safeguard

          g_current_resync = ig.Graph.Adjacency(adj.tolist(), mode="undirected")
          current_k4s = {
              tuple(sorted(c))
              for c in g_current_resync.cliques(min=k_target, max=k_target)
          }
          k4_samples_list = list(current_k4s)
          if not k4_samples_list:

            continue  # Skip this step

        sampled_k4s = random.sample(
            k4_samples_list, min(len(k4_samples_list), 20)
        )  # Sample more K4s
        best_edge_to_remove = None
        candidate_edges = set()
        for clique in sampled_k4s:
          for i in range(k_target):
            for j in range(i + 1, k_target):
              candidate_edges.add(tuple(sorted((clique[i], clique[j]))))

        candidate_moves = []
        # Look-ahead penalty weight
        look_ahead_weight = 0.1

        for u, v in candidate_edges:
          edge_key = tuple(sorted((u, v)))
          if edge_key in tabu_list:
            continue

          # Calculate how many K4s this edge is part of
          common_neighbors = _get_common_neighbors(adj, u, v)
          k4s_involving_uv = 0
          cn_len = len(common_neighbors)
          if cn_len >= 2:
            for i in range(cn_len):
              for j in range(i + 1, cn_len):
                if adj[common_neighbors[i], common_neighbors[j]] == 1:
                  k4s_involving_uv += 1

          if k4s_involving_uv > 0:
            # Look-ahead penalty: penalize if removing (u,v) makes a large IS
            # more likely
            # This is approximated by the number of common non-neighbors of u
            # and v
            non_neighbors_u = np.where(adj[u] == 0)[0]
            non_neighbors_v = np.where(adj[v] == 0)[0]
            common_non_neighbors = np.intersect1d(
                non_neighbors_u, non_neighbors_v, assume_unique=True
            )

            # The new IS component could include u, v, and their common
            # non-neighbors
            is_potential = len(common_non_neighbors) + 2

            # Normalize IS potential: penalize more if potential > alpha_target
            # - 4
            is_penalty = max(0, is_potential - (alpha_target - 4))

            # Calculate gene pool score for removing this edge
            gene_pool_score = 0
            if u < MAX_N_FOR_GENE_POOL and v < MAX_N_FOR_GENE_POOL:
              gene_pool_score = _EDGE_GENE_POOL[
                  u, v
              ]  # Higher means it's a "good" edge

            # Harmony-based scoring: Penalize removing edges whose distance
            # orbit is historically successful
            orbit = get_orbit(u, v, n)
            orbit_score = 0
            if orbit < len(_ORBIT_GENE_POOL) and _ORBIT_HIT_COUNT[orbit] > 0:
              orbit_score = _ORBIT_GENE_POOL[orbit] / _ORBIT_HIT_COUNT[orbit]

            gene_pool_penalty_weight = 0.05
            # High weight for preserving successful "vibrational frequencies"
            orbit_weight = 2.0

            score = (
                k4s_involving_uv
                - look_ahead_weight * is_penalty
                - (gene_pool_penalty_weight * gene_pool_score)
                - (orbit_weight * orbit_score)
            )
            candidate_moves.append(((u, v), score))

        if candidate_moves:
          candidate_moves.sort(
              key=lambda x: x[1], reverse=True
          )  # Higher score is better (more K4s broken, less IS risk)
          best_move = candidate_moves[0]
          chosen_move = best_move

          if len(candidate_moves) > 1 and temperature > 0.01:
            # SA: pick a different (potentially worse) move
            rand_idx = random.randint(0, len(candidate_moves) - 1)
            random_candidate = candidate_moves[rand_idx]

            # "cost" is negative score, so delta_cost = (-rand_score) -
            # (-best_score) = best_score - rand_score
            delta_score = best_move[1] - random_candidate[1]  # >= 0
            if delta_score > 0 and random.random() < np.exp(
                -delta_score / temperature
            ):
              chosen_move = random_candidate

          best_edge_to_remove = chosen_move[0]

        if best_edge_to_remove:
          u, v = best_edge_to_remove
          adj[u, v] = 0
          adj[v, u] = 0
          _update_k4s_remove_edge(u, v, current_k4s)  # Update tracked K4s
          tabu_list.append(tuple(sorted((u, v))))
        else:
          # If all moves tabu or no good moves, random perturbation.
          if tabu_list:
            tabu_list.popleft()
          # Randomly pick an edge from a random K4 to remove, if exists.
          if sampled_k4s:
            clique_to_break = random.choice(sampled_k4s)
            u, v = random.sample(clique_to_break, 2)
            adj[u, v] = 0
            adj[v, u] = 0
            _update_k4s_remove_edge(u, v, current_k4s)
            tabu_list.append(tuple(sorted((u, v))))
          else:  # Completely stuck, just flip a random edge
            u, v = random.sample(range(n), 2)
            if u == v:
              continue
            # Remove if exists, add if not. Aim for sparser initially.
            if adj[u, v] == 1:
              adj[u, v] = 0
              adj[v, u] = 0
              _update_k4s_remove_edge(u, v, current_k4s)
            else:
              adj[u, v] = 1
              adj[v, u] = 1
              _update_k4s_add_edge(adj, u, v, current_k4s, k_target)
            tabu_list.append(tuple(sorted((u, v))))

    stats["simulated_annealing_steps"] += step
    if best_valid_adj is not None:
      if best_valid_adj.shape[0] > best_g1.shape[0]:
        best_g1 = best_valid_adj

        current_n_g1 = best_g1.shape[0]
        if current_n_g1 <= MAX_N_FOR_GENE_POOL:
          for i in range(current_n_g1):
            for j in range(i + 1, current_n_g1):
              if best_g1[i, j] == 1:
                _EDGE_GENE_POOL[i, j] += 1
                _EDGE_GENE_POOL[j, i] += 1
                orbit = get_orbit(i, j, current_n_g1)
                if orbit < len(_ORBIT_GENE_POOL):
                  _ORBIT_GENE_POOL[orbit] += 1
            _GENE_POOL_HIT_COUNT[i] += 1
          for d in range(1, current_n_g1 // 2 + 1):
            if d < len(_ORBIT_HIT_COUNT):
              _ORBIT_HIT_COUNT[d] += 1
        initial_n_start_search = current_n_g1 + 1
      else:
        # If a valid graph was found but it's not larger, still good to record
        # in gene pool
        # This ensures gene pool is updated even if we find multiple valid
        # graphs of the same max size.
        current_n_g1 = best_valid_adj.shape[0]
        if current_n_g1 <= MAX_N_FOR_GENE_POOL:
          for i in range(current_n_g1):
            for j in range(i + 1, current_n_g1):
              if best_valid_adj[i, j] == 1:
                _EDGE_GENE_POOL[i, j] += 1
                _EDGE_GENE_POOL[j, i] += 1
            _GENE_POOL_HIT_COUNT[i] += 1

  return best_g1
