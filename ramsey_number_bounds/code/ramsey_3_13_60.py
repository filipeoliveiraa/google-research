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

import math
import random
import time

import igraph as ig
import numpy as np


def get_counts_and_sets(g, return_sets=False):
  c3_list = g.cliques(min=3, max=3)
  cliques_3 = len(c3_list)

  g_comp = g.complementer(loops=False)
  ac13_list = g_comp.cliques(min=13, max=13)
  anticliques_13 = len(ac13_list)

  if return_sets:
    return cliques_3, anticliques_13, c3_list, ac13_list
  return cliques_3, anticliques_13


def adj_to_igraph(adj):
  n = adj.shape[0]
  rows, cols = np.where(np.triu(adj, 1))
  edges = list(zip(rows, cols))
  g = ig.Graph(n, edges=edges)
  return g


def solve_cyclic(n, duration):
  """Attempts to find a valid cyclic graph of size n.

  Args:
    n: The size of the graph.
    duration: Time limit in seconds.

  Returns:
    The adjacency matrix if found, or None.
  """
  t0 = time.time()
  candidates = list(range(1, n // 2 + 1))

  best_indep = 999

  while time.time() - t0 < duration:
    # Randomized greedy with slight variations
    random.shuffle(candidates)
    s = set()
    s_sym = set()

    # Try to pack as many differences as possible
    for x in candidates:
      if (2 * x) % n in s_sym:
        continue

      conflict = False
      for s_val in s_sym:
        if (x + s_val) % n in s_sym:
          conflict = True
          break
      if conflict:
        continue

      s.add(x)
      s_sym.add(x)
      s_sym.add(n - x)

    # Construct adjacency
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
      for d in s:
        adj[i, (i + d) % n] = 1
        adj[(i + d) % n, i] = 1

    # Check independence number
    g = ig.Graph.Adjacency(adj.tolist(), mode="undirected")

    alpha = g.independence_number()
    if alpha < 13:
      return adj

    if alpha < best_indep:
      best_indep = alpha

  return None


def find_graph():
  """Finds a Ramsey number lower bound graph.

  Returns:
    The adjacency matrix of the best graph found.
  """
  start_time = time.time()
  time_limit = 1900

  best_g1 = np.array([[0]], dtype=int)
  stats = {
      "solve_cyclic_calls": 0,
      "solve_cyclic_successes": 0,
      "sa_iterations": 0,
      "sa_extensions": 0,
  }

  # Start cyclic search from n=45
  curr_n = 45

  # Budget for cyclic search: ~30% of time or until failure
  while time.time() - start_time < time_limit * 0.3:
    stats["solve_cyclic_calls"] += 1
    res = solve_cyclic(curr_n, 3.0)  # Give 3 seconds per size
    if res is not None:
      best_g1 = res
      stats["solve_cyclic_successes"] += 1
      curr_n += 1
    else:
      # Retry once with more time
      stats["solve_cyclic_calls"] += 1
      res = solve_cyclic(curr_n, 8.0)
      if res is not None:
        best_g1 = res
        stats["solve_cyclic_successes"] += 1
        curr_n += 1
      else:
        break

  # -------------------------------------------------------------------------
  # Phase 2: Simulated Annealing Refinement
  # -------------------------------------------------------------------------
  # Start from best_g1 found (or 1x1 if none) and try to extend.

  adj = best_g1.copy()
  current_n = adj.shape[0]

  # If we somehow have nothing substantial, reset to a small valid seed
  if current_n < 2:
    adj = np.array([[0]], dtype=int)
    current_n = 1

  while time.time() - start_time < time_limit:
    # Attempt to extend to current_n + 1
    target_n = current_n + 1
    new_adj = np.zeros((target_n, target_n), dtype=int)
    new_adj[:-1, :-1] = adj

    # Refine new_adj using Simulated Annealing.
    # Cost function: dynamic weights.

    g_old = adj_to_igraph(adj)
    # Adjust SA budget based on current_n, giving more time to larger N
    sa_cap_per_n = 150.0
    # Based on empirical observation that R(3,13) is likely above 57,
    # we dedicate more time to these challenging N values.
    if current_n >= 57:
      sa_cap_per_n = 300.0  # Double time for crucial N values
    if current_n >= 59:
      sa_cap_per_n = 500.0  # Even more time if we push further

    initial_sa_time_budget = min(
        sa_cap_per_n, (time_limit - (time.time() - start_time)) / 1.1
    )
    sa_phase_start_time = (
        time.time()
    )  # This marks the start of the SA phase for this target_n

    candidate_initial_solutions = []

    # Strategy 1: Hitting Set based Independent Set (Smart initialization)
    # We try to find an independent set in g_old that hits as many
    # 12-anticliques as possible.
    # If it hits all 12-anticliques, connecting the new vertex to it avoids
    # new 13-anticliques.
    try:
      t_start_heur = time.time()
      g_old_comp = g_old.complementer(loops=False)
      # Find all 12-anticliques (cliques in complement)
      # This might be large, so we limit or monitor time?
      # Assuming it's fast enough for n < 70 for this specific problem class
      ac12_list = g_old_comp.cliques(min=12, max=12)

      if len(ac12_list) > 0:
        # Greedy hitting set construction
        # We want an IS 'I' that intersects every set in ac12_list
        # Also I must be an IS in g_old

        # Attempt multiple greedy constructions
        for _ in range(5):
          if time.time() - t_start_heur > 5.0:
            break

          i_set = set()
          uncovered = list(range(len(ac12_list)))
          # Valid candidates are all vertices initially
          # But as we add to i_set, we restrict candidates to non-neighbors

          # Optimization: track candidates
          candidates = set(range(current_n))

          while uncovered:
            # Score candidates by how many uncovered sets they hit
            best_v = -1
            best_score = -1

            # Sampling candidates if too many?
            cand_list = list(candidates)
            if len(cand_list) > 100:
              cand_sample = random.sample(cand_list, 100)
            else:
              cand_sample = cand_list

            for v in cand_sample:
              score = 0
              for idx in uncovered:
                if v in ac12_list[idx]:
                  score += 1
              if score > best_score:
                best_score = score
                best_v = v

            if best_v == -1 or best_score == 0:
              break  # Cannot hit remaining sets with valid IS candidates

            i_set.add(best_v)
            # Update candidates: remove best_v and its neighbors in g_old
            candidates.discard(best_v)
            nbrs = g_old.neighbors(best_v)
            candidates.difference_update(nbrs)

            # Update uncovered
            new_uncovered = []
            for idx in uncovered:
              if best_v not in ac12_list[idx]:
                new_uncovered.append(idx)
            uncovered = new_uncovered

          # Construct solution
          init_adj_hitting = new_adj.copy()
          for v in i_set:
            init_adj_hitting[v, target_n - 1] = 1
            init_adj_hitting[target_n - 1, v] = 1
          candidate_initial_solutions.append(init_adj_hitting)

          if not uncovered:
            # Found a perfect hitting set! This is extremely promising.
            break
      else:
        # No 12-anticliques, any IS is safe from creating 13-anticliques
        # involving new vertex. Just use max IS.
        pass

    except (ValueError, RuntimeError):
      pass

    # Strategy 2: Connect to Largest Independent Set (Fallback/Standard)
    try:
      # Get a few largest independent sets
      is_sets = g_old.largest_independent_vertex_sets()
      if is_sets:
        # Pick up to 3 unique ones
        for is_nodes in is_sets[:3]:
          temp_adj = new_adj.copy()
          for v in is_nodes:
            temp_adj[v, target_n - 1] = 1
            temp_adj[target_n - 1, v] = 1
          candidate_initial_solutions.append(temp_adj)
    except (ValueError, RuntimeError):
      pass

    # Strategy 3: Connect new vertex to a random maximal independent set
    for _ in range(2):
      init_adj_random_is = new_adj.copy()
      perm = list(range(current_n))
      random.shuffle(perm)
      is_nodes_random_maximal = []
      blocked = set()
      for v in perm:
        if v not in blocked:
          is_nodes_random_maximal.append(v)
          blocked.add(v)
          blocked.update(g_old.neighbors(v))
      for v in is_nodes_random_maximal:
        init_adj_random_is[v, target_n - 1] = 1
        init_adj_random_is[target_n - 1, v] = 1
      candidate_initial_solutions.append(init_adj_random_is)

    # Strategy 4: Connect to a random set of vertices (average density
    # heuristic)
    init_adj_random_edges = new_adj.copy()
    num_random_edges = random.randint(
        int(current_n * 0.25), int(current_n * 0.45)
    )
    potential_neighbors = list(range(current_n))
    random.shuffle(potential_neighbors)
    for v in potential_neighbors[:num_random_edges]:
      init_adj_random_edges[v, target_n - 1] = 1
      init_adj_random_edges[target_n - 1, v] = 1
    candidate_initial_solutions.append(init_adj_random_edges)

    # Strategy 4: Start with no edges for the new vertex (isolated)
    init_adj_isolated = (
        new_adj.copy()
    )  # new_adj is already mostly zeros for the new vertex
    candidate_initial_solutions.append(init_adj_isolated)

    # Ensure at least one candidate (e.g., if
    # largest_independent_vertex_sets failed and others somehow empty)
    if not candidate_initial_solutions:
      candidate_initial_solutions.append(
          new_adj.copy()
      )  # Fallback to isolated new vertex

    # Limit number of candidates to avoid dilution of effort
    if len(candidate_initial_solutions) > 10:
      candidate_initial_solutions = random.sample(
          candidate_initial_solutions, 10
      )

    best_init_sol = None
    best_init_score = float("inf")

    # Budget for mini-SA on each initialization candidate
    # Allocate 20% of total SA budget to initializations, split evenly.
    # Ensure a minimum and maximum for each mini-SA to prevent it from being
    # too short or too long.
    mini_sa_duration_per_init = (
        initial_sa_time_budget * 0.2 / len(candidate_initial_solutions)
    )
    mini_sa_duration_per_init = max(
        0.5, min(mini_sa_duration_per_init, 15.0)
    )  # Max 15s per mini-SA to allow more thorough initialization

    for initial_sol_candidate in candidate_initial_solutions:
      if time.time() - start_time > time_limit:
        break  # Check overall time limit

      temp_sol = initial_sol_candidate.copy()

      # Mini-SA Parameters
      # Use a higher initial temperature for better exploration in mini-SA
      temp_initial_temp = 1.0
      temp_temp = temp_initial_temp
      temp_cooling = (
          0.995  # Slightly slower cooling for mini-SA to allow more exploration
      )
      temp_sa_start = time.time()

      temp_w_c3 = 10.0
      temp_w_ac13 = 1.0

      temp_graph = adj_to_igraph(temp_sol)
      temp_c3, temp_ac13, temp_c3_list, temp_ac13_list = get_counts_and_sets(
          temp_graph, return_sets=True
      )

      if temp_c3 > temp_ac13 and temp_ac13 > 0:
        temp_w_c3 = temp_c3 / temp_ac13
      elif temp_ac13 > temp_c3 and temp_c3 > 0:
        temp_w_ac13 = temp_ac13 / temp_c3

      temp_score = temp_c3 * temp_w_c3 + temp_ac13 * temp_w_ac13

      temp_best_local_sol = temp_sol.copy()
      temp_best_local_score = temp_score

      temp_iter_count = 0
      temp_no_improvement_count = 0

      while time.time() - temp_sa_start < mini_sa_duration_per_init:
        if temp_score == 0:
          break

        temp_iter_count += 1
        temp_no_improvement_count += 1

        # Adaptive Weight Update for mini-SA
        if (
            temp_iter_count % 20 == 0 and temp_score > 0
        ):  # More frequent weight updates in mini-SA
          if temp_c3 > 0.1 * temp_ac13:
            temp_w_c3 *= 1.2
          elif temp_ac13 > 0.1 * temp_c3:
            temp_w_ac13 *= 1.2

          temp_w_sum = temp_w_c3 + temp_w_ac13
          if temp_w_sum > 0:
            temp_w_c3 = (temp_w_c3 / temp_w_sum) * 20
            temp_w_ac13 = (temp_w_ac13 / temp_w_sum) * 20

          temp_score = temp_c3 * temp_w_c3 + temp_ac13 * temp_w_ac13

        temp_candidate = temp_sol.copy()

        # Move Selection Strategy for mini-SA
        if temp_c3 > 0 and (random.random() < 0.6 or temp_ac13 == 0):
          tri = random.choice(temp_c3_list)
          u, v = random.sample(tri, 2)
          if temp_candidate[u, v] == 1:
            temp_candidate[u, v] = 0
            temp_candidate[v, u] = 0
        elif temp_ac13 > 0:
          ac = random.choice(temp_ac13_list)
          u, v = random.sample(ac, 2)
          if temp_candidate[u, v] == 0:
            temp_candidate[u, v] = 1
            temp_candidate[v, u] = 1
        else:  # Random flip if no conflicts or chosen by probability
          u, v = random.randrange(target_n), random.randrange(target_n)
          if u == v:
            continue
          temp_candidate[u, v] = 1 - temp_candidate[u, v]
          temp_candidate[v, u] = 1 - temp_candidate[v, u]

        temp_cand_graph = adj_to_igraph(temp_candidate)
        temp_nc3, temp_nac13, temp_nc3_list, temp_nac13_list = (
            get_counts_and_sets(temp_cand_graph, return_sets=True)
        )
        temp_new_score = temp_nc3 * temp_w_c3 + temp_nac13 * temp_w_ac13

        temp_delta = temp_new_score - temp_score

        temp_accept = False
        if temp_delta < 0:
          temp_accept = True
        else:
          if random.random() < math.exp(-temp_delta / temp_temp):
            temp_accept = True

        if temp_accept:
          temp_sol = temp_candidate
          temp_score = temp_new_score
          temp_c3, temp_ac13 = temp_nc3, temp_nac13
          temp_c3_list, temp_ac13_list = temp_nc3_list, temp_nac13_list

          if temp_score < temp_best_local_score:
            temp_best_local_score = temp_score
            temp_best_local_sol = temp_sol.copy()
            temp_no_improvement_count = 0

        temp_temp *= temp_cooling

        # Reheating schedule for mini-SA
        # Use a fixed threshold for mini-SAs to be less sensitive to N for
        # short runs
        if temp_no_improvement_count > 50 and temp_score > 0:
          temp_temp = (
              temp_initial_temp * (1 + temp_score / 20)
              if temp_score > 0
              else temp_initial_temp * 0.8
          )  # Reheat based on how bad the score is
          temp_no_improvement_count = 0
        elif temp_temp < temp_initial_temp * 0.005:  # Lower bound for temp
          temp_temp = temp_initial_temp * 0.05

      if temp_best_local_score < best_init_score:
        best_init_score = temp_best_local_score
        best_init_sol = temp_best_local_sol.copy()

    # After all mini-SAs, initialize the main SA with the best solution found
    curr_sol = (
        best_init_sol.copy() if best_init_sol is not None else new_adj.copy()
    )  # Fallback to isolated new vertex

    # SA Parameters for main run
    initial_temp = (
        1.5  # Higher initial temp for main SA for broader exploration
    )
    temp = initial_temp
    cooling = (
        0.9995  # Slower cooling for main SA to allow more time at higher temps
    )
    max_sa_time = initial_sa_time_budget - (
        time.time() - sa_phase_start_time
    )  # Remaining budget after initializations
    if max_sa_time < 0.1:
      max_sa_time = 0.1  # Ensure some minimum time for the main SA
    sa_start = time.time()  # Reset start time for main SA

    # New parameter for random flip probability
    random_flip_prob = (
        0.05  # 5% chance of a random edge flip to escape local minima
    )
    # Probability to perturb an edge involving the newly added vertex
    # (target_n - 1)
    new_vertex_perturb_prob = 0.3  # 30% chance to focus on new vertex's edges

    # Adaptive Weights Parameters for main run
    w_c3 = 10.0
    w_ac13 = 1.0

    weight_update_interval = 50
    weight_increase_factor = 1.5

    curr_graph = adj_to_igraph(curr_sol)
    c3, ac13, c3_list, ac13_list = get_counts_and_sets(
        curr_graph, return_sets=True
    )

    # Initialize weights based on current state
    if c3 > ac13 and ac13 > 0:
      w_c3 = c3 / ac13
    elif ac13 > c3 and c3 > 0:
      w_ac13 = ac13 / c3

    curr_score = c3 * w_c3 + ac13 * w_ac13

    best_local_sol = curr_sol.copy()
    best_local_score = curr_score

    found_next = False

    iter_count = 0
    no_improvement_count = 0
    while time.time() - sa_start < max_sa_time:
      if curr_score == 0:
        found_next = True
        break

      iter_count += 1
      no_improvement_count += 1

      # --- Adaptive Weight Update ---
      if iter_count % weight_update_interval == 0 and curr_score > 0:
        if c3 > 0.1 * ac13:
          w_c3 *= weight_increase_factor
        elif ac13 > 0.1 * c3:
          w_ac13 *= weight_increase_factor

        # Normalize weights to prevent explosion
        w_sum = w_c3 + w_ac13
        w_c3 = (w_c3 / w_sum) * 20
        w_ac13 = (w_ac13 / w_sum) * 20

        # Re-calculate scores with new weights
        curr_score = c3 * w_c3 + ac13 * w_ac13
        best_local_score = (
            get_counts_and_sets(adj_to_igraph(best_local_sol))[0] * w_c3
            + get_counts_and_sets(adj_to_igraph(best_local_sol))[1] * w_ac13
        )

      # Move Selection Strategy
      candidate = curr_sol.copy()

      # Prioritize fixing problems, with a small chance of a pure random flip
      if random.random() < random_flip_prob:
        # Pure random edge flip for broader exploration
        u, v = random.randrange(target_n), random.randrange(target_n)
        if u == v:
          continue
        candidate[u, v] = 1 - candidate[u, v]
        candidate[v, u] = 1 - candidate[v, u]
      elif random.random() < new_vertex_perturb_prob:
        # Prioritize flipping an edge connected to the new vertex
        v_new = target_n - 1
        u_old = random.randrange(v_new)  # Pick a random existing vertex
        candidate[u_old, v_new] = 1 - candidate[u_old, v_new]
        candidate[v_new, u_old] = 1 - candidate[v_new, u_old]
      elif c3 > 0 and (random.random() < 0.6 or ac13 == 0):
        # Break a triangle by removing an edge
        tri = random.choice(c3_list)
        u, v = random.sample(tri, 2)
        # c3_list is always updated when a solution is accepted, so the edge
        # must exist.
        candidate[u, v] = 0
        candidate[v, u] = 0
      elif ac13 > 0:
        # Break an anticlique by adding an edge
        ac = random.choice(ac13_list)
        u, v = random.sample(ac, 2)
        # ac13_list represents non-edges, so the edge must not exist.
        candidate[u, v] = 1
        candidate[v, u] = 1
      else:  # If no conflicts (c3=0 and ac13=0), do a random flip.
        # This branch should ideally only be taken when score is already 0,
        # but serves as a safeguard.
        u, v = random.randrange(target_n), random.randrange(target_n)
        if u == v:
          continue
        candidate[u, v] = 1 - candidate[u, v]
        candidate[v, u] = 1 - candidate[v, u]

      # Re-evaluate
      cand_graph = adj_to_igraph(candidate)
      nc3, nac13, nc3_list, nac13_list = get_counts_and_sets(
          cand_graph, return_sets=True
      )
      new_score = nc3 * w_c3 + nac13 * w_ac13

      delta = new_score - curr_score

      accept = False
      if delta < 0:
        accept = True
      else:
        if random.random() < math.exp(-delta / temp):
          accept = True

      if accept:
        curr_sol = candidate
        curr_score = new_score
        c3, ac13 = nc3, nac13
        c3_list, ac13_list = nc3_list, nac13_list

        if curr_score < best_local_score:
          best_local_score = curr_score
          best_local_sol = curr_sol.copy()
          no_improvement_count = 0  # Reset counter

      temp *= cooling

      # Reheating schedule
      # Reheat if stuck for a period, adapting to graph size
      if (
          no_improvement_count > target_n * 15
      ):  # Increased threshold for deeper search before reheating
        temp = (
            initial_temp * (1 + curr_score / (w_c3 + w_ac13))
            if curr_score > 0
            else initial_temp * 0.8
        )  # Reheat based on how bad the score is
        no_improvement_count = 0
      elif (
          temp < initial_temp * 0.005
      ):  # Gentle reheat if temp gets too low, relative to initial
        temp = initial_temp * 0.05

    stats["sa_iterations"] += iter_count
    if found_next:
      adj = curr_sol
      best_g1 = adj.copy()
      current_n = target_n
      stats["sa_extensions"] += 1
    else:
      pass

  return best_g1
