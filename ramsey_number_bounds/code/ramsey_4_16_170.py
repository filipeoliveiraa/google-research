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
import numpy as np


def get_bit_count(n):
  return n.bit_count() if hasattr(n, "bit_count") else bin(n).count("1")


def count_k4_containing_0(n, s_full_bitset):
  """Counts the number of K4 cliques containing vertex 0.

  Args:
    n: The number of vertices in the graph.
    s_full_bitset: Bitset representing the edges from vertex 0.

  Returns:
    The number of K4 cliques containing vertex 0.
  """
  s_list = [i for i in range(1, n) if (s_full_bitset >> i) & 1]
  nn = len(s_list)
  if nn < 3:
    return 0
  adj = [0] * nn
  for i in range(nn):
    vi = s_list[i]
    for j in range(i + 1, nn):
      if (s_full_bitset >> (s_list[j] - vi)) & 1:
        adj[i] |= 1 << j
  count = 0
  for i in range(nn):
    ai = adj[i]
    for j in range(i + 1, nn):
      if (ai >> j) & 1:
        count += get_bit_count(ai & adj[j])
  return count // 3


def find_one_k4_generators_containing_0(
    n, s_full_bitset
):
  """Finds one K4 clique containing vertex 0 and returns its generators.

  Args:
    n: The number of vertices in the graph.
    s_full_bitset: Bitset representing the edges from vertex 0.

  Returns:
    A list of edge generators for the K4 clique, or None if none found.
  """
  s_list_neighbors = [i for i in range(1, n) if (s_full_bitset >> i) & 1]
  nn = len(s_list_neighbors)
  if nn < 3:
    return None
  adj = [0] * nn
  for i in range(nn):
    vi = s_list_neighbors[i]
    for j in range(i + 1, nn):
      if (s_full_bitset >> (s_list_neighbors[j] - vi)) & 1:
        adj[i] |= 1 << j
  for i in range(nn):
    ai = adj[i]
    for j in range(i + 1, nn):
      if (ai >> j) & 1:
        common_neighbors = ai & adj[j]
        if common_neighbors:
          k = (common_neighbors & -common_neighbors).bit_length() - 1
          v_i, v_j, v_k = (
              s_list_neighbors[i],
              s_list_neighbors[j],
              s_list_neighbors[k],
          )
          gens = {v_i, v_j, v_k, v_j - v_i, v_k - v_i, v_k - v_j}
          norm_gens = set()
          for g in gens:
            g_norm = g % n
            if g_norm > n // 2:
              g_norm = n - g_norm
            if g_norm > 0:
              norm_gens.add(g_norm)
          return list(norm_gens)
  return None


def get_alpha_and_witness_containing_0(
    n, s_full_bitset, k_limit
):
  """Computes alpha (max independent set) and a witness containing vertex 0.

  Args:
    n: The number of vertices in the graph.
    s_full_bitset: Bitset representing the edges from vertex 0.
    k_limit: The limit for the independent set size.

  Returns:
    A tuple (alpha, witness) where alpha is the size of the max independent set
    and witness is a list of vertices in that set.
  """
  s_comp_list = [i for i in range(1, n) if not ((s_full_bitset >> i) & 1)]
  nn = len(s_comp_list)
  if nn < k_limit - 1:
    return nn + 1, [0] + s_comp_list
  adj = [0] * nn
  for i in range(nn):
    vi = s_comp_list[i]
    for j in range(i + 1, nn):
      if not ((s_full_bitset >> (s_comp_list[j] - vi)) & 1):
        adj[i] |= 1 << j
        adj[j] |= 1 << i
  max_c, witness_clique_indices, _clique = 1, [], []

  def backtrack(candidates):
    nonlocal max_c, witness_clique_indices
    if not candidates:
      if len(_clique) + 1 > max_c:
        max_c = len(_clique) + 1
        witness_clique_indices = list(_clique)
      return
    if len(_clique) + 1 + get_bit_count(candidates) <= max_c:
      return
    temp, best_v, max_deg = candidates, -1, -1
    while temp:
      v = (temp & -temp).bit_length() - 1
      deg = get_bit_count(adj[v] & candidates)
      if deg > max_deg:
        max_deg, best_v = deg, v
      temp &= temp - 1
    _clique.append(best_v)
    backtrack(candidates & adj[best_v])
    _clique.pop()
    if max_c >= k_limit:
      return
    backtrack(candidates ^ (1 << best_v))

  backtrack((1 << nn) - 1)
  return max_c, [0] + [s_comp_list[i] for i in witness_clique_indices]


def get_adj_matrix(n, s):
  adj_mat = np.zeros((n, n), dtype=np.int8)
  for s_val in s:
    for i in range(n):
      adj_mat[i, (i + s_val) % n] = 1
      adj_mat[i, (i - s_val) % n] = 1
  return adj_mat


def find_graph():
  """Finds a Ramsey number lower bound using simulated annealing.

  Returns:
      np.ndarray: The adjacency matrix of the found graph.
  """

  start_time = time.time()

  best_g1 = np.array([[0]], dtype=np.int8)
  last_s = None
  for n in range(155, 181):
    now = time.time()
    remaining = 1990 - (now - start_time)
    if remaining < 10:
      break
    time_limit_for_n = (
        min(now + remaining / (181 - n) * 1.2, start_time + 1995)
        if n < 181
        else start_time + 1995
    )
    n_half = n // 2
    if last_s:
      s = {s_val for s_val in last_s if s_val <= n_half}
    else:
      s = set(random.sample(range(1, n_half + 1), max(1, int(n * 0.18))))
    s_full_bitset = 0
    for s_val in s:
      s_full_bitset |= (1 << s_val) | (1 << (n - s_val))
    # Initializing best state for current n
    k4_c_initial = count_k4_containing_0(n, s_full_bitset)
    if k4_c_initial > 0:
      current_score, current_witness = 1000 + k4_c_initial, None

    else:
      alpha_result_initial = get_alpha_and_witness_containing_0(
          n, s_full_bitset, 16
      )
      current_score, current_witness = (
          alpha_result_initial[0],
          alpha_result_initial[1],
      )

    best_s_for_current_n = s.copy()
    best_score_for_current_n = current_score

    initial_temp = 0.5  # Initial temperature
    cooling_rate = 0.9999  # Cooling rate
    temp = initial_temp
    iter_count = 0
    iterations_since_last_improvement = 0

    while time.time() < time_limit_for_n:
      iter_count += 1
      iterations_since_last_improvement += 1

      if iterations_since_last_improvement >= 8000:
        # Symmetry-Invariant Automorphism Teleportation
        s = best_s_for_current_n.copy()
        k_cands = [k_val for k_val in range(2, n) if math.gcd(k_val, n) == 1]
        if k_cands:
          k_val = random.choice(k_cands)
          new_s = set()
          for s_val in s:
            ns = (s_val * k_val) % n
            if ns > n_half:
              ns = n - ns
            if ns != 0:
              new_s.add(ns)
          s = new_s

        # Occasional random perturbation after teleportation
        if random.random() < 0.3:
          for _ in range(random.randint(1, 3)):
            d_p = random.randint(1, n_half)
            if d_p in s:
              s.remove(d_p)
            else:
              s.add(d_p)

        s_full_bitset = 0
        for s_val in s:
          s_full_bitset |= (1 << s_val) | (1 << (n - s_val))
        k4_c = count_k4_containing_0(n, s_full_bitset)
        if k4_c > 0:
          current_score, current_witness = 1000 + k4_c, None
        else:
          current_score, current_witness = get_alpha_and_witness_containing_0(
              n, s_full_bitset, 16
          )

        temp = initial_temp * 1.5
        iterations_since_last_improvement = 0

      d = random.randint(1, n_half)
      if random.random() < 0.75:  # Guided move
        if current_score >= 1000:
          k4_gens = find_one_k4_generators_containing_0(n, s_full_bitset)
          if k4_gens:
            d = random.choice(k4_gens)
        elif current_score >= 16:
          if current_witness and len(current_witness) >= 16:
            v1, v2 = random.sample(current_witness, 2)
            d = min(abs(v1 - v2), n - abs(v1 - v2))

      old_bitset = s_full_bitset
      if d in s:
        s.remove(d)
        s_full_bitset &= ~((1 << d) | (1 << (n - d)))
        added = False
      else:
        s.add(d)
        s_full_bitset |= (1 << d) | (1 << (n - d))
        added = True
      k4_c_val = count_k4_containing_0(n, s_full_bitset)
      if k4_c_val > 0:
        new_score, new_witness = 1000 + k4_c_val, None
      else:
        alpha_size, witness = get_alpha_and_witness_containing_0(
            n, s_full_bitset, 16
        )
        new_score, new_witness = alpha_size, witness

      # If a valid graph for best_g1 is found, update best_g1 and break
      # Found a valid graph for best_g1 (K4-free and alpha < 16)
      if new_score < 16:
        best_g1, last_s = get_adj_matrix(n, s), s.copy()
        break  # Move to try next n

      # Always keep track of the overall best S and its energy for the current n
      # (using g1's energy metric: 1000+k4_c or alpha_c)
      if new_score < best_score_for_current_n:
        best_score_for_current_n = new_score
        best_s_for_current_n = s.copy()

      # Simulated annealing acceptance logic for g1
      delta_e = new_score - current_score
      if delta_e <= 0 or (
          temp > 0 and random.random() < math.exp(-delta_e / temp)
      ):
        if new_score < current_score:
          iterations_since_last_improvement = 0
        current_score = new_score
        current_witness = new_witness
      else:
        # Revert change if not accepted
        if added:
          s.remove(d)
        else:
          s.add(d)
        s_full_bitset = old_bitset
      temp *= cooling_rate  # Apply cooling
  return best_g1
