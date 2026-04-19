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


def _solve_clique(
    p,
    count,
    k,
    adj_bits,
    start_solve,
    time_limit,
):
  """Recursive helper to find a clique of size k."""
  if count >= k:
    return True
  if not p:
    return False
  if count + get_bit_count(p) < k:
    return False
  if time.time() - start_solve > time_limit:
    return True
  u = p.bit_length() - 1
  candidates = p & ~adj_bits[u]
  while candidates:
    v = (candidates & -candidates).bit_length() - 1
    if _solve_clique(
        p & adj_bits[v], count + 1, k, adj_bits, start_solve, time_limit
    ):
      return True
    candidates &= ~(1 << v)
    p &= ~(1 << v)
  return False


def find_clique_bitset(
    adj_bits, k, time_limit = 15.0
):
  if not adj_bits:
    return k <= 0
  n_v = len(adj_bits)
  if n_v < k:
    return False
  start_solve = time.time()
  return _solve_clique((1 << n_v) - 1, 0, k, adj_bits, start_solve, time_limit)


def has_k3(adj_bits):
  for row in adj_bits:
    temp = row
    while temp:
      j = (temp & -temp).bit_length() - 1
      if row & adj_bits[j]:
        return True
      temp &= ~(1 << j)
  return False


def get_orbits(n, g):
  """Computes orbits of elements 1 to n-1 under action of g and negation.

  Args:
    n: The modulus.
    g: The generator of the group action.

  Returns:
    A list of orbits, where each orbit is a list of integers.
  """
  used = [False] * n
  orbits = []
  for i in range(1, n):
    if not used[i]:
      orbit, curr = set(), i
      while curr not in orbit:
        orbit.add(curr)
        orbit.add((n - curr) % n)
        curr = (curr * g) % n
      orbits.append(list(orbit))
      for x in orbit:
        used[x] = True
  return orbits


def find_graph():
  """Finds a Ramsey number lower bound graph.

  Returns:
    The adjacency matrix of the best graph found.
  """
  start_time = time.time()
  best_g1, best_n1 = None, 0
  stats = {
      "n_iterations": 0,
      "g_iterations": 0,
      "samples": 0,
      "k3_filtered": 0,
  }

  for n in range(130, 155):
    stats["n_iterations"] += 1
    gs = [g for g in range(2, n) if math.gcd(g, n) == 1]
    random.shuffle(gs)
    for g in gs:
      stats["g_iterations"] += 1
      if time.time() - start_time > 3950:
        break
      orbits = get_orbits(n, g)
      m = len(orbits)
      if not (4 <= m <= 35):
        continue
      num_samples = 2000 if m > 14 else 2**m
      orbit_indices = list(range(m))
      for s_idx in range(num_samples):
        stats["samples"] += 1
        if time.time() - start_time > 3950:
          break
        if m <= 14:
          indices = [i for i in range(m) if (s_idx >> i) & 1]
        else:
          random.shuffle(orbit_indices)
          indices, curr_size, target_size = [], 0, random.randint(28, 44)
          for o_idx in orbit_indices:
            if curr_size + len(orbits[o_idx]) <= target_size:
              indices.append(o_idx)
              curr_size += len(orbits[o_idx])
          if curr_size < 25:
            continue

        s_set = set()
        for idx in indices:
          for x in orbits[idx]:
            s_set.add(x)
        if not (24 <= len(s_set) <= 48):
          continue

        s_list = sorted(list(s_set))
        adj_s = []
        for a in s_list:
          bits = 0
          for i, b in enumerate(s_list):
            if ((b - a) % n) in s_set:
              bits |= 1 << i
          adj_s.append(bits)
        if has_k3(adj_s):
          stats["k3_filtered"] += 1
          continue

        s_comp_set = set(range(1, n)) - s_set
        s_comp_list = sorted(list(s_comp_set))
        adj_s_comp = []
        for a in s_comp_list:
          bits = 0
          for i, b in enumerate(s_comp_list):
            if ((b - a) % n) in s_comp_set:
              bits |= 1 << i
          adj_s_comp.append(bits)

        if not find_clique_bitset(adj_s_comp, 13):
          adj = np.zeros((n, n), dtype=np.int8)
          for d in s_set:
            for i in range(n):
              adj[i, (i + d) % n] = 1
          best_g1, best_n1 = adj, n
          break
      if best_n1 == n:
        break

  if best_g1 is None:
    best_g1 = np.zeros((1, 1), dtype=np.int8)

  return best_g1
