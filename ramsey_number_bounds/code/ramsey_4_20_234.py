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
import numpy as np


def _solve_max_clique(p, x, cur_r, target, adj, nodes_0, state):
  """Recursive helper to find the maximum clique."""
  if not p:
    if len(cur_r) > state['max_c']:
      state['max_c'] = len(cur_r)
      state['best_v'] = list(cur_r)
    return
  if len(cur_r) + p.bit_count() <= state['max_c']:
    return
  px = p | x
  best_p = -1
  min_c = 1000
  while px:
    u = (px & -px).bit_length() - 1
    c = (p & ~adj[u]).bit_count()
    if c < min_c:
      min_c = c
      best_p = u
      if c <= 1:
        break
    px &= px - 1
  nodes = p & ~adj[best_p]
  while nodes:
    v = (nodes & -nodes).bit_length() - 1
    cur_r.append(nodes_0[v])
    _solve_max_clique(
        p & adj[v], x & adj[v], cur_r, target, adj, nodes_0, state
    )
    if state['max_c'] >= target - 1:
      return
    cur_r.pop()
    p &= ~(1 << v)
    x |= 1 << v
    nodes &= ~(1 << v)


def get_max_clique_and_verts(n, in_jumps, target):
  """Finds the maximum clique and its vertices.

  Args:
    n: The number of vertices.
    in_jumps: A list indicating which jumps are present.
    target: The target clique size.

  Returns:
    A tuple containing the maximum clique size and a list of vertices in it.
  """
  nodes_0_initial = [i for i in range(1, n) if in_jumps[i]]
  m = len(nodes_0_initial)
  if m < target - 1:
    return 1, [0]
  degs = [0] * m
  for i in range(m):
    u = nodes_0_initial[i]
    for j in range(i + 1, m):
      if in_jumps[(nodes_0_initial[j] - u) % n]:
        degs[i] += 1
        degs[j] += 1
  sorted_indices = sorted(range(m), key=lambda x: degs[x], reverse=True)
  nodes_0 = [nodes_0_initial[i] for i in sorted_indices]
  adj = [0] * m
  for i in range(m):
    u = nodes_0[i]
    for j in range(i + 1, m):
      if in_jumps[(nodes_0[j] - u) % n]:
        adj[i] |= 1 << j
        adj[j] |= 1 << i

  state = {'max_c': 0, 'best_v': []}
  _solve_max_clique((1 << m) - 1, 0, [], target, adj, nodes_0, state)
  return state['max_c'] + 1, [0] + state['best_v']


def count_k4_and_verts(n, in_s):
  """Counts K4s and returns vertices of first one found.

  Args:
    n: The number of vertices.
    in_s: A list indicating which elements are in the set.

  Returns:
    A tuple containing the count of K4s and a list of vertices of the first one.
  """
  nodes_0 = [x for x in range(1, n) if in_s[x]]
  m = len(nodes_0)
  adj = [0] * m
  for i in range(m):
    a = nodes_0[i]
    for j in range(i + 1, m):
      if in_s[(nodes_0[j] - a) % n]:
        adj[i] |= 1 << j

  count = 0
  first_v = []
  for i in range(m):
    neighbors = adj[i]
    while neighbors:
      j = (neighbors & -neighbors).bit_length() - 1
      common = adj[i] & adj[j]
      if common:
        count += common.bit_count()
        if not first_v:
          k = (common & -common).bit_length() - 1
          first_v = [0, nodes_0[i], nodes_0[j], nodes_0[k]]
      neighbors &= ~(1 << j)
  return count, first_v


def _apply_harmonic_mutation(
    current_s, n_half, k_min, k_max
):
  """Applies harmonic mutation to the set."""
  temp_s = current_s.copy()
  d = random.randint(2, 8)
  offset = random.randint(0, d - 1)
  jumps = list(range(1, n_half + 1))
  random.shuffle(jumps)
  for j in jumps:
    if j % d == offset:
      if j in temp_s:
        if len(temp_s) > k_min:
          temp_s.remove(j)
      else:
        if len(temp_s) < k_max:
          temp_s.add(j)
  return temp_s


def _roulette_wheel_selection(scores_dict, num_candidates_to_consider=5):
  """Selects a candidate based on scores."""
  if not scores_dict:
    return None
  sorted_items = sorted(
      scores_dict.items(),
      key=lambda x: x[1] + random.uniform(-1e-6, 1e-6),
      reverse=True,
  )
  candidates = sorted_items[
      : min(num_candidates_to_consider, len(sorted_items))
  ]
  scores = np.array([item[1] for item in candidates])
  if len(scores) == 0:
    return None
  min_score = np.min(scores)
  shifted_scores = scores - min_score + 1e-6
  total_score = np.sum(shifted_scores)
  probabilities = shifted_scores / total_score
  return candidates[np.random.choice(len(candidates), p=probabilities)][0]


def _apply_spectral_mutation(
    current_s,
    n,
    n_half,
    k_min,
    k_max,
    current_e,
    cos_matrix,
):
  """Applies spectral mutation to the set."""
  temp_s = current_s.copy()
  lambdas = np.zeros(n)
  for j in temp_s:
    lambdas += 2 * cos_matrix[:, j - 1]
  lambdas[0] = 0

  if current_e >= 1000:  # Focus: Reduce cliques (K4)
    weights = lambdas**3
    grad_j = weights @ cos_matrix
    remove_scores = {j: grad_j[j - 1] for j in temp_s}
    add_scores = {
        j: -grad_j[j - 1] for j in range(1, n_half + 1) if j not in temp_s
    }
  else:  # Focus: Break independent sets (I20)
    weights = np.exp(-(lambdas - np.min(lambdas[1:])) / 2.0)
    weights[0] = 0
    grad_j = weights @ cos_matrix
    add_scores = {
        j: grad_j[j - 1] for j in range(1, n_half + 1) if j not in temp_s
    }
    remove_scores = {j: -grad_j[j - 1] for j in temp_s}

  move_choice = (
      'a'
      if len(temp_s) < k_min
      else ('r' if len(temp_s) > k_max else random.choice(['a', 'r', 's']))
  )
  if move_choice == 'a':
    chosen = _roulette_wheel_selection(add_scores)
    if chosen:
      temp_s.add(chosen)
  elif move_choice == 'r':
    chosen = _roulette_wheel_selection(remove_scores)
    if chosen:
      temp_s.remove(chosen)
  else:  # Swap
    r_chosen = _roulette_wheel_selection(remove_scores)
    a_chosen = _roulette_wheel_selection(add_scores)
    if r_chosen:
      temp_s.remove(r_chosen)
    if a_chosen:
      temp_s.add(a_chosen)
  return temp_s


def _apply_random_mutation(
    current_s, n_half, k_min, k_max
):
  """Applies random mutation to the set."""
  temp_s = current_s.copy()
  force_add = len(temp_s) < k_min
  force_remove = len(temp_s) > k_max

  if force_add:
    move_choice = 'a'
  elif force_remove:
    move_choice = 'r'
  else:
    move_choice = random.choice(['a', 'r', 's'])  # Neutral random choice

  if move_choice == 'a':
    if len(temp_s) < k_max:  # Only add if not already at max size
      candidates = set(range(1, n_half + 1)) - temp_s
      if candidates:
        temp_s.add(random.choice(list(candidates)))
  elif move_choice == 'r':
    if len(temp_s) > k_min:  # Only remove if not already at min size
      if temp_s:
        temp_s.remove(random.choice(list(temp_s)))
  elif move_choice == 's':  # Swap
    if temp_s:  # Must have something to remove
      removed_val = random.choice(list(temp_s))
      temp_s.remove(removed_val)
      candidates = set(range(1, n_half + 1)) - temp_s
      if candidates:
        temp_s.add(random.choice(list(candidates)))
      else:  # If no candidate to add, revert the remove to avoid size change
        temp_s.add(removed_val)  # Revert if no suitable add candidate
  return temp_s


def find_graph():
  """Finds a Ramsey number lower bound."""

  start_time = time.time()

  # Variables to track the best g1 found and its warm-start S set
  best_g1, best_g1_n = np.zeros((1, 1), dtype=np.int8), 0
  current_best_s_for_warm_start = None
  n_for_current_best_s_for_warm_start = None

  # Global search parameters
  total_time_limit = 1960
  p_random_init = 0.25
  num_mutations_on_best_s = 8

  g1_time_fraction_initial = 0.85
  initial_temp = 2.2
  cooling_rate = 0.99992
  base_sa_iters = 140000

  # Prioritized search list: start with safe baseline, then target frontier
  search_ns = [
      180,
      225,
      228,
      230,
      232,
      234,
      236,
      238,
      240,
      226,
      227,
      229,
      231,
      233,
      235,
      220,
      215,
  ]
  for n_idx, n in enumerate(search_ns):
    if n <= best_g1_n:
      continue
    n_half = n // 2
    j_arr = np.arange(1, n_half + 1)
    angles = 2 * np.pi * np.arange(n)[:, None] * j_arr[None, :] / n
    cos_matrix = np.cos(angles)
    if time.time() - start_time > total_time_limit:

      break

    found_for_n = False

    # Consistent k-range, slightly wider
    k_min, k_max = int(n * 0.10), int(n * 0.20)  # Wider range for k_min, k_max
    k_min = max(1, k_min)
    k_max = min(n_half, k_max)
    if k_min > k_max:
      k_max = k_min  # Ensure k_max >= k_min

    # Time allocation for current 'n'

    g1_remaining_time_budget = max(
        0.0,
        total_time_limit * g1_time_fraction_initial
        - (time.time() - start_time),
    )

    # Distribute remaining time among remaining n targets
    rem_search_count = len(search_ns) - n_idx
    current_n_time_budget = g1_remaining_time_budget / max(1, rem_search_count)
    if n > 225:
      current_n_time_budget = max(current_n_time_budget, 180.0)

    start_n_time = time.time()
    s_global_best_for_n, e_global_best_for_n = None, float('inf')

    num_restarts_for_n = (
        10  # Slightly reduced restarts to distribute time more broadly
    )

    for restart_attempt in range(num_restarts_for_n):
      # Check total time limit and specific N's time budget
      if (
          time.time() - start_time > total_time_limit * g1_time_fraction_initial
      ) or (time.time() - start_n_time > current_n_time_budget):
        break

      # Warm start for G1 search on first restart, or if p_random_init fails
      if restart_attempt == 0 or random.random() < p_random_init:
        if (
            current_best_s_for_warm_start is not None
            and n_for_current_best_s_for_warm_start is not None
        ):
          scale = n / n_for_current_best_s_for_warm_start
          current_s = {
              max(1, min(n_half, int(round(j * scale))))
              for j in current_best_s_for_warm_start
          }

          # Ensure current_s is within k_min/k_max bounds
          while (
              len(current_s) < k_min and len(current_s) < k_max
          ):  # Only add if still room
            candidates = set(range(1, n_half + 1)) - current_s
            if candidates:
              current_s.add(random.choice(list(candidates)))
            else:
              break
          while (
              len(current_s) > k_max and len(current_s) > k_min
          ):  # Only remove if still too large
            if current_s:
              current_s.remove(random.choice(list(current_s)))
            else:
              break

          # Evaluate initial warm-started S to check for K4s
          initial_in_s_warm = [False] * n
          for x in current_s:
            initial_in_s_warm[x] = initial_in_s_warm[n - x] = True
          k4_at_warm_start, k4_v_at_warm_start = count_k4_and_verts(
              n, initial_in_s_warm
          )

          # Targeted K4 removal if many K4s exist, else random adaptation
          if k4_at_warm_start > 0:
            for _ in range(
                num_mutations_on_best_s * 2
            ):  # More mutations to fix K4s
              if k4_at_warm_start == 0:
                break  # No K4s left or time is up

              best_rem, min_k4 = None, float('inf')
              # Vertices involved in the K4
              cand = set(k4_v_at_warm_start[1:])
              # Add differences between K4 vertices as potential jumps
              for i in range(1, len(k4_v_at_warm_start)):
                for l in range(i + 1, len(k4_v_at_warm_start)):
                  d = min(
                      (k4_v_at_warm_start[i] - k4_v_at_warm_start[l]) % n,
                      (k4_v_at_warm_start[l] - k4_v_at_warm_start[i]) % n,
                  )
                  if 1 <= d <= n_half:
                    cand.add(d)

              shuffled = list(cand.intersection(current_s))
              random.shuffle(shuffled)

              if (
                  not shuffled
              ):  # Fallback if no specific K4-related jumps to remove
                current_s = _apply_random_mutation(
                    current_s, n_half, k_min, k_max
                )
                initial_in_s_warm = [False] * n  # Recalc state after mutation
                for x in current_s:
                  initial_in_s_warm[x] = initial_in_s_warm[n - x] = True
                k4_at_warm_start, k4_v_at_warm_start = count_k4_and_verts(
                    n, initial_in_s_warm
                )
                continue

              for j_rem in shuffled[
                  : min(len(shuffled), 15)
              ]:  # Increased candidate samples for removal
                test_s = current_s - {j_rem}
                if len(test_s) >= k_min:
                  ts_in = [False] * n
                  for x_ts in test_s:
                    ts_in[x_ts] = ts_in[n - x_ts] = True
                  cnt, _ = count_k4_and_verts(n, ts_in)
                  if cnt < min_k4:
                    min_k4, best_rem = cnt, j_rem

              if best_rem is not None:
                current_s.remove(best_rem)
                # If too small, add back a non-K4 causing jump
                if len(current_s) < k_min and len(current_s) < k_max:
                  # Try to add a jump not in 'cand' (K4-causing jumps)
                  candidates_to_add = list(
                      (set(range(1, n_half + 1)) - current_s) - cand
                  )
                  if candidates_to_add:
                    current_s.add(random.choice(candidates_to_add))
                  else:  # Fallback to any if specific ones not found
                    any_candidates = set(range(1, n_half + 1)) - current_s
                    if any_candidates:
                      current_s.add(random.choice(list(any_candidates)))

                initial_in_s_warm = [False] * n  # Recalc for next iter
                for x_ws in current_s:
                  initial_in_s_warm[x_ws] = initial_in_s_warm[n - x_ws] = True
                k4_at_warm_start, k4_v_at_warm_start = count_k4_and_verts(
                    n, initial_in_s_warm
                )
              else:  # If no good targeted removal, fallback to random
                current_s = _apply_random_mutation(
                    current_s, n_half, k_min, k_max
                )
                initial_in_s_warm = [False] * n
                for x_ws in current_s:
                  initial_in_s_warm[x_ws] = initial_in_s_warm[n - x_ws] = True
                k4_at_warm_start, k4_v_at_warm_start = count_k4_and_verts(
                    n, initial_in_s_warm
                )
          else:  # If no K4s at warm start, adapt size with random mutations
            for _ in range(num_mutations_on_best_s):
              current_s = _apply_random_mutation(
                  current_s, n_half, k_min, k_max
              )
        else:
          # Pure random init if no warm start or not the first restart
          current_s = set(
              random.sample(range(1, n_half + 1), random.randint(k_min, k_max))
          )
      else:
        # Use the best S found so far for the current n, with perturbation
        current_s = s_global_best_for_n.copy()
        for _ in range(
            num_mutations_on_best_s
        ):  # Fewer mutations if perturbing best
          current_s = _apply_random_mutation(current_s, n_half, k_min, k_max)

      # Evaluate initial S for this restart
      in_s = [False] * n
      for x in current_s:
        in_s[x] = in_s[n - x] = True
      # Compute current_in_Sc here, as it's always derived from current_in_S
      in_sc = [not in_s[i] for i in range(n)]
      in_sc[0] = False

      k4_cnt, k4_v = count_k4_and_verts(n, in_s)
      if k4_cnt > 0:
        current_e, i20_v = 1000 + k4_cnt, []
      else:
        current_e, i20_v = get_max_clique_and_verts(n, in_sc, 20)

      if current_e < e_global_best_for_n:
        e_global_best_for_n = current_e
        s_global_best_for_n = current_s.copy()

      # Single SA loop (consolidated from previous multi-stage)
      temp = initial_temp
      cooling = cooling_rate
      e_before_reheat = current_e
      # Scale max_iters for g1
      max_iters = int(base_sa_iters * (n / 100) ** 1.5)
      max_iters = max(
          max_iters, 50000
      )  # Minimum iterations for g1, increased from 40k

      # Reheat check interval, increased to allow more exploration
      reheat_interval = 3000

      for it in range(max_iters):
        # Check time budget more frequently
        if it % 500 == 0 and (
            time.time() - start_n_time > current_n_time_budget
        ):
          break
        if current_e < 20:
          break  # Solution found

        temp_s = current_s.copy()  # Make a copy of the current best S
        move_type_rand = random.random()

        # Mutation probabilities adjusted for more targeted moves
        # Targeted K4 Removal
        if (
            current_e >= 1000 and move_type_rand < 0.55 and k4_v
        ):  # Slightly increased probability for K4
          best_rem, min_k4 = None, float('inf')
          cand = set(k4_v[1:])
          for i in range(1, 4):
            for l in range(i + 1, 4):
              d = min((k4_v[i] - k4_v[l]) % n, (k4_v[l] - k4_v[i]) % n)
              if 1 <= d <= n_half:
                cand.add(d)
          shuffled = list(cand.intersection(current_s))
          random.shuffle(shuffled)
          for j_rem in shuffled[
              : min(len(shuffled), 7)
          ]:  # Increased candidate samples for removal
            test_s = current_s - {j_rem}
            if len(test_s) >= k_min:
              ts_in = [False] * n
              for x in test_s:
                ts_in[x] = ts_in[n - x] = True
              cnt, _ = count_k4_and_verts(n, ts_in)
              if cnt < min_k4:
                min_k4, best_rem = cnt, j_rem
          if best_rem is not None:
            temp_s.remove(best_rem)
            if len(temp_s) < k_min:
              c = list(set(range(1, n_half + 1)) - temp_s)
              if c:
                temp_s.add(
                    random.choice(list(c))
                )  # Use list(c) as random.choice needs sequence
        # Targeted I20 Break
        elif (
            current_e >= 20
            and current_e < 1000
            and move_type_rand < 0.55
            and i20_v
        ):  # Slightly increased probability for I20
          best_add, min_k4 = None, float('inf')
          cand = set()
          # Consider all pairs within the independent set
          for i, rename_me in enumerate(i20_v):
            for l in range(i + 1, len(i20_v)):
              u, v = rename_me, i20_v[l]
              d = min((u - v) % n, (v - u) % n)
              if 1 <= d <= n_half and d not in current_s:
                cand.add(d)
          shuffled = list(cand)
          random.shuffle(shuffled)
          for j_add in shuffled[
              : min(len(shuffled), 10)
          ]:  # Increased candidate samples for addition
            test_s = current_s | {j_add}
            if len(test_s) <= k_max:
              ts_in = [False] * n
              for x in test_s:
                ts_in[x] = ts_in[n - x] = True
              cnt, _ = count_k4_and_verts(n, ts_in)
              if cnt < min_k4:
                min_k4, best_add = cnt, j_add
          if best_add is not None:
            temp_s.add(best_add)
            if len(temp_s) > k_max:
              c = list(temp_s - {best_add})
              if c:
                temp_s.remove(random.choice(list(c)))  # Use list(c)
        elif move_type_rand < 0.88:  # Spectral gets slightly more chances
          temp_s = _apply_spectral_mutation(
              current_s, n, n_half, k_min, k_max, current_e, cos_matrix
          )
        elif move_type_rand < 0.98:  # Harmonic also slightly more chances
          temp_s = _apply_harmonic_mutation(current_s, n_half, k_min, k_max)
        else:  # Random gets fewer chances
          temp_s = _apply_random_mutation(current_s, n_half, k_min, k_max)

        # Evaluate the trial state
        trial_in_s = [False] * n
        for x in temp_s:
          trial_in_s[x] = trial_in_s[n - x] = True
        nk4, nk4v_trial = count_k4_and_verts(n, trial_in_s)
        if nk4 > 0:
          ne, ni20v_trial = 1000 + nk4, []
        else:
          trial_in_sc = [not trial_in_s[i] for i in range(n)]
          trial_in_sc[0] = False
          ne, ni20v_trial = get_max_clique_and_verts(n, trial_in_sc, 20)

        # Reheating if stuck
        if it % reheat_interval == 0 and it > 0:  # Increased reheat_interval
          if current_e >= e_before_reheat:
            temp = initial_temp  # More aggressive reheat: reset to initial temp
          e_before_reheat = current_e

        # Metropolis acceptance criterion
        if ne <= current_e or random.random() < np.exp((current_e - ne) / temp):
          current_s = temp_s  # Update current_S
          current_e = ne  # Update current_e
          # Update all derived state variables
          in_s = trial_in_s
          # Recompute in_Sc as it depends on in_S
          in_sc = [not in_s[i] for i in range(n)]
          in_sc[0] = False

          k4_v = nk4v_trial
          i20_v = ni20v_trial

          if current_e < e_global_best_for_n:
            e_global_best_for_n = current_e
            s_global_best_for_n = current_s.copy()
        temp *= cooling
      # End of single SA loop

      if current_e < 20:
        found_for_n = True
        adj = np.zeros((n, n), dtype=np.int8)
        final_in_s = [False] * n
        for x in s_global_best_for_n:
          final_in_s[x] = final_in_s[n - x] = True
        for i in range(n):
          for j in range(i + 1, n):
            if final_in_s[(j - i) % n]:
              adj[i, j] = adj[j, i] = 1
        best_g1, best_g1_n = adj, n

        current_best_s_for_warm_start = s_global_best_for_n.copy()
        n_for_current_best_s_for_warm_start = n
        break

    if not found_for_n:

      if s_global_best_for_n is not None and e_global_best_for_n <= 21:
        current_best_s_for_warm_start = s_global_best_for_n.copy()
        n_for_current_best_s_for_warm_start = n

  # --- g2 search phase ---
  best_g2_score = float('inf')
  if best_g1_n > 0:
    g2_search_start_n = best_g1_n + 1
    g2_search_end_n = best_g1_n + 4

    for n_g2_val in range(
        g2_search_start_n, g2_search_end_n + 1, 1
    ):  # Iterate N for g2
      if time.time() - start_time > total_time_limit:

        break

      n_half_g2 = n_g2_val // 2
      cos_matrix_g2 = np.cos(
          2
          * np.pi
          * np.arange(n_g2_val)[:, None]
          * np.arange(1, n_half_g2 + 1)[None, :]
          / n_g2_val
      )

      # k-range for g2
      k_min_g2, k_max_g2 = int(n_g2_val * 0.10), int(n_g2_val * 0.20)
      k_min_g2 = max(1, k_min_g2)
      k_max_g2 = min(n_half_g2, k_max_g2)
      if k_min_g2 > k_max_g2:
        k_max_g2 = k_min_g2

      # Allocate remaining time for g2 search
      remaining_g2_time = total_time_limit - (time.time() - start_time)
      remaining_g2_n_count = g2_search_end_n - n_g2_val + 1
      g2_current_n_time_budget = remaining_g2_time / remaining_g2_n_count
      g2_current_n_time_budget = max(
          g2_current_n_time_budget, 5.0
      )  # Min 5s per g2_n

      g2_start_n_time = time.time()

      s_g2_best_for_n = None
      e_g2_best_for_n = float('inf')

      # Simplified restart strategy for g2 search, fewer restarts to save time
      num_restarts_for_n_g2 = 3

      for restart_attempt_g2 in range(num_restarts_for_n_g2):
        if time.time() - g2_start_n_time > g2_current_n_time_budget:
          break

        # Warm start for g2 using current_best_S_for_warm_start
        if (
            restart_attempt_g2 == 0
            and current_best_s_for_warm_start is not None
        ):
          # Scale current_best_S_for_warm_start to the current n_g2_val
          scale_g2 = n_g2_val / n_for_current_best_s_for_warm_start
          current_s_g2 = {
              max(1, min(n_half_g2, int(round(j * scale_g2))))
              for j in current_best_s_for_warm_start
          }

          # Adjust size for new n_g2_val
          while len(current_s_g2) < k_min_g2:
            candidates = set(range(1, n_half_g2 + 1)) - current_s_g2
            if candidates:
              current_s_g2.add(random.choice(list(candidates)))
            else:
              break
          while len(current_s_g2) > k_max_g2:
            if current_s_g2:
              current_s_g2.remove(random.choice(list(current_s_g2)))
            else:
              break
          for _ in range(num_mutations_on_best_s):  # Initial perturbation
            current_s_g2 = _apply_random_mutation(
                current_s_g2, n_half_g2, k_min_g2, k_max_g2
            )
        else:
          # Random init for g2
          current_s_g2 = set(
              random.sample(
                  range(1, n_half_g2 + 1), random.randint(k_min_g2, k_max_g2)
              )
          )

        # Evaluate initial S for this g2 restart
        in_s_g2 = [False] * n_g2_val
        for x in current_s_g2:
          in_s_g2[x] = in_s_g2[n_g2_val - x] = True
        in_sc_g2 = [not in_s_g2[i] for i in range(n_g2_val)]
        in_sc_g2[0] = False

        k4_cnt_g2, k4_v_g2 = count_k4_and_verts(n_g2_val, in_s_g2)
        alpha_val_g2, i20_v_g2 = get_max_clique_and_verts(
            n_g2_val, in_sc_g2, 20
        )
        # G2 Objective: sum of k4 count and alpha value. K4s heavily penalized.
        current_e_g2 = k4_cnt_g2 * 1000 + alpha_val_g2

        if current_e_g2 < e_g2_best_for_n:
          e_g2_best_for_n = current_e_g2
          s_g2_best_for_n = current_s_g2.copy()

        # SA loop for g2
        temp_g2 = initial_temp
        cooling_g2 = cooling_rate
        e_before_reheat_g2 = current_e_g2
        # Scale max_iters for g2 based on n, potentially more aggressively
        # Use different scaling for g2 to allow more focus, but still capped.
        max_iters_g2 = int(
            base_sa_iters * (n_g2_val / 100) ** 1.0
        )  # Lower exponent for g2 to cap iterations
        max_iters_g2 = max(
            max_iters_g2, 8000
        )  # Minimum iterations for g2, increased from 5k

        reheat_interval_g2 = (
            1500  # Shorter reheat interval for g2, allowing faster adaptation
        )

        for it_g2 in range(max_iters_g2):
          # Check time budget frequently
          if it_g2 % 200 == 0 and (
              time.time() - g2_start_n_time > g2_current_n_time_budget
          ):
            break
          if current_e_g2 == 0:
            break  # Optimal g2 found (no k4, no I20)

          temp_s_g2 = current_s_g2.copy()

          # Targeted mutation for g2: prioritize fixing major issues
          move_type_rand_g2 = random.random()

          if (
              k4_cnt_g2 > 0 and move_type_rand_g2 < 0.65 and k4_v_g2
          ):  # Increased probability for K4 in g2
            best_removal_jump_g2 = None
            min_k4_after_removal_g2 = float('inf')
            candidate_jumps_to_remove_g2 = set()
            for i in range(1, 4):
              candidate_jumps_to_remove_g2.add(k4_v_g2[i])
            for i in range(1, 4):
              for l in range(i + 1, 4):
                u, v = k4_v_g2[i], k4_v_g2[l]
                d = min((u - v) % n_g2_val, (v - u) % n_g2_val)
                if 1 <= d <= n_half_g2:
                  candidate_jumps_to_remove_g2.add(d)

            shuffled_jumps_g2 = list(
                candidate_jumps_to_remove_g2.intersection(current_s_g2)
            )
            random.shuffle(shuffled_jumps_g2)

            for j_to_remove_g2 in shuffled_jumps_g2[
                : min(len(shuffled_jumps_g2), 6)
            ]:  # Increased candidate samples for g2
              test_s_g2 = current_s_g2.copy()
              test_s_g2.remove(j_to_remove_g2)
              if len(test_s_g2) >= k_min_g2:  # Only if within size bounds
                test_in_s_g2 = [False] * n_g2_val
                for x_ts in test_s_g2:
                  test_in_s_g2[x_ts] = test_in_s_g2[n_g2_val - x_ts] = True
                current_test_k4_count_g2, _ = count_k4_and_verts(
                    n_g2_val, test_in_s_g2
                )
                if current_test_k4_count_g2 < min_k4_after_removal_g2:
                  min_k4_after_removal_g2 = current_test_k4_count_g2
                  best_removal_jump_g2 = j_to_remove_g2

            if best_removal_jump_g2 is not None:
              temp_s_g2.remove(best_removal_jump_g2)
              # If removing makes it too small, try to add back a different one
              if len(temp_s_g2) < k_min_g2 and len(temp_s_g2) < k_max_g2:
                new_candidates = set(range(1, n_half_g2 + 1)) - temp_s_g2
                if new_candidates:
                  temp_s_g2.add(
                      random.choice(list(new_candidates))
                  )  # Use list(new_candidates)

          elif (
              alpha_val_g2 >= 20 and move_type_rand_g2 < 0.65 and i20_v_g2
          ):  # Increased probability for I20 in g2
            l_g2 = len(i20_v_g2)
            candidate_jumps_to_add_g2 = set()
            for i_idx_g2 in range(l_g2):
              for j_idx_g2 in range(i_idx_g2 + 1, l_g2):
                u, v = i20_v_g2[i_idx_g2], i20_v_g2[j_idx_g2]
                d = min((u - v) % n_g2_val, (v - u) % n_g2_val)
                if 1 <= d <= n_half_g2 and d not in current_s_g2:
                  candidate_jumps_to_add_g2.add(d)

            best_add_jump_g2 = None
            min_k4_after_add_g2 = float('inf')  # Still care about K4

            shuffled_jumps_g2 = list(candidate_jumps_to_add_g2)
            random.shuffle(shuffled_jumps_g2)

            for j_to_add_g2 in shuffled_jumps_g2[
                : min(len(shuffled_jumps_g2), 8)
            ]:  # Limit candidates to check
              test_s_g2 = current_s_g2.copy()
              test_s_g2.add(j_to_add_g2)
              if len(test_s_g2) <= k_max_g2:  # Only if within size bounds
                test_in_s_g2 = [False] * n_g2_val
                for x_ts in test_s_g2:
                  test_in_s_g2[x_ts] = test_in_s_g2[n_g2_val - x_ts] = True
                current_test_k4_count_g2, _ = count_k4_and_verts(
                    n_g2_val, test_in_s_g2
                )
                if current_test_k4_count_g2 < min_k4_after_add_g2:
                  min_k4_after_add_g2 = current_test_k4_count_g2
                  best_add_jump_g2 = j_to_add_g2

            if best_add_jump_g2 is not None:
              temp_s_g2.add(best_add_jump_g2)
              # If adding makes it too large, try to remove a different one
              if len(temp_s_g2) > k_max_g2 and len(temp_s_g2) > k_min_g2:
                rem_candidates = list(temp_s_g2 - {best_add_jump_g2})
                if rem_candidates:
                  temp_s_g2.remove(
                      random.choice(list(rem_candidates))
                  )  # Use list(rem_candidates)
          elif move_type_rand_g2 < 0.90:  # Consistent probabilities with G1
            temp_s_g2 = _apply_spectral_mutation(
                current_s_g2,
                n_g2_val,
                n_half_g2,
                k_min_g2,
                k_max_g2,
                current_e_g2,
                cos_matrix_g2,
            )
          elif move_type_rand_g2 < 0.98:
            temp_s_g2 = _apply_harmonic_mutation(
                current_s_g2, n_half_g2, k_min_g2, k_max_g2
            )
          else:
            # General random move if no targeted move or random roll misses
            temp_s_g2 = _apply_random_mutation(
                current_s_g2, n_half_g2, k_min_g2, k_max_g2
            )

          # Reheating for g2
          if (
              it_g2 % reheat_interval_g2 == 0 and it_g2 > 0
          ):  # New reheat interval for g2
            if current_e_g2 >= e_before_reheat_g2:
              temp_g2 = initial_temp  # More aggressive reheat for g2
            e_before_reheat_g2 = current_e_g2

          # Evaluate trial state for g2
          trial_in_s_g2 = [False] * n_g2_val
          for x in temp_s_g2:
            trial_in_s_g2[x] = trial_in_s_g2[n_g2_val - x] = True
          nk4_g2, nk4v_trial_g2 = count_k4_and_verts(n_g2_val, trial_in_s_g2)
          trial_in_sc_g2 = [not trial_in_s_g2[i] for i in range(n_g2_val)]
          trial_in_sc_g2[0] = False
          na_trial_g2, ni20v_trial_g2 = get_max_clique_and_verts(
              n_g2_val, trial_in_sc_g2, 20
          )
          ne_g2 = nk4_g2 * 1000 + na_trial_g2  # Recalculate g2 energy

          # Metropolis acceptance
          if ne_g2 <= current_e_g2 or random.random() < np.exp(
              (current_e_g2 - ne_g2) / temp_g2
          ):
            current_s_g2 = temp_s_g2
            current_e_g2 = ne_g2
            # Update all derived state variables for g2
            k4_cnt_g2 = nk4_g2
            alpha_val_g2 = na_trial_g2
            k4_v_g2 = nk4v_trial_g2
            i20_v_g2 = ni20v_trial_g2

            if current_e_g2 < e_g2_best_for_n:
              e_g2_best_for_n = current_e_g2
              s_g2_best_for_n = current_s_g2.copy()
          temp_g2 *= cooling_g2

      # After all restarts for current n_g2_val
      # Check if a better graph (in terms of the overall goal) was found in g2
      if s_g2_best_for_n is not None and e_g2_best_for_n < 20:
        # Found a graph with no K4s and alpha < 20 for n_g2_val
        if n_g2_val > best_g1_n:
          best_g2_score = e_g2_best_for_n
          adj_g2 = np.zeros((n_g2_val, n_g2_val), dtype=np.int8)
          final_in_s_g2 = [False] * n_g2_val
          for x in s_g2_best_for_n:
            final_in_s_g2[x] = final_in_s_g2[n_g2_val - x] = True
          for i in range(n_g2_val):
            for j in range(i + 1, n_g2_val):
              if final_in_s_g2[(j - i) % n_g2_val]:
                adj_g2[i, j] = adj_g2[j, i] = 1
          best_g1, best_g1_n = adj_g2, n_g2_val
          current_best_s_for_warm_start = s_g2_best_for_n.copy()
          n_for_current_best_s_for_warm_start = n_g2_val

      # Keep track of best g2 score for warm starting next g2 iteration
      if s_g2_best_for_n is not None and e_g2_best_for_n < best_g2_score:
        best_g2_score = e_g2_best_for_n
        adj_g2 = np.zeros((n_g2_val, n_g2_val), dtype=np.int8)
        final_in_s_g2 = [False] * n_g2_val
        for x in s_g2_best_for_n:
          final_in_s_g2[x] = final_in_s_g2[n_g2_val - x] = True
        for i in range(n_g2_val):
          for j in range(i + 1, n_g2_val):
            if final_in_s_g2[(j - i) % n_g2_val]:
              adj_g2[i, j] = adj_g2[j, i] = 1

        # Update warm start candidate to the best G2 S found so far,
        # as it might be a good starting point for the next g2 N.
        if s_g2_best_for_n is not None:
          current_best_s_for_warm_start = s_g2_best_for_n.copy()
          n_for_current_best_s_for_warm_start = (
              n_g2_val  # Update n for warm start
          )

  return best_g1
