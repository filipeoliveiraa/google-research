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


INITIAL_N = 216
MAX_TIME_SEC = 1910  # Leave more buffer for G2 to ensure it runs
SA_STEPS_FACTOR_PER_N = 50000
SA_TEMP_INITIAL = 1.0
SA_COOLING_RATE = 0.99999  # Faster cooling to allow more S explorations
SA_NODE_LIMIT_FACTOR_MIN = 2000
SA_NODE_LIMIT_FACTOR_MAX = 20000  # More exhaustive anti-clique search
# Devote more search to anti-cliques
SA_NODE_LIMIT_FACTOR_K4_FREE_MULTIPLIER = 5.0
FINAL_NODE_LIMIT_FACTOR_VERIFY = 1000000
TRIAL_MAX_DURATION_SEC = 600
SA_DIVERSITY_RESTART_THRESHOLD = 3500
SA_DIVERSITY_RESTART_PROB = 0.20  # Slightly more aggressive exploration
# Probability of "quantum teleportation" mutation
TELEPORTATION_MUTATION_PROB = 0.03

N_EXPLORATION_PROB = 0.10  # Focus more on the current target N
N_EXPLORATION_RANGE = 3
N_RESTART_PROB_S = 0.08  # Less frequent restarts, rely more on SA and saved S


def count_bits(n):
  return n.bit_count()


def get_mask(n, s):
  mask = 0
  for diff in s:
    mask |= (1 << diff) | (1 << ((n - diff) % n))
  return mask


def get_adj_masks(n, mask):
  full_mask = (1 << n) - 1
  return [((mask << i) | (mask >> (n - i))) & full_mask for i in range(n)]


def get_k4_info(adj_masks, n, max_find=30):
  """Counts K4s and returns differences of first ones found."""
  adj0 = adj_masks[0]
  cnt, first_k4 = 0, []
  temp = adj0
  while temp:
    v = (temp & -temp).bit_length() - 1
    temp &= ~(1 << v)
    c1 = temp & adj_masks[v]
    t2 = c1
    while t2:
      w = (t2 & -t2).bit_length() - 1
      t2 &= ~(1 << w)
      c2 = t2 & adj_masks[w]
      if c2:
        cnt += count_bits(c2)
        if len(first_k4) < max_find * 6:
          t3 = c2
          while t3 and len(first_k4) < max_find * 6:
            u = (t3 & -t3).bit_length() - 1
            t3 &= ~(1 << u)
            for d in [v, w, u, (v - w) % n, (v - u) % n, (w - u) % n]:
              dm = min(d, n - d)
              if dm > 0:
                first_k4.append(dm)
  return cnt, first_k4


def get_max_clique_info(
    adj_masks, limit, node_limit, max_find=40
):
  """Finds the maximum clique using backtracking."""
  best_size, clique_list, nodes = 0, [], 0

  def backtrack(candidates, current_clique):
    nonlocal nodes, best_size, clique_list
    nodes += 1
    l_curr = len(current_clique)
    if l_curr > best_size:
      best_size, clique_list = l_curr, [current_clique[:]]
    elif l_curr == best_size and l_curr > 0:
      if len(clique_list) < max_find:
        clique_list.append(current_clique[:])
    if nodes >= node_limit or (
        best_size >= limit and len(clique_list) >= max_find
    ):
      return
    if l_curr + count_bits(candidates) <= best_size:
      return
    pivot = (candidates & -candidates).bit_length() - 1
    max_deg = count_bits(candidates & adj_masks[pivot])
    temp_cands, p_count = candidates ^ (1 << pivot), 0
    while temp_cands and p_count < 10:
      p = (temp_cands & -temp_cands).bit_length() - 1
      temp_cands ^= 1 << p
      deg = count_bits(candidates & adj_masks[p])
      if deg > max_deg:
        max_deg, pivot = deg, p
      p_count += 1
    search = candidates & ~adj_masks[pivot]
    search_list = []
    temp_search = search
    while temp_search:
      v = (temp_search & -temp_search).bit_length() - 1
      temp_search &= ~(1 << v)
      search_list.append((count_bits(candidates & adj_masks[v]), v))
    search_list.sort(key=lambda x: x[0], reverse=True)

    for _, v in search_list:
      if nodes >= node_limit or (best_size >= limit and l_curr < limit):
        break
      new_candidates = candidates & adj_masks[v]
      if l_curr + 1 + count_bits(new_candidates) > best_size:
        backtrack(new_candidates, current_clique + [v])
      candidates &= ~(1 << v)
      if l_curr + count_bits(candidates) <= best_size:
        break

  backtrack(adj_masks[0], [0])
  return best_size, clique_list, nodes < node_limit


def evaluate_graph_state(
    n, current_mask, node_limit_is_eval
):
  """Evaluates the energy of a graph state."""
  adj_masks = get_adj_masks(n, current_mask)
  nk4, first_k4 = get_k4_info(adj_masks, n)
  if nk4 > 0:
    energy = 2000.0 + nk4 * 20.0
    ni_size, is_cliques, exh = 0, [], True
  else:
    comp_mask = ((1 << n) - 1) ^ current_mask ^ 1
    comp_adj_masks = get_adj_masks(n, comp_mask)
    ni_size, is_cliques, exh = get_max_clique_info(
        comp_adj_masks, 19, node_limit_is_eval, 100
    )
    energy = ni_size + len(is_cliques) / 100.0 + (0.15 if not exh else 0)
  return energy, nk4, first_k4, ni_size, is_cliques, exh


def check_k4_after_adding(n, current_mask, d):
  new_mask = current_mask | (1 << d) | (1 << (n - d))
  mask_d = ((new_mask << d) | (new_mask >> (n - d))) & ((1 << n) - 1)
  common = new_mask & mask_d
  temp = common
  while temp:
    v = (temp & -temp).bit_length() - 1
    temp &= ~(1 << v)
    mask_v = ((new_mask << v) | (new_mask >> (n - v))) & ((1 << n) - 1)
    if mask_v & common:
      return False
  return True


def find_graph():
  """Finds a Ramsey number lower bound graph.

  Returns:
    A numpy array representing the adjacency matrix of the found graph.
  """
  start_time = time.time()

  # Dynamic N Exploration: N is chosen dynamically in each outer loop iteration
  # rather than strictly incrementing. This allows exploring promising Ns more.

  # Global state for our dynamic N search
  global_best_g1 = None
  global_best_n1 = 0

  # Stores the best S found for each N encountered, along with its energy and
  # timestamp
  best_s_per_n = {}  # n -> (energy, S_set, timestamp)

  # Current N we are focusing on in the main loop
  current_target_n = INITIAL_N

  # Feedback accumulation for the entire run

  while time.time() - start_time < MAX_TIME_SEC:

    # --- N Selection for the current iteration ---
    chosen_n = current_target_n  # Default to continue current_target_n

    # If we just found a new global best for current_target_n, try to increment
    if global_best_n1 == current_target_n:
      chosen_n += 1

    # Probabilistic jump to explore other Ns
    if random.random() < N_EXPLORATION_PROB:
      if global_best_n1 > 0:
        # Explore Ns around the best found so far
        chosen_n = random.randint(
            max(INITIAL_N, global_best_n1), global_best_n1 + N_EXPLORATION_RANGE
        )
      else:
        # If no best_n1 yet, explore a wider range around initial N
        chosen_n = random.randint(INITIAL_N, INITIAL_N + N_EXPLORATION_RANGE)

    chosen_n = max(4, chosen_n)  # Ensure n is at least 4 for K4 logic
    current_target_n = (
        chosen_n  # Update current_target_n for the next iteration
    )

    # --- S Initialization for the chosen N ---
    s_hall_of_fame = (
        []
    )  # Hall of fame for the specific 'chosen_n' in this iteration

    if chosen_n in best_s_per_n and random.random() > N_RESTART_PROB_S:
      s = best_s_per_n[chosen_n][1].copy()  # Retrieve saved S
      # Add a small perturbation to prevent being stuck or repetitive
      for _ in range(random.randint(0, 2)):
        s_to_flip = random.randint(1, chosen_n // 2)
        if s_to_flip in s:
          s.remove(s_to_flip)
        else:
          s.add(s_to_flip)
    elif (
        global_best_n1 > 0
        and global_best_n1 in best_s_per_n
        and random.random() < 0.7
    ):
      # Scale S from the globally best N found so far
      s_from_global_best = best_s_per_n[global_best_n1][1]
      s = {
          max(1, min(chosen_n // 2, int(d * chosen_n / global_best_n1 + 0.5)))
          for d in s_from_global_best
      }
      s.discard(0)
      s = {d for d in s if 1 <= d <= chosen_n // 2}
    else:
      # Random initialization
      target_len_s = chosen_n // 13
      num_diffs = random.randint(target_len_s - 1, target_len_s + 1)
      s = set(random.sample(range(1, chosen_n // 2 + 1), num_diffs))

    mask, steps_since_imp = get_mask(chosen_n, s), 0
    node_limit_eval = SA_NODE_LIMIT_FACTOR_MIN * chosen_n

    energy, nk4, first_k4, ni_size, is_cliques, _ = evaluate_graph_state(
        chosen_n, mask, node_limit_eval
    )

    # Increase SA steps for each N trial, allowing more mutations, while
    # TRIAL_MAX_DURATION_SEC bounds total time
    t, current_sa_steps = SA_TEMP_INITIAL, SA_STEPS_FACTOR_PER_N * (
        chosen_n // 10
    )
    current_sa_steps = max(
        200, current_sa_steps
    )  # Ensure a higher minimum number of steps

    trial_start_time = time.time()

    for step in range(current_sa_steps):
      if time.time() - trial_start_time > TRIAL_MAX_DURATION_SEC / 2:
        break  # Shorter duration per SA cycle
      if (energy < 19 and nk4 == 0) or (
          step % 200 == 0 and time.time() - start_time > MAX_TIME_SEC
      ):
        break

      # Aggressive Reheating (Quantum Jump) and Dynamic Cooling Rate
      if steps_since_imp > 8000:
        t = SA_TEMP_INITIAL  # Full reset for aggressive reheating
        steps_since_imp = 0

      # Dynamic adjustment of cooling rate based on graph state
      adaptive_cooling_rate = SA_COOLING_RATE
      if (
          nk4 == 0 and ni_size >= 19
      ):  # K4-free but still has K19 in complement (stuck on plateau)
        adaptive_cooling_rate = min(
            0.999999, SA_COOLING_RATE + 0.000005
        )  # Cool slower (closer to 1) for more thorough exploration
      elif nk4 > 0:  # Still has K4s, need to escape quickly
        adaptive_cooling_rate = max(
            0.99995, SA_COOLING_RATE - 0.00002
        )  # Cool faster (smaller multiplier) to push out of bad states
      # If no specific condition, adaptive_cooling_rate remains SA_COOLING_RATE

      if (
          steps_since_imp > SA_DIVERSITY_RESTART_THRESHOLD
          and random.random() < SA_DIVERSITY_RESTART_PROB
      ):
        if len(s_hall_of_fame) >= 2 and random.random() < 0.7:
          h1, h2 = random.sample(s_hall_of_fame, 2)
          s = {
              d
              for d in range(1, chosen_n // 2 + 1)
              if (d in h1[1] and d in h2[1])
              or ((d in h1[1] or d in h2[1]) and random.random() < 0.5)
          }
        else:
          target_len_s_new = random.randint(
              max(1, chosen_n // 14), chosen_n // 10
          )
          s = set(random.sample(range(1, chosen_n // 2 + 1), target_len_s_new))
        mask = get_mask(chosen_n, s)
        energy, nk4, first_k4, ni_size, is_cliques, _ = evaluate_graph_state(
            chosen_n, mask, node_limit_eval
        )
        t = SA_TEMP_INITIAL * 0.9
        steps_since_imp = 0

      if step > 0 and step % 5000 == 0:
        k_cands = [x for x in range(2, chosen_n) if math.gcd(x, chosen_n) == 1]
        if k_cands:
          k = random.choice(k_cands)
          s = {
              min((s * k) % chosen_n, chosen_n - ((s * k) % chosen_n))
              for s in s
          }
          s.discard(0)
          mask = get_mask(chosen_n, s)
          energy, nk4, first_k4, ni_size, is_cliques, _ = evaluate_graph_state(
              chosen_n, mask, node_limit_eval
          )

      new_s = s.copy()

      # --- Quantum Teleportation Mutation ---
      if random.random() < TELEPORTATION_MUTATION_PROB and len(s) >= 2:
        s_list = list(s)
        d1, d2 = random.sample(s_list, 2)

        # Remove d1 and d2
        new_s.remove(d1)
        new_s.remove(d2)

        n_half = chosen_n // 2

        # Generate two new differences "entangled" from d1, d2
        # Ensure they are within the valid range [1, n_half]
        new_d1 = (d1 + d2) % n_half
        if new_d1 == 0:
          new_d1 = n_half  # Avoid 0, map to max valid difference

        new_d2 = abs(d1 - d2)
        if new_d2 == 0:
          new_d2 = (d1 * d2) % n_half  # If diff is 0, use product
        if new_d2 == 0:  # Fallback if product is also 0
          new_d2 = max(1, chosen_n // 4)  # A non-zero, mid-range difference

        # Add new differences to S
        new_s.add(new_d1)
        new_s.add(new_d2)
      # --- End Quantum Teleportation Mutation ---

      else:  # If not a teleportation mutation, proceed with existing strategies
        change_type = random.random()  # Re-roll change_type for other mutations
        if nk4 > 0:
          if first_k4 and change_type < 0.94:
            e = random.choice(first_k4)
            new_s.remove(e)
          else:
            s = random.randint(1, chosen_n // 2)
            if s in new_s:
              new_s.remove(s)
            else:
              new_s.add(s)
        else:
          if is_cliques and change_type < 0.90:
            diff_counts = {}
            sampled_cliques = random.sample(
                is_cliques, min(len(is_cliques), 100)
            )
            for c in sampled_cliques:
              for i, rename_me in enumerate(c):
                for j in range(i + 1, len(c)):
                  d = min(
                      abs(rename_me - c[j]), chosen_n - abs(rename_me - c[j])
                  )
                  if d not in new_s:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
            sorted_diffs = sorted(
                diff_counts.items(), key=lambda x: x[1], reverse=True
            )
            found_safe = False
            for d, _ in sorted_diffs:
              if check_k4_after_adding(chosen_n, mask, d):
                new_s.add(d)
                found_safe = True
                break
            if not found_safe:
              s = random.choice(list(new_s)) if new_s else None
              if s:
                new_s.remove(s)
          elif change_type < 0.97:
            s = random.randint(1, chosen_n // 2)
            if s in new_s:
              new_s.remove(s)
            elif check_k4_after_adding(chosen_n, mask, s):
              new_s.add(s)
          else:
            s = random.randint(1, chosen_n // 2)
            if s in new_s:
              new_s.remove(s)
            else:
              new_s.add(s)

      new_s.discard(0)
      new_s = {d for d in new_s if 1 <= d <= chosen_n // 2}

      nm = get_mask(chosen_n, new_s)

      time_dilation = 1.0 - (t / SA_TEMP_INITIAL)
      current_node_limit_factor = (
          SA_NODE_LIMIT_FACTOR_MIN
          + (SA_NODE_LIMIT_FACTOR_MAX - SA_NODE_LIMIT_FACTOR_MIN)
          * time_dilation
      )

      if nk4 == 0:
        current_node_limit_factor *= SA_NODE_LIMIT_FACTOR_K4_FREE_MULTIPLIER

      node_limit_eval = int(current_node_limit_factor) * chosen_n
      ne, nk4_new, first_k4_new, ni_size_new, is_cliques_new, _ = (
          evaluate_graph_state(chosen_n, nm, node_limit_eval)
      )

      if ne < energy or random.random() < np.exp((energy - ne) / t):
        if ne < energy:
          steps_since_imp = 0
          if nk4_new == 0:
            s_hall_of_fame.append((ne, new_s.copy()))
            s_hall_of_fame.sort(key=lambda x: x[0])
            s_hall_of_fame = s_hall_of_fame[:8]
        s, mask, energy, nk4, first_k4, ni_size, is_cliques = (
            new_s,
            nm,
            ne,
            nk4_new,
            first_k4_new,
            ni_size_new,
            is_cliques_new,
        )
      else:
        steps_since_imp += 1
      t *= adaptive_cooling_rate

    # --- Post-SA trial evaluation for chosen_n ---
    if (
        energy < 19 and nk4 == 0
    ):  # Candidate found (no K4, ni_size < 19 in SA evaluation)
      # Final verification with a higher node limit
      comp_masks = get_adj_masks(chosen_n, ((1 << chosen_n) - 1) ^ mask ^ 1)
      node_limit_verify = FINAL_NODE_LIMIT_FACTOR_VERIFY * chosen_n
      v_size, _, exh = get_max_clique_info(comp_masks, 19, node_limit_verify)

      if (
          exh and v_size <= 18
      ):  # If verification passed (no K19 found exhaustively)
        adj = np.zeros((chosen_n, chosen_n), dtype=np.int8)
        for r in range(chosen_n):
          for c in range(chosen_n):
            if (mask >> ((r - c) % chosen_n)) & 1:
              adj[r, c] = 1

        # Check if this is a new global best or an improved S for this N
        if chosen_n > global_best_n1:
          global_best_g1 = adj
          global_best_n1 = chosen_n

        elif chosen_n == global_best_n1 and (
            chosen_n not in best_s_per_n or energy < best_s_per_n[chosen_n][0]
        ):
          # If same N, but better energy, update global_best_g1
          global_best_g1 = adj

        # Store this S as the best for chosen_n
        best_s_per_n[chosen_n] = (energy, s.copy(), time.time())

      else:  # Verification failed for this chosen_n

        # If it failed verification, remove it from best_s_per_n if it was there
        if (
            chosen_n in best_s_per_n and chosen_n != global_best_n1
        ):  # Don't remove if it's the global best
          del best_s_per_n[chosen_n]
    else:  # No suitable candidate found for chosen_n in this SA trial
      pass
      # If it had a stored S, and it didn't improve, maybe it's not good enough.
      # No need to remove, just not update.

    # Memory management for best_s_per_n to prevent excessive memory usage.
    # Remove oldest/least promising entries if the dictionary gets too large.
    if len(best_s_per_n) > 20:  # Keep around 20 best S sets across different Ns
      # Remove the oldest S, unless it's for the global_best_n1
      oldest_n = None
      oldest_time = float("inf")
      for n_key, (_, _, timestamp) in best_s_per_n.items():
        if n_key != global_best_n1 and timestamp < oldest_time:
          oldest_time = timestamp
          oldest_n = n_key
      if oldest_n is not None:
        del best_s_per_n[oldest_n]

  return (
      global_best_g1
      if global_best_g1 is not None
      else np.zeros((1, 1), dtype=np.int8)
  )
