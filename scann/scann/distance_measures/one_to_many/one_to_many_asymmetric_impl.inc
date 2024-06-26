

template <int kRegisterIdx, typename T = Simd<float>>
SCANN_SIMD_INLINE auto ExpandToFp32(Sse4<int8_t> int8_vals) {
  if constexpr (IsSame<T, Avx512<float>>()) {
    return Avx512<float>{_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(*int8_vals))};
  }
  if constexpr (IsSame<T, Avx2<float>>()) {
    return Avx2<float>{AvxFunctionsAvx2Fma::Int8ToFloatLower(
        _mm_srli_si128(*int8_vals, 8 * kRegisterIdx))};
  }
  if constexpr (IsSame<T, Avx1<float>>()) {
    return Avx1<float>{AvxFunctionsAvx::Int8ToFloatLower(
        _mm_srli_si128(*int8_vals, 8 * kRegisterIdx))};
  }
  if constexpr (IsSame<T, Sse4<float>>()) {
    return Sse4<float>{_mm_cvtepi32_ps(
        _mm_cvtepi8_epi32(_mm_srli_si128(*int8_vals, 4 * kRegisterIdx)))};
  }
  LOG(FATAL) << "Unhandled: " << SimdName();
}

template <int kRegisterIdx, typename T = Simd<float>>
SCANN_SIMD_INLINE auto ExpandToFp32(Sse4<int16_t> bf16_vals) {
  if constexpr (IsSame<T, Avx2<float>>()) {
    static_assert(kRegisterIdx == 0);
    return Avx2<float>{
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(*bf16_vals), 16)};
  }
  if constexpr (IsSame<T, Avx1<float>>()) {
    static_assert(kRegisterIdx == 0);
    __m128i zeros = _mm_setzero_si128();
    __m128 lo = _mm_castsi128_ps(_mm_unpacklo_epi16(zeros, *bf16_vals));
    __m128 hi = _mm_castsi128_ps(_mm_unpackhi_epi16(zeros, *bf16_vals));
    return Avx1<float>{_mm256_set_m128(hi, lo)};
  }
  if constexpr (IsSame<T, Sse4<float>>()) {
    __m128i zeros = _mm_setzero_si128();
    if constexpr (kRegisterIdx == 0) {
      return Sse4<float>{_mm_unpacklo_epi16(zeros, *bf16_vals)};
    } else {
      static_assert(kRegisterIdx == 1);
      return Sse4<float>{_mm_unpackhi_epi16(zeros, *bf16_vals)};
    }
  }
  LOG(FATAL) << "Unhandled: " << SimdName();
}

template <int kRegisterIdx, typename T = Simd<float>>
SCANN_SIMD_INLINE auto ExpandToFp32(Avx2<int16_t> bf16_vals) {
  if constexpr (IsSame<T, Avx512<float>>()) {
    static_assert(kRegisterIdx == 0);
    return Avx512<float>{
        _mm512_slli_epi32(_mm512_cvtepu16_epi32(*bf16_vals), 16)};
  }
  if constexpr (IsSame<T, Avx2<float>>()) {
    __m256i zeros = _mm256_setzero_si256();
    __m256i permed = _mm256_permute4x64_epi64(*bf16_vals, 0b11'01'10'00);
    if constexpr (kRegisterIdx == 0) {
      return Avx2<float>{_mm256_unpacklo_epi16(zeros, permed)};
    } else {
      static_assert(kRegisterIdx == 1);
      return Avx2<float>{_mm256_unpackhi_epi16(zeros, permed)};
    }
  }
  LOG(FATAL) << "Unhandled: " << SimdName();
}

SCANN_SIMD_INLINE auto SseToSimd(Sse4<float> float_vals) {
  using T = Simd<float>;
  if constexpr (IsSame<T, Avx512<float>>()) {
    return Avx512<float>(
        _mm512_insertf32x4(_mm512_setzero_ps(), *float_vals, 0));
  }
  if constexpr (IsSame<T, Avx2<float>>()) {
    return Avx2<float>(AvxFunctionsAvx2Fma::SseToAvx(*float_vals));
  }
  if constexpr (IsSame<T, Avx1<float>>()) {
    return Avx1<float>(AvxFunctionsAvx::SseToAvx(*float_vals));
  }
  if constexpr (IsSame<T, Sse4<float>>()) {
    return float_vals;
  }
  LOG(FATAL) << "Unhandled: " << SimdName();
}

template <size_t kNumDims>
SCANN_SIMD_INLINE auto LoadXFloats(const float* ptr) {
  if constexpr (kNumDims >= Simd<float>::kNumElements) {
    return SimdFor<float, kNumDims>::Load(ptr);
  } else if constexpr (kNumDims == 8) {
    return Avx512<float>(
        _mm512_insertf32x8(_mm512_setzero_ps(), *Avx2<float>::Load(ptr), 0));
  } else if constexpr (kNumDims == 4) {
    return SseToSimd(Sse4<float>::Load(ptr));
  }
}

template <bool kIsSquaredL2, typename DataT, typename SimdT>
SCANN_SIMD_INLINE SimdT FusedMultiplyOp(SimdT a, SimdT b, SimdT mult,
                                        SimdT accum) {
  if constexpr (!kIsSquaredL2) {
    return FusedMultiplySubtract(a, b, accum);
  } else {
    if constexpr (std::is_same_v<DataT, int8_t>) {
      SimdT diff = (a - b * mult);
      return FusedMultiplyAdd(diff, diff, accum);
    } else {
      SimdT diff = a - b;
      return FusedMultiplyAdd(diff, diff, accum);
    }
  }
}

template <size_t kNumDims, bool kIsSquaredL2, size_t kUnrollBy, typename DataT>
SCANN_SIMD_INLINE Simd<float, kUnrollBy> HandleXDims(
    const float* query, array<const DataT*, kUnrollBy> ptrs,
    const float* inv_multipliers_for_squared_l2, size_t dim,
    Simd<float, kUnrollBy> accums) {
  static_assert(std::is_same_v<DataT, int8_t> ||
                std::is_same_v<DataT, int16_t>);

  auto qq_vals = LoadXFloats<kNumDims>(query + dim);

  static_assert(kNumDims == 4 || kNumDims == 8 || kNumDims == 16);

  static_assert(std::is_same_v<DataT, int8_t> || kNumDims <= 8 ||
                Simd<int16_t>::kRegisterBits >= 256);

  std::conditional_t<kNumDims == 16 && std::is_same_v<DataT, int16_t>,
                     Avx2<DataT, kUnrollBy>, Sse4<DataT, kUnrollBy>>
      db_vals;
  for (size_t jj : Seq(kUnrollBy)) {
    if constexpr (std::is_same_v<DataT, int8_t>) {
      if constexpr (kNumDims == 16) {
        db_vals[jj] = Sse4<int8_t>::Load(ptrs[jj] + dim);
      }
      if constexpr (kNumDims == 8) {
        db_vals[jj] =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptrs[jj] + dim));
      }
      if constexpr (kNumDims == 4) {
        db_vals[jj] =
            _mm_cvtsi32_si128(ABSL_INTERNAL_UNALIGNED_LOAD32(ptrs[jj] + dim));
      }
    } else {
      if constexpr (kNumDims == 16) {
        db_vals[jj] = Avx2<int16_t>::Load(ptrs[jj] + dim);
      }
      if constexpr (kNumDims == 8) {
        db_vals[jj] = Sse4<int16_t>::Load(ptrs[jj] + dim);
      }
      if constexpr (kNumDims == 4) {
        db_vals[jj] =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptrs[jj] + dim));
      }
    }
  }

  decltype(qq_vals) mult;
  if constexpr (kIsSquaredL2 && std::is_same_v<DataT, int8_t>) {
    mult = LoadXFloats<kNumDims>(inv_multipliers_for_squared_l2 + dim);
  }

  asm("" ::: "memory");

  if constexpr (kNumDims == 4) {
    for (size_t jj : Seq(kUnrollBy)) {
      Simd<float> db_vals_float =
          SseToSimd(ExpandToFp32<0, Sse4<float>>(db_vals[jj]));
      accums[jj] = FusedMultiplyOp<kIsSquaredL2, DataT>(qq_vals, db_vals_float,
                                                        mult, accums[jj]);
    }
    return accums;
  }

  if constexpr (decltype(qq_vals)::kNumRegisters >= 1) {
    for (size_t jj : Seq(kUnrollBy)) {
      accums[jj] = FusedMultiplyOp<kIsSquaredL2, DataT>(
          qq_vals[0], ExpandToFp32<0>(db_vals[jj]), mult[0], accums[jj]);
    }
  }
  if constexpr (decltype(qq_vals)::kNumRegisters >= 2) {
    for (size_t jj : Seq(kUnrollBy)) {
      accums[jj] = FusedMultiplyOp<kIsSquaredL2, DataT>(
          qq_vals[1], ExpandToFp32<1>(db_vals[jj]), mult[1], accums[jj]);
    }
  }
  if constexpr (decltype(qq_vals)::kNumRegisters >= 3) {
    for (size_t jj : Seq(kUnrollBy)) {
      accums[jj] = FusedMultiplyOp<kIsSquaredL2, DataT>(
          qq_vals[2], ExpandToFp32<2>(db_vals[jj]), mult[2], accums[jj]);
    }
  }
  if constexpr (decltype(qq_vals)::kNumRegisters >= 4) {
    for (size_t jj : Seq(kUnrollBy)) {
      accums[jj] = FusedMultiplyOp<kIsSquaredL2, DataT>(
          qq_vals[3], ExpandToFp32<3>(db_vals[jj]), mult[3], accums[jj]);
    }
  }
  static_assert(decltype(qq_vals)::kNumRegisters <= 4);

  return accums;
}

SCANN_SIMD_INLINE double StaticallyInvokeOneToOneDenseDotProduct(
    const DatapointPtr<float>& qq, const DatapointPtr<int8_t>& db) {
  using T = Simd<float>;
  if constexpr (IsSame<T, Avx512<float>>()) {
    return ::research_scann::dp_internal::DenseDotProductAvx2(db, qq);
  }
  if constexpr (IsSame<T, Avx2<float>>()) {
    return ::research_scann::dp_internal::DenseDotProductAvx2(db, qq);
  }
  if constexpr (IsSame<T, Avx1<float>>()) {
    return ::research_scann::dp_internal::DenseDotProductAvx1(db, qq);
  }
  if constexpr (IsSame<T, Sse4<float>>()) {
    return ::research_scann::dp_internal::DenseDotProductSse4(db, qq);
  }
  LOG(FATAL) << "Unhandled: " << SimdName();
}

template <size_t kDimensionality, bool kIsSquaredL2>
SCANN_SIMD_INLINE float ComputeOneToOneScore(
    const float* __restrict__ query, const int8_t* ptr,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    size_t dimensionality) {
  if constexpr (kIsSquaredL2) {
    array<const int8_t*, 1> ptrs = {ptr};
    Simd<float, 1> accums = Zeros();
    size_t dim = 0;
    for (; dim + 16 <= dimensionality; dim += 16) {
      accums = HandleXDims<16, kIsSquaredL2>(
          query, ptrs, inv_multipliers_for_squared_l2, dim, accums);
    }

    float dist = HorizontalSum(accums[0]);

    for (; dim < dimensionality; dim++) {
      const float mult = inv_multipliers_for_squared_l2[dim];
      dist = FusedMultiplyOp<kIsSquaredL2, int8_t>(
          query[dim], static_cast<float>(ptr[dim]), mult, dist);
    }
    return dist;
  } else {
    DatapointPtr<float> qq_dptr(nullptr, query, dimensionality, dimensionality);
    DatapointPtr<int8_t> db_dptr(nullptr, ptr, dimensionality, dimensionality);
    return -StaticallyInvokeOneToOneDenseDotProduct(qq_dptr, db_dptr);
  }
}

template <size_t kDimensionality, bool kIsSquaredL2>
SCANN_SIMD_INLINE float ComputeOneToOneScore(
    const float* __restrict__ query, const int16_t* ptr,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    size_t dimensionality) {
  array<const int16_t*, 1> ptrs = {ptr};
  Simd<float, 1> accums = Zeros();
  size_t dim = 0;
  if constexpr (Simd<int16_t>::kRegisterBits >= 256) {
    for (; dim + 16 <= dimensionality; dim += 16) {
      accums = HandleXDims<16, kIsSquaredL2>(
          query, ptrs, inv_multipliers_for_squared_l2, dim, accums);
    }
  } else {
    for (; dim + 8 <= dimensionality; dim += 8) {
      accums = HandleXDims<8, kIsSquaredL2>(
          query, ptrs, inv_multipliers_for_squared_l2, dim, accums);
    }
  }

  float dist = HorizontalSum(accums[0]);

  for (; dim < dimensionality; dim++) {
    dist = FusedMultiplyOp<kIsSquaredL2, int16_t>(
        query[dim], Bfloat16Decompress(ptr[dim]), 0.0f, dist);
  }
  return dist;
}

template <int kDimensionality, size_t kUnrollBy, bool kHasIndices,
          bool kIsSquaredL2, bool kShouldPrefetch, typename DataT,
          typename DatasetViewT, typename IndexT, typename ResultElemT,
          typename CallbackT>
SCANN_SIMD_INLINE void OneToManyAsymmetricTemplate(
    const float* __restrict__ query, DatasetViewT dataset_view,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    const IndexT* indices, MutableSpan<ResultElemT> result,
    CallbackT callback) {
  const size_t dimensionality =
      kDimensionality > 0 ? kDimensionality : dataset_view.dimensionality();

  const size_t num_datapoints = result.size();
  if (num_datapoints == 0 || dimensionality == 0) return;

  constexpr size_t kMinPrefetchAheadBytes = 2304;

  constexpr size_t kCacheLine = 64;
  const size_t cache_lines_per_datapoint =
      DivRoundUp(sizeof(DataT) * dimensionality, kCacheLine);
  size_t num_prefetch_datapoints;
  if (kShouldPrefetch) {
    num_prefetch_datapoints = std::max<size_t>(
        1, kMinPrefetchAheadBytes /
               (kUnrollBy * cache_lines_per_datapoint * kCacheLine));
  }

  auto get_db_ptr = [indices, &dataset_view, result, callback](size_t i)
                        SCANN_INLINE_LAMBDA -> const DataT* {
    using ::research_scann::one_to_many_low_level::GetDatapointIndex;
    const size_t idx = kHasIndices ? indices[i] : GetDatapointIndex(result, i);
    callback.prefetch(idx);
    return dataset_view.GetPtr(idx);
  };

  const size_t num_outer_iters = num_datapoints / kUnrollBy;

  if constexpr (kShouldPrefetch) {
    for (size_t j = num_datapoints / kUnrollBy * kUnrollBy; j < num_datapoints;
         j++) {
      const DataT* prefetch_ptr = get_db_ptr(j);
      for (size_t n : Seq(cache_lines_per_datapoint)) {
        ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_NTA>(
            prefetch_ptr + n * kCacheLine / sizeof(DataT));
      }
    }

    for (size_t j : Seq(std::min(num_prefetch_datapoints, num_outer_iters))) {
      array<const DataT*, kUnrollBy> prefetch_ptrs;
      for (size_t jj : Seq(kUnrollBy)) {
        prefetch_ptrs[jj] = get_db_ptr(j + jj * num_outer_iters);
      }

      for (size_t n : Seq(cache_lines_per_datapoint)) {
        for (size_t jj : Seq(kUnrollBy)) {
          ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_NTA>(
              prefetch_ptrs[jj] + n * kCacheLine / sizeof(DataT));
        }
      }
    }
  }

  std::array<float, kDimensionality> query_storage;
  if constexpr (kDimensionality > 0) {
    DCHECK_EQ(dimensionality, kDimensionality);

    std::copy(query, query + kDimensionality, query_storage.data());
    query = query_storage.data();
  }

  for (size_t j = num_datapoints / kUnrollBy * kUnrollBy; j < num_datapoints;
       j++) {
    const DataT* ptr = get_db_ptr(j);
    callback.invoke(
        j, ComputeOneToOneScore<0, kIsSquaredL2>(
               query, ptr, inv_multipliers_for_squared_l2, dimensionality));
  }

  for (size_t j : Seq(num_outer_iters)) {
    if constexpr (kShouldPrefetch) {
      if (j + num_prefetch_datapoints < num_outer_iters) {
        const size_t prefetch_j = j + num_prefetch_datapoints;

        array<const DataT*, kUnrollBy> prefetch_ptrs;
        for (size_t jj : Seq(kUnrollBy)) {
          prefetch_ptrs[jj] = get_db_ptr(prefetch_j + jj * num_outer_iters);
        }

        for (size_t n : Seq(cache_lines_per_datapoint)) {
          for (size_t jj : Seq(kUnrollBy)) {
            ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_NTA>(
                prefetch_ptrs[jj] + n * kCacheLine / sizeof(DataT));
          }
        }
      }
    }

    array<const DataT*, kUnrollBy> ptrs;
    for (size_t jj : Seq(kUnrollBy)) {
      ptrs[jj] = get_db_ptr(j + jj * num_outer_iters);
    }

    Simd<float, kUnrollBy> accums = Zeros();

    size_t dim = 0;
    if constexpr (std::is_same_v<DataT, int8_t> ||
                  Simd<int16_t>::kRegisterBits >= 256) {
      for (; dim + 16 <= dimensionality; dim += 16) {
        accums = HandleXDims<16, kIsSquaredL2>(
            query, ptrs, inv_multipliers_for_squared_l2, dim, accums);
      }

      if (dim + 8 <= dimensionality) {
        accums = HandleXDims<8, kIsSquaredL2>(
            query, ptrs, inv_multipliers_for_squared_l2, dim, accums);
        dim += 8;
      }
    } else {
      for (; dim + 8 <= dimensionality; dim += 8) {
        accums = HandleXDims<8, kIsSquaredL2>(
            query, ptrs, inv_multipliers_for_squared_l2, dim, accums);
      }
    }

    if (dim + 4 <= dimensionality) {
      accums = HandleXDims<4, kIsSquaredL2>(
          query, ptrs, inv_multipliers_for_squared_l2, dim, accums);
      dim += 4;
    }

    array<float, kUnrollBy> results;
    if constexpr (kUnrollBy == 4) {
      HorizontalSum4X(accums[0], accums[1], accums[2], accums[3], &results[0],
                      &results[1], &results[2], &results[3]);
    } else if constexpr (kUnrollBy == 3) {
      HorizontalSum3X(accums[0], accums[1], accums[2], &results[0], &results[1],
                      &results[2]);
    } else if constexpr (kUnrollBy == 2) {
      HorizontalSum2X(accums[0], accums[1], &results[0], &results[1]);
    } else {
      for (size_t jj : Seq(kUnrollBy)) {
        results[jj] = HorizontalSum(accums[jj]);
      }
    }

    for (; dim < dimensionality; ++dim) {
      for (size_t jj : Seq(kUnrollBy)) {
        float mult;
        if constexpr (kIsSquaredL2 && std::is_same_v<DataT, int8_t>) {
          mult = inv_multipliers_for_squared_l2[dim];
        } else {
          mult = 0.0;
        }
        float decompressed;
        if constexpr (std::is_same_v<DataT, int8_t>) {
          decompressed = static_cast<float>(ptrs[jj][dim]);
        } else {
          decompressed = Bfloat16Decompress(ptrs[jj][dim]);
        }
        results[jj] = FusedMultiplyOp<kIsSquaredL2, DataT>(
            query[dim], decompressed, mult, results[jj]);
      }
    }

    for (size_t jj : Seq(kUnrollBy)) {
      callback.invoke(j + jj * num_outer_iters, results[jj]);
    }
  }
}

template <bool kHasIndices, bool kIsSquaredL2, typename DatasetViewT,
          typename IndexT, typename ResultElemT, typename CallbackT>
SCANN_SIMD_OUTLINE void OneToManyInt8FloatImpl(
    const float* __restrict__ query, DatasetViewT dataset_view,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    const IndexT* indices, MutableSpan<ResultElemT> result,
    CallbackT callback) {
  const size_t dims = dataset_view.dimensionality();
  if (dims == 128) {
    OneToManyAsymmetricTemplate<128, 3, kHasIndices, kIsSquaredL2, true,
                                int8_t>(query, std::move(dataset_view),
                                        inv_multipliers_for_squared_l2, indices,
                                        result, std::move(callback));
  } else if (dims == 64) {
    OneToManyAsymmetricTemplate<64, 3, kHasIndices, kIsSquaredL2, true, int8_t>(
        query, std::move(dataset_view), inv_multipliers_for_squared_l2, indices,
        result, std::move(callback));
  } else {
    OneToManyAsymmetricTemplate<0, 3, kHasIndices, kIsSquaredL2, true, int8_t>(
        query, std::move(dataset_view), inv_multipliers_for_squared_l2, indices,
        result, std::move(callback));
  }
}

template <bool kHasIndices, bool kIsSquaredL2, typename DatasetViewT,
          typename IndexT, typename ResultElemT, typename CallbackT>
SCANN_SIMD_OUTLINE void OneToManyBf16FloatImpl(const float* __restrict__ query,
                                               DatasetViewT dataset_view,
                                               const IndexT* indices,
                                               MutableSpan<ResultElemT> result,
                                               CallbackT callback) {
  constexpr const float* kNoMultipliersForBfloat16 = nullptr;
  const size_t dims = dataset_view.dimensionality();
  if (dims == 128) {
    OneToManyAsymmetricTemplate<128, 3, kHasIndices, kIsSquaredL2, true,
                                int16_t>(query, std::move(dataset_view),
                                         kNoMultipliersForBfloat16, indices,
                                         result, std::move(callback));
  } else if (dims == 64) {
    OneToManyAsymmetricTemplate<64, 3, kHasIndices, kIsSquaredL2, true,
                                int16_t>(query, std::move(dataset_view),
                                         kNoMultipliersForBfloat16, indices,
                                         result, std::move(callback));
  } else {
    OneToManyAsymmetricTemplate<0, 3, kHasIndices, kIsSquaredL2, true, int16_t>(
        query, std::move(dataset_view), kNoMultipliersForBfloat16, indices,
        result, std::move(callback));
  }
}
