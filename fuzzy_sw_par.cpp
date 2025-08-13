#include "fuzzy_sw.hpp"

#include <ranges>
#include <thread>
#include <barrier>
#include <semaphore>
#include <algorithm>
#include <immintrin.h>

namespace fuzzy_sw
{
    namespace{
        template<typename IntType, size_t W>
        struct simd_prims;

        template<typename IntType, size_t W, typename simd_t>
        struct simd_prims_base
        {
            static constexpr size_t Width = W;
            using int_type_t = IntType;
            using simd_base_t = simd_t;

            using input_t = std::array<std::string_view, W>;
            using input_char_src_t = std::array<SIMDParMatcher::CharSource*, W>;
            using scores_t = std::array<IntType, W>;
            using lut_data_t = int8_t;
            static constexpr IntType kMaskVal = IntType(-1);

            static auto set_idx(int idx, int_type_t v, simd_base_t &t) 
            { 
                int_type_t *p = reinterpret_cast<int_type_t *>(&t);
                p[idx] = v;
            }

            static auto unpack(simd_base_t o)
            {
                scores_t s;
                int_type_t *pB = reinterpret_cast<int_type_t *>(&o);
                for(int i = 0; i < Width; ++i)
                    s[i] = pB[i];
                return s;
            }
        };

        template<>
        struct simd_prims<int16_t, 8>: simd_prims_base<int16_t, 8, __m128i>
        {
            static auto set1(int_type_t v) { return _mm_set1_epi16(v); }
            static auto sub(simd_base_t o1, simd_base_t o2) { return _mm_sub_epi16(o1, o2); }
            static auto add(simd_base_t o1, simd_base_t o2) { return _mm_add_epi16(o1, o2); }
            static auto max(simd_base_t o1, simd_base_t o2) { return _mm_max_epi16(o1, o2); }
            static auto blend(simd_base_t o1, simd_base_t o2, simd_base_t m) { return _mm_blendv_epi8(o1, o2, m); }
            static auto eq(simd_base_t o1, simd_base_t o2) { return _mm_cmpeq_epi16(o1, o2); }
            static auto _and(simd_base_t o1, simd_base_t o2) { return _mm_and_si128(o1, o2); }
            static auto _or(simd_base_t o1, simd_base_t o2) { return _mm_or_si128(o1, o2); }
            static auto _xor(simd_base_t o1, simd_base_t o2) { return _mm_xor_si128(o1, o2); }
            static auto load(int_type_t const(&src)[Width]) { return _mm_loadu_epi16(src); }
        };

        template<>
        struct simd_prims<int8_t, 16>: simd_prims_base<int8_t, 16, __m128i>
        {
            static auto set1(int_type_t v) { return _mm_set1_epi8(v); }
            static auto sub(simd_base_t o1, simd_base_t o2) { return _mm_sub_epi8(o1, o2); }
            static auto add(simd_base_t o1, simd_base_t o2) { return _mm_add_epi8(o1, o2); }
            static auto max(simd_base_t o1, simd_base_t o2) { return _mm_max_epi8(o1, o2); }
            static auto blend(simd_base_t o1, simd_base_t o2, simd_base_t m) { return _mm_blendv_epi8(o1, o2, m); }
            static auto eq(simd_base_t o1, simd_base_t o2) { return _mm_cmpeq_epi8(o1, o2); }
            static auto _and(simd_base_t o1, simd_base_t o2) { return _mm_and_si128(o1, o2); }
            static auto _or(simd_base_t o1, simd_base_t o2) { return _mm_or_si128(o1, o2); }
            static auto _xor(simd_base_t o1, simd_base_t o2) { return _mm_xor_si128(o1, o2); }
            static auto load(int_type_t const(&src)[Width]) { return _mm_loadu_epi8(src); }
        };

        template<>
        struct simd_prims<int8_t, 32>: simd_prims_base<int8_t, 32, __m256i>
        {
            static auto set1(int_type_t v) { return _mm256_set1_epi8(v); }
            static auto sub(simd_base_t o1, simd_base_t o2) { return _mm256_sub_epi8(o1, o2); }
            static auto add(simd_base_t o1, simd_base_t o2) { return _mm256_add_epi8(o1, o2); }
            static auto max(simd_base_t o1, simd_base_t o2) { return _mm256_max_epi8(o1, o2); }
            static auto blend(simd_base_t o1, simd_base_t o2, simd_base_t m) { return _mm256_blendv_epi8(o1, o2, m); }
            static auto eq(simd_base_t o1, simd_base_t o2) { return _mm256_cmpeq_epi8(o1, o2); }
            static auto _and(simd_base_t o1, simd_base_t o2) { return _mm256_and_si256(o1, o2); }
            static auto _or(simd_base_t o1, simd_base_t o2) { return _mm256_or_si256(o1, o2); }
            static auto _xor(simd_base_t o1, simd_base_t o2) { return _mm256_xor_si256(o1, o2); }
            static auto load(int_type_t const(&src)[Width]) { return _mm256_lddqu_si256((const simd_base_t*)src); }
        };

        template<>
        struct simd_prims<int16_t, 16>: simd_prims_base<int16_t, 16, __m256i>
        {
            static auto set1(int_type_t v) { return _mm256_set1_epi16(v); }
            static auto sub(simd_base_t o1, simd_base_t o2) { return _mm256_sub_epi16(o1, o2); }
            static auto add(simd_base_t o1, simd_base_t o2) { return _mm256_add_epi16(o1, o2); }
            static auto max(simd_base_t o1, simd_base_t o2) { return _mm256_max_epi16(o1, o2); }
            static auto blend(simd_base_t o1, simd_base_t o2, simd_base_t m) { return _mm256_blendv_epi8(o1, o2, m); }
            static auto eq(simd_base_t o1, simd_base_t o2) { return _mm256_cmpeq_epi16(o1, o2); }
            static auto _and(simd_base_t o1, simd_base_t o2) { return _mm256_and_si256(o1, o2); }
            static auto _or(simd_base_t o1, simd_base_t o2) { return _mm256_or_si256(o1, o2); }
            static auto _xor(simd_base_t o1, simd_base_t o2) { return _mm256_xor_si256(o1, o2); }
            static auto load(int_type_t const(&src)[Width]) { return _mm256_lddqu_si256((const simd_base_t*)src); }
        };

        template<>
        struct simd_prims<int32_t, 8>: simd_prims_base<int32_t, 8, __m256i>
        {
            static auto set1(int_type_t v) { return _mm256_set1_epi32(v); }
            static auto sub(simd_base_t o1, simd_base_t o2) { return _mm256_sub_epi32(o1, o2); }
            static auto add(simd_base_t o1, simd_base_t o2) { return _mm256_add_epi32(o1, o2); }
            static auto max(simd_base_t o1, simd_base_t o2) { return _mm256_max_epi32(o1, o2); }
            static auto blend(simd_base_t o1, simd_base_t o2, simd_base_t m) { return _mm256_blendv_epi8(o1, o2, m); }
            static auto eq(simd_base_t o1, simd_base_t o2) { return _mm256_cmpeq_epi32(o1, o2); }
            static auto _and(simd_base_t o1, simd_base_t o2) { return _mm256_and_si256(o1, o2); }
            static auto _or(simd_base_t o1, simd_base_t o2) { return _mm256_or_si256(o1, o2); }
            static auto _xor(simd_base_t o1, simd_base_t o2) { return _mm256_xor_si256(o1, o2); }
            static auto load(int_type_t const(&src)[Width]) { return _mm256_lddqu_si256((const simd_base_t*)src); }
        };
    }

    constexpr uint8_t tolower_u8(uint8_t c) {
      return (c >= 'A' && c <= 'Z') ? (uint8_t)(c + 32) : c;
    }

    constexpr size_t GetCeilPow2(size_t v)
    {
        if ((v & (v - 1)) == 0)
            return v;//already
        size_t res = 1;
        while(res < v) res <<= 1;
        return res;
    }
    constexpr size_t GetLog2(size_t v)
    {
        size_t res = 0;
        while(v > 1) v >>= 1, ++res;
        return res;
    }

    template<class simd_t, class T = simd_t::lut_data_t, Config cfg = {}>
    struct alignas(64) ScoreCache
    {
        static constexpr auto kCFG = cfg;
        T data[256][256];

        static constexpr ScoreCache<simd_t, T, cfg> GenerateGScores()
        {
            using Cache = ScoreCache<simd_t, T, cfg>;
            Cache res;
            for(int i = 0; i < 256; ++i)
            {
                for(int j = 0; j < 256; ++j)
                {
                    if (i == j)
                        res.data[i][j] = kCFG.full_match_bonus;
                    else if (tolower_u8(uint8_t(i)) == tolower_u8(uint8_t(j)))
                        res.data[i][j] = kCFG.icase_match_bonus;
                    else
                        res.data[i][j] = kCFG.mismatch_penalty;
                }
            }
            return res;
        }

        void GenerateGScores(Config const& _cfg)
        {
            for(int i = 0; i < 256; ++i)
            {
                for(int j = 0; j < 256; ++j)
                {
                    if (i == j)
                        data[i][j] = _cfg.full_match_bonus;
                    else if (tolower_u8(uint8_t(i)) == tolower_u8(uint8_t(j)))
                        data[i][j] = _cfg.icase_match_bonus;
                    else
                        data[i][j] = _cfg.mismatch_penalty;
                }
            }
        }

        void Gather(char q, simd_t::simd_base_t tgt, simd_t::simd_base_t &out_scores)
        {
            typename simd_t::int_type_t *pB = reinterpret_cast<simd_t::int_type_t *>(&tgt);
            typename simd_t::int_type_t *pS = reinterpret_cast<simd_t::int_type_t *>(&out_scores);
            for(int i = 0; i < simd_t::Width; ++i)
                pS[i] = data[q][pB[i]];
        }
    };

    template<class simd_t, char... d>
    struct alignas(64) Delimiters
    {
        static constexpr size_t kPow2 = GetCeilPow2(sizeof...(d));
        static constexpr size_t kLog2 = GetLog2(kPow2);
        simd_t::int_type_t dMasks[sizeof...(d)][simd_t::Width];

        constexpr Delimiters<simd_t, d...>()
        {
            auto fill_d = [](simd_t::int_type_t (&dest)[simd_t::Width], char c)
            {
                for(int i = 0; i < simd_t::Width; ++i) dest[i] = c;
            };
            [&]<size_t... I>(std::index_sequence<I...> s){
                (fill_d(dMasks[I], d),...);
            }(std::make_index_sequence<sizeof...(d)>());
        }

        simd_t::simd_base_t MatchDelimiters(simd_t::simd_base_t syms) const
        {
            using V = simd_t::simd_base_t;
            V temp[kPow2] = {};
            //1. check all masks and store results locally
            [&]<size_t... I>(std::index_sequence<I...> s){
                ((temp[I] = simd_t::eq(syms, simd_t::load(dMasks[I]))),...);
            }(std::make_index_sequence<sizeof...(d)>());

            //2. gradually 'or' all pairs
            auto or_stage = [&]<size_t... I>(std::index_sequence<I...> s){
                ((temp[I] = simd_t::_or(temp[I], temp[I + sizeof...(I)])),...);
            };
            [&]<size_t...I>(std::index_sequence<I...> s)
            {
                ((or_stage(std::make_index_sequence<(1 << (kLog2 - I - 1))>())),...);
            }(std::make_index_sequence<kLog2>());

            return temp[0];
        }

        void Generate()
        {
            using ResT = Delimiters<simd_t, d...>;
            auto fill_d = [](simd_t::int_type_t (&dest)[simd_t::Width], char c)
            {
                for(int i = 0; i < simd_t::Width; ++i) dest[i] = c;
            };
            [&]<size_t... I>(std::index_sequence<I...> s){
                (fill_d(dMasks[I], d),...);
            }(std::make_index_sequence<sizeof...(d)>());
        }
    };


    template<class simd_t>
    struct SIMDImpl
    {
        using vec = simd_t::simd_base_t;
        using input_t = simd_t::input_t;
        using input_char_src_t = simd_t::input_char_src_t;

        static constexpr auto kWidth = simd_t::Width;
        static constexpr auto kTargetCacheSize = 256;
        Config m_Config;
        ScoreCache<simd_t> m_LUTCache;
        const Delimiters<simd_t, ' ', '_', '.', ':', '-', '=', ','> m_Delimiters;

        vec delim_bonus;
        const vec zero = simd_t::set1(0);
        vec extend_gap_penalty_v;
        vec open_extend_gap_penalty_v;

        std::vector<vec> H_prev;
        std::vector<vec> H_cur;
        std::vector<vec> E;

        vec m_CachedTargets[kTargetCacheSize];
        vec m_CachedTargetValidityVecMask[kTargetCacheSize];
        size_t m_CachedValidityMask[kTargetCacheSize];
        size_t m_CacheLastOffset = size_t(-1);

        bool QueryFits(std::string_view const& q) const
        {
            return ((int)std::numeric_limits<typename simd_t::int_type_t>::min() <= (q.length() * (m_Config.open_gap_penalty + m_Config.extend_gap_penalty)));
        }

        void Setup(Config const& cfg)
        {
            m_Config = cfg;
            m_LUTCache.GenerateGScores(m_Config);
            delim_bonus = simd_t::set1(cfg.delimiter_boundary_bonus);
            extend_gap_penalty_v = simd_t::set1(cfg.extend_gap_penalty);
            open_extend_gap_penalty_v = simd_t::set1(cfg.open_gap_penalty + cfg.extend_gap_penalty);

            //std::memset(std::begin(m_CachedValidityMask), 0xff, sizeof(m_CachedValidityMask));
            //std::memset(std::begin(m_CachedTargetValidityVecMask), 0xff, sizeof(m_CachedTargetValidityVecMask));
        }

        void pack_chars(simd_t::input_t const& targets, int j, simd_t::simd_base_t &out_orig_case, simd_t::simd_base_t &out_mask, size_t &out_bitmask)
        {
            for(int i = 0, n = targets.size(); i < n; ++i)
            {
                if (!(out_bitmask & size_t(1) << i))
                    continue;

                auto const& t = targets[i];
                if (j < t.length())
                    simd_t::set_idx(i, t[j], out_orig_case);
                else
                {
                    out_bitmask &= ~(size_t(1) << i);
                    simd_t::set_idx(i, 0, out_mask);
                }
            }
        }

        void fill_cache_from_sources(simd_t::input_char_src_t const& targets, int offset)
        {
            std::memset(std::begin(m_CachedValidityMask), 0, sizeof(m_CachedValidityMask));
            std::memset(std::begin(m_CachedTargetValidityVecMask), 0, sizeof(m_CachedTargetValidityVecMask));

            char local_buf[kTargetCacheSize];
            for(int s = 0; s < kWidth; ++s)
            {
                auto *pSrc = targets[s];
                size_t l = pSrc->read(local_buf, offset, kTargetCacheSize);
                size_t m = 1 << s;
                for(size_t i = 0; i < l; ++i)
                {
                    auto *pInt = (typename simd_t::int_type_t *)(&m_CachedTargets[i]);
                    pInt[s] = local_buf[i];

                    m_CachedValidityMask[i] |= m;
                    pInt = (typename simd_t::int_type_t *)(&m_CachedTargetValidityVecMask[i]);
                    pInt[s] = typename simd_t::int_type_t(-1);
                }
            }
        }

        void pack_chars(simd_t::input_char_src_t const& targets, int j, simd_t::simd_base_t &out_orig_case, simd_t::simd_base_t &out_mask, size_t &out_bitmask)
        {
            if (j < m_CacheLastOffset || j >= (m_CacheLastOffset + kTargetCacheSize))
            {
                m_CacheLastOffset = j;
                fill_cache_from_sources(targets, j);
            }

            size_t off = j - m_CacheLastOffset;
            out_bitmask = m_CachedValidityMask[off];
            out_mask = m_CachedTargetValidityVecMask[off];
            out_orig_case = m_CachedTargets[off];
        }

        int get_max_length(simd_t::input_char_src_t const& targets) const
        {
            int maxTargetLen = 0;
            for(auto const& sv : targets) if (auto l = sv->length(); l > maxTargetLen) maxTargetLen = l;
            return maxTargetLen;
        }

        int get_max_length(simd_t::input_t const& targets) const
        {
            int maxTargetLen = 0;
            for(auto const& sv : targets) if (auto l = sv.length(); l > maxTargetLen) maxTargetLen = l;
            return maxTargetLen;
        }

        template<class InType = simd_t::input_t>
        simd_t::scores_t sw_score_simd(std::string_view const&query, InType const& targets)
        {
            int maxTargetLen = get_max_length(targets);
            const int queryLen = query.length();
            const int k_max_penalty_per_char = (m_Config.extend_gap_penalty + m_Config.open_gap_penalty) < m_Config.mismatch_penalty ? m_Config.extend_gap_penalty + m_Config.open_gap_penalty : m_Config.mismatch_penalty;
            const int k_max_penalty = queryLen * k_max_penalty_per_char;

            const vec max_penalty = simd_t::set1(k_max_penalty);

            vec best = simd_t::set1(0);

            H_prev.resize(maxTargetLen + 1);
            std::fill(H_prev.begin(), H_prev.end(), zero);

            H_cur.resize(maxTargetLen + 1);

            E.resize(maxTargetLen + 1);
            std::fill(E.begin(), E.end(), max_penalty);

            for(int i = 1; i <= queryLen; ++i)
            {
                H_cur[0] = zero;
                vec F = zero;
                char qc = query[i - 1];

                vec prev_is_delim = zero;
                vec valid_targets_mask = simd_t::set1(simd_t::kMaskVal);
                size_t valid_targets_bitmask = size_t(-1);
                for(int j = 1; j <= maxTargetLen; ++j)
                {
                    vec target_chars_lower;
                    vec target_chars_orig;
                    pack_chars(targets, j - 1, target_chars_orig, valid_targets_mask, valid_targets_bitmask);
                    vec is_delim = m_Delimiters.MatchDelimiters(target_chars_orig);
                    E[j] = simd_t::max(simd_t::add(E[j], extend_gap_penalty_v), simd_t::add(H_cur[j - 1], open_extend_gap_penalty_v));
                    F = simd_t::max(simd_t::add(F, extend_gap_penalty_v), simd_t::add(H_prev[j], open_extend_gap_penalty_v));

                    vec match_score;
                    m_LUTCache.Gather(qc, target_chars_orig, match_score);
                    vec mismatches = simd_t::eq(match_score, simd_t::set1(-1));
                    vec delim_bonus_mask = simd_t::blend(simd_t::_xor(is_delim, prev_is_delim), simd_t::set1(0), mismatches);
                    vec delim_bonus_val = simd_t::blend(simd_t::set1(0), delim_bonus, delim_bonus_mask);
                    match_score = simd_t::add(delim_bonus_val, match_score);

                    auto diag = simd_t::add(H_prev[j - 1], match_score);
                    auto H = simd_t::max(simd_t::max(zero, diag), simd_t::max(E[j], F));
                    H = simd_t::blend(zero, H, valid_targets_mask);
                    H_cur[j] = H;

                    best = simd_t::max(H, best);
                    prev_is_delim = is_delim;
                }
                std::swap(H_prev, H_cur);
            }

            return simd_t::unpack(best);
        }
    };

    struct SIMDParMatcher::Impl
    {
        using WorkerFuncT = void(Impl::*)(int wId, void* pSIMD);

        ~Impl();
        void StopThreads();
        void SetupThreads(int threadCount);
        template<class Result, class Input>
        void Do();
        void WorkerFunc(int workerId);
        template<class SIMD, class input_t = typename SIMD::input_t, class Result, class Input>
        void WorkerFuncTpl(int workerId, void *pSIMD);

        SIMDImpl<simd_prims<int32_t,8>> m_AVX2x32;
        SIMDImpl<simd_prims<int16_t,16>> m_AVX2x16;
        SIMDImpl<simd_prims<int8_t, 32>> m_AVX2x8;

        int m_WorkerWidth = 0;
        bool m_WorkerCancel = false;
        std::counting_semaphore<> m_SemWorkStart{0};
        std::binary_semaphore m_SemMain{0};
        std::vector<std::jthread> m_WorkerThreads;
        std::atomic<size_t> m_WorkerChunkOffset{0};//job_idx
        std::atomic<size_t> m_WorkersLeft{0};

        template<class Result, class Input>
        struct WorkerTypedContext
        {
            std::vector<Result> m_WorkerResults;
            Input *m_pInputTargets = nullptr;
            Result *m_pResult = nullptr;
        };
        void *m_pWorkerContext = nullptr;
        std::string_view m_Query;

        WorkerFuncT m_WorkerFunc = nullptr;
        void *m_pSIMDTypeErased = nullptr;
    };

    SIMDParMatcher::SIMDParMatcher(Config const& cfg):
        m_Config(cfg),
        m_Impl(new Impl())
    {
        m_Impl->m_AVX2x8.Setup(cfg);
        m_Impl->m_AVX2x16.Setup(cfg);
        m_Impl->m_AVX2x32.Setup(cfg);
    }


    SIMDParMatcher::~SIMDParMatcher()
    {
    }

    SIMDParMatcher::Result SIMDParMatcher::match(std::string_view const&query, Input &&targets, Param const& params)
    {
        Result res;
        std::ranges::sort(targets, std::greater{}, &std::string_view::length);
        auto Match = [&](auto &impl)
        {
            using simd_impl_t = std::remove_cvref_t<decltype(impl)>;
            res.reserve(targets.size());
            for(size_t i = 0, n = targets.size(); i < n; i += simd_impl_t::kWidth)
            {
                typename simd_impl_t::input_t block;
                size_t l = (i + simd_impl_t::kWidth) < n ? simd_impl_t::kWidth : (n - i);
                for(size_t j = 0; j < l; ++j)
                    block[j] = targets[i + j];
                auto scores = impl.sw_score_simd(query, block);
                for(size_t j = 0; j < l; ++j)
                    if (auto s = scores[i + j]; s > params.scoreThreshold)
                        res.emplace_back(targets[i + j], s);
            }
        };

        if (auto &impl = m_Impl->m_AVX2x8; impl.QueryFits(query))
            Match(impl);
        else if (auto &impl = m_Impl->m_AVX2x16; impl.QueryFits(query))
            Match(impl);
        else
            Match(m_Impl->m_AVX2x32);

        if (params.sortResults)
            std::ranges::sort(res, std::greater{}, &ResultItem::score);
        return res;
    }

    SIMDParMatcher::CharSourceResult SIMDParMatcher::match(std::string_view const&query, CharSourceInput &&targets, Param const& params)
    {
        CharSourceResult res;
        std::ranges::sort(targets, std::greater{}, &CharSource::length);
        auto Match = [&](auto &impl)
        {
            using simd_impl_t = std::remove_cvref_t<decltype(impl)>;
            res.reserve(targets.size());
            for(size_t i = 0, n = targets.size(); i < n; i += simd_impl_t::kWidth)
            {
                typename simd_impl_t::input_char_src_t block;
                size_t l = (i + simd_impl_t::kWidth) < n ? simd_impl_t::kWidth : (n - i);
                for(size_t j = 0; j < l; ++j)
                    block[j] = targets[i + j];
                auto scores = impl.sw_score_simd(query, block);
                for(size_t j = 0; j < l; ++j)
                    if (auto s = scores[i + j]; s > params.scoreThreshold)
                        res.emplace_back(targets[i + j], s);
            }
        };

        if (auto &impl = m_Impl->m_AVX2x8; impl.QueryFits(query))
            Match(impl);
        else if (auto &impl = m_Impl->m_AVX2x16; impl.QueryFits(query))
            Match(impl);
        else
            Match(m_Impl->m_AVX2x32);

        if (params.sortResults)
            std::ranges::sort(res, std::greater{}, &CharSourceResultItem::score);
        return res;
    }

    SIMDParMatcher::CharSourceResult SIMDParMatcher::match_par(std::string_view const&query, CharSourceInput &&targets, Param const& params)
    {
        CharSourceResult res;
        Impl::WorkerTypedContext<CharSourceResult,CharSourceInput> ctx;
        m_Impl->m_pWorkerContext = &ctx;
        ctx.m_pInputTargets = &targets;
        ctx.m_pResult = &res;

        m_Impl->m_Query = query;
        std::ranges::sort(targets, std::greater{}, &CharSource::length);

        if (auto &impl = m_Impl->m_AVX2x8; impl.QueryFits(query))
        {
            using simd_t = decltype(Impl::m_AVX2x8);
            m_Impl->m_WorkerFunc = &Impl::WorkerFuncTpl<simd_t, simd_t::input_char_src_t, CharSourceResult, CharSourceInput>;
            m_Impl->m_pSIMDTypeErased = &impl;
            m_Impl->m_WorkerWidth = std::remove_cvref_t<decltype(impl)>::kWidth;
        }
        else if (auto &impl = m_Impl->m_AVX2x16; impl.QueryFits(query))
        {
            using simd_t = decltype(Impl::m_AVX2x16);
            m_Impl->m_WorkerFunc = &Impl::WorkerFuncTpl<simd_t, simd_t::input_char_src_t, CharSourceResult, CharSourceInput>;
            m_Impl->m_pSIMDTypeErased = &impl;
            m_Impl->m_WorkerWidth = std::remove_cvref_t<decltype(impl)>::kWidth;
        }else
        {
            using simd_t = decltype(Impl::m_AVX2x32);
            m_Impl->m_WorkerFunc = &Impl::WorkerFuncTpl<simd_t, simd_t::input_char_src_t, CharSourceResult, CharSourceInput>;
            m_Impl->m_pSIMDTypeErased = &m_Impl->m_AVX2x32;
            m_Impl->m_WorkerWidth = decltype(Impl::m_AVX2x32)::kWidth;
        }

        m_Impl->Do<CharSourceResult,CharSourceInput>();

        if (params.sortResults)
            std::ranges::sort(res, std::greater{}, &CharSourceResultItem::score);

        m_Impl->m_WorkerFunc = nullptr;
        m_Impl->m_pWorkerContext = &ctx;
        m_Impl->m_pSIMDTypeErased = nullptr;
        return res;
    }

    SIMDParMatcher::Result SIMDParMatcher::match_par(std::string_view const&query, Input &&targets, Param const& params)
    {
        Result res;
        Impl::WorkerTypedContext<Result,Input> ctx;
        m_Impl->m_pWorkerContext = &ctx;
        ctx.m_pInputTargets = &targets;
        ctx.m_pResult = &res;

        m_Impl->m_Query = query;
        std::ranges::sort(targets, std::greater{}, &std::string_view::length);

        if (auto &impl = m_Impl->m_AVX2x8; impl.QueryFits(query))
        {
            using simd_t = decltype(Impl::m_AVX2x8);
            m_Impl->m_WorkerFunc = &Impl::WorkerFuncTpl<simd_t, simd_t::input_t, Result, Input>;
            m_Impl->m_pSIMDTypeErased = &impl;
            m_Impl->m_WorkerWidth = std::remove_cvref_t<decltype(impl)>::kWidth;
        }
        else if (auto &impl = m_Impl->m_AVX2x16; impl.QueryFits(query))
        {
            using simd_t = decltype(Impl::m_AVX2x16);
            m_Impl->m_WorkerFunc = &Impl::WorkerFuncTpl<simd_t, simd_t::input_t, Result, Input>;
            m_Impl->m_pSIMDTypeErased = &impl;
            m_Impl->m_WorkerWidth = std::remove_cvref_t<decltype(impl)>::kWidth;
        }else
        {
            using simd_t = decltype(Impl::m_AVX2x32);
            m_Impl->m_WorkerFunc = &Impl::WorkerFuncTpl<simd_t, simd_t::input_t, Result, Input>;
            m_Impl->m_pSIMDTypeErased = &m_Impl->m_AVX2x32;
            m_Impl->m_WorkerWidth = decltype(Impl::m_AVX2x32)::kWidth;
        }

        m_Impl->Do<Result,Input>();

        if (params.sortResults)
            std::ranges::sort(res, std::greater{}, &ResultItem::score);

        m_Impl->m_WorkerFunc = nullptr;
        m_Impl->m_pWorkerContext = &ctx;
        m_Impl->m_pSIMDTypeErased = nullptr;
        return res;
    }

    void SIMDParMatcher::SetupThreads(int threadCount)
    {
        m_Impl->SetupThreads(threadCount);
    }

    void SIMDParMatcher::StopThreads()
    {
        m_Impl->StopThreads();
    }


    /**********************************************************************/
    /* SIMDParMatcher::Impl                                               */
    /**********************************************************************/
    SIMDParMatcher::Impl::~Impl()
    {
        StopThreads();
    }

    void SIMDParMatcher::Impl::StopThreads()
    {
        m_WorkerCancel = true;
        m_SemWorkStart.release(m_WorkerThreads.size());
        m_WorkerThreads.resize(0);//threads will be joined here
        m_WorkerCancel = false;
    }

    void SIMDParMatcher::Impl::SetupThreads(int threadCount)
    {
        StopThreads();

        for(int i = 0; i < threadCount; ++i)
            m_WorkerThreads.emplace_back(&Impl::WorkerFunc, this, i);
    }


    template<class Result, class Input>
    void SIMDParMatcher::Impl::Do()
    {
        WorkerTypedContext<Result,Input> &ctx = *static_cast<WorkerTypedContext<Result,Input>*>(m_pWorkerContext);
        size_t nThreads = m_WorkerThreads.size();
        ctx.m_WorkerResults.resize(nThreads);
        m_WorkersLeft.store(nThreads, std::memory_order_relaxed);
        m_WorkerChunkOffset.store(nThreads * m_WorkerWidth, std::memory_order_relaxed);
        for(auto &par : ctx.m_WorkerResults) par.clear();
        m_SemWorkStart.release(nThreads);

        //now we wait
        m_SemMain.acquire();

        for(auto &par : ctx.m_WorkerResults)
            for(auto &l : par) ctx.m_pResult->push_back(l);
    }

    template<class simd_t, class input_t, class Result, class Input>
    void SIMDParMatcher::Impl::WorkerFuncTpl(int workerId, void *pSIMD)
    {
        simd_t &simd = *reinterpret_cast<simd_t*>(pSIMD);
        WorkerTypedContext<Result,Input> &ctx = *static_cast<WorkerTypedContext<Result,Input>*>(m_pWorkerContext);
        auto &lines = *ctx.m_pInputTargets;
        auto &results = ctx.m_WorkerResults[workerId];
        size_t i = workerId * m_WorkerWidth;
        size_t n = lines.size();
        while(i < n)
        {
            size_t m = (i + m_WorkerWidth) < n ? i + m_WorkerWidth : n;
            auto *pFrom = &lines[i];
            auto *pTo = &lines[m];
            input_t in;
            std::copy(pFrom, pTo, in.begin());
            auto scores = simd.sw_score_simd(m_Query, in);
            for(int j = 0, tn = m - i; j < tn; ++j)
            {
                if (scores[j])
                    results.emplace_back(lines[i + j], scores[j]);
            }
            i = m_WorkerChunkOffset.fetch_add(m_WorkerWidth, std::memory_order_relaxed);
        }
    }

    void SIMDParMatcher::Impl::WorkerFunc(int workerId)
    {
        while(true)
        {
            m_SemWorkStart.acquire();
            if (m_WorkerCancel)
                break;
            //...useful work
            (this->*m_WorkerFunc)(workerId, m_pSIMDTypeErased);
            if (m_WorkersLeft.fetch_sub(1, std::memory_order_relaxed) == 1)
            {
                //this was the last worker to finish, it sets the main semaphore
                m_SemMain.release();
            }
        }
    }
}
