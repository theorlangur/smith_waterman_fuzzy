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

        template<>
        struct simd_prims<int16_t, 8>
        {
            static constexpr size_t Width = 8;
            using int_type_t = int16_t;
            using simd_base_t = __m128i;
            using input_t = std::array<std::string_view, Width>;
            using lengths_t = std::array<std::string_view, Width>;
            using scores_t = std::array<int_type_t, Width>;
            using lut_data_t = int_type_t;
            static constexpr int_type_t kMaskVal = 0xFFFF;

            static auto set1(int_type_t v) { return _mm_set1_epi16(v); }
            static auto set_idx(int idx, int_type_t v, simd_base_t &t) 
            { 
                int_type_t *p = reinterpret_cast<int_type_t *>(&t);
                p[idx] = v;
            }
            static auto sub(simd_base_t o1, simd_base_t o2) { return _mm_sub_epi16(o1, o2); }
            static auto add(simd_base_t o1, simd_base_t o2) { return _mm_add_epi16(o1, o2); }
            static auto max(simd_base_t o1, simd_base_t o2) { return _mm_max_epi16(o1, o2); }
            static auto blend(simd_base_t o1, simd_base_t o2, simd_base_t m) { return _mm_blendv_epi8(o1, o2, m); }
            static auto eq(simd_base_t o1, simd_base_t o2) { return _mm_cmpeq_epi16(o1, o2); }
            static auto _and(simd_base_t o1, simd_base_t o2) { return _mm_and_si128(o1, o2); }
            static auto _or(simd_base_t o1, simd_base_t o2) { return _mm_or_si128(o1, o2); }
            static auto _xor(simd_base_t o1, simd_base_t o2) { return _mm_xor_si128(o1, o2); }
            static auto load(int_type_t const(&src)[Width]) { return _mm_loadu_epi16(src); }

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
        struct simd_prims<int8_t, 16>
        {
            static constexpr size_t Width = 16;
            using int_type_t = int8_t;
            using simd_base_t = __m128i;
            using input_t = std::array<std::string_view, Width>;
            using lengths_t = std::array<std::string_view, Width>;
            using scores_t = std::array<int_type_t, Width>;
            using lut_data_t = int_type_t;
            static constexpr int_type_t kMaskVal = 0xFF;

            static auto set1(int_type_t v) { return _mm_set1_epi8(v); }
            static auto set_idx(int idx, int_type_t v, simd_base_t &t) 
            { 
                int_type_t *p = reinterpret_cast<int_type_t *>(&t);
                p[idx] = v;
            }
            static auto sub(simd_base_t o1, simd_base_t o2) { return _mm_sub_epi8(o1, o2); }
            static auto add(simd_base_t o1, simd_base_t o2) { return _mm_add_epi8(o1, o2); }
            static auto max(simd_base_t o1, simd_base_t o2) { return _mm_max_epi8(o1, o2); }
            static auto blend(simd_base_t o1, simd_base_t o2, simd_base_t m) { return _mm_blendv_epi8(o1, o2, m); }
            static auto eq(simd_base_t o1, simd_base_t o2) { return _mm_cmpeq_epi8(o1, o2); }
            static auto _and(simd_base_t o1, simd_base_t o2) { return _mm_and_si128(o1, o2); }
            static auto _or(simd_base_t o1, simd_base_t o2) { return _mm_or_si128(o1, o2); }
            static auto _xor(simd_base_t o1, simd_base_t o2) { return _mm_xor_si128(o1, o2); }
            static auto load(int_type_t const(&src)[Width]) { return _mm_loadu_epi8(src); }

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
        struct simd_prims<int8_t, 32>
        {
            static constexpr size_t Width = 32;
            using int_type_t = int8_t;
            using simd_base_t = __m256i;
            using input_t = std::array<std::string_view, Width>;
            using lengths_t = std::array<std::string_view, Width>;
            using scores_t = std::array<int_type_t, Width>;
            using lut_data_t = int8_t;
            static constexpr int_type_t kMaskVal = 0xFF;

            static auto set1(int_type_t v) { return _mm256_set1_epi8(v); }
            static auto set_idx(int idx, int_type_t v, simd_base_t &t) 
            { 
                int_type_t *p = reinterpret_cast<int_type_t *>(&t);
                p[idx] = v;
            }
            static auto sub(simd_base_t o1, simd_base_t o2) { return _mm256_sub_epi8(o1, o2); }
            static auto add(simd_base_t o1, simd_base_t o2) { return _mm256_add_epi8(o1, o2); }
            static auto max(simd_base_t o1, simd_base_t o2) { return _mm256_max_epi8(o1, o2); }
            static auto blend(simd_base_t o1, simd_base_t o2, simd_base_t m) { return _mm256_blendv_epi8(o1, o2, m); }
            static auto eq(simd_base_t o1, simd_base_t o2) { return _mm256_cmpeq_epi8(o1, o2); }
            static auto _and(simd_base_t o1, simd_base_t o2) { return _mm256_and_si256(o1, o2); }
            static auto _or(simd_base_t o1, simd_base_t o2) { return _mm256_or_si256(o1, o2); }
            static auto _xor(simd_base_t o1, simd_base_t o2) { return _mm256_xor_si256(o1, o2); }
            static auto load(int_type_t const(&src)[Width]) { return _mm256_lddqu_si256((const simd_base_t*)src); }

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
        struct simd_prims<int16_t, 16>
        {
            static constexpr size_t Width = 16;
            using int_type_t = int16_t;
            using simd_base_t = __m256i;
            using input_t = std::array<std::string_view, Width>;
            using lengths_t = std::array<std::string_view, Width>;
            using scores_t = std::array<int_type_t, Width>;
            using lut_data_t = int8_t;
            static constexpr int_type_t kMaskVal = 0xFFFF;

            static auto set1(int_type_t v) { return _mm256_set1_epi16(v); }
            static auto set_idx(int idx, int_type_t v, simd_base_t &t) 
            { 
                int_type_t *p = reinterpret_cast<int_type_t *>(&t);
                p[idx] = v;
            }
            static auto sub(simd_base_t o1, simd_base_t o2) { return _mm256_sub_epi16(o1, o2); }
            static auto add(simd_base_t o1, simd_base_t o2) { return _mm256_add_epi16(o1, o2); }
            static auto max(simd_base_t o1, simd_base_t o2) { return _mm256_max_epi16(o1, o2); }
            static auto blend(simd_base_t o1, simd_base_t o2, simd_base_t m) { return _mm256_blendv_epi8(o1, o2, m); }
            static auto eq(simd_base_t o1, simd_base_t o2) { return _mm256_cmpeq_epi16(o1, o2); }
            static auto _and(simd_base_t o1, simd_base_t o2) { return _mm256_and_si256(o1, o2); }
            static auto _or(simd_base_t o1, simd_base_t o2) { return _mm256_or_si256(o1, o2); }
            static auto _xor(simd_base_t o1, simd_base_t o2) { return _mm256_xor_si256(o1, o2); }
            static auto load(int_type_t const(&src)[Width]) { return _mm256_lddqu_si256((const simd_base_t*)src); }

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
        struct simd_prims<int32_t, 8>
        {
            static constexpr size_t Width = 8;
            using int_type_t = int32_t;
            using simd_base_t = __m256i;
            using input_t = std::array<std::string_view, Width>;
            using lengths_t = std::array<std::string_view, Width>;
            using scores_t = std::array<int_type_t, Width>;
            using lut_data_t = int8_t;
            static constexpr int_type_t kMaskVal = 0xFFFFFFFF;

            static auto set1(int_type_t v) { return _mm256_set1_epi32(v); }
            static auto set_idx(int idx, int_type_t v, simd_base_t &t) 
            { 
                int_type_t *p = reinterpret_cast<int_type_t *>(&t);
                p[idx] = v;
            }
            static auto sub(simd_base_t o1, simd_base_t o2) { return _mm256_sub_epi32(o1, o2); }
            static auto add(simd_base_t o1, simd_base_t o2) { return _mm256_add_epi32(o1, o2); }
            static auto max(simd_base_t o1, simd_base_t o2) { return _mm256_max_epi32(o1, o2); }
            static auto blend(simd_base_t o1, simd_base_t o2, simd_base_t m) { return _mm256_blendv_epi8(o1, o2, m); }
            static auto eq(simd_base_t o1, simd_base_t o2) { return _mm256_cmpeq_epi32(o1, o2); }
            static auto _and(simd_base_t o1, simd_base_t o2) { return _mm256_and_si256(o1, o2); }
            static auto _or(simd_base_t o1, simd_base_t o2) { return _mm256_or_si256(o1, o2); }
            static auto _xor(simd_base_t o1, simd_base_t o2) { return _mm256_xor_si256(o1, o2); }
            static auto load(int_type_t const(&src)[Width]) { return _mm256_lddqu_si256((const simd_base_t*)src); }

            static auto unpack(simd_base_t o)
            {
                scores_t s;
                int_type_t *pB = reinterpret_cast<int_type_t *>(&o);
                for(int i = 0; i < Width; ++i)
                    s[i] = pB[i];
                return s;
            }
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
            using ResT = Delimiters<simd_t, d...>;
            ResT r;
            auto fill_d = [](simd_t::int_type_t (&dest)[simd_t::Width], char c)
            {
                for(int i = 0; i < simd_t::Width; ++i) dest[i] = c;
            };
            [&]<size_t... I>(std::index_sequence<I...> s){
                (fill_d(r.dMasks[I], d),...);
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

        static constexpr auto kWidth = simd_t::Width;
        Config m_Config;
        ScoreCache<simd_t> m_LUTCache;
        const Delimiters<simd_t, ' ', '_', '.', ':', '-', '=', ','> m_Delimiters;

        vec delim_bonus;
        const vec zero = simd_t::set1(0);
        vec extend_gap_penalty_v;
        vec open_extend_gap_penalty_v;

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

        simd_t::scores_t sw_score_simd(std::string_view const&query, simd_t::input_t const& targets/*, int dbg_lane*/)
        {
            int maxTargetLen = 0;
            for(auto const& sv : targets) if (auto l = sv.length(); l > maxTargetLen) maxTargetLen = l;
            const int queryLen = query.length();
            const int k_max_penalty_per_char = (m_Config.extend_gap_penalty + m_Config.open_gap_penalty) < m_Config.mismatch_penalty ? m_Config.extend_gap_penalty + m_Config.open_gap_penalty : m_Config.mismatch_penalty;
            const int k_max_penalty = queryLen * k_max_penalty_per_char;

            std::vector<vec> H_prev;
            std::vector<vec> H_cur;
            std::vector<vec> E;

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
                    E[j] = simd_t::max(simd_t::sub(E[j], extend_gap_penalty_v), simd_t::sub(H_cur[j - 1], open_extend_gap_penalty_v));
                    F = simd_t::max(simd_t::sub(F, extend_gap_penalty_v), simd_t::sub(H_prev[j], open_extend_gap_penalty_v));

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
        ~Impl();
        void StopThreads();
        void SetupThreads(int threadCount);
        void Do();
        void WorkerFunc(int workerId);

        //SIMDImpl<simd_prims<int16_t, 8>> m_SSE41x16;
        //SIMDImpl<simd_prims<int8_t, 16>> m_SSE41x8;
        SIMDImpl<simd_prims<int16_t,16>> m_AVX2x16;
        SIMDImpl<simd_prims<int8_t, 32>> m_AVX2x8;

        int m_WorkerWidth = 0;
        bool m_WorkerCancel = false;
        std::counting_semaphore<> m_SemWorkStart{0};
        std::binary_semaphore m_SemMain{0};
        std::vector<std::jthread> m_WorkerThreads;
        std::vector<Result> m_WorkerResults;
        std::atomic<size_t> m_WorkerChunkOffset{0};//job_idx
        std::atomic<size_t> m_WorkersLeft{0};
        Input *m_pInputTargets = nullptr;
    };

    SIMDParMatcher::SIMDParMatcher(Config const& cfg):
        m_Config(cfg),
        m_Impl(new Impl())
    {
        m_Impl->m_AVX2x8.Setup(cfg);
        m_Impl->m_AVX2x16.Setup(cfg);
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

        if (params.sortResults)
            std::ranges::sort(res, std::greater{}, &ResultItem::score);
        return res;
    }

    SIMDParMatcher::Result SIMDParMatcher::match_par(std::string_view const&query, Input &&targets, Param const& params)
    {
        Result res;
        m_Impl->m_pInputTargets = &targets;
        std::ranges::sort(targets, std::greater{}, &std::string_view::length);

        if (params.sortResults)
            std::ranges::sort(res, std::greater{}, &ResultItem::score);
        m_Impl->m_pInputTargets = nullptr;
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
        m_WorkerResults.resize(threadCount);

        for(int i = 0; i < threadCount; ++i)
            m_WorkerThreads.emplace_back(&Impl::WorkerFunc, this, i);
    }

    void SIMDParMatcher::Impl::Do()
    {
        m_WorkersLeft.store(m_WorkerThreads.size(), std::memory_order_relaxed);
        m_SemWorkStart.release(m_WorkerThreads.size());

        //now we wait
        m_SemMain.acquire();
    }

    void SIMDParMatcher::Impl::WorkerFunc(int workerId)
    {
        while(true)
        {
            m_SemWorkStart.acquire();
            if (m_WorkerCancel)
                break;
            //...useful work
            int i = workerId * m_WorkerWidth;
            //int n = (int)lines.size();

            if (m_WorkersLeft.fetch_sub(1, std::memory_order_relaxed) == 1)
            {
                //this was the last worker to finish, it sets the main semaphore
                m_SemMain.release();
            }
        }
    }
}
