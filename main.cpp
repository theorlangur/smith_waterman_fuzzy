#include <immintrin.h>
#include <algorithm>
#include <print>
#include <iostream>
#include <string_view>
#include <vector>
#include <array>
#include <ranges>
#include <chrono>
#include <thread>
#include <barrier>
#include <semaphore>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

class ScopeTimer final {
public:
    explicit ScopeTimer(std::string_view name)
        : name_(std::move(name)), start_(clock::now()) {}

    // non-copyable, non-movable
    ScopeTimer(const ScopeTimer&) = delete;
    ScopeTimer& operator=(const ScopeTimer&) = delete;
    ScopeTimer(ScopeTimer&&) = delete;
    ScopeTimer& operator=(ScopeTimer&&) = delete;

    void cancel() { enabled = false; }
    auto get_duration() 
    { 
        enabled = false;
        const auto end   = clock::now();
        return end - start_;
    } 

    void print_now()
    {
        using namespace std::chrono;
        const auto delta =  get_duration();

        // Print a friendly unit automatically.
        // Destructor must not throw, so swallow any exceptions from printing.
        const auto ns = duration_cast<nanoseconds>(delta).count();
        if (ns < 1'000) {
            std::println("{} took {} ns", name_, ns);
        } else if (ns < 1'000'000) {
            std::println("{} took {:.3f} µs", name_, ns / 1'000.0);
        } else if (ns < 1'000'000'000) {
            std::println("{} took {:.3f} ms", name_, ns / 1'000'000.0);
        } else {
            std::println("{} took {:.3f} s",  name_, ns / 1'000'000'000.0);
        }
    }

    ~ScopeTimer() noexcept {
        if (!enabled) return;

        try {
            print_now();
        }catch(...)
        {
        }
    }

    using clock = std::chrono::high_resolution_clock;
    using duration_t = clock::duration;
private:

    std::string_view name_;
    clock::time_point start_;
    bool enabled = true;
};
int sw_score(std::string_view const& query, std::string_view const& target/*, bool dbg*/)
{
    constexpr int extend_gap_penalty = 1;
    constexpr int open_gap_penalty = 2;

    auto match = [](char c1, char c2)
    {
        if (c1 == c2) return 3;
        if (std::tolower(c1) == std::tolower(c2)) return 2;
        return -1;
    };

    int n = query.length();
    int m = target.length();
    const int max_penalty = -n * open_gap_penalty;
    std::vector<int> H_prev, H_cur;
    std::vector<int> E;
    H_prev.resize(m + 1);
    H_cur.resize(m + 1);
    E.resize(m + 1);
    int best = 0;
    for(int i = 1; i <= n; ++i)
    {
        H_cur[0] = 0;
        E[0] = max_penalty;
        int F = 0;
        for(int j = 1; j <= m; ++j)
        {
            E[j] = std::max(E[j] - extend_gap_penalty, H_cur[j - 1] - open_gap_penalty - extend_gap_penalty);
            F = std::max(F - extend_gap_penalty, H_prev[j] - open_gap_penalty - extend_gap_penalty);
            int diag = H_prev[j - 1] + match(query[i - 1], target[j - 1]);
            int H = std::max({0, diag, E[j], F});
            H_cur[j] = H;
            if (H > best) best = H;
            //if (dbg)
            //    std::println("i={}; j={}; H={}; E[j]={}; F={}; best={};", i, j, H, E[j], F, best);
        }
        std::swap(H_prev, H_cur);
    }

    return best;
}

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

    static auto unpack(simd_base_t o)
    {
        scores_t s;
        int_type_t *pB = reinterpret_cast<int_type_t *>(&o);
        for(int i = 0; i < Width; ++i)
            s[i] = pB[i];
        return s;
    }
};

//using simd_t = simd_prims<int16_t, 8>;
//using simd_t = simd_prims<int8_t, 16>;
using simd_t = simd_prims<int8_t, 32>;

void pack_chars(simd_t::input_t const& targets, int j, simd_t::simd_base_t &out_chars_lower, simd_t::simd_base_t &out_orig_case, simd_t::simd_base_t &out_mask, size_t &out_bitmask)
{
    for(int i = 0, n = targets.size(); i < n; ++i)
    {
        if (!(out_bitmask & size_t(1) << i))
            continue;

        auto const& t = targets[i];
        if (j < t.length())
        {
            char c = t[j];
            simd_t::set_idx(i, c, out_orig_case);
            simd_t::set_idx(i, std::tolower(c), out_chars_lower);
        }
        else
        {
            out_bitmask &= ~(size_t(1) << i);
            simd_t::set_idx(i, 0, out_mask);
        }
    }
}

simd_t::scores_t sw_score_simd(std::string_view const&query, simd_t::input_t const& targets/*, int dbg_lane*/)
{
    constexpr simd_t::int_type_t k_extend_gap_penalty = 1;
    constexpr simd_t::int_type_t k_open_gap_penalty = 2;

    constexpr simd_t::int_type_t k_mismatch_penalty = -1;
    constexpr simd_t::int_type_t k_icase_match_bonus = 2;
    constexpr simd_t::int_type_t k_case_match_add_bonus = 1;

    using vec = simd_t::simd_base_t;
    int maxTargetLen = std::ranges::max(targets, {}, &std::string_view::length).length();
    int queryLen = query.length();
    const int k_max_penalty = -queryLen * k_open_gap_penalty;

    std::vector<vec> H_prev;
    std::vector<vec> H_cur;
    std::vector<vec> E;

    const vec max_penalty = simd_t::set1(k_max_penalty);
    const vec zero = simd_t::set1(0);
    const vec mismatch_penalty = simd_t::set1(k_mismatch_penalty);
    const vec icase_match_bonus = simd_t::set1(k_icase_match_bonus);
    const vec case_match_add_bonus = simd_t::set1(k_case_match_add_bonus + k_icase_match_bonus);
    const vec extend_gap_penalty_v = simd_t::set1(k_extend_gap_penalty);
    const vec open_extend_gap_penalty_v = simd_t::set1(k_open_gap_penalty + k_extend_gap_penalty);


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
        const vec query_chars_orig = simd_t::set1(qc);
        const vec query_chars_lower = simd_t::set1(std::tolower(qc));

        vec valid_targets_mask = simd_t::set1(simd_t::kMaskVal);
        size_t valid_targets_bitmask = size_t(-1);
        for(int j = 1; j <= maxTargetLen; ++j)
        {
            vec target_chars_lower;
            vec target_chars_orig;
            pack_chars(targets, j - 1, target_chars_lower, target_chars_orig, valid_targets_mask, valid_targets_bitmask);
            E[j] = simd_t::max(simd_t::sub(E[j], extend_gap_penalty_v), simd_t::sub(H_cur[j - 1], open_extend_gap_penalty_v));
            F = simd_t::max(simd_t::sub(F, extend_gap_penalty_v), simd_t::sub(H_prev[j], open_extend_gap_penalty_v));


            auto icase_char_match = simd_t::eq(query_chars_lower, target_chars_lower);
            auto icase_match_score = simd_t::blend(mismatch_penalty, icase_match_bonus, icase_char_match);

            auto case_char_match = simd_t::eq(query_chars_orig, target_chars_orig);
            auto case_match_score = simd_t::blend(mismatch_penalty, case_match_add_bonus, case_char_match);

            auto match_score = simd_t::max(icase_match_score, case_match_score);

            auto diag = simd_t::add(H_prev[j - 1], match_score);
            auto H = simd_t::max(simd_t::max(zero, diag), simd_t::max(E[j], F));
            H = simd_t::blend(zero, H, valid_targets_mask);
            H_cur[j] = H;

            best = simd_t::max(H, best);
        }
        std::swap(H_prev, H_cur);
    }

    return simd_t::unpack(best);
}

using scored_line = std::pair<std::string_view, int>;
template<>
struct std::formatter<scored_line, char>
{
    template<class ParseContext>
    constexpr ParseContext::iterator parse(ParseContext& ctx) { return ctx.begin(); }
 
    template<class FmtContext>
    FmtContext::iterator format(scored_line const& s, FmtContext& ctx) const
    {
        return std::format_to(ctx.out(), "Score: {}; {}", s.second, s.first);
    }
};

int main(int argc, char *argv[])
{

    int fd = open(argv[1], O_RDONLY);
    if (fd == -1)
    {
        std::print("Could not open {}", argv[1]);
        return -1;
    }

    struct stat st;
    if (fstat(fd, &st) == -1)
    {
        std::print("fstat for {} has failed", argv[1]);
        return -1;
    }

    const char *pContent = (const char*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    std::string_view content(pContent, st.st_size);

    auto lines = std::views::split(content, std::string_view{"\n"}) 
                | std::views::transform([](auto r) { return std::string_view(r.begin(), r.end()); }) 
                | std::ranges::to<std::vector>();

    std::ranges::sort(lines, std::greater{}, &std::string_view::length);

    char query[256];
    std::vector<scored_line> lines_with_scores_simd;
    std::vector<scored_line> lines_with_scores;
    lines_with_scores_simd.reserve(lines.size());
    lines_with_scores.reserve(lines.size());

    constexpr size_t kThreads = 16;

    std::counting_semaphore<kThreads> work_start(0);
    std::binary_semaphore main_sem(0);
    std::barrier fuzzy_barrier(kThreads, [&]{
            main_sem.release();
    });
    size_t q_size;
    size_t wSize = lines.size() / kThreads;
    bool workerCancel = false;

    std::vector<scored_line> lines_with_scores_simd_par[kThreads];
    for(auto &v : lines_with_scores_simd_par)
        v.reserve(lines.size() / (kThreads - 1));

    auto simd_work = [&](int wId){
        while(true)
        {
            work_start.acquire();
            if (workerCancel)
                break;
            std::string_view q(query, q_size);
            std::vector<scored_line> &results = lines_with_scores_simd_par[wId];
            for(int i = wId * wSize, n = wId == (kThreads - 1) ? (int)lines.size() : i + wSize; i < n; i += simd_t::Width)
            {
                int m = (i + simd_t::Width) < n ? i + simd_t::Width : n;
                auto *pFrom = &lines[i];
                auto *pTo = &lines[m];
                simd_t::input_t in;
                std::copy(pFrom, pTo, in.begin());
                auto scores = sw_score_simd(q, in/*, dbg_lane*/);
                for(int j = 0, tn = m - i; j < tn; ++j)
                {
                    if (scores[j])
                        results.emplace_back(lines[i + j], scores[j]);
                }
            }
            fuzzy_barrier.arrive_and_wait();
        }
    };

    std::vector<std::jthread> workers;
    for(int i = 0; size_t(i) < kThreads; ++i)
        workers.emplace_back(simd_work, i);

    std::print("Enter query: ");
    while(std::cin.getline(query, sizeof(query)))
    {
        std::println("Results: ");
        std::string_view q(query);
        q_size = q.size();
        ScopeTimer::duration_t simd_duration, normal_duration;
        {
            {
                auto run_simd_par = [&]{
                    for(auto &par : lines_with_scores_simd_par) par.clear();

                    ScopeTimer measure("SIMD SW");
                    work_start.release(kThreads);
                    main_sem.acquire();
                    for(auto &par : lines_with_scores_simd_par)
                        for(auto &l : par) lines_with_scores_simd.push_back(l);
                    std::ranges::sort(lines_with_scores_simd, std::less{}, &scored_line::second);
                    return measure.get_duration();
                };
                auto run_simd = [&]{
                    ScopeTimer measure("SIMD SW");
                    for(int i = 0, n = (int)lines.size(); i < n; i += simd_t::Width)
                    {
                        int m = (i + simd_t::Width) < n ? i + simd_t::Width : n;
                        auto *pFrom = &lines[i];
                        auto *pTo = &lines[m];
                        simd_t::input_t in;
                        std::copy(pFrom, pTo, in.begin());
                        auto scores = sw_score_simd(q, in/*, dbg_lane*/);
                        for(int j = 0, tn = m - i; j < tn; ++j)
                        {
                            if (scores[j])
                                lines_with_scores_simd.emplace_back(lines[i + j], scores[j]);
                        }
                    }
                    std::ranges::sort(lines_with_scores_simd, std::less{}, &scored_line::second);
                    return measure.get_duration();
                };
                for(int i = 0; i < 16; ++i)
                {
                    lines_with_scores_simd.clear();
                    auto t = run_simd_par();
                    if (!i || t < simd_duration)
                        simd_duration = t;
                }
            }
            {
                auto run_norm = [&]{
                    ScopeTimer measure("Simple SW");
                    for(const auto &sv : lines)
                    {
                        if (auto s = sw_score(q, sv); s > 0)
                            lines_with_scores.emplace_back(sv, s);
                    }
                    std::ranges::sort(lines_with_scores, std::less{}, &scored_line::second);
                    return measure.get_duration();
                };

                for(int i = 0; i < 16; ++i)
                {
                    lines_with_scores.clear();
                    auto t = run_norm();
                    if (!i || t < normal_duration)
                        normal_duration = t;
                }
            }

            for(const scored_line& l : lines_with_scores)
            {
                std::println("{}", l);
            }
            if (lines_with_scores_simd != lines_with_scores)
            {
                std::println("SIMD failed. Differences:");
                if (lines_with_scores_simd.size() != lines_with_scores.size())
                    std::println("SIMD size {} vs Normal size {}", lines_with_scores_simd.size(), lines_with_scores.size());
                else
                {
                    for(size_t i = 0, n = lines_with_scores.size(); i < n; ++i)
                    {
                        if (lines_with_scores_simd[i] != lines_with_scores[i])
                        {
                            std::println("Position {}", i);
                            std::println("SIMD: Line: {};", lines_with_scores_simd[i].first);
                            std::println("SIMD: Score: {};", lines_with_scores_simd[i].second);
                            std::println("Normal: Line: {};", lines_with_scores[i].first);
                            std::println("Normal: Score: {};", lines_with_scores[i].second);
                        }
                    }
                }
            }else
            {
                std::println("SIMD scoring Ok. SIMD duration {} vs Normal duration {}", simd_duration, normal_duration);
                if (simd_duration < normal_duration)
                    std::println("{}x speedup", double(normal_duration.count()) / double(simd_duration.count()));
                else
                    std::println("{}x slowdown", double(simd_duration.count()) / double(normal_duration.count()));
            }
            std::println("Done!");
        }
        std::print("Enter query: ");
    }
    
    return 0;
}
