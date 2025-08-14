#include <immintrin.h>
#include <algorithm>
#include <print>
#include <iostream>
#include <string_view>
#include <vector>
#include <ranges>
#include <chrono>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "fuzzy_sw.hpp"

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

template<>
struct std::formatter<fuzzy_sw::SIMDParMatcher::ResultItem, char>
{
    template<class ParseContext>
    constexpr ParseContext::iterator parse(ParseContext& ctx) { return ctx.begin(); }
 
    template<class FmtContext>
    FmtContext::iterator format(fuzzy_sw::SIMDParMatcher::ResultItem const& s, FmtContext& ctx) const
    {
        return std::format_to(ctx.out(), "Score: {}; {}", s.score, s.target);
    }
};

struct StringViewSrc: fuzzy_sw::CharSource
{
    std::string_view sv;

    StringViewSrc() = default;
    StringViewSrc(std::string_view _sv):sv(_sv){}

    virtual size_t length() const override { return sv.length(); }
    virtual size_t read(char *pDest, size_t off, size_t n) const override
    {
        if (off >= sv.length()) return 0;

        size_t  r = n;
        if ((off + r) > sv.length())
            r = sv.length() - off;

        std::copy(sv.begin() + off, sv.begin() + off + r, pDest);
        return r;
    }
};

template<>
struct std::formatter<fuzzy_sw::CharSource*, char>
{
    template<class ParseContext>
    constexpr ParseContext::iterator parse(ParseContext& ctx) { return ctx.begin(); }
 
    template<class FmtContext>
    FmtContext::iterator format(fuzzy_sw::CharSource* s, FmtContext& ctx) const
    {
        return std::format_to(ctx.out(), "{}", ((StringViewSrc*)s)->sv);
    }
};

template<>
struct std::formatter<fuzzy_sw::SIMDParMatcher::CharSourceResultItem, char>
{
    template<class ParseContext>
    constexpr ParseContext::iterator parse(ParseContext& ctx) { return ctx.begin(); }
 
    template<class FmtContext>
    FmtContext::iterator format(fuzzy_sw::SIMDParMatcher::CharSourceResultItem const& s, FmtContext& ctx) const
    {
        return std::format_to(ctx.out(), "Score: {}; {}", s.score, s.target);
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

    auto lines_orig = std::views::split(content, std::string_view{"\n"}) 
                | std::views::transform([](auto r) { return std::string_view(r.begin(), r.end()); }) 
                | std::ranges::to<std::vector>();

    auto lines_orig_mut = lines_orig 
                | std::views::transform([](auto r) { return StringViewSrc{r}; }) 
                | std::ranges::to<std::vector>();

    auto lines_orig_mut_ptr = lines_orig_mut 
                | std::views::transform([](auto &r)->fuzzy_sw::CharSource* { return &r; }) 
                | std::ranges::to<std::vector>();

    char query[256];

    constexpr size_t kThreads = 16;
    constexpr size_t kRepeats = 32 * 4;

    auto sort_lines = [](auto/*fuzzy_sw::SIMDParMatcher::Result*/ &l)
    {
        std::ranges::sort(l, 
                [](auto const& l1, auto const& l2){
                if (l1.score != l2.score) return l1.score < l2.score;
                return l1.target < l2.target;
                }
                );
    };

    fuzzy_sw::SIMDParMatcher simdMatcher;
    //fuzzy_sw::SIMDParMatcher::Result lines_with_scores_simd;
    //fuzzy_sw::SIMDParMatcher::Result lines_with_scores_normal;
    fuzzy_sw::SIMDParMatcher::CharSourceResult lines_with_scores_simd;
    fuzzy_sw::SIMDParMatcher::CharSourceResult lines_with_scores_normal;
    simdMatcher.SetupThreads(kThreads);

    std::print("Enter query: ");
    while(std::cin.getline(query, sizeof(query)))
    {
        std::string_view q(query);
        if (q == "!stop")
            break;
        std::println("Results: ");
        ScopeTimer::duration_t simd_duration, normal_duration;
        {
            {
                for(int i = 0; i < kRepeats; ++i)
                {
                    lines_with_scores_simd.clear();
                    {
                        auto lines = lines_orig_mut_ptr/*lines_orig*/;
                        //auto lines = lines_orig;
                        ScopeTimer measure("SIMD SW");
                        lines_with_scores_simd = simdMatcher.match_par(q, std::move(lines), {});
                        auto t = measure.get_duration();
                        if (!i || t < simd_duration)
                            simd_duration = t;
                    }
                    sort_lines(lines_with_scores_simd);
                }
            }
            {
                auto run_norm = [&]{
                    ScopeTimer measure("Simple SW");
                    for(const auto &sv : /*lines_orig*/lines_orig_mut_ptr)
                    //for(const auto &sv : lines_orig)
                    {
                        if (auto s = fuzzy_sw::match(q, sv); s > 0)
                            lines_with_scores_normal.emplace_back(sv, s);
                    }
                    return measure.get_duration();
                };

                for(int i = 0; i < kRepeats; ++i)
                {
                    lines_with_scores_normal.clear();
                    auto t = run_norm();
                    sort_lines(lines_with_scores_normal);
                    if (!i || t < normal_duration)
                        normal_duration = t;
                }
            }

            for(const auto& l : lines_with_scores_normal)
            {
                std::println("{}", l);
            }
            if (lines_with_scores_simd != lines_with_scores_normal)
            {
                std::println("SIMD failed. Differences:");
                if (lines_with_scores_simd.size() != lines_with_scores_normal.size())
                    std::println("SIMD size {} vs Normal size {}", lines_with_scores_simd.size(), lines_with_scores_normal.size());
                else
                {
                    for(size_t i = 0, n = lines_with_scores_normal.size(); i < n; ++i)
                    {
                        if (lines_with_scores_simd[i] != lines_with_scores_normal[i])
                        {
                            std::println("Position {}", i);
                            std::println("SIMD: Line: {};", lines_with_scores_simd[i].target);
                            std::println("SIMD: Score: {};", lines_with_scores_simd[i].score);
                            std::println("Normal: Line: {};", lines_with_scores_normal[i].target);
                            std::println("Normal: Score: {};", lines_with_scores_normal[i].score);
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
    std::println("Finishing...");
    
    return 0;
}
