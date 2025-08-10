#ifndef FUZZY_SW_HPP_
#define FUZZY_SW_HPP_
#include <string_view>
#include <memory>
#include <vector>

namespace fuzzy_sw
{
    struct Config
    {
        int extend_gap_penalty = -1;
        int open_gap_penalty = -2;
        int mismatch_penalty = -1;

        int delimiter_boundary_bonus = 1;
        int full_match_bonus = 3;
        int icase_match_bonus = 2;
    };

    //non-parallel, simple version, no backtracking
    int match(std::string_view const& query, std::string_view const& target, Config const& cfg = {});

    class SIMDParMatcher
    {
        struct Impl;
    public:
        SIMDParMatcher(Config const& cfg = {});
        ~SIMDParMatcher();

        std::vector<int> match(std::string_view const&query, std::vector<std::string_view> const& targets);
        std::vector<int> match_par(std::string_view const&query, std::vector<std::string_view> const& targets);
    private:
        Config m_Config;
        std::unique_ptr<Impl> m_Impl;
    };
};
#endif
