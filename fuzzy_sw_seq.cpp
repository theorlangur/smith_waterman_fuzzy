#include "fuzzy_sw.hpp"
#include <algorithm>

namespace fuzzy_sw
{
    int match(std::string_view const& query, std::string_view const& target, Config const& cfg)
    {
        auto match = [&](char c1, char c2, bool prev_delim, bool delim)
        {
            int delim_bonus = (prev_delim ^ delim) ? cfg.delimiter_boundary_bonus : 0;
            if (c1 == c2) return cfg.full_match_bonus + delim_bonus;
            if (std::tolower(c1) == std::tolower(c2)) return cfg.icase_match_bonus + delim_bonus;
            return cfg.mismatch_penalty;
        };

        auto is_delim = [](char c){
            return c == ' ' || c == '_' || c == '.' || c == ':' || c == '-' || c == '=' || c == ',';
        };

        int n = query.length();
        int m = target.length();
        const int max_penalty = n * cfg.open_gap_penalty;
        const int open_and_extend_penalty = cfg.open_gap_penalty + cfg.extend_gap_penalty;
        std::vector<int> H_prev, H_cur;
        std::vector<int> E;
        H_prev.resize(m + 1);
        std::fill(H_prev.begin(), H_prev.end(), 0);
        H_cur.resize(m + 1);
        E.resize(m + 1);
        std::fill(E.begin(), E.end(), max_penalty);
        int best = 0;
        for(int i = 1; i <= n; ++i)
        {
            H_cur[0] = 0;
            E[0] = max_penalty;
            int F = 0;
            bool prev_delim = false;
            for(int j = 1; j <= m; ++j)
            {
                bool delim = is_delim(target[j - 1]);
                E[j] = std::max(E[j] + cfg.extend_gap_penalty, H_cur[j - 1] + open_and_extend_penalty);
                F = std::max(F + cfg.extend_gap_penalty, H_prev[j] + open_and_extend_penalty);
                auto t_prev_delim = false;
                auto t_delim = false;
                if (i == 1 || i == n)
                {
                    t_prev_delim = prev_delim;
                    t_delim = delim;
                }
                int diag = H_prev[j - 1] + match(query[i - 1], target[j - 1], t_prev_delim, t_delim);
                int H = std::max({0, diag, E[j], F});
                H_cur[j] = H;
                if (H > best) best = H;
                prev_delim = delim;
            }
            std::swap(H_prev, H_cur);
        }

        return best;
    }

    //TODO: refactor, de-duplicate
    int match(std::string_view const& query, CharSource *target, Config const& cfg)
    {
        auto match = [&](char c1, char c2, bool prev_delim, bool delim)
        {
            int delim_bonus = (prev_delim ^ delim) ? cfg.delimiter_boundary_bonus : 0;
            if (c1 == c2) return cfg.full_match_bonus + delim_bonus;
            if (std::tolower(c1) == std::tolower(c2)) return cfg.icase_match_bonus + delim_bonus;
            return cfg.mismatch_penalty;
        };

        auto is_delim = [](char c){
            return c == ' ' || c == '_' || c == '.' || c == ':' || c == '-' || c == '=' || c == ',';
        };

        int n = query.length();
        int m = target->length();
        const int max_penalty = n * cfg.open_gap_penalty;
        const int open_and_extend_penalty = cfg.open_gap_penalty + cfg.extend_gap_penalty;
        std::vector<int> H_prev, H_cur;
        std::vector<int> E;
        H_prev.resize(m + 1);
        std::fill(H_prev.begin(), H_prev.end(), 0);
        H_cur.resize(m + 1);
        E.resize(m + 1);
        std::fill(E.begin(), E.end(), max_penalty);
        int best = 0;
        for(int i = 1; i <= n; ++i)
        {
            H_cur[0] = 0;
            E[0] = max_penalty;
            int F = 0;
            bool prev_delim = false;
            for(int j = 1; j <= m; ++j)
            {
                char tc;
                target->read(&tc, j - 1, 1);
                bool delim = is_delim(tc);
                E[j] = std::max(E[j] + cfg.extend_gap_penalty, H_cur[j - 1] + open_and_extend_penalty);
                F = std::max(F + cfg.extend_gap_penalty, H_prev[j] + open_and_extend_penalty);
                auto t_prev_delim = false;
                auto t_delim = false;
                if ((i == 1) && prev_delim && !delim)
                {
                    t_prev_delim = prev_delim;
                    t_delim = delim;
                }
                int diag = H_prev[j - 1] + match(query[i - 1], tc, t_prev_delim, t_delim);
                int H = std::max({0, diag, E[j], F});
                H_cur[j] = H;
                if (H > best) best = H;
                prev_delim = delim;
            }
            std::swap(H_prev, H_cur);
        }

        return best;
    }
}
