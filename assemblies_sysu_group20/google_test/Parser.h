// Parser.h
#ifndef PARSER_H
#define PARSER_H
// -------------xwr添加，对应brain_parser.py开头的常量和创建词汇表用到的类定义和全局函数------------------
#pragma once

#include <random>
#include <string>
#include <iostream>
#include <unordered_map> // 引入unordered_map容器
#include <map>
#include <unordered_set>
#include <memory> // 引入智能指针
#include <vector>
#include <typeinfo>
#include <sstream>
#include <queue>
#include <boost/math/distributions/binomial.hpp> // Boost.Math 库的头文件
#include <utility>
#include <set>
#include <cmath>
#include <numeric>
#include <boost/random.hpp>
#include <algorithm>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#define AREA_RULE false
#define FIBER_RULE true

// 定义 Rule 类
class Rule {
public:
    std::string action;
    std::string area1; // 适用于 FiberRule
    std::string area2; // 适用于 FiberRule
    std::string area;  // 适用于 AreaRule
    int index;
    bool flag; // false 表示 AreaRule，true 表示 FiberRule

    Rule(std::string a, std::string ar, int i)
        : action(a), area(ar), index(i), flag(AREA_RULE) {}

    Rule(std::string a, std::string a1, std::string a2, int i)
        : action(a), area1(a1), area2(a2), index(i), flag(FIBER_RULE) {}

    Rule(const Rule& other) = default;
    Rule& operator=(const Rule& other) = default;
};

// 定义 GenericRuleSet 结构体
class GenericRuleSet {
public:
    int index;
    std::vector<Rule> pre_rules;
    std::vector<Rule> post_rules;

    GenericRuleSet() = default;
    GenericRuleSet(const GenericRuleSet& other) = default;
    GenericRuleSet& operator=(const GenericRuleSet& other) = default;
};


enum class ReadoutMethod {
    FIXED_MAP_READOUT = 1,
    FIBER_READOUT = 2,
    NATURAL_READOUT = 3
};


// 定义一些常量
extern const std::string DISINHIBIT;
extern const std::string INHIBIT;
extern const std::string LEX;
extern const std::string DET;
extern const std::string SUBJ;
extern const std::string OBJ;
extern const std::string VERB;
extern const std::string PREP;
extern const std::string PREP_P;
extern const std::string ADJ;
extern const std::string ADVERB;
extern const std::vector<std::string> AREAS;
extern const std::vector<std::string> EXPLICIT_AREAS;
extern const std::vector<std::string> RECURRENT_AREAS;
extern const int LEX_SIZE;
// 声明全局变量
// 根据词汇表的数量而变化
extern std::unordered_map<std::string, GenericRuleSet> LEXEME_DICT;

// 函数声明
GenericRuleSet generic_noun(int index);
GenericRuleSet generic_trans_verb(int index);
GenericRuleSet generic_intrans_verb(int index);
GenericRuleSet generic_copula(int index);
GenericRuleSet generic_adverb(int index);
GenericRuleSet generic_determinant(int index);
GenericRuleSet generic_adjective(int index);
GenericRuleSet generic_preposition(int index);

// 根据词性调用不同的初始词汇表的函数
GenericRuleSet generic_noun(int index)
{
    return {
        index,
        {{DISINHIBIT, LEX, SUBJ, 0},
         {DISINHIBIT, LEX, OBJ, 0},
         {DISINHIBIT, LEX, PREP_P, 0},
         {DISINHIBIT, DET, SUBJ, 0},
         {DISINHIBIT, DET, OBJ, 0},
         {DISINHIBIT, DET, PREP_P, 0},
         {DISINHIBIT, ADJ, SUBJ, 0},
         {DISINHIBIT, ADJ, OBJ, 0},
         {DISINHIBIT, ADJ, PREP_P, 0},
         {DISINHIBIT, VERB, OBJ, 0},
         {DISINHIBIT, PREP_P, PREP, 0},
         {DISINHIBIT, PREP_P, SUBJ, 0},
         {DISINHIBIT, PREP_P, OBJ, 0}},
        {{INHIBIT, DET, 0},
         {INHIBIT, ADJ, 0},
         {INHIBIT, PREP_P, 0},
         {INHIBIT, PREP, 0},
         {INHIBIT, LEX, SUBJ, 0},
         {INHIBIT, LEX, OBJ, 0},
         {INHIBIT, LEX, PREP_P, 0},
         {INHIBIT, ADJ, SUBJ, 0},
         {INHIBIT, ADJ, OBJ, 0},
         {INHIBIT, ADJ, PREP_P, 0},
         {INHIBIT, DET, SUBJ, 0},
         {INHIBIT, DET, OBJ, 0},
         {INHIBIT, DET, PREP_P, 0},
         {INHIBIT, VERB, OBJ, 0},
         {INHIBIT, PREP_P, PREP, 0},
         {INHIBIT, PREP_P, VERB, 0},
         {DISINHIBIT, LEX, SUBJ, 1},
         {DISINHIBIT, LEX, OBJ, 1},
         {DISINHIBIT, DET, SUBJ, 1},
         {DISINHIBIT, DET, OBJ, 1},
         {DISINHIBIT, ADJ, SUBJ, 1},
         {DISINHIBIT, ADJ, OBJ, 1},
         {INHIBIT, PREP_P, SUBJ, 0},
         {INHIBIT, PREP_P, OBJ, 0},
         {INHIBIT, VERB, ADJ, 0}} };
}

GenericRuleSet generic_trans_verb(int index)
{
    return {
        index,
        {{DISINHIBIT, LEX, VERB, 0},
         {DISINHIBIT, VERB, SUBJ, 0},
         {DISINHIBIT, VERB, ADVERB, 0},
         {DISINHIBIT, ADVERB, 1}},
        {{INHIBIT, LEX, VERB, 0},
         {DISINHIBIT, OBJ, 0},
         {INHIBIT, SUBJ, 0},
         {INHIBIT, ADVERB, 0},
         {DISINHIBIT, PREP_P, VERB, 0}} };
}

GenericRuleSet generic_intrans_verb(int index)
{
    return {
        index,
        {{DISINHIBIT, LEX, VERB, 0},
         {DISINHIBIT, VERB, SUBJ, 0},
         {DISINHIBIT, VERB, ADVERB, 0},
         {DISINHIBIT, ADVERB, 1}},
        {{INHIBIT, LEX, VERB, 0},
         {INHIBIT, SUBJ, 0},
         {INHIBIT, ADVERB, 0},
         {DISINHIBIT, PREP_P, VERB, 0}} };
}

GenericRuleSet generic_copula(int index)
{
    return {
        index,
        {{DISINHIBIT, LEX, VERB, 0},
         {DISINHIBIT, VERB, SUBJ, 0}},
        {{INHIBIT, LEX, VERB, 0},
         {DISINHIBIT, OBJ, 0},
         {INHIBIT, SUBJ, 0},
         {DISINHIBIT, ADJ, VERB, 0}} };
}

GenericRuleSet generic_adverb(int index)
{
    return {
        index,
        {{DISINHIBIT, ADVERB, 0},
         {DISINHIBIT, LEX, ADVERB, 0}},
        {{INHIBIT, LEX, ADVERB, 0},
         {INHIBIT, ADVERB, 1}} };
}

GenericRuleSet generic_determinant(int index)
{
    return {
        index,
        {{DISINHIBIT, DET, 0},
         {DISINHIBIT, LEX, DET, 0}},
        {{INHIBIT, LEX, DET, 0},
         {INHIBIT, VERB, ADJ, 0}} };
}

GenericRuleSet generic_adjective(int index)
{
    return {
        index,
        {{DISINHIBIT, ADJ, 0},
         {DISINHIBIT, LEX, ADJ, 0}},
        {{INHIBIT, LEX, ADJ, 0},
         {INHIBIT, VERB, ADJ, 0}} };
}

GenericRuleSet generic_preposition(int index)
{
    return {
        index,
        {{DISINHIBIT, PREP, 0},
         {DISINHIBIT, LEX, PREP, 0}},
        {{INHIBIT, LEX, PREP, 0},
         {DISINHIBIT, PREP_P, 0},
         {INHIBIT, LEX, SUBJ, 1},
         {INHIBIT, LEX, OBJ, 1},
         {INHIBIT, DET, SUBJ, 1},
         {INHIBIT, DET, OBJ, 1},
         {INHIBIT, ADJ, SUBJ, 1},
         {INHIBIT, ADJ, OBJ, 1}} };
}

// BrainAreas
const std::string LEX = "LEX";
const std::string DET = "DET";
const std::string SUBJ = "SUBJ";
const std::string OBJ = "OBJ";
const std::string VERB = "VERB";
const std::string PREP = "PREP";
const std::string PREP_P = "PREP_P";
const std::string ADJ = "ADJ";
const std::string ADVERB = "ADVERB";
const std::string NOM = "NOM";
const std::string ACC = "ACC";
const std::string DAT = "DAT";

const int LEX_SIZE = 20;

const std::string DISINHIBIT = "DISINHIBIT";
const std::string INHIBIT = "INHIBIT";
const std::string ACTIVATE_ONLY = "ACTIVATE_ONLY";
const std::string CLEAR_DET = "CLEAR_DET";

const std::vector<std::string> AREAS = { LEX, DET, SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P };
const std::vector<std::string> EXPLICIT_AREAS = { LEX };
const std::vector<std::string> RECURRENT_AREAS = { SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P };

using namespace std;
using defaultdict_set = std::unordered_map<std::string, std::unordered_set<std::string>>;
using defaultdict_list = std::unordered_map<std::string, std::vector<std::string>>;

// 构建词汇表
std::unordered_map<std::string, GenericRuleSet> LEXEME_DICT = {
    {"the", generic_determinant(0)},
    {"a", generic_determinant(1)},
    {"dogs", generic_noun(2)},
    {"cats", generic_noun(3)},
    {"mice", generic_noun(4)},
    {"people", generic_noun(5)},
    {"chase", generic_trans_verb(6)},
    {"love", generic_trans_verb(7)},
    {"bite", generic_trans_verb(8)},
    {"of", generic_preposition(9)},
    {"big", generic_adjective(10)},
    {"bad", generic_adjective(11)},
    {"run", generic_intrans_verb(12)},
    {"fly", generic_intrans_verb(13)},
    {"quickly", generic_adverb(14)},
    {"in", generic_preposition(15)},
    {"are", generic_copula(16)},
    {"man", generic_noun(17)},
    {"woman", generic_noun(18)},
    {"saw", generic_trans_verb(19)},
};

// 定义 ENGLISH_READOUT_RULES
std::unordered_map<std::string, std::vector<std::string>> ENGLISH_READOUT_RULES = {
    {VERB, {LEX, SUBJ, OBJ, PREP_P, ADVERB, ADJ}},
    {SUBJ, {LEX, DET, ADJ, PREP_P}},
    {OBJ, {LEX, DET, ADJ, PREP_P}},
    {PREP_P, {LEX, PREP, ADJ, DET}},
    {PREP, {LEX}},
    {ADJ, {LEX}},
    {DET, {LEX}},
    {ADVERB, {LEX}},
    {LEX, {}} };

std::vector<double> truncnorm_rvs(double a, double b, double scale, std::size_t size)
{

    std::vector<double> results;
    results.reserve(size);

    // 随机数生成器和分布
    boost::mt19937 rng;
    boost::random::normal_distribution<> normal_dist(0, scale);

    for (std::size_t i = 0; i < size; ++i)
    {
        double num;
        do
        {
            num = normal_dist(rng);
        } while (num < a || num > b);
        results.push_back(num);
    }

    return results;
}

std::vector<double> generate_truncated_normal_samples(double mean, double stddev, double lower_bound, double upper_bound, int sample_size)
{
    // Mersenne Twister 随机数生成器
    boost::mt19937 rng;

    // 均匀分布生成器 [0, 1]
    std::uniform_real_distribution<> uniform_dist(0.0, 1.0);

    // 标准正态分布
    boost::math::normal standard_normal;

    // 计算截断后的累积分布函数 (CDF) 值
    double alpha = cdf(standard_normal, (lower_bound - mean) / stddev);
    double beta = cdf(standard_normal, (upper_bound - mean) / stddev);

    // 存储样本的向量
    std::vector<double> samples;
    samples.reserve(sample_size);

    // 生成样本
    for (int i = 0; i < sample_size; ++i)
    {
        // 从均匀分布中生成随机数
        double u = uniform_dist(rng);

        // 将均匀分布随机数映射到截断正态分布上
        double p = alpha + u * (beta - alpha);

        // 使用 try-catch 块捕获可能的溢出错误
        try
        {
            double sample = mean + stddev * quantile(standard_normal, p);
            samples.push_back(sample);
        }
        catch (const std::overflow_error& e)
        {
            //std::cerr << "Overflow error: " << e.what() << std::endl;
            // 可以根据需要处理溢出错误，例如使用一个默认值或者重新生成样本
            // 这里简单地忽略错误
        }
    }

    return samples;
}

std::vector<int> nlargest_indices(const std::vector<double>& all_potential_winner_inputs, int k)
{
    // 定义一个优先队列（最大堆）
    std::priority_queue<std::pair<double, int>> max_heap;

    // 将所有元素及其索引压入堆中
    for (int i = 0; i < all_potential_winner_inputs.size(); ++i)
    {
        max_heap.emplace(all_potential_winner_inputs[i], i);
    }

    // 提取前 k 个最大的元素的索引
    std::vector<int> new_winner_indices;
    for (int i = 0; i < k && !max_heap.empty(); ++i)
    {
        new_winner_indices.push_back(max_heap.top().second); // 获取索引
        max_heap.pop();                                      // 移除当前最大元素
    }

    return new_winner_indices;
}

// 假设已定义Area类
class Area
{
public:
    // 类的属性
    std::string name;                                         // 区域的名称
    int n;                                                    // 神经元数量
    int k;                                                    // 激活时发射的神经元数量
    double beta;                                              // 默认激活 beta
    std::unordered_map<std::string, double> beta_by_stimulus; // 刺激名称与对应 beta 的映射
    std::unordered_map<std::string, double> beta_by_area;     // 区域名称与对应 beta 的映射
    int w;                                                    // 曾经在该区域激活的神经元数量
    std::vector<int> saved_w;                                 // 每轮的支持大小列表
    std::vector<int> winners;                                 // 由先前动作设定的获胜者列表
    std::vector<std::vector<int>> saved_winners;              // 所有获胜者的列表，每轮一个列表
    int num_first_winners;                                    // TODO: Clarify
    bool fixed_assembly;                                      // 该区域的集合（获胜者）是否被冻结
    // bool my_explicit;                                 // 是否完全模拟该区域（而不是仅执行稀疏模拟）
    int _new_w;                    // 自上次调用 .project() 以来的 `w` 值
    std::vector<int> _new_winners; // 自上次调用 .project() 以来的 `winners` 值，仅在 .project() 方法内使用

    int num_ever_fired;
    std::vector<bool> ever_fired;
    bool my_explicit;

    // 构造函数
    Area(const std::string& name, int n, int k, double beta = 0.05, int w = 0, bool my_explicit = false)
        : name(name), n(n), k(k), beta(beta), w(w), num_first_winners(-1), fixed_assembly(false), my_explicit(my_explicit) {}

    // 默认构造函数
    Area() = default;

    void _update_winners()
    {
        winners = _new_winners;
        if (!my_explicit)
        {
            w = _new_w;
        }
    }

    void fix_assembly()
    {
        fixed_assembly = true;
    }

    void unfix_assembly()
    {
        fixed_assembly = false;
    }
};

class Brain
{
public:
    // 类的属性
    std::unordered_map<std::string, Area> area_by_name;                                                             // 大脑区域名称到对应Area实例的映射
    std::unordered_map<std::string, int> stimulus_size_by_name;                                                     // 刺激名称到其神经元数量的映射
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<double>>> connectomes_by_stimulus;  // 刺激名称到区域激活向量的映射
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::vector<double>>>> connectomes; // 源区域名称到目标区域名称的连接映射
    double p;                                                                                                       // 神经元连接概率
    bool save_size;                                                                                                 // 是否保存大小标志
    bool save_winners;                                                                                              // 是否保存获胜者标志
    bool disable_plasticity;                                                                                        // 禁用可塑性调试标志
    std::mt19937 rng;                                                                                               // 随机数生成器
    bool use_normal_ppf;                                                                                            // 调试用途，是否使用标准正态分布

    // 构造函数
    Brain(double p, bool save_size = true, bool save_winners = false, unsigned int seed = 0)
        : p(p), save_size(save_size), save_winners(save_winners), disable_plasticity(false), rng(seed), use_normal_ppf(false) {}

    // 创建了一个区域，赋值给了area_by_name["LEX"]
    // 创建了一个字典new_connectomes，给new_connectomes["LEX"]进行了赋值，赋值了一个随机生成的二维矩阵，矩阵值都是0或1
    // 把new_connectomes赋值给了connectomes["LEX"]
    void add_explicit_area(std::string area_name, int n, int k, double beta, double custom_inner_p = 0, double custom_out_p = 0, double custom_in_p = 0)
    {
        area_by_name[area_name] = Area(area_name, n, k, beta, n, true);
        
        area_by_name[area_name].ever_fired = std::vector<bool>(n, false);
     

        area_by_name[area_name].num_ever_fired = 0;

        double inner_p = p;
        double in_p = p;
        double out_p = p;

        // 区域名称映射到二维数组的字典，数组中值为0或1
        std::unordered_map<std::string, std::vector<std::vector<double>>> new_connectomes;

        for (const auto& other_area_pair : area_by_name)
        {
            const std::string& other_area_name = other_area_pair.first;

            if (other_area_name == area_name)
            {

                // 这里都是在执行new_connectomes[other_area_name] = self._rng.binomial(1, inner_p, size=(n, n)).astype(np.float32)
                std::bernoulli_distribution distribution(inner_p);
                std::vector<std::vector<double>> matrix(n, std::vector<double>(n, 0.0f));
                for (int i = 0; i < n; ++i)
                {
                    for (int j = 0; j < n; ++j)
                    {
                        matrix[i][j] = distribution(rng) ? 1.0f : 0.0f;
                    }
                }

                new_connectomes[other_area_name] = matrix; 
            }

            area_by_name[other_area_name].beta_by_area[area_name] = area_by_name[other_area_name].beta;
            area_by_name[area_name].beta_by_area[other_area_name] = beta;
        }

        connectomes[area_name] = new_connectomes;

    
    }

    void project(int areas_by_stim, defaultdict_set dst_areas_by_src_area, int verbose = 0)
    {
        defaultdict_list stim_in;
        defaultdict_list area_in;

        for (const auto& pair : dst_areas_by_src_area)
        {
            std::string from_area_name = pair.first;
            for (const auto& to_area_name : pair.second)
            {
                area_in[to_area_name].push_back(from_area_name);
       
            }
        }

        // 获取area_in的所有键
        std::vector<std::string> to_update_area_names;
        for (const auto& pair : area_in)
        {
            to_update_area_names.push_back(pair.first);
        }
        // printStringVector(to_update_area_names);

        for (const auto& area_name : to_update_area_names)
        {
            Area& area = area_by_name[area_name];
            int num_first_winners = project_into(area, stim_in[area_name], area_in[area_name], verbose);
            area.num_first_winners = num_first_winners;

            if (save_winners)
            {
                area.saved_winners.push_back(area._new_winners);
            }
        }

        for (const std::string& area_name : to_update_area_names)
        {
            Area& area = area_by_name[area_name];
            area._update_winners();
            if (save_size)
            {
                area.saved_w.push_back(area.w);
            }
        }
    }

    int project_into(Area& target_area, std::vector<std::string> from_stimuli, std::vector<std::string> from_areas, int verbose = 0)
    {
        int num_first_winners_processed = 0;
        std::vector<std::vector<int>> inputs_by_first_winner_index;

        if (target_area.fixed_assembly)
        {
            target_area._new_winners = target_area.winners;
            target_area._new_w = target_area.w;
            num_first_winners_processed = 0;
        }
        else
        {
            std::string target_area_name = target_area.name;

            std::vector<double> prev_winner_inputs(target_area.w, 0.0);
            for (const auto& from_area_name : from_areas) // for from_area_name in from_areas:
            {
                for (const auto& w : area_by_name[from_area_name].winners) // for w in self.area_by_name[from_area_name].winners:
                {
                    // 下面是实现prev_winner_inputs += connectome[w]
                    for (size_t i = 0; i < prev_winner_inputs.size(); ++i)
                    {
                        prev_winner_inputs[i] += connectomes[from_area_name][target_area_name][w][i];
                    }
                }
            }

            std::vector<double> all_potential_winner_inputs;
            int total_k;
            int num_inputs = 0;
            std::vector<int> input_size_by_from_area_index;
            if (!target_area.my_explicit)
            {
                double normal_approx_mean = 0.0;
                double normal_approx_var = 0.0;

                // 使用范围基于 for 循环遍历
                for (const std::string& from_area_name : from_areas) // for from_area_name in from_areas:
                {
                    size_t effective_k = area_by_name[from_area_name].winners.size();

                    input_size_by_from_area_index.push_back(effective_k);
                    num_inputs += 1;
                }

                total_k = std::accumulate(input_size_by_from_area_index.begin(), input_size_by_from_area_index.end(), 0);

                int effective_n = target_area.n - target_area.w;

                if (effective_n <= target_area.k)
                {
                    std::cout << "Remaining size of area " << target_area_name << " too small to sample k new winners." << std::endl;
                }

                double quantile = (double(effective_n) - target_area.k) / effective_n;
  
                // 使用 Boost.Math 计算二项分布的分位点
                boost::math::binomial_distribution<> binom_dist(total_k, p);
                double alpha = boost::math::quantile(binom_dist, quantile);
                double mu = total_k * p;
                double std = std::sqrt(total_k * p * (1.0 - p));
                double a = (alpha - mu) / std;
                double b = (total_k - mu) / std;

                std::vector<double> random_values = generate_truncated_normal_samples(0, std, alpha, total_k, target_area.k);

                // 将每个元素四舍五入为整数
                std::transform(random_values.begin(), random_values.end(), random_values.begin(),
                    [](double x)
                    { return std::round(x); });

                std::vector<double> potential_new_winner_inputs = random_values;

                // std::vector<double> all_potential_winner_inputs;
                all_potential_winner_inputs.reserve(prev_winner_inputs.size() + potential_new_winner_inputs.size()); // 预留足够的空间，避免多次重新分配
                all_potential_winner_inputs.insert(all_potential_winner_inputs.end(), prev_winner_inputs.begin(), prev_winner_inputs.end());
                all_potential_winner_inputs.insert(all_potential_winner_inputs.end(), potential_new_winner_inputs.begin(), potential_new_winner_inputs.end());
            }
            else
            {
                all_potential_winner_inputs = prev_winner_inputs;
            }

            std::vector<int> new_winner_indices = nlargest_indices(all_potential_winner_inputs, target_area.k);

            if (target_area.my_explicit)
            {
                for (int winner : new_winner_indices)
                {
                    if (!target_area.ever_fired[winner])
                    {
                        target_area.ever_fired[winner] = true;
                        target_area.num_ever_fired += 1;
                    }
                }
            }

            num_first_winners_processed = 0;

            std::vector<double> first_winner_inputs;
            if (!target_area.my_explicit)
            {

                for (int i = 0; i < target_area.k; ++i)
                {
                    if (new_winner_indices[i] >= target_area.w)
                    {
                        first_winner_inputs.push_back(all_potential_winner_inputs[new_winner_indices[i]]);
                        new_winner_indices[i] = target_area.w + num_first_winners_processed;
                        num_first_winners_processed += 1;
                    }
                }
            }

            target_area._new_winners = new_winner_indices;
            target_area._new_w = target_area.w + num_first_winners_processed;

            inputs_by_first_winner_index.resize(num_first_winners_processed);
            std::vector<int> num_connections_by_input_index(num_inputs, 0);
            for (int i = 0; i < num_first_winners_processed; i++)
            {
                // 下面都是input_indices = rng.choice(range(total_k), int(first_winner_inputs[i]), replace=False)
                std::vector<int> indices(total_k);
                std::iota(indices.begin(), indices.end(), 0); // 用 0, 1, ..., total_k-1 填充 indices
                int num_to_select = static_cast<int>(first_winner_inputs[i]);
                std::vector<int> input_indices;
                std::sample(indices.begin(), indices.end(), std::back_inserter(input_indices), num_to_select, rng);

                int total_so_far = 0;
                std::vector<int> num_connections_by_input_index(num_inputs, 0);

                for (int j = 0; j < num_inputs; j++)
                {
                    for (int w : input_indices) // 计算满足条件的 w 的数量
                    {
                        if (total_so_far + input_size_by_from_area_index[j] > w && w >= total_so_far)
                        {
                            ++num_connections_by_input_index[j];
                        }
                    }
                    total_so_far += input_size_by_from_area_index[j];
                }
                inputs_by_first_winner_index[i] = num_connections_by_input_index;
            }
        }

        std::string target_area_name = target_area.name;
        int num_inputs_processed = 0;

        for (const auto& from_area_name : from_areas)
        {
            int from_area_w = area_by_name[from_area_name].w;
            std::vector<int> from_area_winners = area_by_name[from_area_name].winners;
            std::set<int> from_area_winners_set(from_area_winners.begin(), from_area_winners.end());
            // 改维度

            // 下面这里是用引用
            auto& from_area_connectomes = connectomes[from_area_name];
            auto& the_connectome = from_area_connectomes[target_area_name];

            int change = num_first_winners_processed;

            size_t num_rows = the_connectome.size(); // 获取行数
            // size_t num_cols = num_rows > 0 ? the_connectome[0].size() : 0; // 获取列数（假设每一行都有相同的列数）
            size_t num_cols = the_connectome[0].size(); // 获取列数（假设每一行都有相同的列数）
            if (num_cols == 1)
            {
                change--;
            }

            if (num_first_winners_processed != 0)
            {
                for (auto& row : the_connectome)
                {
                    row.resize(row.size() + change, 0.0); // 在每行的末尾增加num_first_winners_processed个元素
                }
            }

            for (int i = 0; i < num_first_winners_processed; ++i)
            {
                int total_in = inputs_by_first_winner_index[i][num_inputs_processed]; // 获取每个第一个获胜者的输入总量
                std::vector<int> sample_indices;
                std::sample(from_area_winners.begin(), from_area_winners.end(), std::back_inserter(sample_indices), total_in, rng); // 从源区域的获胜者中选择输入索引

                size_t num_rows = the_connectome.size();                       // 获取行数
                size_t num_cols = num_rows > 0 ? the_connectome[0].size() : 0; // 获取列数（假设每一行都有相同的列数）
   
                for (int j : sample_indices) // 更新连接矩阵
                {
                    the_connectome[j][target_area.w + i] = 1.0;
                }

                // 对于非获胜者，生成二项分布的连接值
                for (int j = 0; j < from_area_w; ++j)
                {
                    if (from_area_winners_set.find(j) == from_area_winners_set.end())
                    {
                        std::bernoulli_distribution binom_dist(p);
                        the_connectome[j][target_area.w + i] = binom_dist(rng) ? 1.0 : 0.0;
                    }
                }
            }

            double area_to_area_beta = (disable_plasticity ? 0.0 : target_area.beta_by_area[from_area_name]);
            for (int i : target_area._new_winners) // 更新连接矩阵
            {
                for (int j : from_area_winners)
                {
                    the_connectome[j][i] *= 1.0 + area_to_area_beta;
                }
            }

            num_inputs_processed += 1;
        }

        for (auto& pair : area_by_name)
        {
            const std::string& other_area_name = pair.first;
            // 改维度

            auto& other_area_connectomes = connectomes[other_area_name];

            if (std::find(from_areas.begin(), from_areas.end(), other_area_name) == from_areas.end()) // if other_area_name not in from_areas:
            {
                auto& the_other_area_connectome = other_area_connectomes[target_area_name];

                int change = num_first_winners_processed;
                size_t num_rows = the_other_area_connectome.size(); // 获取行数
                // size_t num_cols = num_rows > 0 ? the_other_area_connectome[0].size() : 0; // 获取列数（假设每一行都有相同的列数）
                size_t num_cols = the_other_area_connectome[0].size(); // 获取列数（假设每一行都有相同的列数）
                if (num_cols == 1)
                {
                    change--;
                }

                if (num_first_winners_processed != 0)
                {
                    for (auto& row : the_other_area_connectome)
                    {
                        row.resize(row.size() + change, 0.0); // 增加列，并初始化为0.0
                    }
                }

                std::bernoulli_distribution bernoulli(p);
                for (size_t row = 0; row < the_other_area_connectome.size(); ++row)
                {
                    for (int col = target_area.w; col < target_area._new_w; ++col)
                    {
                        the_other_area_connectome[row][col] = bernoulli(rng) ? 1.0 : 0.0;
                    }
                }
            }

            std::unordered_map<std::string, std::vector<std::vector<double>>>& target_area_connectomes = connectomes[target_area_name];
            auto& the_target_area_connectome = target_area_connectomes[other_area_name];

            int change = num_first_winners_processed;
            size_t num_rows = the_target_area_connectome.size(); // 获取行数
            // size_t num_cols = num_rows > 0 ? the_target_area_connectome[0].size() : 0; // 获取列数（假设每一行都有相同的列数）
            size_t num_cols = the_target_area_connectome[0].size(); // 获取列数（假设每一行都有相同的列数）
            if (num_rows == 1)
            {
                change--;
            }

            if (num_first_winners_processed != 0)
            {
                // 添加新的行，并初始化为0
                the_target_area_connectome.resize(the_target_area_connectome.size() + change, std::vector<double>(the_target_area_connectome[0].size(), 0.0));
            }

            std::binomial_distribution<> d(1, p);
            for (int i = target_area.w; i < target_area._new_w; ++i)
            {
                for (size_t j = 0; j < the_target_area_connectome[i].size(); ++j)
                {
                    the_target_area_connectome[i][j] = d(rng);
                }
            }
        }
        return num_first_winners_processed;
    }

    void add_area(std::string area_name, int n, int k, double beta)
    {
        area_by_name[area_name] = Area(area_name, n, k, beta);

        // 初始化新的连接矩阵映射
        std::unordered_map<std::string, std::vector<std::vector<double>>> new_connectomes;
        for (const auto& other_area_pair : area_by_name)
        {
            const std::string& other_area_name = other_area_pair.first;
            Area& other_area = area_by_name[other_area_name];

            int other_area_size = other_area.my_explicit ? other_area.n : 1;

            std::vector<std::vector<double>> empty_matrix(1, std::vector<double>(other_area_size, 0)); // 创建一个大小为 0 x other_area_size 的二维向量
            new_connectomes[other_area_name] = empty_matrix;                                           // 插入到 unordered_map 中

            if (other_area_name != area_name)
            {
                std::vector<std::vector<double>> empty_matrix(other_area_size, std::vector<double>(1, 0)); // 创建一个大小为 other_area_size x 0 的二维向量
                connectomes[other_area_name][area_name] = empty_matrix;                                    // 插入到 connectomes 中
            }

            other_area.beta_by_area[area_name] = other_area.beta;
            area_by_name[area_name].beta_by_area[other_area_name] = beta;
        }

        connectomes[area_name] = new_connectomes;
    }

    void update_plasticity(const std::string& from_area, const std::string& to_area, double new_beta)
    {
        area_by_name[to_area].beta_by_area[from_area] = new_beta;
    }
    void update_plasticities(const std::unordered_map<std::string, std::vector<std::pair<std::string, double>>>& area_update_map = {},
        const std::unordered_map<std::string, std::vector<std::pair<std::string, double>>>& stim_update_map = {})
    {
        for (const auto& to_area_pair : area_update_map)
        {
            const std::string& to_area = to_area_pair.first;
            const std::vector<std::pair<std::string, double>>& update_rules = to_area_pair.second;
            for (const auto& rule : update_rules)
            {
                const std::string& from_area = rule.first;
                double new_beta = rule.second;
                update_plasticity(from_area, to_area, new_beta);
            }
        }
    }
};

class ParserBrain : public Brain
{
public:
    std::unordered_map<std::string, GenericRuleSet> lexeme_dict;
    std::vector<std::string> all_areas;
    std::vector<std::string> recurrent_areas;
    std::vector<std::string> initial_areas;
    std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_set<int>>> fiber_states;
    std::unordered_map<std::string, std::unordered_set<int>> area_states;
    std::unordered_map<std::string, std::unordered_set<std::string>> activated_fibers; // 修改类型
    std::unordered_map<std::string, std::vector<std::string>> readout_rules;

    ParserBrain(double p,
        const std::unordered_map<std::string, GenericRuleSet>& lexeme_dict = {},
        const std::vector<std::string>& all_areas = {},
        const std::vector<std::string>& recurrent_areas = {},
        const std::vector<std::string>& initial_areas = {},
        const std::unordered_map<std::string, std::vector<std::string>>& readout_rules = {})
        : Brain(p), all_areas(all_areas), recurrent_areas(recurrent_areas), initial_areas(initial_areas), readout_rules(readout_rules)
    {
        this->lexeme_dict = lexeme_dict;
        initialize_states();
    }

    void initialize_states()
    {
        // 初始化fiber_states
        for (const auto& from_area : all_areas)
        {
            for (const auto& to_area : all_areas)
            {
                fiber_states[from_area][to_area].insert(0);
            }
        }

        // 初始化area_states
        for (const auto& area : all_areas)
        {
            area_states[area].insert(0);
        }

        // 移除initial_areas中的0
        for (const auto& area : initial_areas)
        {
            area_states[area].erase(0);
        }
    }
    // applyFiberRule 方法实现
    void applyFiberRule(const Rule& rule)
    {
        if (rule.flag == FIBER_RULE)
        {
            if (rule.action == INHIBIT)
            {
                fiber_states[rule.area1][rule.area2].insert(rule.index);
                fiber_states[rule.area2][rule.area1].insert(rule.index);
            }
            else if (rule.action == DISINHIBIT)
            {
                fiber_states[rule.area1][rule.area2].erase(rule.index);
                fiber_states[rule.area2][rule.area1].erase(rule.index);
            }
        }
    }

    void applyAreaRule(const Rule& rule)
    {
        if (rule.flag == AREA_RULE)
        {
            if (rule.action == INHIBIT)
            {
                area_states[rule.area].insert(rule.index);
            }
            else if (rule.action == DISINHIBIT)
            {
                area_states[rule.area].erase(rule.index);
            }
        }
    }

    bool applyRule(const Rule& rule)
    {
        if (rule.flag == FIBER_RULE)
        {
            applyFiberRule(rule);
            return true;
        }
        if (rule.flag == AREA_RULE)
        {
            applyAreaRule(rule);
            return true;
        }
        return false;
    }

    void parse_project()
    {

        // 获取投射映射
        auto project_map = getProjectMap();

        // 记住激活的纤维
        remember_fibers(project_map);

        // 执行投影操作
        project({}, project_map);
    }

    void printProjectMap(const std::unordered_map<std::string, std::unordered_set<std::string>>& project_map)
    {
        if (project_map.empty())
        {
            std::cout << "The project map is empty." << std::endl;
            return;
        }

        // 遍历 unordered_map
        for (const auto& pair : project_map)
        {
            const std::string& project = pair.first;                    // 获取键（项目名）
            const std::unordered_set<std::string>& tasks = pair.second; // 获取值（任务集合）

            std::cout << "Project: " << project << "\nTasks: {";

            // 遍历 unordered_set
            for (auto it = tasks.begin(); it != tasks.end(); ++it)
            {
                std::cout << *it;
                if (std::next(it) != tasks.end())
                {
                    std::cout << ", ";
                }
            }

            std::cout << "}\n";
        }
    }

    void remember_fibers(const std::unordered_map<std::string, std::unordered_set<std::string>>& project_map)
    {

        // printProjectMap(project_map);
        for (const auto& from_area_pair : project_map)
        {
            const std::string& from_area = from_area_pair.first;
            const std::unordered_set<std::string>& to_areas = from_area_pair.second;
            activated_fibers[from_area].insert(to_areas.begin(), to_areas.end());
        }
    }

    bool recurrent(const std::string& area) const
    {
        return std::find(recurrent_areas.begin(), recurrent_areas.end(), area) != recurrent_areas.end();
    }

    std::unordered_map<std::string, std::unordered_set<std::string>> getProjectMap() const
    {
        std::unordered_map<std::string, std::unordered_set<std::string>> proj_map;
        for (const auto& area1 : all_areas)
        {
            if (area_states.at(area1).empty())
            {
                for (const auto& area2 : all_areas)
                {
                    if (area1 == LEX && area2 == LEX)
                    {
                        continue;
                    }
                    if (area_states.at(area2).empty())
                    {
                        if (fiber_states.at(area1).at(area2).empty())
                        {
                            if (!area_by_name.at(area1).winners.empty())
                            {
                                proj_map[area1].insert(area2);
                            }
                            if (!area_by_name.at(area2).winners.empty())
                            {
                                proj_map[area2].insert(area2);
                            }
                        }
                    }
                }
            }
        }

        return proj_map;
    }

    void activateWord(const std::string& area_name, const std::string& word)
    {
        Area& area = area_by_name[area_name];
        int k = area.k;
        int assembly_start = lexeme_dict[word].index * k;

        area.winners.clear();
        for (int i = 0; i < k; ++i)
        {
            area.winners.push_back(assembly_start + i);
        }

        area.fix_assembly();
    }

    std::string interpretAssemblyAsString(const std::string& area_name)
    {
        return getWord(area_name, 0.7);
    }

    std::string getWord(const std::string& area_name, double min_overlap = 0.7)
    {
        if (area_by_name[area_name].winners.empty())
        {
            throw std::runtime_error("Cannot get word because no assembly in " + area_name);
        }

        std::unordered_set<int> winners2(area_by_name[area_name].winners.begin(), area_by_name[area_name].winners.end());
        int area_k = area_by_name[area_name].k;
        int threshold = (int)(min_overlap * area_k);

        // 创建一个vector来存储unordered_map的元素
        std::vector<std::pair<std::string, GenericRuleSet>> vec(lexeme_dict.begin(), lexeme_dict.end());
        std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b)
            { return a.second.index < b.second.index; });

        for (const auto& lexeme_pair : vec)
        {
            const std::string& word = lexeme_pair.first;
            int word_index = lexeme_pair.second.index;
            int word_assembly_start = word_index * area_k;

            std::unordered_set<int> word_assembly;
            for (int i = word_assembly_start; i < word_assembly_start + area_k; ++i)
            {
                word_assembly.insert(i);
            }

            // 计算 winners 和 word_assembly 的交集的大小
            int intersection_count = 0;
            for (const int& element : winners2)
            {
                if (word_assembly.find(element) != word_assembly.end())
                {
                    ++intersection_count;
                }
            }

            if (intersection_count >= threshold)
            {
                return word;
                // 在实际使用中可以返回 word
            }
        }

        return "";
    }

    std::unordered_map<std::string, std::unordered_set<std::string>> getActivatedFibers()
    {
        std::unordered_map<std::string, std::unordered_set<std::string>> pruned_activated_fibers;

        for (const auto& from_area_pair : activated_fibers)
        {
            const std::string& from_area = from_area_pair.first;
            const std::unordered_set<std::string>& to_areas = from_area_pair.second;

            for (const auto& to_area : to_areas)
            {

                auto it = readout_rules.find(from_area);
                if (it != readout_rules.end())
                {
                    const std::vector<std::string>& targets = it->second;
                    if (std::find(targets.begin(), targets.end(), to_area) != targets.end())
                    {
                        pruned_activated_fibers[from_area].insert(to_area);
                    }
                }
            }
        }
        return pruned_activated_fibers;
    }
};

class EnglishParserBrain : public ParserBrain
{
public:
    bool verbose;

    EnglishParserBrain(double p, int non_LEX_n = 10000, int non_LEX_k = 100, int LEX_k = 20,
        double default_beta = 0.2, double LEX_beta = 1.0, double recurrent_beta = 0.05,
        double interarea_beta = 0.5, bool verbose = false)
        : ParserBrain(p, LEXEME_DICT, AREAS, RECURRENT_AREAS, { LEX, SUBJ, VERB }, ENGLISH_READOUT_RULES), verbose(verbose)
    {

        int LEX_n = LEX_SIZE * LEX_k;
        add_explicit_area(LEX, LEX_n, LEX_k, default_beta);

        int DET_k = LEX_k;
        add_area(SUBJ, non_LEX_n, non_LEX_k, default_beta);
        add_area(OBJ, non_LEX_n, non_LEX_k, default_beta);
        add_area(VERB, non_LEX_n, non_LEX_k, default_beta);
        add_area(ADJ, non_LEX_n, non_LEX_k, default_beta);
        add_area(PREP, non_LEX_n, non_LEX_k, default_beta);
        add_area(PREP_P, non_LEX_n, non_LEX_k, default_beta);
        add_area(DET, non_LEX_n, DET_k, default_beta);
        add_area(ADVERB, non_LEX_n, non_LEX_k, default_beta);

        std::unordered_map<std::string, std::vector<std::pair<std::string, double>>> custom_plasticities;
        for (const auto& area : RECURRENT_AREAS)
        {
            custom_plasticities[LEX].emplace_back(area, LEX_beta);
            custom_plasticities[area].emplace_back(LEX, LEX_beta);
            custom_plasticities[area].emplace_back(area, recurrent_beta);
            for (const auto& other_area : RECURRENT_AREAS)
            {
                if (other_area == area)
                    continue;
                custom_plasticities[area].emplace_back(other_area, interarea_beta);
            }
        }

        update_plasticities(custom_plasticities);
    }

    std::unordered_map<std::string, std::unordered_set<std::string>> getProjectMap() const
    {
        auto proj_map = ParserBrain::getProjectMap();
        if (proj_map.find(LEX) != proj_map.end() && proj_map[LEX].size() > 2)
        {
            throw std::runtime_error("Got that LEX projecting into many areas: " + std::to_string(proj_map[LEX].size()));
        }
        return proj_map;
    }

    std::string getWord(const std::string& area_name, double min_overlap = 0.7)
    {
        auto word = ParserBrain::getWord(area_name, min_overlap);
        if (!word.empty())
        {
            return word;
        }
        return "<NON-WORD>";
    }
};

std::vector<std::vector<std::string>> parseHelper(ParserBrain& b, const std::string& sentence, double p, int LEX_k, int project_rounds, bool verbose, bool debug,
    const std::unordered_map<std::string, GenericRuleSet>& lexeme_dict,
    const std::vector<std::string>& all_areas,
    const std::vector<std::string>& explicit_areas,
    ReadoutMethod readout_method,
    const std::unordered_map<std::string, std::vector<std::string>>& readout_rules);

std::vector<std::vector<std::string>> parse(const std::string& sentence = "the dogs love a man quickly", const std::string& language = "English",
    double p = 0.1, int LEX_k = 20, int project_rounds = 20, bool verbose = false, bool debug = false,
    ReadoutMethod readout_method = ReadoutMethod::FIBER_READOUT)
{
    //cout << "input sentence: " << sentence << endl;
    EnglishParserBrain b(p, 10000, 100, LEX_k, 0.2, 1.0, 0.05, 0.5, verbose);
    auto lexeme_dict = LEXEME_DICT;
    auto all_areas = AREAS;
    auto explicit_areas = EXPLICIT_AREAS;
    auto readout_rules = ENGLISH_READOUT_RULES;

    return parseHelper(b, sentence, p, LEX_k, project_rounds, verbose, debug,
        lexeme_dict, all_areas, explicit_areas, readout_method, readout_rules);
    
}

void read_out(ParserBrain& b, const std::string& area, const std::unordered_map<std::string, std::unordered_set<std::string>>& mapping, std::vector<std::vector<std::string>>& dependencies);

std::vector<std::vector<std::string>> parseHelper(ParserBrain& b, const std::string& sentence, double p, int LEX_k, int project_rounds, bool verbose, bool debug,
    const std::unordered_map<std::string, GenericRuleSet>& lexeme_dict,
    const std::vector<std::string>& all_areas,
    const std::vector<std::string>& explicit_areas,
    ReadoutMethod readout_method,
    const std::unordered_map<std::string, std::vector<std::string>>& readout_rules)
{

    // 将句子按空格拆分成单词列表
    std::istringstream iss(sentence);
    std::vector<std::string> words((std::istream_iterator<std::string>(iss)),
        std::istream_iterator<std::string>());

    bool extreme_debug = false;

    // 遍历句子中的每个单词
    for (const auto& word : words)
    {
        // 从词典中获取当前单词的词素信息
        const auto& lexeme = lexeme_dict.at(word);

        // 激活词汇区域中的当前单词
        b.activateWord(LEX, word);

        // 如果启用了详细模式，打印激活的单词及其对应的获胜神经元
        if (verbose)
        {
            std::cout << "Activated word: " << word << std::endl;
            for (const auto& winner : b.area_by_name[LEX].winners)
            {
                std::cout << winner << " ";
            }
            std::cout << std::endl;
        }

        // 应用词素的预规则
        for (const auto& rule : lexeme.pre_rules)
        {
            b.applyRule(rule);
        }

        // 获取投射映射（神经区域之间的连接）
        auto proj_map = b.getProjectMap();

        // printUnorderedMap(proj_map);

        // 遍历投射映射中的每个区域
        for (const auto& area : proj_map)
        {
            // 如果区域不在 LEX 的投射映射中，固定其神经元集合
            if (proj_map[LEX].find(area.first) == proj_map[LEX].end())
            {
                b.area_by_name[area.first].fix_assembly();

                // 如果启用了详细模式，打印固定的区域
                if (verbose)
                {
                    std::cout << "FIXED assembly bc not LEX->this area in: " << area.first << std::endl;
                }
            }
            // 否则，如果区域不是 LEX，则取消固定其神经元集合并清空获胜神经元
            else if (area.first != LEX)
            {
                b.area_by_name[area.first].unfix_assembly();
                b.area_by_name[area.first].winners.clear();

                // 如果启用了详细模式，打印被清除的区域
                if (verbose)
                {
                    std::cout << "ERASED assembly because LEX->this area in " << area.first << std::endl;
                }
            }
        }

        // 获取投射映射（神经区域之间的连接）
        proj_map = b.getProjectMap();

        // 如果启用了详细模式，打印投射映射
        if (verbose)
        {
            std::cout << "Got proj_map = " << std::endl;
            for (const auto& proj : proj_map)
            {
                std::cout << proj.first << ": ";
                for (const auto& area : proj.second)
                {
                    std::cout << area << " ";
                }
                std::cout << std::endl;
            }
        }

        // 执行指定轮次的投射
        for (int i = 0; i < project_rounds; ++i)
            // for (int i = 0; i < 50; ++i)
        {
            // 执行一次解析投射
            b.parse_project();

            proj_map = b.getProjectMap();

            // 如果启用了详细模式，打印每轮投射后的投射映射
            if (verbose)
            {
                proj_map = b.getProjectMap();
                std::cout << "Got proj_map = " << std::endl;
                for (const auto& proj : proj_map)
                {
                    std::cout << proj.first << ": ";
                    for (const auto& area : proj.second)
                    {
                        std::cout << area << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }

        // 应用词素的后置规则
        for (const auto& rule : lexeme.post_rules)
        {
            b.applyRule(rule);
        }
    }

    // 读取解析结果
    // 对于所有的读出方法，解锁所有区域的神经元集合，并禁用可塑性
    b.disable_plasticity = true;
    for (const auto& area : all_areas)
    {
        b.area_by_name[area].unfix_assembly();
    }

    // 初始化依存关系列表
    std::vector<std::vector<std::string>> dependencies;

    if (readout_method == ReadoutMethod::FIBER_READOUT)
    {
        // 获取已激活的纤维
        auto activated_fibers = b.getActivatedFibers();

        // 如果启用了详细模式，打印已激活的纤维
        if (verbose)
        {
            std::cout << "Got activated fibers for readout:" << std::endl;
            for (const auto& pair : activated_fibers)
            {
                std::cout << pair.first << ": ";
                for (const auto& area : pair.second)
                {
                    std::cout << area << " ";
                }
                std::cout << std::endl;
            }
        }

        // 读取动词的依存关系
        read_out(b, VERB, activated_fibers, dependencies);

        // 打印获取的依存关系
        // std::cout << "Got dependencies: " << std::endl;
        // for (const auto& dep : dependencies)
        // {
        //     for (const auto& word : dep)
        //     {
        //         std::cout << word << " ";
        //     }
        //     std::cout << std::endl;
        // }

        return dependencies;
    }
}

void read_out(ParserBrain& b, const std::string& area, const std::unordered_map<std::string, std::unordered_set<std::string>>& mapping, std::vector<std::vector<std::string>>& dependencies)
{
    const auto& to_areas = mapping.at(area); // 获取映射中当前区域对应的目标区域

    // 执行从当前区域到目标区域的投射
    std::unordered_map<std::string, std::unordered_set<std::string>> project_map;

    project_map[area] = std::unordered_set<std::string>(to_areas.begin(), to_areas.end());

    b.project(0, project_map);

    // 获取当前区域在LEX中的词汇
    std::string this_word = b.getWord(LEX);

    // 遍历目标区域
    for (const auto& to_area : to_areas)
    {
        // 如果目标区域是LEX，跳过
        if (to_area == LEX)
        {
            continue;
        }

        // 执行从目标区域到LEX的投射
        project_map.clear();
        project_map[to_area].insert(LEX);
        b.project(0, project_map);

        // 获取目标区域在LEX中的词汇
        std::string other_word = b.getWord(LEX);

        // 将当前词汇、目标词汇和目标区域添加到依存关系中
        dependencies.push_back({ this_word, other_word, to_area });
    }

    // 递归调用read_out函数，处理目标区域
    for (const auto& to_area : to_areas)
    {
        if (to_area != LEX)
        {
            read_out(b, to_area, mapping, dependencies);
        }
    }
}

// int main()
// {
//     parse("cats chase mice");
//     parse("cats chase mice quickly");
//     parse("the cats chase a man");
//     parse("the cats of people chase mice");
//     parse("cats chase big mice");
//     parse("cats run");
//     parse("cats are mice");

//     parse("the dogs love a man quickly");

//     return 0;
// }


#endif // PARSER_H
