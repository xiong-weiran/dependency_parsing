#include "brain.h"
using namespace std;
using defaultdict_set = std::unordered_map<std::string, std::unordered_set<std::string>>;
using defaultdict_list = std::unordered_map<std::string, std::vector<std::string>>;

// 构建词汇表
// 创建一个无序映射，将字符串映射到 GenericRuleSet 对象。
// 这些字符串表示英语单词，它们被映射到具有特定语法功能的规则集。
std::unordered_map<std::string, GenericRuleSet> LEXEME_DICT = {
    {"the", generic_determinant(0)},     // "the" 是一个限定词，对应规则集 generic_determinant，索引为 0
    {"a", generic_determinant(1)},       // "a" 是一个限定词，对应规则集 generic_determinant，索引为 1
    {"dogs", generic_noun(2)},           // "dogs" 是一个名词，对应规则集 generic_noun，索引为 2
    {"cats", generic_noun(3)},           // "cats" 是一个名词，对应规则集 generic_noun，索引为 3
    {"mice", generic_noun(4)},           // "mice" 是一个名词，对应规则集 generic_noun，索引为 4
    {"people", generic_noun(5)},         // "people" 是一个名词，对应规则集 generic_noun，索引为 5
    {"chase", generic_trans_verb(6)},    // "chase" 是一个及物动词，对应规则集 generic_trans_verb，索引为 6
    {"love", generic_trans_verb(7)},     // "love" 是一个及物动词，对应规则集 generic_trans_verb，索引为 7
    {"bite", generic_trans_verb(8)},     // "bite" 是一个及物动词，对应规则集 generic_trans_verb，索引为 8
    {"of", generic_preposition(9)},      // "of" 是一个介词，对应规则集 generic_preposition，索引为 9
    {"big", generic_adjective(10)},      // "big" 是一个形容词，对应规则集 generic_adjective，索引为 10
    {"bad", generic_adjective(11)},      // "bad" 是一个形容词，对应规则集 generic_adjective，索引为 11
    {"run", generic_intrans_verb(12)},   // "run" 是一个不及物动词，对应规则集 generic_intrans_verb，索引为 12
    {"fly", generic_intrans_verb(13)},   // "fly" 是一个不及物动词，对应规则集 generic_intrans_verb，索引为 13
    {"quickly", generic_adverb(14)},     // "quickly" 是一个副词，对应规则集 generic_adverb，索引为 14
    {"in", generic_preposition(15)},     // "in" 是一个介词，对应规则集 generic_preposition，索引为 15
    {"are", generic_copula(16)},         // "are" 是一个系动词，对应规则集 generic_copula，索引为 16
    {"man", generic_noun(17)},           // "man" 是一个名词，对应规则集 generic_noun，索引为 17
    {"woman", generic_noun(18)},         // "woman" 是一个名词，对应规则集 generic_noun，索引为 18
    {"saw", generic_trans_verb(19)},     // "saw" 是一个及物动词，对应规则集 generic_trans_verb，索引为 19
};

// 定义英语句法解析规则，映射从语法角色到可接受的成分序列。
// 例如，动词 (VERB) 可能由词素 (LEX)、主语 (SUBJ)、宾语 (OBJ)、介词短语 (PREP_P)、副词 (ADVERB)、形容词 (ADJ) 组成。
std::unordered_map<std::string, std::vector<std::string>> ENGLISH_READOUT_RULES = {
    {VERB, {LEX, SUBJ, OBJ, PREP_P, ADVERB, ADJ}},    // 动词规则
    {SUBJ, {LEX, DET, ADJ, PREP_P}},                  // 主语规则
    {OBJ, {LEX, DET, ADJ, PREP_P}},                   // 宾语规则
    {PREP_P, {LEX, PREP, ADJ, DET}},                  // 介词短语规则
    {PREP, {LEX}},                                    // 介词规则
    {ADJ, {LEX}},                                     // 形容词规则
    {DET, {LEX}},                                     // 限定词规则
    {ADVERB, {LEX}},                                  // 副词规则
    {LEX, {}} };                                      // 词素规则

// 生成一个给定区间 [a, b] 和尺度 (scale) 的截断正态分布随机数序列。
// size 参数指定生成的随机数个数。
std::vector<double> truncnorm_rvs(double a, double b, double scale, std::size_t size)
{
    std::vector<double> results;          // 存储结果的向量
    results.reserve(size);                // 预分配存储空间

    // 随机数生成器和正态分布定义
    boost::mt19937 rng;                   // 随机数生成器
    boost::random::normal_distribution<> normal_dist(0, scale);  // 均值为0，标准差为 scale 的正态分布

    for (std::size_t i = 0; i < size; ++i) // 生成 size 个随机数
    {
        double num;
        do
        {
            num = normal_dist(rng);       // 生成一个正态分布的随机数
        } while (num < a || num > b);     // 只保留在区间 [a, b] 内的数
        results.push_back(num);           // 将数添加到结果中
    }

    return results;                       // 返回生成的随机数序列
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
    double beta;                                              // 默认激活阈值 beta
    std::unordered_map<std::string, double> beta_by_stimulus; // 刺激名称与对应 beta 的映射
    std::unordered_map<std::string, double> beta_by_area;     // 区域名称与对应 beta 的映射
    int w;                                                    // 曾经在该区域激活的神经元数量
    std::vector<int> saved_w;                                 // 每轮的支持大小列表
    std::vector<int> winners;                                 // 由先前动作设定的获胜者列表
    std::vector<std::vector<int>> saved_winners;              // 所有获胜者的列表，每轮一个列表
    int num_first_winners;                                    // 第一次获胜者数量
    bool fixed_assembly;                                      // 该区域的集合（获胜者）是否被冻结
    int _new_w;                                               // 自上次调用 .project() 以来的 `w` 值
    std::vector<int> _new_winners;                            // 自上次调用 .project() 以来的 `winners` 值，仅在 .project() 方法内使用
    int num_ever_fired;                                       // 曾经被激活的神经元数量
    std::vector<bool> ever_fired;                             // 记录每个神经元是否曾经被激活过
    bool my_explicit;                                         // 是否完全模拟该区域（而不是仅执行稀疏模拟）

    // 构造函数
    Area(const std::string& name, int n, int k, double beta = 0.05, int w = 0, bool my_explicit = false)
        : name(name), n(n), k(k), beta(beta), w(w), num_first_winners(-1), fixed_assembly(false), my_explicit(my_explicit) {}

    // 默认构造函数
    Area() = default;

    // 更新获胜者列表
    void _update_winners()
    {
        winners = _new_winners;   // 将 `_new_winners` 赋值给 `winners`
        if (!my_explicit)         // 如果不是完全模拟
        {
            w = _new_w;           // 更新 `w` 值
        }
    }

    // 冻结获胜者集合
    void fix_assembly()
    {
        fixed_assembly = true;    // 将 `fixed_assembly` 设为 true
    }

    // 解除获胜者集合的冻结
    void unfix_assembly()
    {
        fixed_assembly = false;   // 将 `fixed_assembly` 设为 false
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
        // 在 area_by_name 中创建一个新的区域 Area 对象，并初始化相关属性
        area_by_name[area_name] = Area(area_name, n, k, beta, n, true);

        // 初始化区域的 ever_fired 属性，记录每个神经元是否曾经被激活过，初始值为 false
        area_by_name[area_name].ever_fired = std::vector<bool>(n, false);

        // 初始化区域的 num_ever_fired 属性，记录曾经被激活的神经元数量，初始值为 0
        area_by_name[area_name].num_ever_fired = 0;

        // 默认连接概率，如果没有提供自定义概率，则使用默认概率 p
        double inner_p = p;
        double in_p = p;
        double out_p = p;

        // 区域名称映射到二维数组的字典，数组中值为0或1，表示连接矩阵
        std::unordered_map<std::string, std::vector<std::vector<double>>> new_connectomes;

        // 遍历现有的区域，更新连接矩阵和 beta 值映射
        for (const auto& other_area_pair : area_by_name)
        {
            const std::string& other_area_name = other_area_pair.first;

            // 如果当前区域是新添加的区域
            if (other_area_name == area_name)
            {
                // 使用伯努利分布生成新的连接矩阵，矩阵大小为 (n, n)，值为 0 或 1
                std::bernoulli_distribution distribution(inner_p);
                std::vector<std::vector<double>> matrix(n, std::vector<double>(n, 0.0f));
                for (int i = 0; i < n; ++i)
                {
                    for (int j = 0; j < n; ++j)
                    {
                        matrix[i][j] = distribution(rng) ? 1.0f : 0.0f;
                    }
                }

                // 将生成的连接矩阵添加到 new_connectomes 中
                new_connectomes[other_area_name] = matrix;
            }

            // 更新其他区域的 beta_by_area 属性，记录与新添加区域的 beta 映射
            area_by_name[other_area_name].beta_by_area[area_name] = area_by_name[other_area_name].beta;
            area_by_name[area_name].beta_by_area[other_area_name] = beta;
        }

        // 将新的连接矩阵添加到 connectomes 中
        connectomes[area_name] = new_connectomes;
    }


    void project(int areas_by_stim, defaultdict_set dst_areas_by_src_area, int verbose = 0)
    {
        // 定义两个默认列表，分别用于存储刺激输入和区域输入
        defaultdict_list stim_in;
        defaultdict_list area_in;

        // 遍历所有源区域到目标区域的映射
        for (const auto& pair : dst_areas_by_src_area)
        {
            std::string from_area_name = pair.first; // 获取源区域名称
            for (const auto& to_area_name : pair.second) // 获取目标区域名称列表
            {
                area_in[to_area_name].push_back(from_area_name); // 将源区域名称添加到目标区域的输入列表中
            }
        }

        // 获取需要更新的区域名称
        std::vector<std::string> to_update_area_names;
        for (const auto& pair : area_in)
        {
            to_update_area_names.push_back(pair.first); // 将每个目标区域名称添加到更新列表中
        }

        // 对每个需要更新的区域进行投影操作
        for (const auto& area_name : to_update_area_names)
        {
            Area& area = area_by_name[area_name]; // 获取区域对象的引用
            int num_first_winners = project_into(area, stim_in[area_name], area_in[area_name], verbose); // 投影到区域中
            area.num_first_winners = num_first_winners; // 更新区域的第一次获胜者数量

            if (save_winners)
            {
                area.saved_winners.push_back(area._new_winners); // 保存新的获胜者列表
            }
        }

        // 更新每个需要更新的区域的获胜者，并保存支持大小
        for (const std::string& area_name : to_update_area_names)
        {
            Area& area = area_by_name[area_name]; // 获取区域对象的引用
            area._update_winners(); // 更新区域的获胜者列表
            if (save_size)
            {
                area.saved_w.push_back(area.w); // 保存区域的支持大小
            }
        }
    }


    int project_into(Area& target_area, std::vector<std::string> from_stimuli, std::vector<std::string> from_areas, int verbose = 0)
    {
        // 定义一个变量用于记录处理的第一次获胜者的数量
        int num_first_winners_processed = 0;

        // 定义一个二维向量，用于存储按第一次获胜者索引分组的输入
        std::vector<std::vector<int>> inputs_by_first_winner_index;

        // 如果目标区域的集合被冻结
        if (target_area.fixed_assembly)
        {
            // 将新的获胜者设置为当前的获胜者
            target_area._new_winners = target_area.winners;

            // 将新的激活神经元数量设置为当前的激活神经元数量
            target_area._new_w = target_area.w;

            // 设置处理的第一次获胜者数量为 0
            num_first_winners_processed = 0;
        }

        else
        {
            // 获取目标区域的名称
            std::string target_area_name = target_area.name;

            // 初始化向量以存储以前的获胜者输入，初始值为0
            std::vector<double> prev_winner_inputs(target_area.w, 0.0);

            // 遍历所有的源区域名称
            for (const auto& from_area_name : from_areas)
            {
                // 遍历源区域的所有获胜者
                for (const auto& w : area_by_name[from_area_name].winners)
                {
                    // 累加每个获胜者的输入值到 prev_winner_inputs 中
                    for (size_t i = 0; i < prev_winner_inputs.size(); ++i)
                    {
                        prev_winner_inputs[i] += connectomes[from_area_name][target_area_name][w][i];
                    }
                }
            }

            // 初始化向量以存储所有潜在的获胜者输入
            std::vector<double> all_potential_winner_inputs;
            int total_k;                          // 总的有效获胜者数量
            int num_inputs = 0;                   // 输入数量
            std::vector<int> input_size_by_from_area_index; // 每个源区域的输入大小

            // 如果目标区域不是显式的
            if (!target_area.my_explicit)
            {
                double normal_approx_mean = 0.0;  // 正态分布近似的均值
                double normal_approx_var = 0.0;   // 正态分布近似的方差

                // 遍历所有源区域名称
                for (const std::string& from_area_name : from_areas)
                {
                    size_t effective_k = area_by_name[from_area_name].winners.size(); // 获取源区域的有效获胜者数量
                    input_size_by_from_area_index.push_back(effective_k); // 将有效获胜者数量添加到输入大小向量中
                    num_inputs += 1; // 增加输入数量
                }

                // 计算总的有效获胜者数量
                total_k = std::accumulate(input_size_by_from_area_index.begin(), input_size_by_from_area_index.end(), 0);

                // 计算目标区域的有效神经元数量
                int effective_n = target_area.n - target_area.w;

                // 如果有效神经元数量小于目标区域的激活神经元数量，输出警告信息
                if (effective_n <= target_area.k)
                {
                    std::cout << "Remaining size of area " << target_area_name << " too small to sample k new winners." << std::endl;
                }

                // 计算分位点
                double quantile = (double(effective_n) - target_area.k) / effective_n;

                // 使用 Boost.Math 计算二项分布的分位点
                boost::math::binomial_distribution<> binom_dist(total_k, p);
                double alpha = boost::math::quantile(binom_dist, quantile);
                double mu = total_k * p;
                double std = std::sqrt(total_k * p * (1.0 - p));
                double a = (alpha - mu) / std;
                double b = (total_k - mu) / std;

                // 生成截断正态分布的样本
                std::vector<double> random_values = generate_truncated_normal_samples(0, std, alpha, total_k, target_area.k);

                // 将每个元素四舍五入为整数
                std::transform(random_values.begin(), random_values.end(), random_values.begin(),
                    [](double x) { return std::round(x); });

                // 存储潜在的新获胜者输入
                std::vector<double> potential_new_winner_inputs = random_values;

                // 预留足够的空间，避免多次重新分配
                all_potential_winner_inputs.reserve(prev_winner_inputs.size() + potential_new_winner_inputs.size());
                all_potential_winner_inputs.insert(all_potential_winner_inputs.end(), prev_winner_inputs.begin(), prev_winner_inputs.end());
                all_potential_winner_inputs.insert(all_potential_winner_inputs.end(), potential_new_winner_inputs.begin(), potential_new_winner_inputs.end());
            }
            else
            {
                // 如果目标区域是显式的，只使用以前的获胜者输入
                all_potential_winner_inputs = prev_winner_inputs;
            }


            // 获取具有最大值的前 k 个索引，即新的获胜者索引
            std::vector<int> new_winner_indices = nlargest_indices(all_potential_winner_inputs, target_area.k);

            // 如果目标区域是显式的
            if (target_area.my_explicit)
            {
                // 遍历新的获胜者索引
                for (int winner : new_winner_indices)
                {
                    // 如果该获胜者以前未被激活过
                    if (!target_area.ever_fired[winner])
                    {
                        target_area.ever_fired[winner] = true; // 标记为已激活
                        target_area.num_ever_fired += 1; // 增加已激活神经元数量
                    }
                }
            }

            // 初始化处理的第一次获胜者数量为 0
            num_first_winners_processed = 0;

            // 定义一个向量来存储第一次获胜者的输入
            std::vector<double> first_winner_inputs;

            // 如果目标区域不是显式的
            if (!target_area.my_explicit)
            {
                // 遍历新的获胜者索引
                for (int i = 0; i < target_area.k; ++i)
                {
                    // 如果当前获胜者索引大于或等于当前激活神经元数量
                    if (new_winner_indices[i] >= target_area.w)
                    {
                        // 将对应的潜在获胜者输入添加到第一次获胜者输入中
                        first_winner_inputs.push_back(all_potential_winner_inputs[new_winner_indices[i]]);
                        // 更新新的获胜者索引，使其指向新的激活神经元位置
                        new_winner_indices[i] = target_area.w + num_first_winners_processed;
                        // 增加处理的第一次获胜者数量
                        num_first_winners_processed += 1;
                    }
                }
            }


            // 设置目标区域的新获胜者和新的激活神经元数量
            target_area._new_winners = new_winner_indices;
            target_area._new_w = target_area.w + num_first_winners_processed;

            // 调整 inputs_by_first_winner_index 的大小为处理的第一次获胜者数量
            inputs_by_first_winner_index.resize(num_first_winners_processed);

            // 初始化一个向量来存储每个输入索引的连接数量，初始值为0
            std::vector<int> num_connections_by_input_index(num_inputs, 0);

            // 遍历处理的每一个第一次获胜者
            for (int i = 0; i < num_first_winners_processed; i++)
            {
                // 创建一个向量，填充为 0 到 total_k-1 的值
                std::vector<int> indices(total_k);
                std::iota(indices.begin(), indices.end(), 0);

                // 获取要选择的数量，等于第一次获胜者输入的值
                int num_to_select = static_cast<int>(first_winner_inputs[i]);

                // 随机选择 num_to_select 个索引
                std::vector<int> input_indices;
                std::sample(indices.begin(), indices.end(), std::back_inserter(input_indices), num_to_select, rng);

                int total_so_far = 0;
                // 再次初始化每个输入索引的连接数量
                std::vector<int> num_connections_by_input_index(num_inputs, 0);

                // 遍历所有输入
                for (int j = 0; j < num_inputs; j++)
                {
                    // 遍历每个选择的输入索引
                    for (int w : input_indices)
                    {
                        // 如果 w 在当前输入索引范围内，则增加对应的连接数量
                        if (total_so_far + input_size_by_from_area_index[j] > w && w >= total_so_far)
                        {
                            ++num_connections_by_input_index[j];
                        }
                    }
                    // 更新总计值
                    total_so_far += input_size_by_from_area_index[j];
                }
                // 将连接数量添加到 inputs_by_first_winner_index 中
                inputs_by_first_winner_index[i] = num_connections_by_input_index;
            }

        }

        std::string target_area_name = target_area.name; // 获取目标区域的名称
        int num_inputs_processed = 0; // 初始化已处理的输入数量

        // 遍历所有来源区域
        for (const auto& from_area_name : from_areas)
        {
            // 获取来源区域的激活神经元数量和获胜者
            int from_area_w = area_by_name[from_area_name].w;
            std::vector<int> from_area_winners = area_by_name[from_area_name].winners;
            std::set<int> from_area_winners_set(from_area_winners.begin(), from_area_winners.end());

            // 引用来源区域到目标区域的连接矩阵
            auto& from_area_connectomes = connectomes[from_area_name];
            auto& the_connectome = from_area_connectomes[target_area_name];

            int change = num_first_winners_processed; // 需要增加的列数
            size_t num_rows = the_connectome.size(); // 获取行数
            size_t num_cols = the_connectome[0].size(); // 获取列数

            // 如果连接矩阵只有一列，则不增加列数
            if (num_cols == 1)
            {
                change--;
            }

            // 如果有新的获胜者，增加连接矩阵的列数
            if (num_first_winners_processed != 0)
            {
                for (auto& row : the_connectome)
                {
                    row.resize(row.size() + change, 0.0); // 在每行的末尾增加 change 个元素
                }
            }

            // 遍历所有新的获胜者
            for (int i = 0; i < num_first_winners_processed; ++i)
            {
                int total_in = inputs_by_first_winner_index[i][num_inputs_processed]; // 获取当前获胜者的输入总量
                std::vector<int> sample_indices;

                // 从来源区域的获胜者中随机选择 total_in 个索引
                std::sample(from_area_winners.begin(), from_area_winners.end(), std::back_inserter(sample_indices), total_in, rng);

                size_t num_rows = the_connectome.size(); // 获取行数
                size_t num_cols = num_rows > 0 ? the_connectome[0].size() : 0; // 获取列数（假设每一行都有相同的列数）

                // 更新连接矩阵，设置所选索引的连接值为 1.0
                for (int j : sample_indices)
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

            // 获取区域之间的 beta 值，如果未禁用可塑性则使用实际值，否则为 0.0
            double area_to_area_beta = (disable_plasticity ? 0.0 : target_area.beta_by_area[from_area_name]);

            // 更新连接矩阵，根据区域之间的 beta 值调整连接强度
            for (int i : target_area._new_winners)
            {
                for (int j : from_area_winners)
                {
                    the_connectome[j][i] *= 1.0 + area_to_area_beta;
                }
            }

            num_inputs_processed += 1; // 增加已处理的输入数量
        }


        // 遍历所有区域
        for (auto& pair : area_by_name)
        {
            const std::string& other_area_name = pair.first; // 获取其他区域的名称

            // 获取其他区域的连接矩阵
            auto& other_area_connectomes = connectomes[other_area_name];

            // 如果其他区域不在 from_areas 列表中
            if (std::find(from_areas.begin(), from_areas.end(), other_area_name) == from_areas.end())
            {
                // 获取其他区域到目标区域的连接矩阵
                auto& the_other_area_connectome = other_area_connectomes[target_area_name];

                int change = num_first_winners_processed; // 需要增加的列数
                size_t num_rows = the_other_area_connectome.size(); // 获取行数
                size_t num_cols = the_other_area_connectome[0].size(); // 获取列数（假设每一行都有相同的列数）

                // 如果列数为1，则减少 change 的值
                if (num_cols == 1)
                {
                    change--;
                }

                // 如果有需要增加的列
                if (num_first_winners_processed != 0)
                {
                    // 增加列，并初始化为 0.0
                    for (auto& row : the_other_area_connectome)
                    {
                        row.resize(row.size() + change, 0.0);
                    }
                }

                // 使用伯努利分布为新增的列赋值
                std::bernoulli_distribution bernoulli(p);
                for (size_t row = 0; row < the_other_area_connectome.size(); ++row)
                {
                    for (int col = target_area.w; col < target_area._new_w; ++col)
                    {
                        the_other_area_connectome[row][col] = bernoulli(rng) ? 1.0 : 0.0;
                    }
                }
            }

            // 获取目标区域到其他区域的连接矩阵
            std::unordered_map<std::string, std::vector<std::vector<double>>>& target_area_connectomes = connectomes[target_area_name];
            auto& the_target_area_connectome = target_area_connectomes[other_area_name];

            int change = num_first_winners_processed; // 需要增加的行数
            size_t num_rows = the_target_area_connectome.size(); // 获取行数
            size_t num_cols = the_target_area_connectome[0].size(); // 获取列数（假设每一行都有相同的列数）

            // 如果行数为1，则减少 change 的值
            if (num_rows == 1)
            {
                change--;
            }

            // 如果有需要增加的行
            if (num_first_winners_processed != 0)
            {
                // 添加新的行，并初始化为 0.0
                the_target_area_connectome.resize(the_target_area_connectome.size() + change, std::vector<double>(the_target_area_connectome[0].size(), 0.0));
            }

            // 使用二项分布为新增的行赋值
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

    // 添加新区域的函数
    void add_area(std::string area_name, int n, int k, double beta)
    {
        // 创建一个新的 Area 对象并添加到 area_by_name 字典中
        area_by_name[area_name] = Area(area_name, n, k, beta);

        // 初始化新的连接矩阵映射
        std::unordered_map<std::string, std::vector<std::vector<double>>> new_connectomes;

        // 遍历所有现有区域
        for (const auto& other_area_pair : area_by_name)
        {
            const std::string& other_area_name = other_area_pair.first; // 获取其他区域的名称
            Area& other_area = area_by_name[other_area_name]; // 获取其他区域的引用

            // 如果其他区域是显式区域，使用其 n 值，否则使用 1 作为大小
            int other_area_size = other_area.my_explicit ? other_area.n : 1;

            // 创建一个大小为 1 x other_area_size 的二维向量，初始值为 0
            std::vector<std::vector<double>> empty_matrix(1, std::vector<double>(other_area_size, 0));
            new_connectomes[other_area_name] = empty_matrix; // 插入到 new_connectomes 中

            // 如果其他区域不是当前添加的区域
            if (other_area_name != area_name)
            {
                // 创建一个大小为 other_area_size x 1 的二维向量，初始值为 0
                std::vector<std::vector<double>> empty_matrix(other_area_size, std::vector<double>(1, 0));
                connectomes[other_area_name][area_name] = empty_matrix; // 插入到 connectomes 中
            }

            // 更新其他区域和新添加区域之间的 beta 值
            other_area.beta_by_area[area_name] = other_area.beta;
            area_by_name[area_name].beta_by_area[other_area_name] = beta;
        }

        // 将新区域的连接矩阵添加到 connectomes 中
        connectomes[area_name] = new_connectomes;
    }


    // 更新两个区域之间塑性值的函数
    void update_plasticity(const std::string& from_area, const std::string& to_area, double new_beta)
    {
        // 将目标区域的 beta 值更新为新的 beta 值
        area_by_name[to_area].beta_by_area[from_area] = new_beta;
    }

    // 更新多个区域之间塑性值的函数
    void update_plasticities(const std::unordered_map<std::string, std::vector<std::pair<std::string, double>>>& area_update_map = {},
        const std::unordered_map<std::string, std::vector<std::pair<std::string, double>>>& stim_update_map = {})
    {
        // 遍历 area_update_map 中的每个键值对
        for (const auto& to_area_pair : area_update_map)
        {
            const std::string& to_area = to_area_pair.first; // 获取目标区域的名称
            const std::vector<std::pair<std::string, double>>& update_rules = to_area_pair.second; // 获取更新规则

            // 遍历更新规则
            for (const auto& rule : update_rules)
            {
                const std::string& from_area = rule.first; // 获取源区域的名称
                double new_beta = rule.second; // 获取新的 beta 值
                update_plasticity(from_area, to_area, new_beta); // 更新两个区域之间的塑性值
            }
        }

        // stim_update_map 目前未使用，保留为扩展用
    }

};

class ParserBrain : public Brain
{
public:
    // 词汇字典，映射字符串到规则集
    std::unordered_map<std::string, GenericRuleSet> lexeme_dict;

    // 所有区域的向量
    std::vector<std::string> all_areas;

    // 递归区域的向量
    std::vector<std::string> recurrent_areas;

    // 初始区域的向量
    std::vector<std::string> initial_areas;

    // 光纤状态映射，嵌套的 unordered_map 映射每对区域到一组整数状态
    std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_set<int>>> fiber_states;

    // 区域状态映射，映射每个区域到一组整数状态
    std::unordered_map<std::string, std::unordered_set<int>> area_states;

    // 激活的光纤映射，修改后的类型，映射每个区域到一组目标区域
    std::unordered_map<std::string, std::unordered_set<std::string>> activated_fibers;

    // 读出规则映射，映射每个区域到一组字符串规则
    std::unordered_map<std::string, std::vector<std::string>> readout_rules;


    // ParserBrain 构造函数
    ParserBrain(double p,
        const std::unordered_map<std::string, GenericRuleSet>& lexeme_dict = {}, // 词汇字典，默认值为空
        const std::vector<std::string>& all_areas = {}, // 所有区域的向量，默认值为空
        const std::vector<std::string>& recurrent_areas = {}, // 递归区域的向量，默认值为空
        const std::vector<std::string>& initial_areas = {}, // 初始区域的向量，默认值为空
        const std::unordered_map<std::string, std::vector<std::string>>& readout_rules = {}) // 读出规则映射，默认值为空
        : Brain(p), // 调用基类 Brain 的构造函数，传递参数 p
        all_areas(all_areas), // 使用参数 all_areas 初始化成员变量 all_areas
        recurrent_areas(recurrent_areas), // 使用参数 recurrent_areas 初始化成员变量 recurrent_areas
        initial_areas(initial_areas), // 使用参数 initial_areas 初始化成员变量 initial_areas
        readout_rules(readout_rules) // 使用参数 readout_rules 初始化成员变量 readout_rules
    {
        this->lexeme_dict = lexeme_dict; // 将参数 lexeme_dict 赋值给成员变量 lexeme_dict
        initialize_states(); // 调用初始化状态的函数
    }


    // 初始化状态的函数
    void initialize_states()
    {
        // 初始化 fiber_states，将每对区域之间的光纤状态设置为 0
        for (const auto& from_area : all_areas)
        {
            for (const auto& to_area : all_areas)
            {
                fiber_states[from_area][to_area].insert(0);
            }
        }

        // 初始化 area_states，将所有区域的状态设置为 0
        for (const auto& area : all_areas)
        {
            area_states[area].insert(0);
        }

        // 从 initial_areas 中移除状态 0
        for (const auto& area : initial_areas)
        {
            area_states[area].erase(0);
        }
    }

    // 应用光纤规则的函数
    void applyFiberRule(const Rule& rule)
    {
        // 检查规则是否为光纤规则
        if (rule.flag == FIBER_RULE)
        {
            // 如果规则的动作是 INHIBIT
            if (rule.action == INHIBIT)
            {
                // 在光纤状态中插入规则的索引，适用于两个区域之间
                fiber_states[rule.area1][rule.area2].insert(rule.index);
                fiber_states[rule.area2][rule.area1].insert(rule.index);
            }
            // 如果规则的动作是 DISINHIBIT
            else if (rule.action == DISINHIBIT)
            {
                // 从光纤状态中移除规则的索引，适用于两个区域之间
                fiber_states[rule.area1][rule.area2].erase(rule.index);
                fiber_states[rule.area2][rule.area1].erase(rule.index);
            }
        }
    }


    // 应用区域规则的函数
    void applyAreaRule(const Rule& rule)
    {
        // 检查规则是否为区域规则
        if (rule.flag == AREA_RULE)
        {
            // 如果规则的动作是 INHIBIT
            if (rule.action == INHIBIT)
            {
                // 在区域状态中插入规则的索引，适用于单个区域
                area_states[rule.area].insert(rule.index);
            }
            // 如果规则的动作是 DISINHIBIT
            else if (rule.action == DISINHIBIT)
            {
                // 从区域状态中移除规则的索引，适用于单个区域
                area_states[rule.area].erase(rule.index);
            }
        }
    }

    // 应用规则的函数，根据规则类型调用相应的处理函数
    bool applyRule(const Rule& rule)
    {
        // 检查规则是否为光纤规则
        if (rule.flag == FIBER_RULE)
        {
            applyFiberRule(rule); // 应用光纤规则
            return true; // 返回 true 表示成功应用规则
        }

        // 检查规则是否为区域规则
        if (rule.flag == AREA_RULE)
        {
            applyAreaRule(rule); // 应用区域规则
            return true; // 返回 true 表示成功应用规则
        }

        return false; // 返回 false 表示规则类型不匹配，未应用规则
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

    // 打印投射映射的函数
    void printProjectMap(const std::unordered_map<std::string, std::unordered_set<std::string>>& project_map)
    {
        // 检查投射映射是否为空
        if (project_map.empty())
        {
            std::cout << "The project map is empty." << std::endl; // 如果为空，打印提示信息
            return;
        }

        // 遍历 unordered_map 中的每个键值对
        for (const auto& pair : project_map)
        {
            const std::string& project = pair.first; // 获取键（项目名）
            const std::unordered_set<std::string>& tasks = pair.second; // 获取值（任务集合）

            std::cout << "Project: " << project << "\nTasks: {"; // 打印项目名和起始花括号

            // 遍历任务集合中的每个任务
            for (auto it = tasks.begin(); it != tasks.end(); ++it)
            {
                std::cout << *it; // 打印任务名
                if (std::next(it) != tasks.end())
                {
                    std::cout << ", "; // 如果不是最后一个任务，打印逗号和空格
                }
            }

            std::cout << "}\n"; // 打印结束花括号和换行符
        }
    }

    // 记录激活的光纤的函数
    void remember_fibers(const std::unordered_map<std::string, std::unordered_set<std::string>>& project_map)
    {
        // 可选：打印投射映射
        // printProjectMap(project_map);

        // 遍历投射映射中的每个键值对
        for (const auto& from_area_pair : project_map)
        {
            const std::string& from_area = from_area_pair.first; // 获取源区域的名称
            const std::unordered_set<std::string>& to_areas = from_area_pair.second; // 获取目标区域的集合

            // 将目标区域集合插入到激活光纤的映射中
            activated_fibers[from_area].insert(to_areas.begin(), to_areas.end());
        }
    }


    // 判断一个区域是否是递归区域的函数
    bool recurrent(const std::string& area) const
    {
        // 检查区域是否在 recurrent_areas 向量中
        return std::find(recurrent_areas.begin(), recurrent_areas.end(), area) != recurrent_areas.end();
    }

    // 获取投射映射的函数
    std::unordered_map<std::string, std::unordered_set<std::string>> getProjectMap() const
    {
        std::unordered_map<std::string, std::unordered_set<std::string>> proj_map; // 初始化投射映射

        // 遍历所有区域
        for (const auto& area1 : all_areas)
        {
            // 如果区域1的状态为空
            if (area_states.at(area1).empty())
            {
                // 遍历所有区域
                for (const auto& area2 : all_areas)
                {
                    // 跳过区域1和区域2都为 LEX 的情况
                    if (area1 == LEX && area2 == LEX)
                    {
                        continue;
                    }

                    // 如果区域2的状态为空
                    if (area_states.at(area2).empty())
                    {
                        // 如果光纤状态为空
                        if (fiber_states.at(area1).at(area2).empty())
                        {
                            // 如果区域1的胜者不为空，将区域2添加到投射映射中
                            if (!area_by_name.at(area1).winners.empty())
                            {
                                proj_map[area1].insert(area2);
                            }

                            // 如果区域2的胜者不为空，将区域2添加到投射映射中
                            if (!area_by_name.at(area2).winners.empty())
                            {
                                proj_map[area2].insert(area2);
                            }
                        }
                    }
                }
            }
        }

        return proj_map; // 返回投射映射
    }


    // 激活指定区域中的词汇
    void activateWord(const std::string& area_name, const std::string& word)
    {
        // 获取指定区域的引用
        Area& area = area_by_name[area_name];
        int k = area.k; // 获取区域中的集群大小
        int assembly_start = lexeme_dict[word].index * k; // 计算词汇在区域中的起始索引

        // 清空区域的胜者列表
        area.winners.clear();

        // 将词汇对应的索引添加到区域的胜者列表中
        for (int i = 0; i < k; ++i)
        {
            area.winners.push_back(assembly_start + i);
        }

        // 固定区域中的胜者集群
        area.fix_assembly();
    }

    // 解释指定区域中的集群为字符串
    std::string interpretAssemblyAsString(const std::string& area_name)
    {
        return getWord(area_name, 0.7); // 调用 getWord 函数，使用默认最小重叠度 0.7
    }


    // 获取指定区域中集群对应的词汇
    std::string getWord(const std::string& area_name, double min_overlap = 0.7)
    {
        // 如果指定区域的胜者列表为空，抛出异常
        if (area_by_name[area_name].winners.empty())
        {
            throw std::runtime_error("Cannot get word because no assembly in " + area_name);
        }

        // 将胜者列表转换为集合
        std::unordered_set<int> winners2(area_by_name[area_name].winners.begin(), area_by_name[area_name].winners.end());
        int area_k = area_by_name[area_name].k; // 获取区域中的集群大小
        int threshold = static_cast<int>(min_overlap * area_k); // 计算满足条件的最小重叠数

        // 创建一个 vector 来存储 lexeme_dict 的元素，并按索引排序
        std::vector<std::pair<std::string, GenericRuleSet>> vec(lexeme_dict.begin(), lexeme_dict.end());
        std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b)
        { return a.second.index < b.second.index; });

        // 遍历排序后的词汇字典
        for (const auto& lexeme_pair : vec)
        {
            const std::string& word = lexeme_pair.first; // 获取词汇
            int word_index = lexeme_pair.second.index; // 获取词汇的索引
            int word_assembly_start = word_index * area_k; // 计算词汇集群的起始索引

            // 将词汇对应的集群索引添加到集合中
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

            // 如果交集大小满足最小重叠数条件，返回词汇
            if (intersection_count >= threshold)
            {
                return word;
            }
        }

        return ""; // 如果没有满足条件的词汇，返回空字符串
    }


    // 获取激活光纤的函数
    std::unordered_map<std::string, std::unordered_set<std::string>> getActivatedFibers()
    {
        std::unordered_map<std::string, std::unordered_set<std::string>> pruned_activated_fibers; // 初始化修剪后的激活光纤映射

        // 遍历所有激活光纤的键值对
        for (const auto& from_area_pair : activated_fibers)
        {
            const std::string& from_area = from_area_pair.first; // 获取源区域的名称
            const std::unordered_set<std::string>& to_areas = from_area_pair.second; // 获取目标区域的集合

            // 遍历目标区域
            for (const auto& to_area : to_areas)
            {
                // 查找源区域的读出规则
                auto it = readout_rules.find(from_area);
                if (it != readout_rules.end())
                {
                    const std::vector<std::string>& targets = it->second; // 获取源区域的读出目标集合

                    // 如果目标区域在源区域的读出规则中
                    if (std::find(targets.begin(), targets.end(), to_area) != targets.end())
                    {
                        // 将目标区域插入到修剪后的激活光纤映射中
                        pruned_activated_fibers[from_area].insert(to_area);
                    }
                }
            }
        }

        return pruned_activated_fibers; // 返回修剪后的激活光纤映射
    }

};

// 定义 EnglishParserBrain 类，继承自 ParserBrain 类
class EnglishParserBrain : public ParserBrain
{
public:
    bool verbose; // 用于控制详细输出的布尔变量

    // 构造函数，初始化 EnglishParserBrain 对象
    EnglishParserBrain(double p, int non_LEX_n = 10000, int non_LEX_k = 100, int LEX_k = 20,
        double default_beta = 0.2, double LEX_beta = 1.0, double recurrent_beta = 0.05,
        double interarea_beta = 0.5, bool verbose = false)
        : ParserBrain(p, LEXEME_DICT, AREAS, RECURRENT_AREAS, { LEX, SUBJ, VERB }, ENGLISH_READOUT_RULES), verbose(verbose)
    {
        // 初始化 LEX 区域的大小
        int LEX_n = LEX_SIZE * LEX_k;
        add_explicit_area(LEX, LEX_n, LEX_k, default_beta); // 添加显式的 LEX 区域

        int DET_k = LEX_k;
        // 添加其他区域
        add_area(SUBJ, non_LEX_n, non_LEX_k, default_beta);
        add_area(OBJ, non_LEX_n, non_LEX_k, default_beta);
        add_area(VERB, non_LEX_n, non_LEX_k, default_beta);
        add_area(ADJ, non_LEX_n, non_LEX_k, default_beta);
        add_area(PREP, non_LEX_n, non_LEX_k, default_beta);
        add_area(PREP_P, non_LEX_n, non_LEX_k, default_beta);
        add_area(DET, non_LEX_n, DET_k, default_beta);
        add_area(ADVERB, non_LEX_n, non_LEX_k, default_beta);

        // 自定义塑性值的映射
        std::unordered_map<std::string, std::vector<std::pair<std::string, double>>> custom_plasticities;
        for (const auto& area : RECURRENT_AREAS)
        {
            custom_plasticities[LEX].emplace_back(area, LEX_beta); // 设置 LEX 与各递归区域的塑性值
            custom_plasticities[area].emplace_back(LEX, LEX_beta);
            custom_plasticities[area].emplace_back(area, recurrent_beta);
            for (const auto& other_area : RECURRENT_AREAS)
            {
                if (other_area == area)
                    continue;
                custom_plasticities[area].emplace_back(other_area, interarea_beta); // 设置递归区域之间的塑性值
            }
        }

        update_plasticities(custom_plasticities); // 更新塑性值
    }

    // 获取投射映射的函数
    std::unordered_map<std::string, std::unordered_set<std::string>> getProjectMap() const
    {
        auto proj_map = ParserBrain::getProjectMap(); // 调用基类的 getProjectMap 函数
        // 检查投射映射中 LEX 区域的投射数量是否超过 2
        if (proj_map.find(LEX) != proj_map.end() && proj_map[LEX].size() > 2)
        {
            throw std::runtime_error("Got that LEX projecting into many areas: " + std::to_string(proj_map[LEX].size())); // 抛出异常并打印相关信息
        }
        return proj_map; // 返回投射映射
    }

    // 获取指定区域中集群对应的词汇
    std::string getWord(const std::string& area_name, double min_overlap = 0.7)
    {
        auto word = ParserBrain::getWord(area_name, min_overlap); // 调用基类的 getWord 函数
        if (!word.empty())
        {
            return word; // 返回找到的词汇
        }
        return "<NON-WORD>"; // 如果没有找到词汇，返回 "<NON-WORD>"
    }
};


void parseHelper(ParserBrain& b, const std::string& sentence, double p, int LEX_k, int project_rounds, bool verbose, bool debug,
    const std::unordered_map<std::string, GenericRuleSet>& lexeme_dict,
    const std::vector<std::string>& all_areas,
    const std::vector<std::string>& explicit_areas,
    ReadoutMethod readout_method,
    const std::unordered_map<std::string, std::vector<std::string>>& readout_rules);

// 解析给定句子的函数
void parse(const std::string& sentence = "the dogs love a man quickly", const std::string& language = "English",
    double p = 0.1, int LEX_k = 20, int project_rounds = 20, bool verbose = false, bool debug = false,
    ReadoutMethod readout_method = ReadoutMethod::FIBER_READOUT)
{
    // 打印输入句子
    cout << "input sentence: " << sentence << endl;

    // 创建 EnglishParserBrain 对象 b，传入相关参数
    EnglishParserBrain b(p, 10000, 100, LEX_k, 0.2, 1.0, 0.05, 0.5, verbose);

    // 获取相关的全局变量
    auto lexeme_dict = LEXEME_DICT;
    auto all_areas = AREAS;
    auto explicit_areas = EXPLICIT_AREAS;
    auto readout_rules = ENGLISH_READOUT_RULES;

    // 调用辅助解析函数 parseHelper，传入解析相关参数
    parseHelper(b, sentence, p, LEX_k, project_rounds, verbose, debug,
        lexeme_dict, all_areas, explicit_areas, readout_method, readout_rules);
}


void read_out(ParserBrain& b, const std::string& area, const std::unordered_map<std::string, std::unordered_set<std::string>>& mapping, std::vector<std::vector<std::string>>& dependencies);

void parseHelper(ParserBrain& b, const std::string& sentence, double p, int LEX_k, int project_rounds, bool verbose, bool debug,
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
        std::cout << "Got dependencies: " << std::endl;
        for (const auto& dep : dependencies)
        {
            for (const auto& word : dep)
            {
                std::cout << word << " ";
            }
            std::cout << std::endl;
        }
        /*for (const auto& dep : dependencies)
        {
            cout << "{";
            for (const auto& word : dep)
            {
                std::cout <<"\"" << word << "\"" << ", ";
            }
            std::cout << "}, ";
        }*/
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
//
//int main()
//{
//    parse("cats chase mice");
//    parse("people run");
//    parse("dogs fly");
//    parse("dogs bite people");
//    parse("the cats chase the mice");
//    parse("a man saw a woman");
//    parse("big cats chase bad mice");
//    parse("cats chase mice quickly");
//    parse("cats chase mice in people");
//    parse("dogs are big");
//    parse("cats are dogs");
//    parse("the big cats of people chase a bad mice quickly");
//
//
//    return 0;
//}

int main()
{
    std::vector<std::string> sentences = {
        "cats chase mice",
        "people run",
        "dogs fly",
        "dogs bite people",
        "the cats chase the mice",
        "a man saw a woman",
        "big cats chase bad mice",
        "cats chase mice quickly",
        "cats chase mice in people",
        "dogs are big",
        "cats are dogs",
        "the big cats of people chase a bad mice quickly"
    };
    double total_time_per_word = 0;
    for (const auto& sentence : sentences)
    {
        auto start = std::chrono::high_resolution_clock::now();

        // 调用解析函数
        parse(sentence);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // 计算句子中的单词数量
        int word_count = std::count(sentence.begin(), sentence.end(), ' ') + 1;

        // 计算解析时长/句子单词数
        double time_per_word = elapsed.count() / word_count;

        total_time_per_word += time_per_word;

        //std::cout << "Sentence: \"" << sentence << "\"" << std::endl;
        std::cout << "解析用时: " << elapsed.count() << " 秒" << std::endl;
        std::cout << "平均每个单词用时: " << time_per_word << " 秒每词" << std::endl;

    }
    double time_per_word_average = total_time_per_word / sentences.size();
    std::cout << "每个句子平均每个单词用时的平均值: " << time_per_word_average << " 秒每词" << std::endl;
    return 0;
}
