// -------------xwr添加，对应brain_parser.py开头的常量和创建词汇表用到的类定义和全局函数------------------
#pragma once
#include <chrono>
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


