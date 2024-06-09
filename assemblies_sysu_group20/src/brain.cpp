#include "brain.h"

// 根据词性调用不同的初始词汇表的函数

// 生成名词规则集
GenericRuleSet generic_noun(int index)
{
    return {
        index,
        // 预规则
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
         // 后规则
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
          {INHIBIT, VERB, ADJ, 0}}
    };
}

// 生成及物动词规则集
GenericRuleSet generic_trans_verb(int index)
{
    return {
        index,
        // 预规则
        {{DISINHIBIT, LEX, VERB, 0},
         {DISINHIBIT, VERB, SUBJ, 0},
         {DISINHIBIT, VERB, ADVERB, 0},
         {DISINHIBIT, ADVERB, 1}},
         // 后规则
         {{INHIBIT, LEX, VERB, 0},
          {DISINHIBIT, OBJ, 0},
          {INHIBIT, SUBJ, 0},
          {INHIBIT, ADVERB, 0},
          {DISINHIBIT, PREP_P, VERB, 0}}
    };
}

// 生成不及物动词规则集
GenericRuleSet generic_intrans_verb(int index)
{
    return {
        index,
        // 预规则
        {{DISINHIBIT, LEX, VERB, 0},
         {DISINHIBIT, VERB, SUBJ, 0},
         {DISINHIBIT, VERB, ADVERB, 0},
         {DISINHIBIT, ADVERB, 1}},
         // 后规则
         {{INHIBIT, LEX, VERB, 0},
          {INHIBIT, SUBJ, 0},
          {INHIBIT, ADVERB, 0},
          {DISINHIBIT, PREP_P, VERB, 0}}
    };
}

// 生成系动词规则集
GenericRuleSet generic_copula(int index)
{
    return {
        index,
        // 预规则
        {{DISINHIBIT, LEX, VERB, 0},
         {DISINHIBIT, VERB, SUBJ, 0}},
         // 后规则
         {{INHIBIT, LEX, VERB, 0},
          {DISINHIBIT, OBJ, 0},
          {INHIBIT, SUBJ, 0},
          {DISINHIBIT, ADJ, VERB, 0}}
    };
}

// 生成副词规则集
GenericRuleSet generic_adverb(int index)
{
    return {
        index,
        // 预规则
        {{DISINHIBIT, ADVERB, 0},
         {DISINHIBIT, LEX, ADVERB, 0}},
         // 后规则
         {{INHIBIT, LEX, ADVERB, 0},
          {INHIBIT, ADVERB, 1}}
    };
}

// 生成限定词规则集
GenericRuleSet generic_determinant(int index)
{
    return {
        index,
        // 预规则
        {{DISINHIBIT, DET, 0},
         {DISINHIBIT, LEX, DET, 0}},
         // 后规则
         {{INHIBIT, LEX, DET, 0},
          {INHIBIT, VERB, ADJ, 0}}
    };
}

// 生成形容词规则集
GenericRuleSet generic_adjective(int index)
{
    return {
        index,
        // 预规则
        {{DISINHIBIT, ADJ, 0},
         {DISINHIBIT, LEX, ADJ, 0}},
         // 后规则
         {{INHIBIT, LEX, ADJ, 0},
          {INHIBIT, VERB, ADJ, 0}}
    };
}

// 生成介词规则集
GenericRuleSet generic_preposition(int index)
{
    return {
        index,
        // 预规则
        {{DISINHIBIT, PREP, 0},
         {DISINHIBIT, LEX, PREP, 0}},
         // 后规则
         {{INHIBIT, LEX, PREP, 0},
          {DISINHIBIT, PREP_P, 0},
          {INHIBIT, LEX, SUBJ, 1},
          {INHIBIT, LEX, OBJ, 1},
          {INHIBIT, DET, SUBJ, 1},
          {INHIBIT, DET, OBJ, 1},
          {INHIBIT, ADJ, SUBJ, 1},
          {INHIBIT, ADJ, OBJ, 1}}
    };
}

// 定义大脑区域的常量
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

const int LEX_SIZE = 20; // LEX 区域的大小

// 定义动作类型的常量
const std::string DISINHIBIT = "DISINHIBIT";
const std::string INHIBIT = "INHIBIT";
const std::string ACTIVATE_ONLY = "ACTIVATE_ONLY";
const std::string CLEAR_DET = "CLEAR_DET";

// 定义区域的向量
const std::vector<std::string> AREAS = { LEX, DET, SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P };

// 定义显式区域的向量
const std::vector<std::string> EXPLICIT_AREAS = { LEX };

// 定义递归区域的向量
const std::vector<std::string> RECURRENT_AREAS = { SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P };
