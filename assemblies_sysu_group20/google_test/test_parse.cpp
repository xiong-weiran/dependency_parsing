#include <gtest/gtest.h>
#include "Parser.h"  // 假设Parser类定义在这个头文件中
#include <vector>
#include <string>

// 使用的词库
// 词汇字典，包含一系列词汇
std::vector<std::string> lexeme_dict = {
    "the", "a", "dogs", "cats", "mice", "people", "chase", "love", "bite",
    "of", "big", "bad", "run", "fly", "quickly", "in", "are", "man", "woman", "saw"
};

// 获取预期解析结果的函数
std::vector<std::vector<std::string>> getExpectedParseResult(const std::string& sentence) {
    // 定义一个模拟结果的映射，根据句子返回解析结果
    std::unordered_map<std::string, std::vector<std::vector<std::string>>> mock_results = {
        {"cats chase mice", {{"chase", "cats", "SUBJ"}, {"chase", "mice", "OBJ"}}},
        {"people run", {{"run", "people", "SUBJ"}}},
        {"dogs fly", {{"fly", "dogs", "SUBJ"}}},
        {"dogs bite people", {{"bite", "dogs", "SUBJ"}, {"bite", "people", "OBJ"}}},
        {"the cats chase the mice", {{"chase", "cats", "SUBJ"}, {"chase", "mice", "OBJ"}, {"cats", "the", "DET"}, {"mice", "the", "DET"}}},
        {"a man saw a woman", {{"saw", "man", "SUBJ"}, {"saw", "woman", "OBJ"}, {"man", "a", "DET"}, {"woman", "a", "DET"}}},
        {"big cats chase bad mice", {{"chase", "cats", "SUBJ"}, {"chase", "mice", "OBJ"}, {"cats", "big", "ADJ"}, {"mice", "bad", "ADJ"}}},
        {"cats chase mice quickly", {{"chase", "cats", "SUBJ"}, {"chase", "quickly", "ADVERB"}, {"chase", "mice", "OBJ"}}},
        {"cats chase mice in people", {{"chase", "cats", "SUBJ"}, {"chase", "mice", "OBJ"}, {"mice", "people", "PREP_P"}, {"people", "in", "PREP"}}},
        {"dogs are big", {{"are", "big", "ADJ"}, {"are", "dogs", "SUBJ"}}},
        {"cats are dogs", {{"are", "cats", "SUBJ"}, {"are", "dogs", "OBJ"}}},
        {"the big cats of people chase a bad mice quickly", {{"chase", "cats", "SUBJ"}, {"chase", "quickly", "ADVERB"}, {"chase", "mice", "OBJ"}, {"cats", "people", "PREP_P"}, {"cats", "big", "ADJ"}, {"cats", "the", "DET"}, {"people", "of", "PREP"}, {"mice", "a", "DET"}, {"mice", "bad", "ADJ"}}}
    };
    return mock_results[sentence]; // 根据输入句子返回对应的解析结果
}



// 比较两个二维字符串向量是否相等（忽略外部向量的顺序）
bool areVectorsEqual(std::vector<std::vector<std::string>> v1, std::vector<std::vector<std::string>> v2) {
    // 定义一个哈希函数，用于计算每个内部向量的哈希值
    auto vecHash = [](const std::vector<std::string>& vec) {
        std::string hashStr;
        // 将内部向量的每个字符串拼接在一起，中间使用"#"作为分隔符，确保唯一性
        for (const auto& str : vec) {
            hashStr += str + "#";
        }
        // 返回拼接后的字符串的哈希值
        return std::hash<std::string>{}(hashStr);
    };

    // 定义两个unordered_set，分别存储两个二维向量的哈希值
    std::unordered_set<size_t> set1, set2;
    // 将第一个二维向量的每个内部向量的哈希值插入set1
    for (const auto& vec : v1) {
        set1.insert(vecHash(vec));
    }
    // 将第二个二维向量的每个内部向量的哈希值插入set2
    for (const auto& vec : v2) {
        set2.insert(vecHash(vec));
    }

    // 比较两个set是否相等，如果相等则返回true，否则返回false
    return set1 == set2;
}

// 测试解析器是否能够解析给定的句子
void TestSentenceParsing(const std::string& sentence) {
    // 调用解析方法，获取解析结果
    std::vector<std::vector<std::string>> result = parse(sentence);

    // 获取预期结果
    std::vector<std::vector<std::string>> expected = getExpectedParseResult(sentence);

    // 如果结果和预期不相等，打印解析结果
    if (!areVectorsEqual(expected, result)) {
        cout << "result:" << endl;
        for (const auto& dep : result) {
            for (const auto& word : dep) {
                std::cout << word << " ";
            }
            std::cout << std::endl;
        }
    }

    // 使用 Google Test 的 EXPECT_TRUE 宏断言结果和预期相等，如果不相等，输出错误信息
    EXPECT_TRUE(areVectorsEqual(expected, result)) << "Sentence failed to parse correctly: " << sentence;
}


// 测试简单句
TEST(ParserTest, SimpleSentences) {
    TestSentenceParsing("cats chase mice");
    TestSentenceParsing("people run");
    TestSentenceParsing("dogs fly");
    TestSentenceParsing("dogs bite people");

    // TestSentenceParsing("the dogs chase the cats");
    // TestSentenceParsing("a man saw a woman");
    // TestSentenceParsing("people love mice");
}

TEST(ParserTest, SentencesWithDet) {
    TestSentenceParsing("the cats chase the mice");
    TestSentenceParsing("a man saw a woman");

    // TestSentenceParsing("the dogs chase the cats");
    // TestSentenceParsing("a man saw a woman");
    // TestSentenceParsing("people love mice");
}


// 测试带有形容词的句子
TEST(ParserTest, SentencesWithAdjectives) {
    TestSentenceParsing("big cats chase bad mice");

    // TestSentenceParsing("the big dogs bite the bad cats");
    // TestSentenceParsing("a big man saw a bad woman");
}

// 测试带有副词的句子
TEST(ParserTest, SentencesWithAdverbs) {
    TestSentenceParsing("cats chase mice quickly");

    // TestSentenceParsing("the dogs run quickly");
    // TestSentenceParsing("the cats fly quickly");
}

// 测试带有介词的句子
TEST(ParserTest, SentencesWithPrepositions) {
    TestSentenceParsing("cats chase mice in people");
    //TestSentenceParsing("the dogs chase the cats in the house");
    //TestSentenceParsing("a man saw a woman in the park");
}

// 测试带有连系动词的句子
TEST(ParserTest, SentencesWithCopulas) {
    TestSentenceParsing("dogs are big");
    TestSentenceParsing("cats are dogs");


    // TestSentenceParsing("the dogs are big");
    // TestSentenceParsing("the man is bad");
}

// 测试更复杂的句子
TEST(ParserTest, ComplexSentences) {
    TestSentenceParsing("the big cats of people chase a bad mice quickly");

    
    //TestSentenceParsing("the big dogs quickly chase the bad cats");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
