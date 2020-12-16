#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <map>
#include <sstream>
#include <vector>
#include <set>
#include <algorithm>
#include <regex>
#include <iomanip>
using namespace std;

int total_reviews = 0;
int total_positive_reviews = 0; // 긍정 리뷰의 총 개수
int total_negative_reviews = 0; // 부정 리뷰의 총 개수

int total_positive_words = 0; // 긍정 리뷰에 출현한 단어의 총 개수(중복허용)
int total_negative_words = 0; // 부정 리뷰에 출현한 단어의 총 개수(중복허용)

map<string, pair<int, int>> words;             // 특정 단어의 (긍정 리뷰에 출현 횟수, 부정 리뷰 출현 횟수) 쌍
map<string, pair<double, double>> probability; // 특정 단어의 (긍정 리뷰일 확률, 부정 리뷰일 확률) 쌍

vector<int> correct;     // 테스트 데이터의 정답
vector<int> predictions; // 테스트 데이터의 예측 답
vector<pair<double, double>> predict_probability;

vector<string> josa_and_eomi;

void train(string filename);
void get_josa_and_eomi();
void calculate_probability();

void test(string filename);
int predict(vector<string> term);

void calculate_accuracy();

void print_info();

bool compare_length(const string &a, const string &b);
template <typename T> bool compare_value(const pair<string, T> &a, const pair<string, T> &b);
template <typename T> bool compare_first_value(const pair<string, pair<T, T>> &a, const pair<string, pair<T, T>> &b);
template <typename T> bool compare_second_value(const pair<string, pair<T, T>> &a, const pair<string, pair<T, T>> &b);

regex pattern("[^0-9a-zA-z가-힣]+");

void train(string filename)
{
    ifstream trainfile;
    trainfile.open(filename);

    string line;
    stringstream lineStream;
    get_josa_and_eomi();
    
    while (!trainfile.eof())
    {
        getline(trainfile, line);
        if (line == "")
            continue;

        int category = line.at(0) - '0';

        if (category == 1) total_positive_reviews++;
        else total_negative_reviews++;

        line.erase(0, 2);
        line = regex_replace(line, pattern, " ");

        lineStream.clear();
        lineStream.str(line);

        string word;
        set<string> temp_words;
        vector<string> uni_bigram_words; 
        while (!lineStream.eof())
        {
            getline(lineStream, word, ' ');

            if (word == "")
                continue;

            for (auto i : josa_and_eomi)
            {
                int len_word = word.length();
                int len_i = i.length();

                if (len_word <= len_i) continue;

                if (word.find(i, len_word-len_i) != string::npos)
                {
                    word.erase(len_word-len_i, len_i);
                    break;
                }
            }
            temp_words.insert(word);
            uni_bigram_words.push_back(word);
        }

        int size = uni_bigram_words.size();
        for (int i = 0; i < size - 1; i++)
        {
            uni_bigram_words.push_back(uni_bigram_words[i] + " " + uni_bigram_words[i + 1]);
            temp_words.insert(uni_bigram_words.back()); // 현재 리뷰에서 출현한 word 집합
        }

        if (category == 1)
        {
            for (auto word : uni_bigram_words)
            {
                if (words.find(word) != words.end())
                    words[word].first++;
                else
                    words[word] = make_pair(1, 0);
                total_positive_words++;
            }
        }
        else
        {
            for (auto word : uni_bigram_words)
            {
                if (words.find(word) != words.end())
                    words[word].second++;
                else
                    words[word] = make_pair(0, 1);
                total_negative_words++;
            }
        }
    }
    total_reviews = total_positive_reviews + total_negative_reviews;
    trainfile.close();
    calculate_probability();
}

void get_josa_and_eomi()
{   
    ifstream file;
    file.open("josa_and_eomi.txt");

    string line;
    while (!file.eof())
    {
        getline(file, line);
        if (line == "")
            continue;

        josa_and_eomi.push_back(line);
    }
    file.close();

    sort(josa_and_eomi.begin(), josa_and_eomi.end(), compare_length);
}

void calculate_probability()
{
    for (auto it : words)
    {
        string word = it.first;
        double count_positive = double((it.second).first);
        double count_negative = double((it.second).second);

        probability[word].first = (count_positive + 1.0) / double(total_positive_words + words.size());
        probability[word].second = (count_negative + 1.0) / double(total_negative_words + words.size());
    }
}

void test(string filename)
{
    ifstream testfile;
    testfile.open(filename);

    ofstream result("prediction.txt");

    stringstream lineStream;
    string line;
    string word;

    while (!testfile.eof())
    {
        getline(testfile, line);
        if (line == "")
            continue;

        int category = line.at(0) - '0';

        line.erase(0, 2);
        line = regex_replace(line, pattern, " ");
        
        lineStream.clear();
        lineStream.str(line);

        vector<string> uni_bigram_words;
        while (!lineStream.eof())
        {
            getline(lineStream, word, ' ');

            if (word == "")
                continue;

            for (auto i : josa_and_eomi)
            {
                int len_word = word.length();
                int len_i = i.length();

                if (len_word <= len_i) continue;

                if (word.find(i, len_word-len_i) != string::npos)
                {
                    word.erase(len_word-len_i, len_i);
                    break;
                }
            }
            uni_bigram_words.push_back(word);
        }

        int size = uni_bigram_words.size();
        for (int i = 0; i < size - 1; i++)
        {
            uni_bigram_words.push_back(uni_bigram_words[i] + " " + uni_bigram_words[i + 1]);
        }

        if (!uni_bigram_words.empty())
        {
            correct.push_back(category);
            predictions.push_back(predict(uni_bigram_words));
            result << predictions.back() << '\t' << predict_probability.back().first << '\t' << predict_probability.back().second << '\t' << line << endl;
            uni_bigram_words.clear();
        }
    }
    result.close();
    calculate_accuracy();
}

int predict(vector<string> term)
{
    double positive_prob = 1;
    double negative_prob = 1;
    for (int i = 0; i < term.size(); i++)
    {
        if (probability.find(term[i]) == probability.end()) continue;
        positive_prob *= probability[term[i]].first;
        negative_prob *= probability[term[i]].second;
    }
    positive_prob *= (double)total_positive_reviews/(double)total_reviews;
    negative_prob *= (double)total_negative_reviews/(double)total_reviews;
    
    predict_probability.push_back(make_pair(positive_prob, negative_prob));

    if (positive_prob > negative_prob)
        return 1; // 긍정
    else if (positive_prob < negative_prob)
        return 0; // 부정
    else
        return 2; // 중립 또는 정보가 부족해서 알수없음
}

void calculate_accuracy()
{
    int count_correct = 0;
    for (int i = 0; i < predictions.size(); i++)
    {
        if (predictions[i] == correct[i])
            count_correct++;
    }
    print_info();
    cout << count_correct << "/" << predictions.size() << "개의 감정을 맞췄습니다." << endl;
    cout << "정확도 : " << ((double)count_correct / (double)predictions.size()) * 100 << '%' << endl;
}

bool compare_length(const string &a, const string &b)
{
    return a.length() > b.length();
}

template <typename T>
bool compare_value(const pair<string, T> &a, const pair<string, T> &b)
{
    return a.second > b.second;
}

template <typename T>
bool compare_first_value(const pair<string, pair<T, T>> &a, const pair<string, pair<T, T>> &b)
{
    return a.second.first > b.second.first;
}

template <typename T>
bool compare_second_value(const pair<string, pair<T, T>> &a, const pair<string, pair<T, T>> &b)
{
    return a.second.second > b.second.second;
}

void print_top10_word_count()
{
    vector<pair<string, pair<int, int>>> v(words.begin(), words.end());
    
    sort(v.begin(), v.end(), compare_first_value<int>);

    cout << "***** 긍정 리뷰에서 많이 출현한 단어 TOP 10 *****" << endl;
    cout << setw(9) << "순위" << setw(20) << "단어" << '\t' << setw(35) << "긍정 리뷰에 출현한 횟수" << setw(25) << "부정 리뷰에 출현한 횟수" << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << setw(7) << i+1 << setw(20) << v[i].first << '\t' << setw(23) << v[i].second.first << setw(20) << v[i].second.second << endl;
    }
    cout << endl;

    sort(v.begin(), v.end(), compare_second_value<int>);

    cout << "***** 부정 리뷰에서 많이 출현한 단어 TOP 10 *****" << endl;
    cout << setw(9) << "순위" << setw(20) << "단어" << '\t' << setw(35) << "부정 리뷰에 출현한 횟수" << setw(25) << "긍정 리뷰에 출현한 횟수" << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << setw(7) << i+1 << setw(20) << v[i].first << '\t' << setw(23) << v[i].second.second << setw(20) << v[i].second.first << endl;
    }
    cout << endl;
}

void print_top10_word_probability()
{
    vector<pair<string, pair<double, double>>> v(probability.begin(), probability.end());
    sort(v.begin(), v.end(), compare_first_value<double>);

    cout << "***** 긍정 리뷰에 출현한 확률이 높은 단어 TOP 10 *****" << endl;
    cout << setw(9) << "순위" << setw(20) << "단어" << '\t' << setw(35) << "긍정 리뷰에 출현한 확률" << setw(25) << "부정 리뷰에 출현할 확률" << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << setw(7) << i+1 << setw(20) << v[i].first << '\t' << setw(23) << v[i].second.first << setw(20) << v[i].second.second << endl;
    }
    cout << endl;

    sort(v.begin(), v.end(), compare_second_value<double>);
    
    cout << "***** 부정 리뷰에 출현한 확률이 높은 단어 TOP 10 *****" << endl;
    cout << setw(9) << "순위" << setw(20) << "단어" << '\t' << setw(35) << "부정 리뷰에 출현한 확률" << setw(25) << "긍정 리뷰에 출현할 확률" << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << setw(7) << i+1 << setw(20) << v[i].first << '\t' << setw(23) << v[i].second.second << setw(20) << v[i].second.first << endl;
    }
    cout << endl;
}

void print_info()
{
    cout.setf(ios::left);
    cout << "리뷰의 총 개수(긍정 리뷰+부정 리뷰) : " << total_reviews << "(" << total_positive_reviews << "+" << total_negative_reviews << ")" << endl;
    cout << "긍정 리뷰의 단어 총 개수 : " << total_positive_words << endl;
    cout << "부정 리뷰의 단어 총 개수 : " << total_negative_words << endl;
    cout << endl;
    print_top10_word_count();
    print_top10_word_probability();
}

int main()
{
    train("train_data.txt");
    test("test_data.txt");
}