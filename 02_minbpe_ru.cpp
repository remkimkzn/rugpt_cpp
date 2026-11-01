// Сборка g++ -O3 02_minbpe_ru.cpp -o 03_minbpe_ru ./02_minbpe_ru

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// Токенизатор уровня байтов для русского языка (UTF-8)
struct RussianBPE {
  // Правила слияния пара токенов (id1, id2) -> новый id токена (>=256)
  std::vector<std::pair<std::pair<int, int>, int>> merges;
  // Словарь для декодирования id -> строка байтов utf-8
  std::map<int, std::string> vocab;

  RussianBPE() {
    // Базовый алфавит 256 сырых байтов (0...255)
    for (int i = 0; i < 256; ++i) {
      vocab[i] = std::string(1, static_cast<char>(i));
    }
  }

  // Обучение словаря BPE на русском тексте
  void train(const std::string& text, int num_merges) {
    std::vector<int> ids;
    for (unsigned char c : text) ids.push_back(static_cast<int>(c));

    for (int m = 0; m < num_merges; ++m) {
      std::map<std::pair<int, int>, int> counts;
      for (size_t i = 0; i + 1 < ids.size(); ++i) {
        counts[{ids[i], ids[i + 1]}]++;
      }
      if (counts.empty()) break;

      // Ищем частую пару
      auto best_pair = counts.begin()->first;
      int max_count = 0;
      for (const auto& kv : counts) {
        if (kv.second > max_count) {
          max_count = kv.second;
          best_pair = kv.first;
        }
      }
      if (max_count < 2) break;  // больше нет повторяющихся пар

      int new_id = 256 + m;
      merges.push_back({best_pair, new_id});
      vocab[new_id] = vocab[best_pair.first] + vocab[best_pair.second];

      // заменяем пару на new_id во всем тексте
      std::vector<int> next_ids;
      for (size_t i = 0; i < ids.size(); ++i) {
        if (i + 1 < ids.size() && ids[i] == best_pair.first &&
            ids[i + 1] == best_pair.second) {
          next_ids.push_back(new_id);
          ++i;
        } else {
          next_ids.push_back(ids[i]);
        }
      }
      ids = std::move(next_ids);
    }
  }

  // Кодирование строки в список ID токенов
  std::vector<int> encode(const std::string& text) const {
    std::vector<int> ids;
    for (unsigned char c : text) ids.push_back(static_cast<int>(c));

    for (const auto& rule : merges) {
      int p1 = rule.first.first;
      int p2 = rule.first.second;
      int new_id = rule.second;

      std::vector<int> next_ids;
      for (size_t i = 0; i < ids.size(); ++i) {
        if (i + 1 < ids.size() && ids[i] == p1 && ids[i + 1] == p2) {
          next_ids.push_back(new_id);
          ++i;
        } else {
          next_ids.push_back(ids[i]);
        }
      }
      ids = std::move(next_ids);
    }
    return ids;
  }

  // Декодирование списка ID обратно в русский текст UTF-8
  std::string decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
      auto it = vocab.find(id);
      if (it != vocab.end()) out += it->second;
    }
    return out;
  }
};

int main() {
  std::cout << "=== ЭТАП 2: minbpe_ru.cpp}