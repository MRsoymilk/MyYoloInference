#include "metadata.h"

#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

namespace my_yolo {

Metadata::Metadata() {}

Metadata::~Metadata() {}

std::string Metadata::readFileTail(const std::string& file_path,
                                   const size_t& tail_size) {
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "Failed to open the file!" << std::endl;
    return "";
  }

  size_t file_size = file.tellg();
  size_t tail = tail_size;
  if (file_size < tail_size) {
    tail = file_size;
  }

  file.seekg(file_size - tail, std::ios::beg);

  std::string file_tail(tail, '\0');
  file.read(&file_tail[0], tail);
  file.close();

  int loc = file_tail.find("description");
  if (loc != std::string::npos) {
    int start = loc - 4;
    return file_tail.substr(start);
  }

  std::cerr << "No Description Found!" << std::endl;
  std::cerr << "Please set classes and model size manullay!(or increase "
               "read size and try again)"
            << std::endl;
  return "";
}

void Metadata::analysis(const std::string& data) {
  int begin = 0;

  for (size_t i = 0; i < v_key.size(); ++i) {
    m_data[v_key[i]] = extract(begin, data, v_key[i]);
  }

  // get batch
  m_batch = std::stoi(m_data["batch"]);

  // get stride
  m_stride = std::stoi(m_data["stride"]);

  // get task
  std::string task = m_data["task"];
  m_task = TASK::UNKNOWN;
  if ("segment" == task) {
    m_task = TASK::SEGMENT;
  } else if ("detect" == task) {
    m_task = TASK::DETECT;
  }

  // get imgsz
  std::smatch match;
  std::regex re_imgsz(R"(\[(\d+),\s*(\d+)\])");
  if (std::regex_match(m_data["imgsz"], match, re_imgsz)) {
    int height = std::stoi(match[1].str());
    int width = std::stoi(match[2].str());
    std::cout << "Height: " << height << ", Width: " << width << std::endl;
    m_imgsz.w = width;
    m_imgsz.h = height;
  } else {
    std::cout << "Invalid format!" << std::endl;
  }

  // get names
  std::regex re_names(R"('([^']+)')");
  auto iter = m_data["names"].cbegin();
  std::vector<std::string> list;
  while (std::regex_search(iter, m_data["names"].cend(), match, re_names)) {
    std::cout << match[1] << std::endl;
    list.push_back(match[1].str());
    iter = match[0].second;
  }
  m_names = list;
}

int Metadata::getBatch() { return m_batch; }

int Metadata::getStride() { return m_stride; }

std::vector<std::string> Metadata::getNames() { return m_names; }

Metadata::TASK Metadata::getTask() { return m_task; }

Metadata::IMGSZ Metadata::getImgsz() { return m_imgsz; }

uint32_t Metadata::decodeULEB128(int& jmp, const std::string& data,
                                 const int& begin) {
  uint32_t result = 0;
  int shift = 0;
  int pos = begin;

  while (pos < data.size()) {
    uint8_t byte = static_cast<uint8_t>(data[pos]);
    pos++;
    // extract the lower 7 bits and merge
    result |= (byte & 0x7F) << shift;
    // break if MSB is 0
    if ((byte & 0x80) == 0) {
      break;
    }
    shift += 7;
    jmp += 1;
  }

  return result;
}

std::string Metadata::extract(int& begin, const std::string& data,
                              const std::string& key) {
  int JMP = 2;
  int size = decodeULEB128(JMP, data, begin + 1);
  std::string str = data.substr(begin + 2, size);
  std::string val = str.substr(JMP + key.size() + JMP);
  begin += JMP + size;
  return val;
}

std::string Metadata::getMetadata(const std::string& key) {
  if (m_data.find(key) != m_data.end()) {
    return m_data[key];
  }
  return "null";
}

}  // namespace my_yolo
