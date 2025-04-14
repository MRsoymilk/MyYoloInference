#include "metadata.h"

#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "definitions.h"

namespace my_yolo {

Metadata::Metadata() {}

Metadata::~Metadata() {}

std::string Metadata::readFileTail(const std::string& file_path, const size_t& tail_size) {
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
  if (!m_data["batch"].empty()) {
    m_batch = std::stoi(m_data["batch"]);
  }

  // get stride
  if (!m_data["stride"].empty()) {
    m_stride = std::stoi(m_data["stride"]);
  }

  // get task
  if (!m_data["task"].empty()) {
    std::string task = m_data["task"];
    m_task = TASK::UNKNOWN;
    if ("segment" == task) {
      m_task = TASK::SEGMENT;
    } else if ("detect" == task) {
      m_task = TASK::DETECT;
    } else if ("classify" == task) {
      m_task = TASK::CLASSIFY;
    } else if ("pose" == task) {
      m_task = TASK::POSE;
    } else if ("obb" == task) {
      m_task = TASK::OBB;
    }
  }

  // get imgsz
  std::smatch match;
  if (!m_data["imgsz"].empty()) {
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
  }

  // get names
  if (!m_data["names"].empty()) {
    std::string names = m_data["names"];
    std::vector<std::string> list;
    // remove '{', '}'
    names = names.substr(1, names.size() - 2);
    std::stringstream ss(names);
    std::string item;

    while (std::getline(ss, item, ',')) {
      item.erase(std::remove(item.begin(), item.end(), '\''), item.end());
      item.erase(0, item.find_first_not_of(" "));

      size_t pos = item.find(": ");
      if (pos != std::string::npos) {
        item = item.substr(pos + 2);  // 取 `:` 后面的部分
      }

      list.push_back(item);
    }
    m_names = list;
  }

  // get keypoints
  if (!m_data["kpt_shape"].empty()) {
    std::regex re_kpt(R"(\[(\d+),\s*(\d+)\])");
    if (std::regex_match(m_data["kpt_shape"], match, re_kpt)) {
      int nums = std::stoi(match[1].str());
      int dims = std::stoi(match[2].str());
      std::cout << "Nums: " << nums << ", Dims: " << dims << std::endl;
      m_keypoint.num = nums;
      m_keypoint.dim = dims;
    } else {
      std::cout << "Invalid format!" << std::endl;
    }
  }
}

int Metadata::getBatch() { return m_batch; }

int Metadata::getStride() { return m_stride; }

std::vector<std::string> Metadata::getNames() { return m_names; }

TASK Metadata::getTask() { return m_task; }

IMGSZ Metadata::getImgsz() { return m_imgsz; }

KEYPOINT Metadata::getKeypoint() { return m_keypoint; }

uint32_t Metadata::decodeULEB128(int& jmp, const std::string& data, const int& begin) {
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

std::string Metadata::extract(int& begin, const std::string& data, const std::string& key) {
  int JMP = 2;
  int size = decodeULEB128(JMP, data, begin + 1);
  if (size) {
    std::string str = data.substr(begin + JMP, size);
    std::string val = str.substr(2 + key.size() + JMP);
    begin += JMP + size;
    return val;
  }
  return "";
}

std::string Metadata::getMetadata(const std::string& key) {
  if (m_data.find(key) != m_data.end()) {
    return m_data[key];
  }
  return "";
}

}  // namespace my_yolo
