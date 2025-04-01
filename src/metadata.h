#ifndef METADATA_H
#define METADATA_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace my_yolo {

class Metadata {
 public:
  struct IMGSZ {
    int w;
    int h;
  };

  enum class TASK { UNKNOWN = 0, DETECT, SEGMENT };
  Metadata();
  ~Metadata();
  std::string readFileTail(const std::string &file_path,
                           const size_t &tail_size = 2048);
  void analysis(const std::string &data = "");

 public:
  int getBatch();
  int getStride();
  std::vector<std::string> getNames();
  TASK getTask();
  IMGSZ getImgsz();

 private:
  uint32_t decodeULEB128(int &jmp, const std::string &data, const int &begin);
  std::string extract(int &begin, const std::string &data,
                      const std::string &key);

  std::string getMetadata(const std::string &key);

 private:
  std::vector<std::string> v_key{"description", "author", "date",   "version",
                                 "license",     "docs",   "stride", "task",
                                 "batch",       "imgsz",  "names",  "args"};
  std::unordered_map<std::string, std::string> m_data;
  int m_batch;
  int m_stride;
  TASK m_task;
  IMGSZ m_imgsz;
  std::vector<std::string> m_names;
};
}  // namespace my_yolo

#endif  // METADATA_H
