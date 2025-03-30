#ifndef MY_YOLO_INFERENCE_H
#define MY_YOLO_INFERENCE_H

#include "global.h"

namespace my_yolo {
class MYYOLOINFERENCE_API MyYoloInference {
 private:
  MyYoloInference();

 public:
  static MyYoloInference& getInstance();
  virtual ~MyYoloInference();
  bool loadModel(const char* path);
  bool inference(const char* input_path, const char* output_path);
  void setModelImgSize(const int& width, const int& height);
  void setNMS(const float& threshold);
  void setConfidence(const float& threshold);
  void setClasses(const char** classes, const int& count);

 private:
  class Impl;
  Impl* m_impl;
};
}  // namespace my_yolo

#define MY_YOLO my_yolo::MyYoloInference::getInstance()
extern "C" {
MYYOLOINFERENCE_API bool loadModel(const char* path);
MYYOLOINFERENCE_API bool inference(const char* input_path,
                                   const char* output_path);
}
#endif
