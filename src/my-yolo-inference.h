#ifndef MY_YOLO_INFERENCE_H
#define MY_YOLO_INFERENCE_H

#include "global.h"

namespace my_yolo {
class ImageData;

class MYYOLOINFERENCE_API MyYoloInference {
 private:
  MyYoloInference();

 public:
  static MyYoloInference& getInstance();
  virtual ~MyYoloInference();
  bool loadModel(const char* path, const int& metadata_size = 2048);
  bool inference(const char* input_path, const char* output_path);
  bool inference(const void* image_data, unsigned int image_size, char* out_json, unsigned int* out_json_size);
  bool inference(ImageData* image_data);
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
MYYOLOINFERENCE_API bool loadModel(const char* path, int metadata_size = 2048);
MYYOLOINFERENCE_API bool inference(const char* input_path, const char* output_path);
MYYOLOINFERENCE_API bool inference_binary(const void* image_data, unsigned int image_size, char* out_json,
                                          unsigned int* out_json_size);
MYYOLOINFERENCE_API bool inference_ImageData(my_yolo::ImageData* image_data);
MYYOLOINFERENCE_API void setModelImgSize(int width, int height);
MYYOLOINFERENCE_API void setNMS(float threshold);
MYYOLOINFERENCE_API void setConfidence(float threshold);
MYYOLOINFERENCE_API void setClasses(const char** classes, int count);
}
#endif
