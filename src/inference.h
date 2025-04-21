#ifndef INFERENCE_H
#define INFERENCE_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "definitions.h"

namespace my_yolo {

class Inference {
 public:
  Inference() = default;
  virtual ~Inference() = default;

 public:
  virtual std::vector<YOLO_RESULT> process(const std::vector<cv::Mat>&) = 0;
  virtual cv::Mat draw() { return cv::Mat(); };
  virtual std::string str() { return ""; };

  MODEL_INFO m_info;
  cv::Mat m_image;
  std::vector<YOLO_RESULT> m_result;
};

}  // namespace my_yolo

#endif  // INFERENCE_H
