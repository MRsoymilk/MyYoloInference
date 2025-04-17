#ifndef INFERENCEDETECT_H
#define INFERENCEDETECT_H

#include "inference.h"

namespace my_yolo {
class InferenceDetect : public Inference {
 public:
  InferenceDetect() = default;

  // Inference interface
 public:
  std::vector<YOLO_RESULT> process(const std::vector<cv::Mat>& v) override;
  void draw() override;
  std::string str() override;
};

}  // namespace my_yolo

#endif  // INFERENCEDETECT_H
