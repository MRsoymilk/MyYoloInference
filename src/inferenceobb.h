#ifndef INFERENCEOBB_H
#define INFERENCEOBB_H

#include "inference.h"

namespace my_yolo {
class InferenceOBB : public Inference {
 public:
  InferenceOBB() = default;

  // Inference interface
 public:
  std::vector<YOLO_RESULT> process(const std::vector<cv::Mat> &) override;
  void draw() override;
  std::string str() override;
};

}  // namespace my_yolo

#endif  // INFERENCEOBB_H
