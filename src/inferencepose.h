#ifndef INFERENCEPOSE_H
#define INFERENCEPOSE_H

#include "inference.h"

namespace my_yolo {
class InferencePose : public Inference {
 public:
  InferencePose() = default;

  // Inference interface
 public:
  std::vector<YOLO_RESULT> process(const std::vector<cv::Mat> &) override;
  cv::Mat draw() override;
  std::string str() override;
};

}  // namespace my_yolo

#endif  // INFERENCEPOSE_H
