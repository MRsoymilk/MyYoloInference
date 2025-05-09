#ifndef INFERENCECLASSIFY_H
#define INFERENCECLASSIFY_H

#include "inference.h"

namespace my_yolo {
class InferenceClassify : public Inference {
 public:
  InferenceClassify() = default;

  // Inference interface
 public:
  std::vector<YOLO_RESULT> process(const std::vector<cv::Mat> &ptr) override;
  cv::Mat draw() override;
  std::string str() override;
};

}  // namespace my_yolo

#endif  // INFERENCECLASSIFY_H
