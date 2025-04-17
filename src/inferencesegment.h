#ifndef INFERENCESEGMENT_H
#define INFERENCESEGMENT_H

#include "inference.h"

namespace my_yolo {
class InferenceSegment : public Inference {
 public:
  InferenceSegment() = default;

  // Inference interface
 public:
  std::vector<YOLO_RESULT> process(const std::vector<cv::Mat> &) override;
  void draw() override;
  std::string str() override;

 private:
  cv::Mat getMask(const cv::Mat &masks_features, const cv::Mat &proto, const cv::Mat &image, const cv::Rect bound);
};

}  // namespace my_yolo

#endif  // INFERENCESEGMENT_H
