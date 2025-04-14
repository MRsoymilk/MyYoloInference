#ifndef INFERENCEFACTORY_H
#define INFERENCEFACTORY_H

#include <memory>

#include "definitions.h"

namespace my_yolo {
class Inference;

class InferenceFactory {
 public:
  InferenceFactory() = default;
  ~InferenceFactory() = default;
  static std::unique_ptr<Inference> Process(cv::Mat img, MODEL_INFO info);
};
}  // namespace my_yolo

#endif  // INFERENCEFACTORY_H
