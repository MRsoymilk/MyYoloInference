#include "inferencefactory.h"

#include "definitions.h"
#include "inferenceclassify.h"
#include "inferencedetect.h"
#include "inferenceobb.h"
#include "inferencepose.h"
#include "inferencesegment.h"

namespace my_yolo {

std::unique_ptr<Inference> InferenceFactory::Process(cv::Mat img, MODEL_INFO info) {
  std::unique_ptr<Inference> inf;
  switch (info.task) {
    case TASK::DETECT:
      inf = std::make_unique<InferenceDetect>();
      break;
    case TASK::SEGMENT:
      inf = std::make_unique<InferenceSegment>();
      break;
    case TASK::CLASSIFY:
      inf = std::make_unique<InferenceClassify>();
      break;
    case TASK::POSE:
      inf = std::make_unique<InferencePose>();
      break;
    case TASK::OBB:
      inf = std::make_unique<InferenceOBB>();
      break;
    case TASK::UNKNOWN:
      return nullptr;
  }
  if (inf) {
    inf->m_info = info;
    inf->m_image = img;
  }
  return inf;
}

}  // namespace my_yolo
