#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <opencv2/core/mat.hpp>
namespace my_yolo {

enum class TASK { UNKNOWN = 0, DETECT, SEGMENT, CLASSIFY, POSE, OBB };

struct IMGSZ {
  int w;
  int h;
};

struct KEYPOINT {
  int num;
  int dim;
};

struct BBOX {
  float x;
  float y;
  float w;
  float h;
};

struct YOLO_RESULT {
  int class_idx;
  float confidence;
  cv::Rect bbox;
  cv::RotatedRect obb;
  cv::Mat mask;
  float angle;
  std::vector<cv::Point2f> keypoints;
};

struct MODEL_INFO {
  float confidence_threshold = 0.5;
  float nms_threshold = 0.5;
  float mask_threshold = 0.5;
  std::vector<std::string> class_names;
  int nc = 0;
  int model_width;
  int model_height;
  int mask_width;
  int mask_height;
  int mask_features;
  TASK task;
  KEYPOINT kpt;
};

}  // namespace my_yolo

#endif  // DEFINITIONS_H
