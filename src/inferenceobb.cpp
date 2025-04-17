#include "inferenceobb.h"

#include <numeric>

#include "utils.h"

namespace my_yolo {

float rotatedIoU(const cv::RotatedRect& a, const cv::RotatedRect& b) {
  std::vector<cv::Point2f> inter_pts;
  int ret = cv::rotatedRectangleIntersection(a, b, inter_pts);
  if (ret == cv::INTERSECT_NONE || inter_pts.empty()) return 0.0f;
  float inter_area = static_cast<float>(cv::contourArea(inter_pts));
  float union_area = a.size.area() + b.size.area() - inter_area;
  return inter_area / union_area;
}

void rotatedNMS(const std::vector<cv::RotatedRect>& boxes, const std::vector<float>& scores, float iou_threshold,
                std::vector<int>& indices) {
  std::vector<int> order(boxes.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });

  std::vector<bool> suppressed(boxes.size(), false);

  for (size_t i = 0; i < order.size(); ++i) {
    if (suppressed[order[i]]) continue;
    indices.push_back(order[i]);
    for (size_t j = i + 1; j < order.size(); ++j) {
      if (suppressed[order[j]]) continue;
      float iou = rotatedIoU(boxes[order[i]], boxes[order[j]]);
      if (iou > iou_threshold) suppressed[order[j]] = true;
    }
  }
}

static cv::RotatedRect scaleOBB(const cv::RotatedRect& obb, const cv::Size& model_shape, const cv::Size& image_shape) {
  float gain = std::min(static_cast<float>(model_shape.height) / image_shape.height,
                        static_cast<float>(model_shape.width) / image_shape.width);
  float pad_x = (model_shape.width - image_shape.width * gain) / 2.0f;
  float pad_y = (model_shape.height - image_shape.height * gain) / 2.0f;

  cv::Point2f center = obb.center;
  center.x -= pad_x;
  center.y -= pad_y;
  center.x /= gain;
  center.y /= gain;

  cv::Size2f size = obb.size;
  size.width /= gain;
  size.height /= gain;

  return cv::RotatedRect(center, size, obb.angle);
}

std::vector<YOLO_RESULT> InferenceOBB::process(const std::vector<cv::Mat>& v) {
  std::vector<YOLO_RESULT> output;
  if (v.empty() || v[0].dims != 3) {
    std::cerr << "Invalid OBB model output shape!" << std::endl;
    return output;
  }

  int batch_size = v[0].size[0];
  int features = v[0].size[1];
  int num_preds = v[0].size[2];
  if (batch_size != 1) {
    std::cerr << "Only batch_size = 1 is supported!" << std::endl;
    return output;
  }

  cv::Mat output_box = v[0].reshape(1, features).t();  // shape: [num_preds, features]
  int data_width = m_info.nc + 5;                      // 6: x, y, w, h, angle
  float* pdata = (float*)output_box.data;
  int rows = output_box.rows;

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::RotatedRect> boxes;

  for (int r = 0; r < rows; ++r) {
    cv::Mat scores(1, m_info.nc, CV_32FC1, pdata + 4);
    cv::Point class_id;
    double max_conf;
    cv::minMaxLoc(scores, 0, &max_conf, 0, &class_id);
    if (max_conf > m_info.confidence_threshold) {
      class_ids.emplace_back(class_id.x);
      confidences.emplace_back(max_conf);

      float out_x = pdata[0];
      float out_y = pdata[1];
      float out_w = pdata[2];
      float out_h = pdata[3];
      float angle = pdata[data_width - 1] * 180 / CV_PI;

      // build RotatedRect
      cv::RotatedRect obb(cv::Point2f(out_x, out_y), cv::Size2f(out_w, out_h), angle);

      // scale OBB to original
      cv::RotatedRect scaled_obb = scaleOBB(obb, cv::Size(m_info.model_width, m_info.model_height), m_image.size());

      boxes.emplace_back(scaled_obb);
    }

    pdata += data_width;
  }

  // NMS
  std::vector<int> nms_result;
  rotatedNMS(boxes, confidences, m_info.nms_threshold, nms_result);

  for (int idx : nms_result) {
    YOLO_RESULT result;
    result.class_idx = class_ids[idx];
    result.confidence = confidences[idx];
    result.obb = boxes[idx];
    result.angle = boxes[idx].angle;
    result.bbox = boxes[idx].boundingRect();
    output.emplace_back(result);
  }
  m_result = output;
  return m_result;
}

void InferenceOBB::draw() {
  int thickness = 1;

  for (const auto& res : m_result) {
    float left = res.bbox.x;
    float top = res.bbox.y;
    // hadle box
    cv::Point2f pts[4];
    res.obb.points(pts);
    for (int i = 0; i < 4; ++i) {
      cv::line(m_image, pts[i], pts[(i + 1) % 4], Utils::Color(res.class_idx), 2);
    }
    // handle label
    std::string label =
        cv::format("%s %.2f, angle %.2f", m_info.class_names[res.class_idx].c_str(), res.confidence, res.angle);
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
    cv::Rect rect_to_fill(left - 1, top - text_size.height - 5, text_size.width + 2, text_size.height + 5);
    cv::Scalar text_color = cv::Scalar(255.0, 255.0, 255.0);
    cv::rectangle(m_image, rect_to_fill, Utils::Color(res.class_idx), -1);
    cv::putText(m_image, label, cv::Point(left - 1.5, top - 2.5), cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, thickness);
  }
}
std::string InferenceOBB::str() {
  std::stringstream ss;
  ss << "{";
  ss << "\"obb\":[";

  for (size_t i = 0; i < m_result.size(); ++i) {
    const auto& res = m_result[i];
    ss << "{";
    ss << "\"" << m_info.class_names[res.class_idx] << "\":{";
    ss << "\"x\":" << res.bbox.x << ",";
    ss << "\"y\":" << res.bbox.y << ",";
    ss << "\"w\":" << res.bbox.width << ",";
    ss << "\"h\":" << res.bbox.height << ",";
    ss << "\"angle\":" << res.angle;
    ss << "}}";
    if (i != m_result.size() - 1) {
      ss << ",";
    }
  }

  ss << "]";
  ss << "}";
  return ss.str();
}

}  // namespace my_yolo
