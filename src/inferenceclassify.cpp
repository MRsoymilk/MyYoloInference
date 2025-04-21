#include "inferenceclassify.h"

namespace my_yolo {

std::vector<YOLO_RESULT> InferenceClassify::process(const std::vector<cv::Mat> &v) {
  std::vector<YOLO_RESULT> output;

  if (v.empty() || v[0].dims != 2) {
    std::cerr << "Invalid output shape for classification model!" << std::endl;
    return output;
  }

  cv::Mat output_scores = v[0];  // shape: (batch_size, num_classes)
  int batch_size = output_scores.rows;
  int num_classes = output_scores.cols;

  for (int i = 0; i < batch_size; ++i) {
    cv::Mat scores_row = output_scores.row(i);
    cv::Point class_id;
    double max_conf;
    minMaxLoc(scores_row, 0, &max_conf, 0, &class_id);

    if (max_conf > m_info.confidence_threshold) {
      YOLO_RESULT result;
      result.class_idx = class_id.x;
      result.confidence = max_conf;

      output.emplace_back(result);
    }
  }
  m_result = output;
  return m_result;
}

cv::Mat InferenceClassify::draw() {
  int base_line = 0;
  int padding = 5;
  int start_x = 10;
  int start_y = m_image.rows - 10;
  int font_scale = 1;
  int thickness = 1;

  for (const auto &res : m_result) {
    // handle label
    std::string label = cv::format("%s %.2f", m_info.class_names[res.class_idx].c_str(), res.confidence);
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &base_line);
    if (start_x + text_size.width > m_image.cols - 10) {
      start_x = 10;
      start_y -= text_size.height + padding;
    }
    cv::Rect bg_rect(start_x - 2, start_y - text_size.height - 2, text_size.width + 4, text_size.height + 4);
    cv::rectangle(m_image, bg_rect, cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(m_image, label, cv::Point(start_x, start_y), cv::FONT_HERSHEY_SIMPLEX, font_scale,
                cv::Scalar(255, 255, 255), thickness);
    start_x += text_size.width + padding;
  }
  return m_image;
}

std::string InferenceClassify::str() {
  std::stringstream ss;
  ss << "{";
  ss << "\"classify\":[";

  for (size_t i = 0; i < m_result.size(); ++i) {
    const auto &res = m_result[i];
    ss << "{\"" << m_info.class_names[res.class_idx] << "\":" << res.confidence << "}";
    if (i != m_result.size() - 1) {
      ss << ",";
    }
  }

  ss << "]";
  ss << "}";
  return ss.str();
}

}  // namespace my_yolo
