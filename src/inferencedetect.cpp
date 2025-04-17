#include "inferencedetect.h"

#include "utils.h"

namespace my_yolo {

std::vector<YOLO_RESULT> InferenceDetect::process(const std::vector<cv::Mat>& v) {
  std::vector<int64_t> box_shape = {v[0].size[0], v[0].size[1], v[0].size[2]};  // [bs, features, preds_num]
  cv::Mat output_box = v[0].reshape(1, box_shape[1]).t();
  std::vector<YOLO_RESULT> output;
  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  // 4 - your default number of rect parameters {x, y, w, h}
  int data_width = m_info.nc + 4;
  int rows = output_box.rows;
  float* pdata = (float*)output_box.data;
  for (int r = 0; r < rows; ++r) {
    cv::Mat scores(1, m_info.nc, CV_32FC1, pdata + 4);
    cv::Point class_id;
    double max_conf;
    cv::minMaxLoc(scores, 0, &max_conf, 0, &class_id);
    if (max_conf > m_info.confidence_threshold) {
      class_ids.emplace_back(class_id.x);
      confidences.emplace_back(max_conf);

      float out_w = pdata[2];
      float out_h = pdata[3];
      float out_left = MAX((pdata[0] - 0.5 * out_w + 0.5), 0);
      float out_top = MAX((pdata[1] - 0.5 * out_h + 0.5), 0);
      cv::Rect_<float> bbox = cv::Rect(out_left, out_top, (out_w + 0.5), (out_h + 0.5));
      cv::Rect_<float> scaled_bbox =
          Utils::ScaleBox(cv::Size(m_info.model_width, m_info.model_height), bbox, m_image.size());

      boxes.emplace_back(scaled_bbox);
    }
    pdata += data_width;  // next pred
  }

  std::vector<int> nms_result;
  cv::dnn::NMSBoxes(boxes, confidences, m_info.confidence_threshold, m_info.nms_threshold, nms_result);

  cv::Mat proto;

  for (int i = 0; i < nms_result.size(); ++i) {
    int idx = nms_result[i];
    boxes[idx] = boxes[idx] & cv::Rect(0, 0, m_image.cols, m_image.rows);
    YOLO_RESULT result = {class_ids[idx], confidences[idx], boxes[idx]};
    output.emplace_back(result);
  }
  m_result = output;
  return m_result;
}

void InferenceDetect::draw() {
  int thickness = 1;
  int line_width = 2;

  for (const auto& res : m_result) {
    float left = res.bbox.x;
    float top = res.bbox.y;
    // handle box
    cv::rectangle(m_image, res.bbox, Utils::Color(res.class_idx), line_width);
    // handle label
    std::string label = cv::format("%s %.2f", m_info.class_names[res.class_idx].c_str(), res.confidence);
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, thickness, nullptr);
    cv::Rect rect_to_fill(left - 1, top - text_size.height - 5, text_size.width + 2, text_size.height + 5);
    cv::Scalar text_color = cv::Scalar(255.0, 255.0, 255.0);
    cv::rectangle(m_image, rect_to_fill, Utils::Color(res.class_idx), -1);
    cv::putText(m_image, label, cv::Point(left - 1.5, top - 2.5), cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, thickness);
  }
}

std::string InferenceDetect::str() {
  std::stringstream ss;
  ss << "{";
  ss << "\"detect\":[";

  for (size_t i = 0; i < m_result.size(); ++i) {
    const auto& res = m_result[i];
    ss << "{";
    ss << "\"" << m_info.class_names[res.class_idx] << "\":{";
    ss << "\"confidence\":" << res.confidence << ",";
    ss << "\"x\":" << res.bbox.x << ",";
    ss << "\"y\":" << res.bbox.y << ",";
    ss << "\"w\":" << res.bbox.width << ",";
    ss << "\"h\":" << res.bbox.height;
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
