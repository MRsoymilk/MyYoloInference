#include "inferencepose.h"

#include "utils.h"

namespace my_yolo {

std::vector<YOLO_RESULT> InferencePose::process(const std::vector<cv::Mat> &v) {
  std::vector<YOLO_RESULT> results;
  if (v.empty() || v[0].dims != 3) {
    std::cerr << "Invalid Pose model output shape!" << std::endl;
    return results;
  }

  int batch = v[0].size[0];
  int feature_dim = v[0].size[1];  // x, y, w, h, conf, keypoints, scores...
  int num_preds = v[0].size[2];
  if (batch != 1) {
    std::cerr << "Only batch size = 1 is supported!" << std::endl;
    return results;
  }

  int kpt_num = m_info.kpt.num;
  int data_width = 4 + kpt_num * m_info.kpt.dim + m_info.nc;

  cv::Mat pred = v[0].reshape(1, feature_dim).t();
  float *pdata = (float *)pred.data;
  int rows = pred.rows;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> v_keypoints;
  std::vector<float> confidences;
  std::vector<int> class_ids;
  for (int i = 0; i < rows; ++i) {
    cv::Mat scores(1, m_info.nc, CV_32FC1, pdata + 4);
    cv::Point class_id;
    double max_conf;
    cv::minMaxLoc(scores, 0, &max_conf, 0, &class_id);
    if (max_conf > m_info.confidence_threshold) {
      confidences.push_back(max_conf);
      class_ids.push_back(class_id.x);
      float cx = pdata[0];
      float cy = pdata[1];
      float w = pdata[2];
      float h = pdata[3];
      cv::Rect_<float> box(cx - w / 2, cy - h / 2, w, h);
      cv::Rect_<float> scaled_bbox =
          Utils::ScaleBox(cv::Size(m_info.model_width, m_info.model_height), box, m_image.size());
      std::vector<cv::Point2f> keypoints;
      for (int k = 0; k < kpt_num; ++k) {
        float kx = pdata[4 + m_info.nc + k * 3];
        float ky = pdata[4 + m_info.nc + k * 3 + 1];
        float kconf = pdata[4 + m_info.nc + k * 3 + 2];
        if (kconf > m_info.confidence_threshold) {
          auto kp = Utils::ScalePoint(cv::Size(m_info.model_width, m_info.model_height), m_image.size(), {kx, ky});
          keypoints.emplace_back(kp);
        } else {
          keypoints.emplace_back(-1, -1);
        }
      }

      YOLO_RESULT result;
      result.class_idx = class_id.x;
      result.confidence = static_cast<float>(max_conf);
      result.bbox = scaled_bbox;
      result.keypoints = keypoints;
      v_keypoints.push_back(keypoints);
      boxes.emplace_back(scaled_bbox);
      results.emplace_back(result);
    }
    pdata += data_width;
  }

  std::vector<int> nms_result;
  cv::dnn::NMSBoxes(boxes, confidences, m_info.confidence_threshold, m_info.nms_threshold, nms_result);

  std::vector<YOLO_RESULT> output;
  for (int i = 0; i < nms_result.size(); ++i) {
    int idx = nms_result[i];
    boxes[idx] = boxes[idx] & cv::Rect(0, 0, m_image.cols, m_image.rows);
    YOLO_RESULT result = {class_ids[idx], confidences[idx], boxes[idx]};
    result.keypoints = v_keypoints[idx];
    output.emplace_back(result);
  }
  m_result = output;
  return m_result;
}

static const std::vector<std::pair<int, int>> limb_pairs = {
    // head
    {0, 1},  // nose - left eye
    {0, 2},  // nose - right eye
    {1, 3},  // left eye - left ear
    {2, 4},  // right eye - right ear
    {1, 2},  // left eye - right eye
    {3, 4},  // left ear - right ear
    // upper limbs
    {5, 7},   // left shoulder - left elbow
    {7, 9},   // left elbow - left wrist
    {6, 8},   // right shoulder - right elbow
    {8, 10},  // right elbow - right wrist
    {5, 6},   // left shoulder - right shoulder
    // torse
    {5, 11},  // left shoulder - left hip
    {6, 12},  // right shoulder - right hip
    // lower limbs
    {11, 13},  // left hip - left knee
    {13, 15},  // left knee - left ankle
    {12, 14},  // right hip - right knee
    {14, 16},  // right knee - right ankle
    {11, 12}   // left hip - right hip
};

void InferencePose::draw() {
  int thickness = 1;
  // float scale_x = 1.0;
  // float scale_y = 1.0;

  for (const auto &res : m_result) {
    float left = res.bbox.x;
    float top = res.bbox.y;
    // handle box
    cv::Rect scaled_bbox(cv::Point(res.bbox.x, res.bbox.y), cv::Size(res.bbox.width, res.bbox.height));
    cv::rectangle(m_image, scaled_bbox, Utils::Color(res.class_idx), 2);

    // handle label
    std::string label = cv::format("%s %.2f", m_info.class_names[res.class_idx].c_str(), res.confidence);
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
    cv::Rect rect_to_fill(left - 1, top - text_size.height - 5, text_size.width + 2, text_size.height + 5);
    cv::Scalar text_color = cv::Scalar(255.0, 255.0, 255.0);
    cv::rectangle(m_image, rect_to_fill, Utils::Color(res.class_idx), -1);
    cv::putText(m_image, label, cv::Point(left - 1.5, top - 2.5), cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, thickness);
    // handle joints
    for (const auto &pair : limb_pairs) {
      int idx1 = pair.first;
      int idx2 = pair.second;
      if (idx1 >= res.keypoints.size() || idx2 >= res.keypoints.size()) continue;

      const auto &pt1 = res.keypoints[idx1];
      const auto &pt2 = res.keypoints[idx2];

      if (pt1.x < 0 || pt1.y < 0 || pt2.x < 0 || pt2.y < 0) {
        continue;
      }

      cv::Point2f p1(pt1.x, pt1.y);
      cv::Point2f p2(pt2.x, pt2.y);
      cv::line(m_image, p1, p2, cv::Scalar(0, 255, 0), 2);
    }

    // handle keypoints
    for (int i = 0; i < res.keypoints.size(); ++i) {
      auto pt = res.keypoints.at(i);
      if (pt.x < 0 || pt.y < 0) {
        continue;
      }
      cv::circle(m_image, cv::Point2f{pt.x, pt.y}, 3, cv::Scalar(0, 0, 255), -1);
    }
  }
}

std::string InferencePose::str() {
  std::stringstream ss;
  ss << "{";
  ss << "\"pose\":[";

  for (size_t i = 0; i < m_result.size(); ++i) {
    const auto &res = m_result[i];
    ss << "{";
    ss << "\"" << m_info.class_names[res.class_idx] << "\":{";
    ss << "\"confidence\":" << res.confidence << ",";
    ss << "\"x\":" << res.bbox.x << ",";
    ss << "\"y\":" << res.bbox.y << ",";
    ss << "\"w\":" << res.bbox.width << ",";
    ss << "\"h\":" << res.bbox.height << ",";
    ss << "\"keypoints\":{";
    for (int j = 0; j < res.keypoints.size(); ++j) {
      const auto &pt = res.keypoints.at(j);
      if (pt.x < 0 || pt.y < 0) {
        continue;
      }
      ss << "\"" << j << "\":";
      ss << "{" << "\"x\":" << pt.x << "," << "\"y\":" << pt.y << "}";
      if (j != res.keypoints.size() - 1) {
        ss << ",";
      }
    }
    ss << "}";
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
