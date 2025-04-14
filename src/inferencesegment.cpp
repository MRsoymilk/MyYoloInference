#include "inferencesegment.h"

#include "utils.h"

namespace my_yolo {

cv::Mat InferenceSegment::getMask(const cv::Mat &masks_features, const cv::Mat &proto, const cv::Mat &image,
                                  const cv::Rect bound) {
  cv::Size img_shape = image.size();
  cv::Size model_shape = cv::Size(m_info.model_width, m_info.model_height);
  cv::Size downsampled_size = cv::Size(m_info.mask_width, m_info.mask_height);

  cv::Rect_<float> bound_float(static_cast<float>(bound.x), static_cast<float>(bound.y),
                               static_cast<float>(bound.width), static_cast<float>(bound.height));

  cv::Rect_<float> downsampled_bbox = Utils::ScaleBox(img_shape, bound_float, downsampled_size);
  cv::Size bound_size = cv::Size(m_info.mask_width, m_info.mask_height);
  Utils::ClipBox(downsampled_bbox, bound_size);

  cv::Mat matmul_res = (masks_features * proto).t();
  matmul_res = matmul_res.reshape(1, {downsampled_size.height, downsampled_size.width});
  // apply sigmoid to the mask:
  cv::Mat sigmoid_mask;
  exp(-matmul_res, sigmoid_mask);
  sigmoid_mask = 1.0 / (1.0 + sigmoid_mask);
  cv::Mat resized_mask;
  cv::Rect_<float> input_bbox = Utils::ScaleBox(img_shape, bound_float, model_shape);
  cv::resize(sigmoid_mask, resized_mask, model_shape, 0, 0, cv::INTER_LANCZOS4);
  cv::Mat pre_out_mask = resized_mask(input_bbox);
  cv::Mat scaled_mask;
  Utils::ScaleImage(scaled_mask, resized_mask, img_shape);
  cv::Mat mask_out;
  cv::resize(scaled_mask, mask_out, img_shape);
  mask_out = mask_out(bound) > m_info.mask_threshold;
  return mask_out;
}

std::vector<YOLO_RESULT> InferenceSegment::process(const std::vector<cv::Mat> &outputs) {
  cv::Mat output_boxes, output_masks;
  output_boxes = outputs[0];
  output_masks = outputs[1];
  std::vector<int64_t> box_shape = {output_boxes.size[0], output_boxes.size[1],
                                    output_boxes.size[2]};  // [bs, features, preds_num]
  cv::Mat output_box = output_boxes.reshape(1, box_shape[1]).t();
  if (outputs.size() > 1) {
    auto mask_shape = output_masks.size;
    m_info.mask_features = mask_shape[1];
    m_info.mask_height = mask_shape[2];
    m_info.mask_width = mask_shape[3];
  }

  std::vector<YOLO_RESULT> output;
  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<float>> masks;
  // 4 - your default number of rect parameters {x, y, w, h}
  int data_width = m_info.nc + 4 + m_info.mask_features;
  int rows = output_box.rows;
  float *pdata = (float *)output_box.data;
  for (int r = 0; r < rows; ++r) {
    cv::Mat scores(1, m_info.nc, CV_32FC1, pdata + 4);
    cv::Point class_id;
    double max_conf;
    minMaxLoc(scores, 0, &max_conf, 0, &class_id);
    if (max_conf > m_info.confidence_threshold) {
      masks.emplace_back(std::vector<float>(pdata + 4 + m_info.nc, pdata + data_width));
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
  if (!output_masks.empty()) {
    // select all of the protos tensor
    cv::Size downsampled_size = cv::Size(m_info.mask_width, m_info.mask_height);
    std::vector<cv::Range> roi_rangs = {cv::Range(0, 1), cv::Range::all(), cv::Range(0, downsampled_size.height),
                                        cv::Range(0, downsampled_size.width)};
    cv::Mat temp_mask = output_masks(roi_rangs).clone();
    proto = temp_mask.reshape(0, {m_info.mask_features, downsampled_size.width * downsampled_size.height});
  }

  for (int i = 0; i < nms_result.size(); ++i) {
    int idx = nms_result[i];
    boxes[idx] = boxes[idx] & cv::Rect(0, 0, m_image.cols, m_image.rows);
    YOLO_RESULT result = {class_ids[idx], confidences[idx], boxes[idx]};
    if (!output_masks.empty()) {
      result.mask = getMask(cv::Mat(masks[idx]).t(), proto, m_image, boxes[idx]);
    }
    output.emplace_back(result);
  }
  m_result = output;
  return output;
}

void InferenceSegment::draw() {
  int thickness = 1;
  cv::Mat mask = m_image.clone();
  for (const auto &res : m_result) {
    float left = res.bbox.x;
    float top = res.bbox.y;
    // Draw bounding box
    cv::rectangle(m_image, res.bbox, Utils::Color(res.class_idx), 2);
    // Draw mask if available
    if (res.mask.rows && res.mask.cols > 0) {
      mask(res.bbox).setTo(Utils::Color(res.class_idx), res.mask);
      cv::addWeighted(m_image, 0.6, mask, 0.4, 0, m_image);
    }
    // Create label
    std::string label = cv::format("%s %.2f", m_info.class_names[res.class_idx].c_str(), res.confidence);
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
    cv::Rect rect_to_fill(left - 1, top - text_size.height - 5, text_size.width + 2, text_size.height + 5);
    cv::Scalar text_color = cv::Scalar(255.0, 255.0, 255.0);
    cv::rectangle(m_image, rect_to_fill, Utils::Color(res.class_idx), -1);
    cv::putText(m_image, label, cv::Point(left - 1.5, top - 2.5), cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, thickness);
  }
}

}  // namespace my_yolo
