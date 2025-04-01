#include "my-yolo-inference.h"

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "metadata.h"
#include "vendor/base64.h"

namespace my_yolo {

struct YoloResults {
  int class_idx{};
  float conf{};
  cv::Rect_<float> bbox;
  cv::Mat mask;
};

struct detect_info {
  float confidence_threshold = 0.5;
  float nms_threshold = 0.5;
  float mask_threshold = 0.5;
  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<int> nms_indices;
  std::vector<std::string> class_names;
  int nc = 0;
  int model_width;
  int model_height;
  int mask_width;
  int mask_height;
  int mask_features;
};

detect_info m_info;

cv::dnn::Net m_net;
bool m_loaded = false;
// std::unordered_map<std::string, std::string> m_metadata;

cv::Mat getMask(const cv::Mat& masks_features, const cv::Mat& proto,
                const cv::Mat& image, const cv::Rect bound);
std::vector<YoloResults> handleOutput(cv::Mat& output_box,
                                      cv::Mat& output_segment, cv::Mat& image);
cv::Mat letterbox(const cv::Mat& img);

std::vector<cv::Scalar> COLORS = {
    cv::Scalar(0, 255, 0),      // class 0 - Green
    cv::Scalar(255, 0, 0),      // class 1 - Blue
    cv::Scalar(0, 0, 255),      // class 2 - Red
    cv::Scalar(255, 0, 255),    // class 3 - Magenta
    cv::Scalar(255, 255, 0),    // class 4 - Cyan
    cv::Scalar(0, 128, 255),    // class 5 - Orange
    cv::Scalar(128, 0, 128),    // class 6 - Purple
    cv::Scalar(0, 255, 255),    // class 7 - Light Green
    cv::Scalar(255, 165, 0),    // class 8 - Orange Yellow
    cv::Scalar(255, 105, 180),  // class 9 - Hot Pink
    cv::Scalar(60, 179, 113),   // class 10 - Medium Sea Green
    cv::Scalar(255, 20, 147),   // class 11 - Deep Pink
    cv::Scalar(138, 43, 226),   // class 12 - Blue Violet
    cv::Scalar(75, 0, 130),     // class 13 - Indigo
    cv::Scalar(0, 191, 255),    // class 14 - Deep Sky Blue
    cv::Scalar(255, 69, 0)      // class 15 - Orange Red
};

cv::Scalar generateColor(int index) {
  uint8_t r = (index * 41) % 256;
  uint8_t g = (index * 73) % 256;
  uint8_t b = (index * 97) % 256;

  return cv::Scalar(b, g, r);
}

cv::Scalar colors(const int& index) {
  if (index < COLORS.size()) {
    return COLORS[index];
  } else {
    return generateColor(index);
  }
}

void drawInfo(cv::Mat img, std::vector<YoloResults>& results,
              const cv::Size& shape) {
  cv::Mat mask = img.clone();
  for (const auto& res : results) {
    float left = res.bbox.x;
    float top = res.bbox.y;
    // Draw bounding box
    cv::rectangle(img, res.bbox, colors(res.class_idx), 2);
    // Draw mask if available
    if (res.mask.rows && res.mask.cols > 0) {
      mask(res.bbox).setTo(colors(res.class_idx), res.mask);
      cv::addWeighted(img, 0.6, mask, 0.4, 0, img);
    }
    // Create label
    std::stringstream labelStream;
    labelStream << m_info.class_names[res.class_idx] << " " << std::fixed
                << std::setprecision(2) << res.conf;
    std::string label = labelStream.str();
    cv::Size text_size =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
    cv::Rect rect_to_fill(left - 1, top - text_size.height - 5,
                          text_size.width + 2, text_size.height + 5);
    cv::Scalar text_color = cv::Scalar(255.0, 255.0, 255.0);
    cv::rectangle(img, rect_to_fill, colors(res.class_idx), -1);
    cv::putText(img, label, cv::Point(left - 1.5, top - 2.5),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
  }
}

const int DEFAULT_LETTERBOX_PAD_VALUE = 114;

cv::Mat letterbox(const cv::Mat& img) {
  cv::Mat base(
      m_info.model_width, m_info.model_height, CV_8UC3,
      cv::Scalar(DEFAULT_LETTERBOX_PAD_VALUE, DEFAULT_LETTERBOX_PAD_VALUE,
                 DEFAULT_LETTERBOX_PAD_VALUE));
  int input_height = img.rows;
  int input_width = img.cols;
  float scale = std::min(m_info.model_width * 1.0f / input_width,
                         m_info.model_height * 1.0f / input_height);
  if (scale < 1.0f) {
    cv::Mat resizedImg;
    cv::resize(img, resizedImg, cv::Size(), scale, scale);
    input_height = resizedImg.rows;
    input_width = resizedImg.cols;
    int top = (base.rows - input_height) / 2;
    int left = (base.cols - input_width) / 2;
    resizedImg.copyTo(base(cv::Rect(left, top, input_width, input_height)));
  } else {
    int top = (base.rows - input_height) / 2;
    int left = (base.cols - input_width) / 2;
    img.copyTo(base(cv::Rect(left, top, input_width, input_height)));
  }
  return base;
}

int64_t vectorProduct(const std::vector<int64_t>& vec) {
  int64_t result = 1;
  for (int64_t value : vec) {
    result *= value;
  }
  return result;
}

void fillBlob(cv::Mat& image, float*& blob) {
  cv::Mat float_image;
  image.convertTo(float_image, CV_32FC3, 1.0f / 255.0);
  blob =
      new float[float_image.cols * float_image.rows * float_image.channels()];
  cv::Size float_img_size{float_image.cols, float_image.rows};
  // hwc -> chw
  std::vector<cv::Mat> chw(float_image.channels());
  for (int i = 0; i < float_image.channels(); ++i) {
    chw[i] = cv::Mat(float_img_size, CV_32FC1,
                     blob + i * float_img_size.width * float_img_size.height);
  }
  cv::split(float_image, chw);
}

void clipBoxes(cv::Rect_<float>& box, const cv::Size& shape) {
  box.x = std::max(0.0f, std::min(box.x, static_cast<float>(shape.width)));
  box.y = std::max(0.0f, std::min(box.y, static_cast<float>(shape.height)));
  box.width = std::max(
      0.0f, std::min(box.width, static_cast<float>(shape.width - box.x)));
  box.height = std::max(
      0.0f, std::min(box.height, static_cast<float>(shape.height - box.y)));
}

cv::Rect_<float> scaleBoxes(const cv::Size& img1_shape, cv::Rect_<float>& box,
                            const cv::Size& img0_shape,
                            std::pair<float, cv::Point2f> ratio_pad =
                                std::make_pair(-1.0f,
                                               cv::Point2f(-1.0f, -1.0f)),
                            bool padding = true) {
  float gain, pad_x, pad_y;
  if (ratio_pad.first < 0.0f) {
    gain = std::min(static_cast<float>(img1_shape.height) /
                        static_cast<float>(img0_shape.height),
                    static_cast<float>(img1_shape.width) /
                        static_cast<float>(img0_shape.width));
    pad_x = roundf((img1_shape.width - img0_shape.width * gain) / 2.0f - 0.1f);
    pad_y =
        roundf((img1_shape.height - img0_shape.height * gain) / 2.0f - 0.1f);
  } else {
    gain = ratio_pad.first;
    pad_x = ratio_pad.second.x;
    pad_y = ratio_pad.second.y;
  }

  cv::Rect_<float> scaled_coords(box);

  if (padding) {
    scaled_coords.x -= pad_x;
    scaled_coords.y -= pad_y;
  }

  scaled_coords.x /= gain;
  scaled_coords.y /= gain;
  scaled_coords.width /= gain;
  scaled_coords.height /= gain;

  // Clip the box to the bounds of the image
  clipBoxes(scaled_coords, img0_shape);

  return scaled_coords;
}

void scaleImage(cv::Mat& scaled_mask, const cv::Mat& resized_mask,
                const cv::Size& im0_shape,
                const std::pair<float, cv::Point2f>& ratio_pad =
                    std::make_pair(-1.0f, cv::Point2f(-1.0f, -1.0f))) {
  cv::Size im1_shape = resized_mask.size();

  // Check if resizing is needed
  if (im1_shape == im0_shape) {
    scaled_mask = resized_mask.clone();
    return;
  }

  float gain, pad_x, pad_y;

  if (ratio_pad.first < 0.0f) {
    gain = std::min(static_cast<float>(im1_shape.height) /
                        static_cast<float>(im0_shape.height),
                    static_cast<float>(im1_shape.width) /
                        static_cast<float>(im0_shape.width));
    pad_x = (im1_shape.width - im0_shape.width * gain) / 2.0f;
    pad_y = (im1_shape.height - im0_shape.height * gain) / 2.0f;
  } else {
    gain = ratio_pad.first;
    pad_x = ratio_pad.second.x;
    pad_y = ratio_pad.second.y;
  }

  int top = static_cast<int>(pad_y);
  int left = static_cast<int>(pad_x);
  int bottom = static_cast<int>(im1_shape.height - pad_y);
  int right = static_cast<int>(im1_shape.width - pad_x);

  // Clip and resize the mask
  cv::Rect clipped_rect(left, top, right - left, bottom - top);
  cv::Mat clipped_mask = resized_mask(clipped_rect);
  cv::resize(clipped_mask, scaled_mask, im0_shape);
}

cv::Mat getMask(const cv::Mat& masks_features, const cv::Mat& proto,
                const cv::Mat& image, const cv::Rect bound) {
  cv::Size img_shape = image.size();
  cv::Size model_shape = cv::Size(m_info.model_width, m_info.model_height);
  cv::Size downsampled_size = cv::Size(m_info.mask_width, m_info.mask_height);

  cv::Rect_<float> bound_float(
      static_cast<float>(bound.x), static_cast<float>(bound.y),
      static_cast<float>(bound.width), static_cast<float>(bound.height));

  cv::Rect_<float> downsampled_bbox =
      scaleBoxes(img_shape, bound_float, downsampled_size);
  cv::Size bound_size = cv::Size(m_info.mask_width, m_info.mask_height);
  clipBoxes(downsampled_bbox, bound_size);

  cv::Mat matmul_res = (masks_features * proto).t();
  matmul_res =
      matmul_res.reshape(1, {downsampled_size.height, downsampled_size.width});
  // apply sigmoid to the mask:
  cv::Mat sigmoid_mask;
  exp(-matmul_res, sigmoid_mask);
  sigmoid_mask = 1.0 / (1.0 + sigmoid_mask);
  cv::Mat resized_mask;
  cv::Rect_<float> input_bbox = scaleBoxes(img_shape, bound_float, model_shape);
  cv::resize(sigmoid_mask, resized_mask, model_shape, 0, 0, cv::INTER_LANCZOS4);
  cv::Mat pre_out_mask = resized_mask(input_bbox);
  cv::Mat scaled_mask;
  scaleImage(scaled_mask, resized_mask, img_shape);
  cv::Mat mask_out;
  cv::resize(scaled_mask, mask_out, img_shape);
  mask_out = mask_out(bound) > m_info.mask_threshold;
  return mask_out;
}

std::vector<YoloResults> handleOutput(cv::Mat& output_box,
                                      cv::Mat& output_segment, cv::Mat& image) {
  std::vector<YoloResults> output;
  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<float>> masks;
  // 4 - your default number of rect parameters {x, y, w, h}
  int data_width = m_info.nc + 4 + m_info.mask_features;
  int rows = output_box.rows;
  float* pdata = (float*)output_box.data;
  for (int r = 0; r < rows; ++r) {
    cv::Mat scores(1, m_info.nc, CV_32FC1, pdata + 4);
    cv::Point class_id;
    double max_conf;
    minMaxLoc(scores, 0, &max_conf, 0, &class_id);
    if (max_conf > m_info.confidence_threshold) {
      masks.emplace_back(
          std::vector<float>(pdata + 4 + m_info.nc, pdata + data_width));
      class_ids.emplace_back(class_id.x);
      confidences.emplace_back(max_conf);

      float out_w = pdata[2];
      float out_h = pdata[3];
      float out_left = MAX((pdata[0] - 0.5 * out_w + 0.5), 0);
      float out_top = MAX((pdata[1] - 0.5 * out_h + 0.5), 0);
      cv::Rect_<float> bbox =
          cv::Rect(out_left, out_top, (out_w + 0.5), (out_h + 0.5));
      cv::Rect_<float> scaled_bbox =
          scaleBoxes(cv::Size(m_info.model_width, m_info.model_height), bbox,
                     image.size());
      boxes.emplace_back(scaled_bbox);
    }
    pdata += data_width;  // next pred
  }

  std::vector<int> nms_result;
  cv::dnn::NMSBoxes(boxes, confidences, m_info.confidence_threshold,
                    m_info.nms_threshold, nms_result);

  cv::Mat proto;
  if (!output_segment.empty()) {
    // select all of the protos tensor
    cv::Size downsampled_size = cv::Size(m_info.mask_width, m_info.mask_height);
    std::vector<cv::Range> roi_rangs = {cv::Range(0, 1), cv::Range::all(),
                                        cv::Range(0, downsampled_size.height),
                                        cv::Range(0, downsampled_size.width)};
    cv::Mat temp_mask = output_segment(roi_rangs).clone();
    proto = temp_mask.reshape(
        0, {m_info.mask_features,
            downsampled_size.width * downsampled_size.height});
  }

  for (int i = 0; i < nms_result.size(); ++i) {
    int idx = nms_result[i];
    boxes[idx] = boxes[idx] & cv::Rect(0, 0, image.cols, image.rows);
    YoloResults result = {class_ids[idx], confidences[idx], boxes[idx]};
    if (!output_segment.empty()) {
      result.mask = getMask(cv::Mat(masks[idx]).t(), proto, image, boxes[idx]);
    }
    output.emplace_back(result);
  }

  return output;
}

std::string mapToString(const std::map<int, std::string>& myMap) {
  std::ostringstream oss;
  oss << "{";
  for (auto it = myMap.begin(); it != myMap.end(); ++it) {
    if (it != myMap.begin()) {
      oss << ", ";
    }
    oss << "\"" << it->first << "\": \"" << it->second << "\"";
  }
  oss << "}";
  return oss.str();
}

struct MY_JSON {
  std::string name;
  size_t idx;
  struct {
    size_t h;
    size_t w;
    size_t x;
    size_t y;
  } bbox;
  float confidence;
  std::string mask;
};

std::string to_json(const MY_JSON& obj) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"name\": \"" << obj.name << "\", ";
  oss << "\"id\": " << obj.idx << ", ";
  oss << "\"bbox\": {"
      << "\"h\": " << obj.bbox.h << ", "
      << "\"w\": " << obj.bbox.w << ", "
      << "\"x\": " << obj.bbox.x << ", "
      << "\"y\": " << obj.bbox.y << "}, ";
  oss << "\"confidence\": " << obj.confidence << ", ";
  oss << "\"mask\": \"" << obj.mask << "\"";
  oss << "}";
  return oss.str();
}

std::string imgToBase64(const cv::Mat& img) {
  // encode as PNG
  std::vector<uchar> buf;
  cv::imencode(".png", img, buf);
  std::string encoded =
      "data:image/png;base64," + base64_encode(buf.data(), buf.size(), false);
  return encoded;
}

std::vector<YoloResults> runInterface(const cv::Mat& img) {
  cv::Mat image;
  if (cv::COLOR_BGR2RGB >= 0) {
    cv::cvtColor(img, image, cv::COLOR_BGR2RGB);
  }

  cv::Mat preprocessed_img = letterbox(image);
  cv::Mat blob;
  cv::dnn::blobFromImage(preprocessed_img, blob, 1.0 / 255.0,
                         cv::Size(m_info.model_width, m_info.model_height),
                         cv::Scalar(0, 0, 0), false, false);
  m_net.setInput(blob);

  std::vector<cv::Mat> outputs;
  try {
    m_net.forward(outputs, m_net.getUnconnectedOutLayersNames());
  } catch (cv::Exception e) {
    std::cerr << e.what() << std::endl;
  }

  cv::Mat output_boxes, output_masks;
  output_boxes = outputs[0];
  if (outputs.size() > 1) {
    output_masks = outputs[1];
  }
  std::vector<int64_t> box_shape = {
      output_boxes.size[0], output_boxes.size[1],
      output_boxes.size[2]};  // [bs, features, preds_num]
  cv::Mat output_box = output_boxes.reshape(1, box_shape[1]).t();
  if (outputs.size() > 1) {
    auto mask_shape = output_masks.size;
    m_info.mask_features = mask_shape[1];
    m_info.mask_height = mask_shape[2];
    m_info.mask_width = mask_shape[3];
  }

  std::vector<YoloResults> results =
      handleOutput(output_box, output_masks, image);
  return results;
}

int jsonInfo(const std::vector<YoloResults>& results, char* rlt) {
  std::vector<MY_JSON> v;
  for (auto x : results) {
    MY_JSON j;
    j.confidence = x.conf;
    j.idx = x.class_idx;
    j.bbox.h = x.bbox.height;
    j.bbox.w = x.bbox.width;
    j.bbox.x = x.bbox.x;
    j.bbox.y = x.bbox.y;
    j.name = m_info.class_names.at(x.class_idx);
    j.mask = imgToBase64(x.mask);
    v.push_back(j);
  }
  std::ostringstream oss;
  oss << "{ \"objects\": [";
  for (size_t i = 0; i < v.size(); ++i) {
    oss << to_json(v[i]);
    if (i != v.size() - 1) {
      oss << ", ";
    }
  }
  oss << "] }";
  std::string str = oss.str();
  str.copy(rlt, str.size());
  return str.size();
}

class MyYoloInference::Impl {
 public:
  Impl()
      : m_width(0),
        m_height(0),
        m_nms_threshold(0.5f),
        m_confidence_threshold(0.5f),
        m_classes({}) {}

  virtual ~Impl() {}

  bool loadModel(const char* path, const int& metadata_size = 2048) {
    if (m_loaded) {
      return true;
    }
    Metadata metadata;
    std::string data = metadata.readFileTail(path, metadata_size);
    if (data.empty()) {
      std::cerr << "no description found!" << std::endl;
      return false;
    }
    metadata.analysis(data);
    m_info.class_names = metadata.getNames();
    m_info.nc = m_info.class_names.size();
    m_info.model_height = metadata.getImgsz().h;
    m_info.model_width = metadata.getImgsz().w;

    std::ifstream file(path, std::ios::binary);
    if (!file) {
      std::cerr << "Failed to load model: " << path << std::endl;
      return false;
    }

    file.seekg(0, std::ios::end);
    size_t model_data_length = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> model_data(model_data_length);
    file.read(model_data.data(), model_data_length);
    file.close();

    m_net = cv::dnn::readNetFromONNX(model_data.data(), model_data_length);

    if (m_net.empty()) {
      return false;
    }
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
      m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
      m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
      m_net.enableFusion(false);
    } else {
      m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
      m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    m_loaded = true;
    return true;
  }

  bool inference(const char* input_path, const char* output_path) {
    if (!m_loaded) {
      std::cerr << "No ONNX model loaded!" << std::endl;
      return false;
    }
    // preprocess
    cv::Mat image = cv::imread(input_path);
    if (image.empty()) {
      std::cerr << "Failed to read image!" << std::endl;
      return false;
    }
    std::vector<YoloResults> results = runInterface(image);
    if (results.empty()) {
      std::cerr << "Failed to run Interface!" << std::endl;
      return false;
    }
    drawInfo(image, results, image.size());
    cv::imshow("image", image);
    cv::imwrite(output_path, image);
    for (int i = 0; i < results.size(); ++i) {
      cv::Mat mask = results.at(i).mask;
      if (!mask.empty()) {
        cv::imshow(std::to_string(i), mask);
        cv::imwrite("mask_" + std::to_string(i) + ".png", mask);
      }
    }
    cv::waitKey();
    m_net = cv::dnn::Net();
    return true;
  }

  void setModelImgSize(const int& width, const int& height) {
    m_width = width;
    m_height = height;
    std::cout << "Model input size set to: " << width << "x" << height
              << std::endl;
  }

  void setNMS(const float& threshold) {
    m_nms_threshold = threshold;
    std::cout << "NMS threshold set to: " << threshold << std::endl;
  }

  void setConfidence(const float& threshold) {
    m_confidence_threshold = threshold;
    std::cout << "Confidence threshold set to: " << threshold << std::endl;
  }

  void setClasses(const char** classes, const int& count) {
    m_classes.clear();
    for (size_t i = 0; i < count; ++i) {
      m_classes.emplace_back(classes[i]);
    }

    std::cout << "Classes set: ";
    for (const auto& cls : m_classes) {
      std::cout << cls << " ";
    }
    std::cout << std::endl;
  }

 private:
  int m_width;
  int m_height;
  float m_nms_threshold;
  float m_confidence_threshold;
  std::vector<std::string> m_classes;
};

MyYoloInference::MyYoloInference() : m_impl(new Impl()) {}

MyYoloInference& MyYoloInference::getInstance() {
  static MyYoloInference instance;
  return instance;
}

MyYoloInference::~MyYoloInference() { delete m_impl; }

bool MyYoloInference::loadModel(const char* path, const int& metadata_size) {
  return m_impl->loadModel(path, metadata_size);
}

bool MyYoloInference::inference(const char* input_path,
                                const char* output_path) {
  return m_impl->inference(input_path, output_path);
}

void MyYoloInference::setModelImgSize(const int& width, const int& height) {
  m_impl->setModelImgSize(width, height);
}

void MyYoloInference::setNMS(const float& threshold) {
  m_impl->setNMS(threshold);
}

void MyYoloInference::setConfidence(const float& threshold) {
  m_impl->setConfidence(threshold);
}

void MyYoloInference::setClasses(const char** classes, const int& count) {
  m_impl->setClasses(classes, count);
}

}  // namespace my_yolo

bool loadModel(const char* path, int metadata_size) {
  if (metadata_size == 0) {
    metadata_size = 2048;
  }
  return MY_YOLO.loadModel(path, metadata_size);
}

bool inference(const char* input_path, const char* output_path) {
  return MY_YOLO.inference(input_path, output_path);
}

void setModelImgSize(int width, int height) {
  MY_YOLO.setModelImgSize(width, height);
}

void setNMS(float threshold) { MY_YOLO.setNMS(threshold); }

void setConfidence(float threshold) { MY_YOLO.setConfidence(threshold); }

void setClasses(const char** classes, int count) {
  MY_YOLO.setClasses(classes, count);
}
