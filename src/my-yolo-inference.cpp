#include "my-yolo-inference.h"

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "definitions.h"
#include "inference.h"
#include "inferencefactory.h"
#include "metadata.h"

namespace my_yolo {

class MyYoloInference::Impl {
 private:
  MODEL_INFO m_info;
  cv::dnn::Net m_net;
  std::unordered_map<const char*, bool> m_model_loaded;

 public:
  Impl() {}

  virtual ~Impl() { m_net = cv::dnn::Net(); }

  bool loadModel(const char* path, const int& metadata_size = 2048) {
    if (m_model_loaded[path]) {
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
    m_info.task = metadata.getTask();
    m_info.kpt = metadata.getKeypoint();

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
    m_model_loaded[path] = true;
    return true;
  }

  bool inference(const char* input_path, const char* output_path) {
    // 1. read image
    cv::Mat image = cv::imread(input_path);
    if (image.empty()) {
      std::cerr << "Failed to read image!" << std::endl;
      return false;
    }

    // 2. preprocess
    cv::Mat blob = preprocess(image);
    m_net.setInput(blob);

    // 3. opencv inference
    std::vector<cv::Mat> outputs;
    try {
      m_net.forward(outputs, m_net.getUnconnectedOutLayersNames());
    } catch (const cv::Exception& e) {
      std::cerr << e.what() << std::endl;
    }

    // 4. handle outputs
    std::unique_ptr<Inference> fc = InferenceFactory::Process(image, m_info);
    std::vector<YOLO_RESULT> results = fc->process(outputs);

    if (results.empty()) {
      std::cerr << "Failed to run Interface!" << std::endl;
      return false;
    }

    fc->draw();
    try {
      cv::imwrite(output_path, image);
    } catch (const cv::Exception& e) {
      std::cerr << e.what() << std::endl;
    }
    cv::imshow("img", image);
    cv::waitKey();
    return true;
  }

  bool inference(const void* image_data, unsigned int image_size, char* out_json, unsigned int* out_json_size) {
    // 1. decode image
    std::vector<uint8_t> buf((const uint8_t*)image_data, (const uint8_t*)image_data + image_size);
    cv::Mat image = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (image.empty()) {
      std::cerr << "Failed to decode image from memory!" << std::endl;
      return false;
    }

    // 2. preprocess
    cv::Mat blob = preprocess(image);
    m_net.setInput(blob);

    // 3. opencv inference
    std::vector<cv::Mat> outputs;
    try {
      m_net.forward(outputs, m_net.getUnconnectedOutLayersNames());
    } catch (const cv::Exception& e) {
      std::cerr << e.what() << std::endl;
      return false;
    }

    // 4. postprocess
    std::unique_ptr<Inference> fc = InferenceFactory::Process(image, m_info);
    std::vector<YOLO_RESULT> results = fc->process(outputs);

    if (results.empty()) {
      std::cerr << "Inference result is empty!" << std::endl;
      return false;
    }

    // 5. get json
    auto val = fc->str();
    *out_json_size = val.size();
    std::strncpy(out_json, val.c_str(), *out_json_size - 1);
    out_json[*out_json_size - 1] = '\0';
    return true;
  }

  bool inference(ImageData* img_data) {
    // 1. decode image
    if (img_data == nullptr || img_data->data == nullptr) {
      std::cerr << "Invalid image data!" << std::endl;
      return false;
    }

    cv::Mat image(img_data->height, img_data->width, CV_8UC3, img_data->data);

    // 2. preprocess
    cv::Mat blob = preprocess(image);
    m_net.setInput(blob);

    // 3. inference
    std::vector<cv::Mat> outputs;
    try {
      m_net.forward(outputs, m_net.getUnconnectedOutLayersNames());
    } catch (const cv::Exception& e) {
      std::cerr << e.what() << std::endl;
      return false;
    }

    // 4. postprocess
    std::unique_ptr<Inference> fc = InferenceFactory::Process(image, m_info);
    std::vector<YOLO_RESULT> results = fc->process(outputs);
    if (results.empty()) {
      std::cerr << "Inference result is empty!" << std::endl;
      return false;
    }

    // 5. get result
    cv::Mat res = fc->draw();
    res.copyTo(cv::Mat(img_data->height, img_data->width, CV_8UC3, img_data->data));
    return true;
  }

  void setModelImgSize(const int& width, const int& height) {
    m_info.model_width = width;
    m_info.model_height = height;
    std::cout << "Model input size set to: " << width << "x" << height << std::endl;
  }

  void setNMS(const float& threshold) {
    m_info.nms_threshold = threshold;
    std::cout << "NMS threshold set to: " << threshold << std::endl;
  }

  void setConfidence(const float& threshold) {
    m_info.confidence_threshold = threshold;
    std::cout << "Confidence threshold set to: " << threshold << std::endl;
  }

  void setClasses(const char** classes, const int& count) {
    m_info.class_names.clear();
    for (size_t i = 0; i < count; ++i) {
      m_info.class_names.emplace_back(classes[i]);
    }

    std::cout << "Classes set: ";
    for (const auto& cls : m_info.class_names) {
      std::cout << cls << " ";
    }
    std::cout << std::endl;
  }

 private:
  cv::Mat preprocess(const cv::Mat& image) {
    // cv::Mat img;
    // cv::cvtColor(image, img, cv::COLOR_RGB2BGR);

    // cv::Mat preprocessed_img = Utils::Letterbox(img, {m_info.model_width, m_info.model_height});
    // cv::Mat blob =
    //     cv::dnn::blobFromImage(preprocessed_img, 1.0 / 255.0, cv::Size(m_info.model_width, m_info.model_height));

    cv::dnn::Image2BlobParams params;
    params.scalefactor = cv::Scalar(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0);
    params.size = cv::Size(m_info.model_width, m_info.model_height);
    params.swapRB = true;
    params.datalayout = cv::dnn::DNN_LAYOUT_NCHW;
    params.paddingmode = cv::dnn::ImagePaddingMode::DNN_PMODE_LETTERBOX;
    params.borderValue = cv::Scalar(114, 114, 114);

    cv::Mat blob = cv::dnn::blobFromImageWithParams(image, params);
    return blob;
  }
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

bool MyYoloInference::inference(const char* input_path, const char* output_path) {
  return m_impl->inference(input_path, output_path);
}

bool MyYoloInference::inference(const void* image_data, unsigned int image_size, char* out_json,
                                unsigned int* out_json_size) {
  return m_impl->inference(image_data, image_size, out_json, out_json_size);
}

bool MyYoloInference::inference(ImageData* image_data) { return m_impl->inference(image_data); }

void MyYoloInference::setModelImgSize(const int& width, const int& height) { m_impl->setModelImgSize(width, height); }

void MyYoloInference::setNMS(const float& threshold) { m_impl->setNMS(threshold); }

void MyYoloInference::setConfidence(const float& threshold) { m_impl->setConfidence(threshold); }

void MyYoloInference::setClasses(const char** classes, const int& count) { m_impl->setClasses(classes, count); }

}  // namespace my_yolo

bool loadModel(const char* path, int metadata_size) {
  if (0 == metadata_size) {
    metadata_size = 2048;
  }
  return MY_YOLO.loadModel(path, metadata_size);
}

bool inference(const char* input_path, const char* output_path) { return MY_YOLO.inference(input_path, output_path); }

void setModelImgSize(int width, int height) { MY_YOLO.setModelImgSize(width, height); }

void setNMS(float threshold) { MY_YOLO.setNMS(threshold); }

void setConfidence(float threshold) { MY_YOLO.setConfidence(threshold); }

void setClasses(const char** classes, int count) { MY_YOLO.setClasses(classes, count); }
