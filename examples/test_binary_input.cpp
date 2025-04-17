#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "my-yolo-inference.h"
#include "test.h"

int main(int argc, char* argv[]) {
  const std::string dir_model = "../examples/res/";
  const std::string dir_output = "./";

  std::vector<PARAMS> params{
      {dir_model + "yolo11n-seg.onnx", 2048, dir_model + "apple.jpg", dir_output + "test_seg.jpg"},
      {dir_model + "yolo11n-cls.onnx", 20480, dir_model + "strawberry.jpg", dir_output + "test_cls.jpg"},
      {dir_model + "yolo11n-obb.onnx", 2048, dir_model + "airport.jpg", dir_output + "test_obb.jpg"},
      {dir_model + "yolo11n-pose.onnx", 2048, dir_model + "ikun-dance.jpg", dir_output + "test_pose.jpg"},
      {dir_model + "yolo11n.onnx", 2048, dir_model + "ikun-play.jpg", dir_output + "test_detect.jpg"},
  };

  for (const auto& input : params) {
    bool loaded = MY_YOLO.loadModel(input.model_path.c_str(), input.metadata_size);

    std::ifstream file(input.input_img, std::ios::binary | std::ios::ate);
    if (!file) {
      std::cerr << "Failed to open image: " << input.input_img << std::endl;
      continue;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
      std::cerr << "Failed to read image: " << input.input_img << std::endl;
      continue;
    }

    char json_buffer[20480] = {0};
    unsigned int json_buffer_len = 0;
    bool ok = MY_YOLO.inference(buffer.data(), size, json_buffer, &json_buffer_len);
    std::cout << "result size: " << json_buffer_len << std::endl;
    if (!ok) {
      std::cerr << "Inference failed: " << input.input_img << std::endl;
      continue;
    }
    std::cout << "JSON result for " << input.input_img << ":\n" << json_buffer << std::endl;
  }
  return 0;
}
