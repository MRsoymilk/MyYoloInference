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
    if (loaded) {
      MY_YOLO.inference(input.input_img.c_str(), input.output_img.c_str());
    } else {
      std::cerr << "error: " << input.model_path << std::endl;
    }
  }
  return 0;
}
