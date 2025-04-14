#include <iostream>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <vector>

#include "test.h"

typedef bool (*LoadModelFunc)(const char*, int);
typedef bool (*InferenceFunc)(const char*, const char*);

int main() {
  const char* dll_name =
#ifdef _WIN32
      "./MyYoloInference.dll";
#else
      "./MyYoloInference.so";
#endif

#ifdef _WIN32
  HMODULE handle = LoadLibrary(dll_name);
  if (!handle) {
    std::cerr << "Failed to load library: " << GetLastError() << std::endl;
    return -1;
  }
  LoadModelFunc loadModel = (LoadModelFunc)GetProcAddress(handle, "loadModel");
  InferenceFunc inference = (InferenceFunc)GetProcAddress(handle, "inference");
#else
  void* handle = dlopen(dll_name, RTLD_LAZY);
  if (!handle) {
    std::cerr << "Failed to load library: " << dlerror() << std::endl;
    return -1;
  }
  LoadModelFunc loadModel = (LoadModelFunc)dlsym(handle, "loadModel");
  if (!loadModel) {
    std::cerr << "Failed to get function: loadModel" << std::endl;
  }
  InferenceFunc inference = (InferenceFunc)dlsym(handle, "inference");
  if (!inference) {
    std::cerr << "Failed to get function: inference" << std::endl;
  }
#endif

  if (!loadModel || !inference) {
    std::cerr << "Failed to get function!" << std::endl;
#ifdef _WIN32
    FreeLibrary(handle);
#else
    dlclose(handle);
#endif
    return -1;
  }

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
    bool loaded = loadModel(input.model_path.c_str(), input.metadata_size);
    if (loaded) {
      inference(input.input_img.c_str(), input.output_img.c_str());
    } else {
      std::cerr << "error: " << input.model_path << std::endl;
    }
  }

#ifdef _WIN32
  FreeLibrary(handle);
#else
  dlclose(handle);
#endif

  return 0;
}
