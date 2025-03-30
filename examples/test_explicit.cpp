#include <iostream>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

typedef void (*LoadModelFunc)(const char*);
typedef void (*InferenceFunc)(const char*, const char*);

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

  loadModel("../examples/res/yolo11n-seg.onnx");
  inference("../examples/res/apple.jpg", "result.jpg");

#ifdef _WIN32
  FreeLibrary(handle);
#else
  dlclose(handle);
#endif

  return 0;
}
