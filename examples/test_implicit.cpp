#include "my-yolo-inference.h"

int main(int argc, char *argv[]) {
  MY_YOLO.loadModel("../examples/res/yolo11n-seg.onnx");
  MY_YOLO.inference("../examples/res/apple.jpg", "result.jpg");
  return 0;
}
