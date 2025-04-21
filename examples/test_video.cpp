
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "definitions.h"
#include "my-yolo-inference.h"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "Correct Usage: ./test_video your_model your_video" << std::endl;
    return -1;
  }
  std::string model = argv[1];
  std::string video = argv[2];
  std::cout << "model: " << model << std::endl;
  std::cout << "video :" << video << std::endl;

  if (!MY_YOLO.loadModel(model.c_str())) {
    std::cerr << "Error loading model: " << model << std::endl;
    return -1;
  }

  cv::VideoCapture cap(video);
  if (!cap.isOpened()) {
    std::cerr << "Error loading video: " << video << std::endl;
    return -1;
  }

  double fps = cap.get(cv::CAP_PROP_FPS);
  int delay = static_cast<int>(1000.0 / fps);
  cv::Mat frame;
  int frame_count = 0;

  while (true) {
    double start_time = cv::getTickCount();
    cap >> frame;
    if (frame.empty()) {
      std::cout << "video read complete" << std::endl;
      break;
    }

    my_yolo::ImageData img_data;
    img_data.width = frame.cols;
    img_data.height = frame.rows;
    img_data.channels = frame.channels();
    img_data.data = frame.data;

    if (!MY_YOLO.inference(&img_data)) {
      std::cerr << "Failed to inference at frame " << frame_count << std::endl;
    }
    ++frame_count;

    double elapsed_time = (cv::getTickCount() - start_time) / cv::getTickFrequency();
    int frame_delay = std::max(1, delay - static_cast<int>(elapsed_time * 1000));

    cv::imshow("test_video", frame);

    if (cv::waitKey(frame_delay) == 'q') {
      break;
    }
  }

  cap.release();
  cv::destroyAllWindows();

  return 0;
}
