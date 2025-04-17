#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>

#include "base64.h"

class Utils {
 public:
  Utils();
  ~Utils();

  static std::string Img2Base64(const cv::Mat& img) {
    // encode as PNG
    std::vector<uchar> buf;
    cv::imencode(".png", img, buf);
    std::string encoded = "data:image/png;base64," + base64_encode(buf.data(), buf.size(), false);
    return encoded;
  }

  static cv::Mat Letterbox(const cv::Mat& img, const cv::Size& new_shape,
                           const cv::Scalar& color = cv::Scalar(114, 114, 114), bool scale_up = true) {
    int img_w = img.cols;
    int img_h = img.rows;

    float r = std::min((float)new_shape.width / img_w, (float)new_shape.height / img_h);
    if (!scale_up) {
      r = std::min(r, 1.0f);
    }

    int resized_w = std::round(img_w * r);
    int resized_h = std::round(img_h * r);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resized_w, resized_h));

    int dw = new_shape.width - resized_w;
    int dh = new_shape.height - resized_h;

    // Padding: left, right, top, bottom
    int top = std::floor(dh / 2.0);
    int bottom = dh - top;
    int left = std::floor(dw / 2.0);
    int right = dw - left;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT, color);

    return padded;
  }

  static void ScaleImage(cv::Mat& scaled_mask, const cv::Mat& resized_mask, const cv::Size& im0_shape,
                         const std::pair<float, cv::Point2f>& ratio_pad = std::make_pair(-1.0f,
                                                                                         cv::Point2f(-1.0f, -1.0f))) {
    cv::Size im1_shape = resized_mask.size();

    // Check if resizing is needed
    if (im1_shape == im0_shape) {
      scaled_mask = resized_mask.clone();
      return;
    }

    float gain, pad_x, pad_y;

    if (ratio_pad.first < 0.0f) {
      gain = std::min(static_cast<float>(im1_shape.height) / static_cast<float>(im0_shape.height),
                      static_cast<float>(im1_shape.width) / static_cast<float>(im0_shape.width));
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

  static cv::Scalar Color(const int& index) {
    uint8_t r = (index * 41) % 256;
    uint8_t g = (index * 73) % 256;
    uint8_t b = (index * 97) % 256;

    return cv::Scalar(b, g, r);
  }

  static void ClipBox(cv::Rect_<float>& box, const cv::Size& shape) {
    box.x = std::max(0.0f, std::min(box.x, static_cast<float>(shape.width)));
    box.y = std::max(0.0f, std::min(box.y, static_cast<float>(shape.height)));
    box.width = std::max(0.0f, std::min(box.width, static_cast<float>(shape.width - box.x)));
    box.height = std::max(0.0f, std::min(box.height, static_cast<float>(shape.height - box.y)));
  }

  static cv::Rect_<float> ScaleBox(const cv::Size& img1_shape, cv::Rect_<float>& box, const cv::Size& img0_shape,
                                   std::pair<float, cv::Point2f> ratio_pad = std::make_pair(-1.0f,
                                                                                            cv::Point2f(-1.0f, -1.0f)),
                                   bool padding = true) {
    float gain, pad_x, pad_y;
    if (ratio_pad.first < 0.0f) {
      gain = std::min(static_cast<float>(img1_shape.height) / static_cast<float>(img0_shape.height),
                      static_cast<float>(img1_shape.width) / static_cast<float>(img0_shape.width));
      pad_x = roundf((img1_shape.width - img0_shape.width * gain) / 2.0f - 0.1f);
      pad_y = roundf((img1_shape.height - img0_shape.height * gain) / 2.0f - 0.1f);
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
    ClipBox(scaled_coords, img0_shape);

    return scaled_coords;
  }

  static cv::Point2f ScalePoint(const cv::Size& input_size, const cv::Size& image_size, const cv::Point2f& pt) {
    float r_w = input_size.width / (float)image_size.width;
    float r_h = input_size.height / (float)image_size.height;

    float scale = std::min(r_w, r_h);

    float new_unpad_w = scale * image_size.width;
    float new_unpad_h = scale * image_size.height;
    float dw = (input_size.width - new_unpad_w) / 2;
    float dh = (input_size.height - new_unpad_h) / 2;

    float x = (pt.x - dw) / scale;
    float y = (pt.y - dh) / scale;

    return cv::Point2f(x, y);
  }
};

#endif  // UTILS_H
