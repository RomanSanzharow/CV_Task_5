#include <string>
#include <fstream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

std::vector<cv::Rect> get_region_of_interest(const cv::Mat &original, cv::Scalar lover, cv::Scalar upper) {
    cv::Mat thresholdedMat;
    cv::cvtColor(original, thresholdedMat, cv::COLOR_BGR2HSV_FULL);

    cv::inRange(
            thresholdedMat,
            lover,
            upper,
            thresholdedMat
    );

    cv::erode(
            thresholdedMat,
            thresholdedMat,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5))
    );

    cv::dilate(
            thresholdedMat,
            thresholdedMat,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5))
    );

    cv::dilate(
            thresholdedMat,
            thresholdedMat,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5))
    );
    cv::erode(
            thresholdedMat,
            thresholdedMat,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5))
    );

    cv::Canny(thresholdedMat, thresholdedMat, 100, 50, 5);

    std::vector<std::vector<cv::Point>> countours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(
            thresholdedMat,
            countours,
            hierarchy,
            CV_RETR_TREE,
            CV_CHAIN_APPROX_SIMPLE,
            cv::Point(0, 0)
    );

    std::vector<cv::Rect> rects;
    for (uint i = 0; i < countours.size(); ++i) {
        if (0 <= hierarchy[i][3]) {
            continue;
        }
        rects.push_back(cv::boundingRect(countours[i]));
    }

    return rects;
}

void detect_text(string input) {
    Mat original = imread(input);
    imshow("Source", original);

    auto red_regions = get_region_of_interest(original, cv::Scalar(0, 100, 100), cv::Scalar(5, 255, 255));
    for(auto red_region: red_regions) {
        cv::Mat red_roi = original(red_region);

        auto black_regions = get_region_of_interest(red_roi, cv::Scalar(0, 0, 0), cv::Scalar(140, 140, 60));
        for(auto black_region: black_regions) {
            cv::Mat black_roi = red_roi(black_region);
            Mat small;
            cvtColor(black_roi, small, CV_BGR2GRAY);

            Mat grad;
            Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
            morphologyEx(small, grad, MORPH_GRADIENT, morphKernel);

            Mat bw;
            threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);

            Mat connected;
            morphKernel = getStructuringElement(MORPH_RECT, Size(9, 1));
            morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);

            Mat mask = Mat::zeros(bw.size(), CV_8UC1);
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(connected, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

            if(hierarchy.empty()) {
                continue;
            }

            for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
                Rect rect = boundingRect(contours[idx]);
                Mat maskROI(mask, rect);

                drawContours(mask, contours, idx, Scalar(255, 255, 255), CV_FILLED);

                double r = (double) countNonZero(maskROI) / (rect.width * rect.height);

                if (r > 0.45 && (rect.height > 8 && rect.width > 8)) {
                    rectangle(black_roi, rect, Scalar(0, 255, 0), 2);
                }
            }
        }
    }

    imshow("text", original);
    waitKey(0);
}

int main(int argc, char *argv[]) {
    detect_text(string("../image/test_marker.jpg"));
    return 0;
}
