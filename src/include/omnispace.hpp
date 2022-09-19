#pragma once
#ifndef _OMNIAZ_CAMERA_HPP_
#define _OMNIAZ_CAMERA_HPP_

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream> 
#include "stdlib.h"

namespace omnispace {
    class Camera {
         public: 
            Camera(std::string);
            ~Camera(void);
            bool captureVideo(void);
            bool isRunning = false;
            cv::Mat frame;
            double dWidth; 
            double dHeight; 
            double fps; 
            
        private:
            cv::VideoCapture cap;
    };

    class Detector {
        public:
            Detector(std::string);
            ~Detector();
            void inferImage(cv::Mat&);
            void drawPredictions(cv::Mat&);
            std::vector<cv::Rect> filteredBoxes;

        private:
            cv::dnn::Net net;
            const float IM_WIDTH = 640.0;
            const float IM_HEIGHT = 640.0;
            const float SCORE_THRESH = 0.35;
            const float NMS_THRESH = 0.4;
            const float CONF_THRESH = 0.35;
            std::vector<cv::Mat> predictNet(cv::Mat&);
            void postProcess(cv::Mat&, std::vector<cv::Mat>&);
    };

    class Tracker {
        public:
            Tracker();
            Tracker(float, int, float, float, int, int, int);
            ~Tracker();
            void initTracker(cv::Mat&, cv::Rect, int);
            void updateTracker(cv::Mat&);
            void resetTracker();
            int trackingId = -1;
            cv::Rect trackedBox;

        private: 
            cv::Ptr<cv::Tracker> tracker;
            int untracked_history = 0;
            int UNTRACKED_THRESH = 5;
    };

    void updateTrackers(std::vector<omnispace::Tracker>&, cv::Mat& frame);
    void assignTrackers(cv::Mat&, std::vector<cv::Rect>&, std::vector<omnispace::Tracker>&,int&);
    std::vector<cv::Rect> refineTrackerMatches(std::vector<omnispace::Tracker>&, std::vector<cv::Rect>);
    void drawTrackerPredictions(cv::Mat&, std::vector<omnispace::Tracker>&);
}

#endif