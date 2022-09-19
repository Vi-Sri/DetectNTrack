#include "./include/omnispace.hpp"
#include "./include/omniqueue.hpp"
#include <thread>
#include <vector>
#include <chrono>

int main()
{
	std::string model_path = cv::samples::findFile("/Users/srinivas_v/Work/Projects/Omniaz/models/yolov5n.onnx");
	omnispace::Detector car_detector(model_path);

	std::string video_path = cv::samples::findFile("/Users/srinivas_v/Work/Projects/Omniaz/data/Cars-1900.mp4");
	omnispace::Camera cam(video_path);
    cam.captureVideo();

	int frame_counter = 0;
	static int car_tracking_id = 0;
	static bool is_started_tracking = false;
	std::vector<omnispace::Tracker> trackers;
	std::vector<cv::Rect> unmatched_detections;    
    cv::Mat frame;

	while(true) {
		bool ret = cam.captureVideo();
		if (ret){
			frame = cam.frame;
			if (frame_counter % 3 == 0) {
				car_detector.inferImage(frame);
				if (!is_started_tracking) {
					omnispace::assignTrackers(frame, car_detector.filteredBoxes, trackers, car_tracking_id);
					is_started_tracking = true;
				} else {
					omnispace::updateTrackers(trackers, frame);
					if (car_detector.filteredBoxes.size() != trackers.size()){
						unmatched_detections = omnispace::refineTrackerMatches(trackers, car_detector.filteredBoxes);
						if (unmatched_detections.size() > 0) {
							std::cout<<"\nUnmatched detections : "<<unmatched_detections.size();
							omnispace::assignTrackers(frame, unmatched_detections, trackers, car_tracking_id);
						}
					}
				}
			} 
			omnispace::updateTrackers(trackers, frame);
			car_detector.drawPredictions(frame);
			omnispace::drawTrackerPredictions(frame, trackers);
			cv::imshow("predicted", frame);
			frame_counter += 1;
			if (cv::waitKey(1) == 27) 		
				{
					std::cout << "\nEsc pressed\n" << std::endl;
					break; 
				}
		}
	}
    return 0;
}