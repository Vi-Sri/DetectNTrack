#include "./include/omnispace.hpp"
#include "./include/omniqueue.hpp"
#include <thread>
#include <vector>
#include <chrono>

bool STOP_SIGNAL = false;
int FRAME_BUFFER = 50;
omnispace::OmniQueue<cv::Mat> frame_queue;
omnispace::OmniQueue<cv::Mat> process_queue;

void detectFrame(omnispace::Detector* car_detector) {
	cv::Mat frame;
	::process_queue.clear();
	static int car_tracking_id = 0;
	static bool is_started_tracking = false;
	int frame_counter = 0;
	std::vector<omnispace::Tracker> trackers;
	while(!::STOP_SIGNAL) {
		std::vector<cv::Rect> unmatched_detections;
		if(!::frame_queue.empty())  {    
			frame = ::frame_queue.dequeue();
			if (frame_counter % 3 == 0) {
				car_detector->inferImage(frame);
				if (!is_started_tracking) {
					omnispace::assignTrackers(frame, car_detector->filteredBoxes, trackers, car_tracking_id);
					is_started_tracking = true;
				} else {
					omnispace::updateTrackers(trackers, frame);
					if (car_detector->filteredBoxes.size() != trackers.size()){
						unmatched_detections = omnispace::refineTrackerMatches(trackers, car_detector->filteredBoxes);
						if (unmatched_detections.size() > 0) {
							std::cout<<"\nUnmatched detections : "<<unmatched_detections.size();
							omnispace::assignTrackers(frame, unmatched_detections, trackers, car_tracking_id);
						}
					}
				}
			} 
			omnispace::updateTrackers(trackers, frame);
			car_detector->drawPredictions(frame);
			omnispace::drawTrackerPredictions(frame, trackers);
			unmatched_detections.clear();
			::process_queue.enqueue(frame);
			frame_counter+=1;
		}
		if(::process_queue.size() >= ::FRAME_BUFFER) { 
			::process_queue.clear();	
		}
	}
	std::cout<<"\nStopping Detector thread"<<std::endl;
	return;
}

void grabFrame(omnispace::Camera* cam) {
	cv::Mat frame;
	::frame_queue.clear();
	while(!::STOP_SIGNAL) {
		frame = cam->captureVideo();
		if (::frame_queue.size() < ::FRAME_BUFFER) { 
			::frame_queue.enqueue(cam->frame);
		} else {
			::frame_queue.clear();
		}
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
	std::cout << "Stopping grabber thread" << std::endl;
	return;
}



int main()
{
	std::string model_path = cv::samples::findFile("/Users/srinivas_v/Work/Projects/Omniaz/models/yolov5n.onnx");
	omnispace::Detector Detector(model_path);

	std::string video_path = cv::samples::findFile("/Users/srinivas_v/Work/Projects/Omniaz/data/Cars-1900.mp4");
	omnispace::Camera cam(video_path);
    cam.captureVideo();
    
    cv::Mat pred_image; 
    cv::Mat frame;

	std::thread t1(detectFrame, &Detector);
	std::thread t2(grabFrame, &cam);


	while(true) {
		if(::process_queue.size() > 0)  {
			pred_image = ::process_queue.dequeue();
			if (!pred_image.empty()){
				cv::imshow("predicted", pred_image);
				cv::waitKey(1);
			}
		}
	}
	t1.join();
	t2.join();
    return 0;
}