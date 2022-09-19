#include "./include/omnispace.hpp"

omnispace::Camera::Camera(std::string video_path) {
    std::cout<<"Initializing Camera..";
    if (not cap.open(video_path)) {
        std::cerr<<"Cannot open video file"<<std::endl;
        exit(0);
    }
    cap.set(cv::CAP_PROP_FPS, 30);
    dWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    dHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    fps = cap.get(cv::CAP_PROP_FPS);
}

omnispace::Camera::~Camera() {
    std::cout<<"Shutting down camera..";
    cap.release();
    frame = NULL;
}

bool omnispace::Camera::captureVideo(void) {
    if (cap.isOpened()){
        isRunning = true;
        cap >> frame;
        return true;
    }
    isRunning = false;
    return false;
}

omnispace::Detector::Detector(std::string model_path) {
    net = cv::dnn::readNet(model_path);
    if(!net.empty()) {
        std::cout<<"\nModel Loaded successfully\n";
    } else {
        std::cerr<<"\nModel not loaded\n";
    }
}

omnispace::Detector::~Detector() {
    net.~Net();
    std::cout<<"\nClearing Detector"<<std::endl;

}

std::vector<cv::Mat> omnispace::Detector::predictNet(cv::Mat& input_image) {
    cv::Mat blob;
    cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(IM_WIDTH, IM_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    return outputs;
}

void omnispace::Detector::postProcess(cv::Mat& input_image, std::vector<cv::Mat> &outputs){
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        float x_factor = input_image.cols / IM_WIDTH;
        float y_factor = input_image.rows / IM_HEIGHT;

        float *data = (float *)outputs[0].data;

        const int dimensions = 85;
        const int rows = 25200;
        for (int i = 0; i < rows; ++i) 
        {
            float confidence = data[4];
            if (confidence >= CONF_THRESH) 
            {
                float * classes_scores = data + 5;
                cv::Mat scores(1, 80, CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;
                cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
                if (max_class_score > SCORE_THRESH) 
                {
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);
                    float cx = data[0];
                    float cy = data[1];
                    float w = data[2];
                    float h = data[3];
                    int left = int((cx - 0.5 * w) * x_factor);
                    int top = int((cy - 0.5 * h) * y_factor);
                    int width = int(w * x_factor);
                    int height = int(h * y_factor);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
            data += dimensions;
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESH, NMS_THRESH, indices);

        for (int i = 0; i < indices.size(); i++) 
        {
            int idx = indices[i];
            if (class_ids[idx] == 2){
                cv::Rect box = boxes[idx];
                filteredBoxes.push_back(box);
            }  
        }
        boxes.clear();
        indices.clear();
        class_ids.clear();
        
    }

void omnispace::Detector::drawPredictions(cv::Mat& input_image){
    for (cv::Rect& box: filteredBoxes){
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        cv::rectangle(input_image, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(255, 178, 50), 2);
    }
}

void omnispace::Detector::inferImage(cv::Mat &frame) {
    std::vector<cv::Mat> detections;
    filteredBoxes.clear();
    detections = predictNet(frame);
    postProcess(frame, detections);
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000; 
    double t = net.getPerfProfile(layersTimes) / freq;
    std::string label = cv::format("\nInference time : %.2f ms", t);
    std::cout<<label<<std::endl;
    detections.clear();
    layersTimes.clear();
}

omnispace::Tracker::Tracker() {
    tracker = cv::TrackerMIL::create();
}

omnispace::Tracker::Tracker(
    float samplerInitInRadius,
    int samplerInitMaxNegNum, 
    float samplerSearchWinSize,
    float samplerTrackInRadius,
    int samplerTrackMaxPosNum,
    int samplerTrackMaxNegNum,
    int featureSetNumFeatures
) {
    cv::TrackerMIL::Params params;
    params.samplerInitInRadius = samplerInitInRadius; 
    params.samplerInitMaxNegNum = samplerInitMaxNegNum;
    params.samplerSearchWinSize = samplerSearchWinSize;
    params.samplerTrackInRadius = samplerTrackInRadius;
    params.samplerTrackMaxPosNum = samplerTrackMaxPosNum;
    params.samplerTrackMaxNegNum = samplerTrackMaxNegNum;
    params.featureSetNumFeatures = featureSetNumFeatures;
    tracker = cv::TrackerMIL::create(params);
}

omnispace::Tracker::~Tracker() {
    tracker.release();
    tracker = nullptr;
}

void omnispace::Tracker::initTracker(cv::Mat& frame, cv::Rect box, int id) {
    tracker->init(frame, box);
    trackingId = id; 
}

void omnispace::Tracker::updateTracker(cv::Mat& frame) {
    bool state; 
    cv::Rect bbox(0,0,0,0);
    state = tracker->update(frame, bbox);
    if (state) {
        trackedBox = bbox;
        untracked_history = 0;
    } else {
        untracked_history += 1;
        if (untracked_history > UNTRACKED_THRESH ) {
            resetTracker();
        }
    }
    
}

void omnispace::Tracker::resetTracker(){
    cv::Rect empty_box(0,0,0,0);
    trackingId = -1; 
    trackedBox = empty_box;
    untracked_history = 0;
}

void omnispace::assignTrackers(cv::Mat& frame, std::vector<cv::Rect>& boxes, std::vector<omnispace::Tracker>& trackers, int& track_id) {
    for (cv::Rect& box:boxes) {
        omnispace::Tracker tracker = omnispace::Tracker();
        tracker.initTracker(frame, box, track_id);
        trackers.push_back(tracker);
        track_id += 1;
    }
}

void omnispace::updateTrackers(std::vector<omnispace::Tracker>& trackers, cv::Mat& frame) {
    for (omnispace::Tracker& tracker: trackers){
        tracker.updateTracker(frame);
    }
}

void omnispace::drawTrackerPredictions(cv::Mat& frame, std::vector<omnispace::Tracker>& trackers) {
    for (omnispace::Tracker& tracker: trackers){
        cv::Rect box = tracker.trackedBox;
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        cv::rectangle(frame, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 255, 255), 2);
        cv::putText(frame, std::to_string(tracker.trackingId), cv::Point(left+20, top+20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 255), 2);
    }
}

float euclideanDist(cv::Point& p, cv::Point& q) {
    cv::Point diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

std::vector<cv::Rect> omnispace::refineTrackerMatches(std::vector<omnispace::Tracker>& trackers, std::vector<cv::Rect> boxes) {
    cv::Point centre_det(0,0);
    cv::Point centre_trk(0,0);
    int box_counter = 0;
    int track_counter = 0;
    while(track_counter < trackers.size()){
        centre_trk.x = trackers[track_counter].trackedBox.x + (int)(trackers[track_counter].trackedBox.width/2);
        centre_trk.y = trackers[track_counter].trackedBox.y + (int)(trackers[track_counter].trackedBox.height/2);
        if (boxes.size() > 0) {
            while(box_counter < boxes.size()) {
                centre_det.x = boxes[box_counter].x + (int)(boxes[box_counter].width/2);
                centre_det.y = boxes[box_counter].y + (int)(boxes[box_counter].height/2);
                if ( euclideanDist(centre_det, centre_trk) < 30.0) {
                    boxes.erase(boxes.begin() + box_counter);
                    std::cout<<"Here 1 : "<<trackers[track_counter].trackingId<<std::endl;
                    break;
                }
                box_counter += 1;
            }
            if (box_counter == boxes.size()-1) {
                std::cout<<"Here 2 : "<<trackers[track_counter].trackingId<<std::endl;
                trackers[track_counter].resetTracker();
                trackers[track_counter].~Tracker();
                trackers.erase(trackers.begin() + track_counter);
                track_counter -= 1;
            }
        }
        box_counter = 0;
        track_counter += 1;
    }
    return boxes;
}

