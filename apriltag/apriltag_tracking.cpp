#include <opencv2/opencv.hpp>
#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>
#include <apriltag/apriltag_pose.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <cmath>
#include <cstdlib>
#ifdef __unix__
  #include <pwd.h>
  #include <unistd.h>
#endif
namespace fs = std::filesystem;

//8-6-25 JML
//augustine's aprilTag.py code converted into C++ format courtesy of chatgpt
//

// JSON-like structure for logging (simple CSV alternative)
struct TrackingEntry {
    std::string timestamp;
    std::string camera;
    int tag_id;
    std::string arm;
    double x, y, z;
    cv::Mat transform_matrix;
};

class AprilTagTracker {
private:
    // Configuration
    static constexpr double TAG_SIZE = 0.0355; // meters, aka 3 inches (i regret my decision JML)
    static constexpr double fx = 600.0, fy = 600.0, cx = 320.0, cy = 240.0;
    
    // AprilTag detector
    apriltag_detector_t* td;
    apriltag_family_t* tf;
    
    // Cameras
    cv::VideoCapture cap0, cap1;
    
    // Arm mapping
    //0, 2, and 8 are unhealthy arm tags of the 36h11 family
    //3, 5, and 5 are healthy arm tags of the ibid family
    std::map<int, std::string> arm_map = {
        {0, "impacted"}, {8, "impacted"}, {2, "impacted"},
        {3, "healthy"}, {4, "healthy"}, {5, "healthy"}
    };
    
    // Chains for skeleton drawing
    std::vector<std::pair<int, int>> impacted_chain = {{0, 8}, {8, 2}};
    std::vector<std::pair<int, int>> healthy_chain = {{3, 4}, {4, 5}};
    
    // Data logs
    std::vector<TrackingEntry> impacted_log;
    std::vector<TrackingEntry> healthy_log;
    
    // Output directory
    std::string output_dir;
    
    // Camera intrinsics
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;

public:
    AprilTagTracker() {
        // Initialize AprilTag detector
        tf = tag36h11_create();
        td = apriltag_detector_create();
        apriltag_detector_add_family(td, tf);
        
        // Camera parameters
        camera_matrix = (cv::Mat_<double>(3,3) << 
            fx, 0, cx,
            0, fy, cy,
            0, 0, 1);
        dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
        
        // Setup output directory
        setupOutputDirectory();
        
        // Initialize cameras
        if (!initializeCameras()) {
            throw std::runtime_error("Failed to initialize cameras");
        }
    }
    
    ~AprilTagTracker() {
        apriltag_detector_destroy(td);
        tag36h11_destroy(tf);
        cap0.release();
        cap1.release();
        cv::destroyAllWindows();
    }
    
    bool initializeCameras() {
        cap0.open(0); // impacted arm camera
        cap1.open(2); // healthy arm camera
        
        if (!cap0.isOpened() || !cap1.isOpened()) {
            std::cerr << "Error: Could not open cameras" << std::endl;
            return false;
        }
        
        // Set camera properties for better performance
        cap0.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap0.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap1.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        
        return true;
    }
    
void setupOutputDirectory() {
    const std::string home = resolveHomeDir();
    fs::path base = fs::path(home) / "Documents/StrokeRehabProject/chai3d-3.3.0/chai3d-3.3.0/bin/apriltagData/";

    output_dir = base.string();

    fs::create_directories(base);
    fs::create_directories(base / "session_graphs");
    fs::create_directories(base / "joint_plots");
    fs::create_directories(base / "joint_overlays");

    // Debug so you can confirm at runtime
    std::cout << "[CARES] Output directory: " << output_dir << "\n";
}

    
    std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }


    static std::string resolveHomeDir() {
#ifdef __unix__
    // Prefer the invoking user's home if running under sudo
    if (const char* sudo_user = std::getenv("SUDO_USER")) {
        if (passwd* pw = getpwnam(sudo_user)) {
            if (pw->pw_dir && pw->pw_dir[0] != '\0') return std::string(pw->pw_dir);
        }
    }
    // Normal HOME
    if (const char* home = std::getenv("HOME")) {
        if (home[0] != '\0') return std::string(home);
    }
    // Fallback to current user's passwd entry
    if (passwd* pw = getpwuid(getuid())) {
        if (pw->pw_dir && pw->pw_dir[0] != '\0') return std::string(pw->pw_dir);
    }
#else
    // Windows
    if (const char* prof = std::getenv("USERPROFILE")) {
        if (prof[0] != '\0') return std::string(prof);
    }
#endif
    // Last resort
    return fs::current_path().string();
}
    
    void processFrame(cv::Mat& frame, const std::string& cam_name) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        // AprilTag detection
        image_u8_t im = { 
            .width = gray.cols,
            .height = gray.rows,
            .stride = gray.cols,
            .buf = gray.data
        };
        
        zarray_t* detections = apriltag_detector_detect(td, &im);
        
        std::map<int, cv::Point3d> tag_positions;
        std::map<int, cv::Point2i> tag_centers;
        
        for (int i = 0; i < zarray_size(detections); i++) {
            apriltag_detection_t* det;
            zarray_get(detections, i, &det);
            
            int tag_id = det->id;
            if (arm_map.find(tag_id) == arm_map.end() || 
                arm_map[tag_id] != cam_name) {
                continue; // Skip tags not belonging to this camera/arm
            }
            
            // Draw tag outline and center
            cv::Point2i center(det->c[0], det->c[1]);
            tag_centers[tag_id] = center;
            
            // Draw corners
            std::vector<cv::Point2i> corners;
            for (int j = 0; j < 4; j++) {
                corners.push_back(cv::Point2i(det->p[j][0], det->p[j][1]));
            }
            
            for (int j = 0; j < 4; j++) {
                cv::line(frame, corners[j], corners[(j+1)%4], cv::Scalar(0, 255, 0), 2);
            }
            cv::circle(frame, center, 5, cv::Scalar(0, 0, 255), -1);
            cv::putText(frame, "ID:" + std::to_string(tag_id), 
                       cv::Point(center.x + 10, center.y),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
            
            // Pose estimation
            apriltag_detection_info_t info;
            info.det = det;
            info.tagsize = TAG_SIZE;
            info.fx = fx; info.fy = fy;
            info.cx = cx; info.cy = cy;
            
            apriltag_pose_t pose;
            double err = estimate_tag_pose(&info, &pose);
            
            // Calculate distance from camera
            double dist_cam = sqrt(pose.t->data[0]*pose.t->data[0] + 
                                 pose.t->data[1]*pose.t->data[1] + 
                                 pose.t->data[2]*pose.t->data[2]);
            
            cv::putText(frame, std::to_string(dist_cam).substr(0, 4) + "m",
                       cv::Point(center.x + 10, center.y + 20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(128, 0, 255), 2);
            
            // Store position
            cv::Point3d pos(pose.t->data[0], pose.t->data[1], pose.t->data[2]);
            tag_positions[tag_id] = pos;
            
            // Create log entry
            TrackingEntry entry;
            entry.timestamp = getCurrentTimestamp();
            entry.camera = cam_name;
            entry.tag_id = tag_id;
            entry.arm = cam_name;
            entry.x = pos.x;
            entry.y = pos.y;
            entry.z = pos.z;
            
            // Store transform matrix (simplified)
            entry.transform_matrix = cv::Mat::eye(4, 4, CV_64F);
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++) {
                    entry.transform_matrix.at<double>(r, c) = pose.R->data[r*3 + c];
                }
                entry.transform_matrix.at<double>(r, 3) = pose.t->data[r];
            }
            
            if (cam_name == "impacted") {
                impacted_log.push_back(entry);
            } else {
                healthy_log.push_back(entry);
            }
        }
        
        // Draw skeleton chains
        auto& chain = (cam_name == "impacted") ? impacted_chain : healthy_chain;
        cv::Scalar color = (cam_name == "impacted") ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
        
        for (const auto& link : chain) {
            int id1 = link.first, id2 = link.second;
            if (tag_positions.find(id1) != tag_positions.end() && 
                tag_positions.find(id2) != tag_positions.end()) {
                
                cv::Point2i p1 = tag_centers[id1];
                cv::Point2i p2 = tag_centers[id2];
                
                cv::Point3d pos1 = tag_positions[id1];
                cv::Point3d pos2 = tag_positions[id2];
                double dist = cv::norm(pos2 - pos1);
                
                cv::Point2i mid((p1.x + p2.x)/2, (p1.y + p2.y)/2);
                cv::line(frame, p1, p2, color, 2);
                cv::putText(frame, std::to_string(dist).substr(0, 5) + "m", mid,
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            }
        }
        
        apriltag_detections_destroy(detections);
    }
    
    void run() {
        cv::Mat frame0, frame1;
        
        std::cout << "AprilTag Tracker Started. Press ESC to exit." << std::endl;
        
        while (true) {
            bool ret0 = cap0.read(frame0);
            bool ret1 = cap1.read(frame1);
            
            if (!ret0 || !ret1) {
                std::cerr << "Failed to read from cameras" << std::endl;
                break;
            }
            
            // Process both cameras
            processFrame(frame0, "impacted");
            processFrame(frame1, "healthy");
            
            // Display frames
            cv::imshow("Impacted Arm Tracker", frame0);
            cv::imshow("Healthy Arm Tracker", frame1);
            
            // Check for ESC key
            int key = cv::waitKey(1) & 0xFF;
            if (key == 27) { // ESC key
                break;
            }
        }
        
        // Save data after session
        saveToCSV();
        std::cout << "Session data saved to CSV files." << std::endl;
    }
    
std::string getUniqueFilename(const std::string& basePath, const std::string& filename) {
    namespace fs = std::filesystem;
    fs::path fullPath = fs::path(basePath) / filename;
    if (!fs::exists(fullPath)) {
        return fullPath.string();
    }

    std::string stem = fullPath.stem().string();
    std::string ext  = fullPath.extension().string();
    int counter = 1;
    while (true) {
        fs::path trial = fs::path(basePath) / (stem + "_" + std::to_string(counter) + ext);
        if (!fs::exists(trial)) {
            return trial.string();
        }
        ++counter;
    }
}

void saveToCSV() {
    namespace fs = std::filesystem;
    fs::create_directories(output_dir); // ensure output_dir exists

    // impacted arm file
    std::string impactedPath = getUniqueFilename(output_dir, "/impacted_arm.csv");
    std::ofstream impacted_file(impactedPath);
    impacted_file << "timestamp,camera,tag_id,arm,x,y,z\n";
    for (const auto& entry : impacted_log) {
        impacted_file << entry.timestamp << "," << entry.camera << "," 
                      << entry.tag_id << "," << entry.arm << ","
                      << entry.x << "," << entry.y << "," << entry.z << "\n";
    }
    impacted_file.close();
    std::cout << "Saved impacted arm data to " << impactedPath << "\n";

    // healthy arm file
    std::string healthyPath = getUniqueFilename(output_dir, "/healthy_arm.csv");
    std::ofstream healthy_file(healthyPath);
    healthy_file << "timestamp,camera,tag_id,arm,x,y,z\n";
    for (const auto& entry : healthy_log) {
        healthy_file << entry.timestamp << "," << entry.camera << "," 
                     << entry.tag_id << "," << entry.arm << ","
                     << entry.x << "," << entry.y << "," << entry.z << "\n";
    }
    healthy_file.close();
    std::cout << "Saved healthy arm data to " << healthyPath << "\n";
}
};

int main() {
    try {
        AprilTagTracker tracker;
        tracker.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
