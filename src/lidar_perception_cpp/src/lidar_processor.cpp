#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <atomic>
#include <cmath>
#include <limits>

class LidarProcessor : public rclcpp::Node {

    public:
    explicit LidarProcessor(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
        : Node("lidar_processor", options) {
        this->declare_parameter<float>("gradientThresh", 0.26f);
        this->declare_parameter<float>("hMin", -0.78f);
        this->declare_parameter<int>("horizSamples", 512);
        this->declare_parameter<int>("vertSamples", 16);
        this->declare_parameter<float>("cropMinX", -0.90f);  
        this->declare_parameter<float>("cropMaxX",  0.38f); 
        this->declare_parameter<float>("cropMinY", -0.48f);
        this->declare_parameter<float>("cropMaxY",  0.48f);
        this->declare_parameter<float>("cropMinZ", -1.50f); 
        this->declare_parameter<float>("cropMaxZ",  0.50f);
        this->declare_parameter<float>("lidarHeight", 0.88f);
        this->declare_parameter<float>("maxRange", 15.0f);
        // Must match the <horizontal>/<vertical> min_angle/max_angle in steve.urdf's
        // gpu_ray sensor. Used to reconstruct each beam's direction so that "no
        // return" beams (NaN/Inf in the organized cloud) can still be turned into
        // valid free-space clearing endpoints instead of being silently dropped.
        this->declare_parameter<float>("hAngleMin", -3.14159f);
        this->declare_parameter<float>("hAngleMax",  3.14159f);
        this->declare_parameter<float>("vAngleMin", -0.25f);
        this->declare_parameter<float>("vAngleMax",  0.05f);

        lidarHeight = this->get_parameter("lidarHeight").as_double();
        gradientThresh = this->get_parameter("gradientThresh").as_double();
        hMin = this->get_parameter("hMin").as_double();
        horizSamples = this->get_parameter("horizSamples").as_int();
        vertSamples = this->get_parameter("vertSamples").as_int();
        minX = this->get_parameter("cropMinX").as_double();
        maxX = this->get_parameter("cropMaxX").as_double();
        minY = this->get_parameter("cropMinY").as_double();
        maxY = this->get_parameter("cropMaxY").as_double();
        minZ = this->get_parameter("cropMinZ").as_double();
        maxZ = this->get_parameter("cropMaxZ").as_double();
        maxRange_ = this->get_parameter("maxRange").as_double();
        maxRangeSq = maxRange_ * maxRange_;

        float hAngleMin = this->get_parameter("hAngleMin").as_double();
        float hAngleMax = this->get_parameter("hAngleMax").as_double();
        float vAngleMin = this->get_parameter("vAngleMin").as_double();
        float vAngleMax = this->get_parameter("vAngleMax").as_double();

        // Precompute a unit direction vector for every (v, u) beam in the
        // organized scan grid, in the same v*horizSamples + u order the rest
        // of this node already assumes the cloud is laid out in.
        beamDirections.resize(static_cast<size_t>(horizSamples) * vertSamples);
        for (int v = 0; v < vertSamples; v++) {
            float vAngle = vAngleMin + (vertSamples > 1 ? v * (vAngleMax - vAngleMin) / (vertSamples - 1) : 0.0f);
            float cv = std::cos(vAngle);
            float sv = std::sin(vAngle);
            for (int u = 0; u < horizSamples; u++) {
                float hAngle = hAngleMin + (horizSamples > 1 ? u * (hAngleMax - hAngleMin) / (horizSamples - 1) : 0.0f);
                Eigen::Vector3f dir(cv * std::cos(hAngle), cv * std::sin(hAngle), sv);
                beamDirections[(static_cast<size_t>(v) * horizSamples) + u] = dir;
            }
        }

        lidarSub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "scan", rclcpp::SensorDataQoS(), std::bind(&LidarProcessor::lidarCallback, this, std::placeholders::_1)
        );
        imuSub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "imu", rclcpp::SensorDataQoS(), std::bind(&LidarProcessor::imuCallback, this, std::placeholders::_1)
        );

        obstacleCloudPub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "obstacle_cloud", rclcpp::SensorDataQoS()
        );
        clearingCloudPub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "clearing_cloud", rclcpp::SensorDataQoS()
        );

        RCLCPP_INFO(this->get_logger(), "Successfully created Lidar Processor Node.");
    }

    private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidarSub_; 
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imuSub_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr obstacleCloudPub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr clearingCloudPub_;

    std::atomic<double> currentPitch{0.0};
    float lidarHeight, gradientThresh, hMin, minX, maxX, minY, maxY, minZ, maxZ, maxRangeSq, maxRange_;
    int horizSamples, vertSamples;
    std::vector<Eigen::Vector3f> beamDirections;


    void lidarCallback(sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr rawPclCloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *rawPclCloud);

        // Rotating the point cloud using the pitch
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.pretranslate(Eigen::Vector3f(0.0f, 0.0f,  lidarHeight));
        pcl::PointCloud<pcl::PointXYZ>::Ptr rotatedRawCloud(new pcl::PointCloud<pcl::PointXYZ>());
        transform.rotate(Eigen::AngleAxisf(currentPitch, Eigen::Vector3f::UnitY()));
        transform.pretranslate(Eigen::Vector3f(0.0f, 0.0f, -lidarHeight));
        pcl::transformPointCloud(*rawPclCloud, *rotatedRawCloud, transform);

        // Appplying gradient thresholding and minimum height filter.
        pcl::PointCloud<pcl::PointXYZ>::Ptr obstacleCloud(new pcl::PointCloud<pcl::PointXYZ>());
        int maxIdx = rotatedRawCloud->points.size() - 1;
        for(uint32_t v = 0; v < vertSamples-1; v++) {
            for(uint32_t u = 0; u < horizSamples; u++) {
                int currentIdx = (v * horizSamples) + u;
                int neighbourIdx = ((v + 1) * horizSamples) + u;
                if (neighbourIdx > maxIdx) {
                    continue;
                }
                auto point1 = rotatedRawCloud->points[currentIdx];
                auto point2 = rotatedRawCloud->points[neighbourIdx];
                if (!std::isfinite(point1.x) || !std::isfinite(point2.x) || 
                    !std::isfinite(point1.z) || !std::isfinite(point2.z)) {
                    continue;
                }
                float dZ = std::abs(point2.z - point1.z);
                float dX = point1.x - point2.x;
                float dY = point1.y - point2.y;
                float run  = std::sqrt((dX * dX) + (dY * dY));
                float grad;
                if(run == 0.0f) {
                    if(dZ == 0.0f) {
                        continue;
                    }
                    else {
                        grad = 999;
                    }
                }
                else {
                    grad = dZ / run;
                }
                if (grad > gradientThresh && point2.z >= hMin) {
                    const auto &rawPoint1 = rawPclCloud->points[currentIdx];
                    const auto &rawPoint2 = rawPclCloud->points[neighbourIdx];
                    if (point1.z >= hMin && (point1.x * point1.x + point1.y * point1.y) <= maxRangeSq) {
                        obstacleCloud->points.push_back(rawPoint1);
                    }
                    if (point2.z >= hMin && (point2.x * point2.x + point2.y * point2.y) <= maxRangeSq) {
                        obstacleCloud->points.push_back(rawPoint2);
                    }
                }
            }
        }

        // Cropping the points to prevent robot's chasis being classified as obstacle.
        pcl::PointCloud<pcl::PointXYZ>::Ptr croppedObstacleCloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::CropBox<pcl::PointXYZ> boxFilter;

        boxFilter.setMin(Eigen::Vector4f(minX, minY, minZ, 1.0f));
        boxFilter.setMax(Eigen::Vector4f(maxX, maxY, maxZ, 1.0f));

        boxFilter.setNegative(true); 
        boxFilter.setInputCloud(obstacleCloud);
        boxFilter.filter(*croppedObstacleCloud);

        // Build the free-space/clearing cloud. Every beam in the organized scan
        // (hit or miss) gets exactly one point here:
        //  - a real return within maxRange is kept as-is.
        //  - a beam with no return (NaN/Inf) or a return beyond maxRange is
        //    replaced with a synthetic endpoint at maxRange along that beam's
        //    known direction.
        // Previously, beams with no return were simply dropped (NaN comparisons
        // are always false), which meant any direction with nothing to reflect
        // off never produced a clearing ray at all — so stale lethal cells in
        // open space, or anything that had rotated out of the vision clearer's
        // forward FOV, could never be raytraced away again.
        pcl::PointCloud<pcl::PointXYZ>::Ptr rawClearingCloud(new pcl::PointCloud<pcl::PointXYZ>());
        rawClearingCloud->points.resize(rawPclCloud->points.size());
        for (size_t i = 0; i < rawPclCloud->points.size(); i++) {
            const auto &pt = rawPclCloud->points[i];
            float rangeSq = (pt.x * pt.x) + (pt.y * pt.y);
            if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z) && rangeSq <= maxRangeSq) {
                rawClearingCloud->points[i] = pt;
            } else if (i < beamDirections.size()) {
                const auto &dir = beamDirections[i];
                rawClearingCloud->points[i].x = dir.x() * maxRange_;
                rawClearingCloud->points[i].y = dir.y() * maxRange_;
                rawClearingCloud->points[i].z = dir.z() * maxRange_;
            } else {
                rawClearingCloud->points[i].x = std::numeric_limits<float>::quiet_NaN();
                rawClearingCloud->points[i].y = std::numeric_limits<float>::quiet_NaN();
                rawClearingCloud->points[i].z = std::numeric_limits<float>::quiet_NaN();
            }
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr rangedClearingCloud(new pcl::PointCloud<pcl::PointXYZ>());
        boxFilter.setInputCloud(rawClearingCloud);
        boxFilter.filter(*rangedClearingCloud);

        // Downsampling using voxel filtering
        pcl::VoxelGrid<pcl::PointXYZ> voxelGrid;
        voxelGrid.setInputCloud(croppedObstacleCloud);
        voxelGrid.setLeafSize(0.05f, 0.05f, 0.05f);
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampledCloud(new pcl::PointCloud<pcl::PointXYZ>());
        voxelGrid.filter(*downsampledCloud);
        auto finalObstacleCloud = std::make_unique<sensor_msgs::msg::PointCloud2>();
        pcl::toROSMsg(*downsampledCloud, *finalObstacleCloud);
        finalObstacleCloud->header = msg->header;
        obstacleCloudPub_->publish(std::move(finalObstacleCloud));

        voxelGrid.setInputCloud(rangedClearingCloud);
        voxelGrid.setLeafSize(0.1f, 0.1f, 0.1f);
        voxelGrid.filter(*downsampledCloud);
        auto finalClearingCloud = std::make_unique<sensor_msgs::msg::PointCloud2>();
        pcl::toROSMsg(*downsampledCloud, *finalClearingCloud);
        finalClearingCloud->header = msg->header;
        clearingCloudPub_->publish(std::move(finalClearingCloud));
    }

    void imuCallback(sensor_msgs::msg::Imu::SharedPtr msg) {
        tf2::Quaternion quaternion(
            msg->orientation.x,
            msg->orientation.y,
            msg->orientation.z,
            msg->orientation.w
        );
        tf2::Matrix3x3 quatMat(quaternion);

        double roll, pitch, yaw;
        quatMat.getRPY(roll, pitch, yaw);
        currentPitch = pitch;
    }
};

RCLCPP_COMPONENTS_REGISTER_NODE(LidarProcessor)

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LidarProcessor>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}