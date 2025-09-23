/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#pragma once

#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "dynosam/common/GroundTruthPacket.hpp"

namespace dyno {

void throwExceptionIfPathInvalid(const std::string& image_path);

void loadRGB(const std::string& image_path, cv::Mat& img);

// CV_32F (float)
void loadFlow(const std::string& image_path, cv::Mat& img);

// CV_64F (double)
void loadDepth(const std::string& image_path, cv::Mat& img);

// CV_32SC1
// this is old kitti style and loads from a .txt file (which is why we need the
// size)
void loadSemanticMask(const std::string& image_path, const cv::Size& size,
                      cv::Mat& mask);

// CV_32SC1
void loadMask(const std::string& image_path, cv::Mat& mask);

/**
 * @brief Considers the conversion from the left-handed coordiante system (e.g.
 * unreal, VIODE, carla...) to the right-handed coordinate system (robotic).
 *
 * Taken from:
 * https://github.com/carla-simulator/ros-bridge/blob/master/carla_common/src/carla_common/transforms.py#L41
 *
 * @param right_handed_linear_velocity gtsam::Vector3&
 * @param right_handed_angular_velocity gtsam::Vector3&
 * @param left_handed_linear_velocity const gtsam::Vector3&
 * @param left_handed_angular_velocity const gtsam::Vector3&
 * @param left_handed_rotation std::optional<gtsam::Rot3>
 */
void toRightHandedTwist(gtsam::Vector3& right_handed_linear_velocity,
                        gtsam::Vector3& right_handed_angular_velocity,
                        const gtsam::Vector3& left_handed_linear_velocity,
                        const gtsam::Vector3& left_handed_angular_velocity,
                        std::optional<gtsam::Rot3> left_handed_rotation = {});

/**
 * @brief Considers the conversion from left-handed system (e.g. unreal, VIODE,
 * carla...) to right-handed system.
 *
 * @param left_handed_rotation const gtsam::Rot3&
 * @return gtsam::Rot3
 */
gtsam::Rot3 toRightHandedRotation(const gtsam::Rot3& left_handed_rotation);

/**
 * @brief Considers the conversion from a left-handed system (e.g. unreal,
 * VIODE, carla...) vector to right-handed system with optional rotation
 * provided in the left-handed system.
 *
 * @param left_handed_vector const gtsam::Vector3&
 * @param left_handed_rotation std::optional<gtsam::Rot3>
 * @return gtsam::Vector3
 */
gtsam::Vector3 toRightHandedVector(
    const gtsam::Vector3& left_handed_vector,
    std::optional<gtsam::Rot3> left_handed_rotation);

/**
 * @brief Returns a ORDERED vector of all files in the given directory.
 * (jesse) is this the file name or the absolute file path?
 *
 * @param folder_path
 * @return std::vector<std::filesystem::path>
 */
std::vector<std::filesystem::path> getAllFilesInDir(
    const std::string& folder_path);

void loadPathsInDirectory(
    std::vector<std::string>& file_paths, const std::string& folder_path,
    const std::function<bool(const std::string&)>& condition =
        std::function<bool(const std::string&)>());

/**
 * @brief From an instance semantic mask (one that satisfies the requirements
 * for a SemanticMask), ie. all detected obejcts in the scene, with unique
 * instance labels as pixel values starting from 1 (background is 0). Using the
 * information in the ground truth, masks of detected objects that are not
 * moving are removed. Masks are removed by setting the pixle values to 0
 * (background).
 *
 * Expects the gt packet to be fully formed (ie. have all motion information set
 * from setMotions()).
 *
 *
 *
 *
 * NOTE: function expects to find a match between the object id's in the ground
 * truth packet, and the pixel values (instance mask values) in the image!! If
 * there is not a 1-to-1 match between ground truth packets and the pixels, the
 * function will fail
 *
 *
 * @param instance_mask const cv::Mat&
 * @param motion_mask cv::Mat&
 * @param gt_packet const GroundTruthInputPacket&
 */
void removeStaticObjectFromMask(const cv::Mat& instance_mask,
                                cv::Mat& motion_mask,
                                const GroundTruthInputPacket& gt_packet);

/**
 * @brief Gets the next line from the input ifstream and returns it as split
 * string (using white space as the delimieter). Any newline/carriage
 * return/trailing white space values are trimmed.
 *
 * @param fstream std::ifstream&
 * @param split_lines std::vector<std::string>&
 * @return true
 * @return false
 */
bool getLine(std::ifstream& fstream, std::vector<std::string>& split_lines);

/**
 * @brief Takes an input string and splits it using white space (" ") as the
 * delimiter. Trims any newline/carriage return/trailing white space values.
 *
 * @param input const std::string&
 * @return std::vector<std::string>
 */
std::vector<std::string> trimAndSplit(const std::string& input,
                                      const std::string& delimiter = " ");

}  // namespace dyno
