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

#include <gtsam/geometry/Pose3.h>

#include <optional>

#include "dynosam/common/StructuredContainers.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/frontend/RGBDInstance-Definitions.hpp"
#include "dynosam/utils/GtsamUtils.hpp"

namespace dyno_testing {

using namespace dyno;

class ScenarioBodyBase {
 public:
  DYNO_POINTER_TYPEDEFS(ScenarioBodyBase)

  virtual ~ScenarioBodyBase() {}

  virtual gtsam::Pose3 pose(FrameId frame_id) const = 0;  ///< pose at time t
  virtual gtsam::Pose3 motionWorld(
      FrameId frame_id) const = 0;  ///< motion in world frame from t-1 to t
  virtual gtsam::Pose3 motionBody(FrameId frame_id)
      const = 0;  ///< motion local frame from t-1 to t, in ^{t-1}X_{t-1}
  virtual gtsam::Pose3 motionWorldFromInitial(
      FrameId frame_id) const = 0;  ///< motion in world frame from 0 to t

  gtsam::Rot3 rotation(FrameId frame_id) const {
    return this->pose(frame_id).rotation();
  }
  gtsam::Vector3 translation(FrameId frame_id) const {
    return this->pose(frame_id).translation();
  }
};

class ScenarioBodyVisitor : public ScenarioBodyBase {
 public:
  DYNO_POINTER_TYPEDEFS(ScenarioBodyVisitor)
  virtual ~ScenarioBodyVisitor() {}

  virtual gtsam::Pose3 pose(FrameId frame_id) const = 0;  ///< pose at time t
  virtual gtsam::Pose3 motionWorld(
      FrameId frame_id) const = 0;  ///< motion in world frame from t-1 to t

  ///< motion local frame from t-1 to t, in ^{t-1}X_{t-1}
  virtual gtsam::Pose3 motionBody(FrameId frame_id) const override {
    // from t-1 to t
    const gtsam::Pose3 motion_k = motionWorld(frame_id);
    const gtsam::Pose3 pose_k = pose(frame_id);
    // TODO: check
    return pose_k.inverse() * motion_k * pose_k.inverse();
  }
  virtual gtsam::Pose3 motionWorldFromInitial(
      FrameId frame_id) const = 0;  ///< motion in world frame from 0 to t
};

class ScenarioBody : public ScenarioBodyBase {
 public:
  DYNO_POINTER_TYPEDEFS(ScenarioBody)

  ScenarioBody(ScenarioBodyVisitor::UniquePtr body_visitor)
      : body_visitor_(std::move(body_visitor)) {}

  gtsam::Pose3 pose(FrameId frame_id) const override {
    return body_visitor_->pose(frame_id);
  }
  gtsam::Pose3 motionWorld(FrameId frame_id) const override {
    return body_visitor_->motionWorld(frame_id);
  }
  gtsam::Pose3 motionBody(FrameId frame_id) const override {
    return body_visitor_->motionBody(frame_id);
  }
  gtsam::Pose3 motionWorldFromInitial(FrameId frame_id) const override {
    return body_visitor_->motionWorldFromInitial(frame_id);
  }

 protected:
  ScenarioBodyVisitor::UniquePtr body_visitor_;
};

using TrackedPoint = std::pair<TrackletId, gtsam::Point3>;
using TrackedPoints = std::vector<TrackedPoint>;

struct PointsGenerator {
  /**
   * @brief Static function to generate a unique tracklet for any generator. If
   * increment is true, the global tracklet id will be incremented
   *
   * @param increment
   * @return TrackletId
   */
  static TrackletId getTracklet(bool increment = true) {
    static TrackletId global_tracklet = 0;

    auto tracklet_id = global_tracklet;

    if (increment) global_tracklet++;
    return tracklet_id;
  }

  static TrackedPoint generateNewPoint(const gtsam::Point3& mean, double sigma,
                                       int32_t seed = 42) {
    gtsam::Point3 point = dyno::utils::perturbWithNoise(mean, sigma, seed);
    return std::make_pair(PointsGenerator::getTracklet(true), point);
  }
};

/**
 * @brief Base class that knows how to generate points given the
 * ScenarioBodyVisitor for an object
 *
 */
class ObjectPointGeneratorVisitor {
 public:
  DYNO_POINTER_TYPEDEFS(ObjectPointGeneratorVisitor)

  virtual ~ObjectPointGeneratorVisitor() = default;
  virtual TrackedPoints getPointsWorld(
      const ScenarioBodyVisitor::UniquePtr& body_visitor,
      FrameId frame_id) const = 0;
};

class StaticPointGeneratorVisitor {
 public:
  DYNO_POINTER_TYPEDEFS(StaticPointGeneratorVisitor)

  virtual ~StaticPointGeneratorVisitor() = default;
  virtual TrackedPoints getPointsWorld(FrameId frame_id) const = 0;
};

using Range = FrameRange<int>;

class RangesWithEnd : public std::vector<Range::Ptr> {
 public:
  using Base = std::vector<Range>;

  Range::Ptr find(FrameId query) const {
    for (const Range::Ptr& r : *this) {
      if (r->contains(query)) {
        return r;
      }
    }
    return nullptr;
  }

  RangesWithEnd& add(FrameId start, FrameId end) {
    this->push_back(std::make_shared<Range>(start, end, 0, false));
    return *this;
  }
};

struct ObjectBodyParams {
  // FrameId enters_scenario = 0;
  // FrameId leaves_scenario = std::numeric_limits<FrameId>::max();
  RangesWithEnd ranges;

  ObjectBodyParams(
      FrameId enters_scenario = 0,
      FrameId leaves_scenario = std::numeric_limits<FrameId>::max()) {
    addRange(enters_scenario, leaves_scenario);
  }

  ObjectBodyParams& addRange(FrameId enters_scenario, FrameId leaves_scenario) {
    ranges.add(enters_scenario, leaves_scenario);
    return *this;
  }
};

class ObjectBody : public ScenarioBody {
 public:
  DYNO_POINTER_TYPEDEFS(ObjectBody)

  ObjectBody(ScenarioBodyVisitor::UniquePtr body_visitor,
             ObjectPointGeneratorVisitor::UniquePtr points_visitor,
             const ObjectBodyParams& params = ObjectBodyParams())
      : ScenarioBody(std::move(body_visitor)),
        points_visitor_(std::move(points_visitor)),
        params_(params) {}

  virtual bool inFrame(FrameId frame_id) const {
    return (bool)params_.ranges.find(frame_id);
  }
  virtual TrackedPoints getPointsWorld(FrameId frame_id) const {
    return points_visitor_->getPointsWorld(body_visitor_, frame_id);
  };

 protected:
  ObjectPointGeneratorVisitor::UniquePtr points_visitor_;
  ObjectBodyParams params_;
};

// Motion and pose visotors
class ConstantMotionBodyVisitor : public ScenarioBodyVisitor {
 public:
  DYNO_POINTER_TYPEDEFS(ConstantMotionBodyVisitor)
  ConstantMotionBodyVisitor(const gtsam::Pose3& pose_0,
                            const gtsam::Pose3& motion)
      : pose_0_(pose_0), motion_(motion) {}

  virtual gtsam::Pose3 pose(FrameId frame_id) const override {
    // from Pose Changes From a Different Point of View
    return motionWorldFromInitial(frame_id) * pose_0_;
  }

  virtual gtsam::Pose3 motionWorld(FrameId) const override { return motion_; }

  // TODO: I have no idea if this is right for constant motion but whatevs...
  gtsam::Pose3 motionWorldFromInitial(FrameId frame_id) const {
    return gtsam::Pose3::Expmap(frame_id * gtsam::Pose3::Logmap(motion_));
  }

 private:
  const gtsam::Pose3 pose_0_;
  const gtsam::Pose3 motion_;
};

class RandomOverlapObjectPointsVisitor : public ObjectPointGeneratorVisitor {
 public:
  RandomOverlapObjectPointsVisitor(size_t num_points, size_t overlap)
      : num_points_(num_points),
        overlap_(overlap),
        overlap_dist_(2, std::max(2, (int)overlap)) {}

  TrackedPoints getPointsWorld(
      const ScenarioBodyVisitor::UniquePtr& body_visitor,
      FrameId frame_id) const override {
    // minimum n_points per frame
    // get points for frame + 1
    // if size points < num points -> generate N = (num points - len(points))
    // points are generated with 2 <-> O(verlap) as number of points to exist
    std::vector<Point> points_next = getPoints(frame_id + 1);
    // LOG(INFO) << "Points next=" <<points_next.size();
    std::vector<Point> points_current = getPoints(frame_id);
    // LOG(INFO) << "Points current=" <<points_current.size();
    if (points_next.size() < num_points_) {
      auto required_points = num_points_ - points_next.size();
      // LOG(INFO) << "Required points=" << required_points;
      std::vector<Point> points_new = generatePoints(frame_id, required_points);
      points_current.insert(points_current.end(), points_new.begin(),
                            points_new.end());
    }
    // LOG(INFO) << "New Points current=" <<points_current.size();
    // CHECK_EQ(points_current.size(), num_points_);

    TrackedPoints tracked_points(points_current.size());
    std::transform(points_current.begin(), points_current.end(),
                   tracked_points.begin(),
                   [&body_visitor, &frame_id](const Point& p_body) {
                     // LOG(INFO) <<
                     CHECK_NOTNULL(body_visitor);
                     CHECK(p_body.contains(frame_id));
                     const gtsam::Point3 P_world =
                         body_visitor->pose(frame_id) * p_body.P_body_.second;
                     return std::make_pair(p_body.P_body_.first, P_world);
                   });

    return tracked_points;
  }

 private:
  struct Point {
    FrameId starting_frame_;
    FrameId ending_frame_;
    TrackedPoint P_body_;  //! Point in the object body frame

    Point(FrameId starting_frame, FrameId ending_frame,
          const TrackedPoint& P_body)
        : starting_frame_(starting_frame),
          ending_frame_(ending_frame),
          P_body_(P_body) {
      CHECK_GT(ending_frame_, starting_frame_);
    }

    bool contains(FrameId frame_id) const {
      return frame_id >= starting_frame_ && frame_id <= ending_frame_;
    }
  };

  Point generatePoint(FrameId frame) const {
    auto O = overlap_dist_(gen);
    auto ending_frame = frame + O;

    auto tracked_point = PointsGenerator::generateNewPoint(
        gtsam::Point3(0, 0, 0), 0.1, seed_dist_(gen));
    Point p(frame, ending_frame, tracked_point);
    all_points_.push_back(p);
    return p;
  }

  std::vector<Point> getPoints(FrameId frame) const {
    std::vector<Point> points;
    for (const Point& point : all_points_) {
      if (point.contains(frame)) {
        points.push_back(point);
      }
    }
    return points;
  }

  std::vector<Point> generatePoints(FrameId frame, size_t N) const {
    std::vector<Point> points;
    for (size_t i = 0; i < N; i++) {
      points.push_back(generatePoint(frame));
    }

    return points;
  }

  const size_t num_points_;
  const size_t overlap_;

  static std::random_device rd;
  static std::mt19937 gen;

  mutable std::uniform_int_distribution<> overlap_dist_;
  mutable std::uniform_int_distribution<int> seed_dist_{0, 100};

  mutable std::vector<Point> all_points_;
};
inline std::random_device RandomOverlapObjectPointsVisitor::rd;
inline std::mt19937 RandomOverlapObjectPointsVisitor::gen{rd()};

// class BetterRandomOverlapObjectPointsVisitor : public
// ObjectPointGeneratorVisitor {
//   public:
//    RandomOverlapObjectPointsVisitor(size_t num_points, int max_frame, size_t
//    overlap)
//        : num_points_(num_points), max_frames_(max_frame) overlap_(overlap) {}

//    TrackedPoints getPointsWorld(
//        const ScenarioBodyVisitor::UniquePtr& body_visitor,
//        FrameId frame_id) const override {
//      // minimum n_points per frame
//      // get points for frame + 1
//      // if size points < num points -> generate N = (num points -
//      len(points))
//      // points are generated with 2 <-> O(verlap) as number of points to
//      exist std::vector<Point> points_next = getPoints(frame_id + 1);
//      // LOG(INFO) << "Points next=" <<points_next.size();
//      std::vector<Point> points_current = getPoints(frame_id);
//      // LOG(INFO) << "Points current=" <<points_current.size();
//      if (points_next.size() < num_points_) {
//        auto required_points = num_points_ - points_next.size();
//        // LOG(INFO) << "Required points=" << required_points;
//        std::vector<Point> points_new = generatePoints(frame_id,
//        required_points); points_current.insert(points_current.end(),
//        points_new.begin(),
//                              points_new.end());
//      }
//      // LOG(INFO) << "New Points current=" <<points_current.size();
//      // CHECK_EQ(points_current.size(), num_points_);

//      TrackedPoints tracked_points(points_current.size());
//      std::transform(points_current.begin(), points_current.end(),
//                     tracked_points.begin(),
//                     [&body_visitor, &frame_id](const Point& p_body) {
//                       // LOG(INFO) <<
//                       CHECK_NOTNULL(body_visitor);
//                       CHECK(p_body.contains(frame_id));
//                       const gtsam::Point3 P_world =
//                           body_visitor->pose(frame_id) *
//                           p_body.P_body_.second;
//                       return std::make_pair(p_body.P_body_.first, P_world);
//                     });

//      return tracked_points;
//    }

//   private:
//    struct Point {
//      FrameId starting_frame_;
//      FrameId ending_frame_;
//      TrackedPoint P_body_;  //! Point in the object body frame

//      Point(FrameId starting_frame, FrameId ending_frame,
//            const TrackedPoint& P_body)
//          : starting_frame_(starting_frame),
//            ending_frame_(ending_frame),
//            P_body_(P_body) {
//        CHECK_GT(ending_frame_, starting_frame_);
//      }

//      bool contains(FrameId frame_id) const {
//        return frame_id >= starting_frame_ && frame_id <= ending_frame_;
//      }
//    };

//    void generateAllInitialPoints() {
//     for(size_t i = 0; i < num_points_; i++) {
//       auto tracked_point = PointsGenerator::generateNewPoint(
//           gtsam::Point3(0, 0, 0), 0.1, seed_dist(gen));
//     }
//    }

//    Point generatePoint(FrameId frame) const {
//      std::uniform_int_distribution<> distrib(std::max(1, (int)overlap_ - 2),
//                                              (int)overlap_ + 2);
//      auto O = distrib(gen);
//      auto ending_frame = frame + O;

//      std::uniform_int_distribution<int> seed_dist(0, 100);
//      auto tracked_point = PointsGenerator::generateNewPoint(
//          gtsam::Point3(0, 0, 0), 0.1, seed_dist(gen));
//      Point p(frame, ending_frame, tracked_point);
//      all_points_.push_back(p);
//      return p;
//    }

//    std::vector<Point> getPoints(FrameId frame) const {
//      std::vector<Point> points;
//      for (const Point& point : all_points_) {
//        if (point.contains(frame)) {
//          points.push_back(point);
//        }
//      }
//      return points;
//    }

//    std::vector<Point> generatePoints(FrameId frame, size_t N) const {
//      std::vector<Point> points;
//      for (size_t i = 0; i < N; i++) {
//        points.push_back(generatePoint(frame));
//      }

//      return points;
//    }

//    const size_t num_points_;
//    const size_t overlap_;
//    const int max_frames_;

//    static std::random_device rd;
//    static std::mt19937 gen;

//    mutable std::vector<Point> all_points_;
//  };

// Points generator visitor
class ConstantObjectPointsVisitor : public ObjectPointGeneratorVisitor {
 public:
  ConstantObjectPointsVisitor(size_t num_points) : num_points_(num_points) {}

  // TODO: this assumes that the points we get from the object are ALWAYS the
  // same and ALWAYS the same order
  //
  TrackedPoints getPointsWorld(
      const ScenarioBodyVisitor::UniquePtr& body_visitor,
      FrameId frame_id) const override {
    if (!is_init) {
      initalisePoints(body_visitor->pose(0));
    }

    TrackedPoints points_world_t;  // points in world frame at time t
    for (const auto& tracked_point : points_world_0_) {
      auto tracklet_id = tracked_point.first;
      auto point = tracked_point.second;
      points_world_t.push_back(std::make_pair(
          tracklet_id, body_visitor->motionWorldFromInitial(frame_id) * point));
    }

    return points_world_t;
  }

 private:
  void initalisePoints(const gtsam::Pose3& P0) const {
    std::mt19937 engine(42);
    std::uniform_real_distribution<double> normal(0.0, 1.0);

    for (size_t i = 0; i < num_points_; i++) {
      // generate around pose0 with a normal distrubution around the translation
      // component
      // gtsam::Point3 p(P0.x() + normal(engine), P0.y() + normal(engine),
      //           P0.z() + normal(engine));

      // points_world_0_.push_back(std::make_pair(PointsGenerator::getTracklet(true),
      // p));
      points_world_0_.push_back(
          PointsGenerator::generateNewPoint(P0.translation(), 1.0));
    }

    is_init = true;
  }

  const size_t num_points_;

  // mutable so can be changed in the initalised poitns function, which is
  // called once
  mutable TrackedPoints points_world_0_;  // points in the world frame at time 0
  mutable bool is_init{false};
};

// I think this only ever means that a point can be seen by a max o
class SimpleStaticPointsGenerator : public StaticPointGeneratorVisitor {
 public:
  SimpleStaticPointsGenerator(size_t num_points_per_frame, size_t overlap)
      : num_points_per_frame_(num_points_per_frame),
        overlap_(overlap),
        has_overlap_(overlap < num_points_per_frame) {}

  TrackedPoints getPointsWorld(FrameId frame_id) const override {
    // expect we always start at zero
    if (frame_id == 0) {
      generateNewPoints(num_points_per_frame_);
      return points_world_0_;
    } else {
      // must have at least this many points after the first (zeroth) frame
      CHECK_GE(points_world_0_.size(), num_points_per_frame_);

      CHECK(has_overlap_) << "not implemented";
      int diff = (int)num_points_per_frame_ - (int)overlap_;
      CHECK(diff > 0);
      generateNewPoints((size_t)diff);

      size_t start_i = frame_id * ((size_t)diff);
      CHECK_GT(start_i, 0);

      size_t end_i = start_i + num_points_per_frame_ - 1;
      CHECK_LT(end_i, points_world_0_.size());

      TrackedPoints points_in_window;
      for (size_t i = start_i; i <= end_i; i++) {
        points_in_window.push_back(points_world_0_.at(i));
      }

      CHECK_EQ(points_in_window.size(), num_points_per_frame_);
      return points_in_window;
    }
  }

 private:
  void generateNewPoints(size_t num_new) const {
    // points can be distributed over this distance
    constexpr double point_distance_sigma = 40;
    for (size_t i = 0; i < num_new; i++) {
      points_world_0_.push_back(PointsGenerator::generateNewPoint(
          gtsam::Point3(0, 0, 0), point_distance_sigma));
    }
  }

  const size_t num_points_per_frame_;
  const size_t overlap_;
  const bool has_overlap_;
  mutable TrackedPoints
      points_world_0_;  // all points in the world frame at time 0. This may be
                        // uppdated overtime within the getPointsWorld
};

class Scenario {
 public:
  Scenario(ScenarioBody::Ptr camera_body,
           StaticPointGeneratorVisitor::Ptr static_points_generator)
      : camera_body_(camera_body),
        static_points_generator_(static_points_generator) {}

  void addObjectBody(ObjectId object_id, ObjectBody::Ptr object_body) {
    CHECK_GT(object_id, background_label);
    object_bodies_.insert2(object_id, object_body);
  }

  gtsam::Pose3 cameraPose(FrameId frame_id) const {
    return camera_body_->pose(frame_id);
  }

  ObjectIds getObjectIds(FrameId frame_id) const {
    ObjectIds object_ids;
    for (const auto& [object_id, obj] : object_bodies_) {
      if (objectInScenario(object_id, frame_id))
        object_ids.push_back(object_id);
    }

    return object_ids;
  }

  bool objectInScenario(ObjectId object_id, FrameId frame_id) const {
    if (object_bodies_.exists(object_id)) {
      const auto& object = object_bodies_.at(object_id);

      return object->inFrame(frame_id);
    }
    return false;
  }

 protected:
  ScenarioBody::Ptr camera_body_;
  StaticPointGeneratorVisitor::Ptr static_points_generator_;
  gtsam::FastMap<ObjectId, ObjectBody::Ptr> object_bodies_;

  mutable ObjectMotionMap object_motions_;
  mutable ObjectMotionMap noisy_object_motions_;
  // ObjectPoseMap object_poses_;
};

class RGBDScenario : public Scenario {
 public:
  struct NoiseParams {
    double H_R_sigma{0.0};
    double H_t_sigma{0.0};

    //! rotation noise on relative camera motion
    double X_R_sigma{0.0};
    //! translation noise on relative camera motion
    double X_t_sigma{0.0};

    double dynamic_point_sigma{0.0};
    double static_point_sigma{0.0};

    NoiseParams() {}
  };

  RGBDScenario(ScenarioBody::Ptr camera_body,
               StaticPointGeneratorVisitor::Ptr static_points_generator,
               const NoiseParams& noise_params = NoiseParams())
      : Scenario(camera_body, static_points_generator),
        noise_params_(noise_params) {}

  // first is gt, second is with noisy
  using Output =
      std::pair<RGBDInstanceOutputPacket::Ptr, RGBDInstanceOutputPacket::Ptr>;

  Output getOutput(FrameId frame_id) const {
    StatusLandmarkVector static_landmarks, dynamic_landmarks,
        noisy_static_landmarks, noisy_dynamic_landmarks;
    StatusKeypointVector static_keypoint_measurements,
        dynamic_keypoint_measurements;

    GroundTruthInputPacket gt_packet;
    gt_packet.frame_id_ = frame_id;

    MotionEstimateMap motions, noisy_motions;
    const gtsam::Pose3 X_world_k = cameraPose(frame_id);
    gt_packet.X_world_ = X_world_k;

    gtsam::Pose3 noisy_X_world_k;
    gtsam::Pose3 w_T_k_1_k;
    gtsam::Pose3 noisy_w_T_k_1_k;
    if (frame_id > 0) {
      // add noise on relative transformation of camera pose using gt poses to
      // calculate gt realtive pose
      const gtsam::Pose3 X_world_k_1 = cameraPose(frame_id - 1u);
      w_T_k_1_k = X_world_k_1.inverse() * X_world_k;

      gtsam::Vector6 pose_sigmas;
      pose_sigmas.head<3>().setConstant(noise_params_.X_R_sigma);
      pose_sigmas.tail<3>().setConstant(noise_params_.X_t_sigma);
      noisy_w_T_k_1_k = dyno::utils::perturbWithNoise(w_T_k_1_k, pose_sigmas);

      CHECK(noisy_camera_poses_.exists(frame_id - 1u));
      noisy_X_world_k = noisy_camera_poses_.at(frame_id - 1u) * noisy_w_T_k_1_k;
    } else {
      noisy_X_world_k = X_world_k;
    }

    // tracklets should be uniqyue but becuase we use the DynamicPointSymbol
    // they only need to be unique per frame
    for (const auto& [object_id, object] : object_bodies_) {
      if (objectInScenario(object_id, frame_id)) {
        const gtsam::Pose3 H_world_k = object->motionWorld(frame_id);
        const gtsam::Pose3 L_world_k = object->pose(frame_id);
        TrackedPoints points_world = object->getPointsWorld(frame_id);

        ObjectPoseGT object_pose_gt;
        object_pose_gt.frame_id_ = frame_id;
        object_pose_gt.object_id_ = object_id;
        object_pose_gt.L_world_ = L_world_k;
        object_pose_gt.prev_H_current_world_ = H_world_k;
        gt_packet.object_poses_.push_back(object_pose_gt);

        FrameId previous_frame;
        if (frame_id > 0) {
          previous_frame = frame_id - 1u;
        } else {
          previous_frame = 0u;  // hack? should actually skip this case
        }

        motions.insert2(object_id,
                        Motion3ReferenceFrame(
                            H_world_k, Motion3ReferenceFrame::Style::F2F,
                            ReferenceFrame::GLOBAL, previous_frame, frame_id));

        gtsam::Vector6 motion_sigmas;
        motion_sigmas.head<3>().setConstant(noise_params_.H_R_sigma);
        motion_sigmas.tail<3>().setConstant(noise_params_.H_t_sigma);
        const gtsam::Pose3 noisy_H_world_k =
            dyno::utils::perturbWithNoise(H_world_k, motion_sigmas);
        noisy_motions.insert2(
            object_id, Motion3ReferenceFrame(
                           noisy_H_world_k, Motion3ReferenceFrame::Style::F2F,
                           ReferenceFrame::GLOBAL, previous_frame, frame_id));

        // convert to status vectors
        for (const TrackedPoint& tracked_p_world : points_world) {
          auto tracklet_id = tracked_p_world.first;
          auto p_world = tracked_p_world.second;
          const Point3Measurement p_camera(X_world_k.inverse() * p_world);
          const Point3Measurement noisy_p_camera(dyno::utils::perturbWithNoise(
              p_camera.measurement(), noise_params_.dynamic_point_sigma));

          // LOG(INFO) << p_camera;
          // LOG(INFO) << noisy_p_camera;

          auto landmark_status = dyno::LandmarkStatus::DynamicInLocal(
              p_camera, frame_id, tracklet_id, object_id);
          dynamic_landmarks.push_back(landmark_status);

          auto noisy_landmark_status = dyno::LandmarkStatus::DynamicInLocal(
              noisy_p_camera, frame_id, tracklet_id, object_id);
          noisy_dynamic_landmarks.push_back(noisy_landmark_status);

          // the keypoint sttatus should be unused in the RGBD case but
          // we need it to fill out the data structures
          auto keypoint_status = dyno::KeypointStatus::DynamicInLocal(
              KeypointMeasurement(dyno::Keypoint()), frame_id, tracklet_id,
              object_id);
          dynamic_keypoint_measurements.push_back(keypoint_status);
        }
      }
    }

    // add static points
    const TrackedPoints static_points_world =
        static_points_generator_->getPointsWorld(frame_id);

    // convert to status vectors
    for (const TrackedPoint& tracked_p_world : static_points_world) {
      auto tracklet_id = tracked_p_world.first;
      auto p_world = tracked_p_world.second;
      const Point3Measurement p_camera(X_world_k.inverse() * p_world);
      const Point3Measurement noisy_p_camera(dyno::utils::perturbWithNoise(
          p_camera.measurement(), noise_params_.static_point_sigma));

      auto landmark_status =
          dyno::LandmarkStatus::StaticInLocal(p_camera, frame_id, tracklet_id);
      static_landmarks.push_back(landmark_status);

      auto noisy_landmark_status = dyno::LandmarkStatus::StaticInLocal(
          noisy_p_camera, frame_id, tracklet_id);
      noisy_static_landmarks.push_back(noisy_landmark_status);

      // the keypoint sttatus should be unused in the RGBD case but
      // we need it to fill out the data structures
      auto keypoint_status = dyno::KeypointStatus::StaticInLocal(
          KeypointMeasurement(dyno::Keypoint()), frame_id, tracklet_id);
      static_keypoint_measurements.push_back(keypoint_status);
    }

    ground_truths_.insert2(frame_id, gt_packet);
    noisy_camera_poses_.insert2(frame_id, noisy_X_world_k);

    object_motions_.insert2(frame_id, motions);
    noisy_object_motions_.insert2(frame_id, noisy_motions);

    auto gt_output = std::make_shared<RGBDInstanceOutputPacket>(
        static_keypoint_measurements, dynamic_keypoint_measurements,
        static_landmarks, dynamic_landmarks, X_world_k, frame_id, frame_id,
        object_motions_, ObjectPoseMap{}, gtsam::Pose3Vector{}, nullptr,
        gt_packet);
    gt_output->T_k_1_k_ = w_T_k_1_k;

    auto noisy_output = std::make_shared<RGBDInstanceOutputPacket>(
        static_keypoint_measurements, dynamic_keypoint_measurements,
        noisy_static_landmarks, noisy_dynamic_landmarks, noisy_X_world_k,
        frame_id, frame_id, noisy_object_motions_, ObjectPoseMap{},
        gtsam::Pose3Vector{}, nullptr, gt_packet);
    noisy_output->T_k_1_k_ = noisy_w_T_k_1_k;

    return {gt_output, noisy_output};
  }

  const GroundTruthPacketMap& getGroundTruths() const { return ground_truths_; }

 private:
  NoiseParams noise_params_;
  mutable GroundTruthPacketMap ground_truths_;
  mutable gtsam::FastMap<FrameId, gtsam::Pose3> noisy_camera_poses_;
};

}  // namespace dyno_testing
