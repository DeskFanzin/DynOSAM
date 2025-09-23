/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
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

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendInputPacket.hpp"
#include "dynosam/backend/BackendModule.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/backend/RGBDBackendDefinitions.hpp"
#include "dynosam/backend/VisionImuBackendModule.hpp"
#include "dynosam/backend/optimizers/ISAM2.hpp"
#include "dynosam/backend/optimizers/IncrementalOptimization.hpp"
#include "dynosam/backend/optimizers/SlidingWindowOptimization.hpp"
#include "dynosam/backend/rgbd/WorldMotionEstimator.hpp"
#include "dynosam/backend/rgbd/WorldPoseEstimator.hpp"
#include "dynosam/common/Flags.hpp"
#include "dynosam/common/Map.hpp"

namespace dyno {

class RegularBackendModule
    : public VisionImuBackendModule<RGBDBackendModuleTraits> {
 public:
  DYNO_POINTER_TYPEDEFS(RegularBackendModule)

  using Base = VisionImuBackendModule<RGBDBackendModuleTraits>;
  using RGBDMap = Base::MapType;
  using FormulationType = Base::FormulationType;

  // for backwards compatability!
  using UpdaterType = RGBDFormulationType;

  RegularBackendModule(const BackendParams& backend_params, Camera::Ptr camera,
                       const UpdaterType& updater_type,
                       ImageDisplayQueue* display_queue = nullptr);
  ~RegularBackendModule();

  using SpinReturn = Base::SpinReturn;

  const FormulationType* formulation() const { return formulation_.get(); }

  // also provide non-const access (this should only be used with caution and is
  // really only there to enable specific unit-tests!)
  FormulationType* formulation() { return formulation_.get(); }

  using PostFormulationUpdateCallback = std::function<void(
      const Formulation<RGBDMap>::UniquePtr&, FrameId, const gtsam::Values&,
      const gtsam::NonlinearFactorGraph&)>;
  void registerPostFormulationUpdateCallback(
      const PostFormulationUpdateCallback& cb) {
    post_formulation_update_cb_ = cb;
  }

 protected:
  void setupUpdates();

  void updateAndOptimize(FrameId frame_id_k, const gtsam::Values& new_values,
                         const gtsam::NonlinearFactorGraph& new_factors);
  void updateIncremental(FrameId frame_id_k, const gtsam::Values& new_values,
                         const gtsam::NonlinearFactorGraph& new_factors);
  void updateBatch(FrameId frame_id_k, const gtsam::Values& new_values,
                   const gtsam::NonlinearFactorGraph& new_factors);
  void updateSlidingWindow(FrameId frame_id_k, const gtsam::Values& new_values,
                           const gtsam::NonlinearFactorGraph& new_factors);

  void logIncrementalStats(
      FrameId frame_id_k,
      const IncrementalInterface<dyno::ISAM2>& smoother_interface) const;

  // TODO: for now
 protected:
  SpinReturn boostrapSpinImpl(
      RGBDInstanceOutputPacket::ConstPtr input) override;
  SpinReturn nominalSpinImpl(RGBDInstanceOutputPacket::ConstPtr input) override;

  void addInitialStates(const RGBDInstanceOutputPacket::ConstPtr& input,
                        FormulationType* formulation, gtsam::Values& new_values,
                        gtsam::NonlinearFactorGraph& new_factors);
  void addStates(const RGBDInstanceOutputPacket::ConstPtr& input,
                 FormulationType* formulation, gtsam::Values& new_values,
                 gtsam::NonlinearFactorGraph& new_factors);

  // initial pose can come from many sources
  void updateMapWithMeasurements(
      FrameId frame_id_k, const RGBDInstanceOutputPacket::ConstPtr& input,
      const gtsam::Pose3& X_k_w);

 private:
  // Also sets up error hooks based on the formulation
  Formulation<RGBDMap>::UniquePtr makeFormulation();

  BackendMetaData createBackendMetadata() const;
  FormulationHooks createFormulationHooks() const;
  BackendOutputPacket::Ptr constructOutputPacket(FrameId frame_k,
                                                 Timestamp timestamp) const;
  static BackendOutputPacket::Ptr constructOutputPacket(
      const Formulation<RGBDMap>::UniquePtr& formulation, FrameId frame_k,
      Timestamp timestamp);

  Camera::Ptr camera_;
  const UpdaterType updater_type_;
  Formulation<RGBDMap>::UniquePtr formulation_;
  // new calibration every time
  inline auto getGtsamCalibration() const {
    const CameraParams& camera_params = camera_->getParams();
    return boost::make_shared<Camera::CalibrationType>(
        camera_params.constructGtsamCalibration<Camera::CalibrationType>());
  }

  // logger here!!
  BackendLogger::UniquePtr logger_{nullptr};
  DebugInfo debug_info_;
  ErrorHandlingHooks error_hooks_;

  // optimizers are set in setupUpdates() depending on
  SlidingWindowOptimization::UniquePtr sliding_window_opt_;
  std::unique_ptr<dyno::ISAM2> smoother_;

  //! External callback containing formulation data and new values and factors
  PostFormulationUpdateCallback post_formulation_update_cb_;
};

}  // namespace dyno
