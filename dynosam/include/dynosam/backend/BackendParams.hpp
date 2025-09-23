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

#include <gflags/gflags.h>

#include <string>

// defined in BackendParamd.cc
DECLARE_double(static_point_sigma);
DECLARE_double(dynamic_point_sigma);

DECLARE_double(constant_object_motion_rotation_sigma);
DECLARE_double(constant_object_motion_translation_sigma);

DECLARE_double(motion_ternary_factor_noise_sigma);

DECLARE_double(odometry_rotation_sigma);
DECLARE_double(odometry_translation_sigma);

DECLARE_double(static_point_noise_sigma);
DECLARE_double(dynamic_point_noise_sigma);

DECLARE_bool(use_smoothing_factor);

namespace dyno {

struct BackendParams {
  //! Only Mono
  double static_smart_projection_noise_sigma_ =
      FLAGS_static_point_sigma;  //! Isotropic noise used for the smart
                                 //! projection factor (mono) on static points
  double dynamic_smart_projection_noise_sigma_ =
      FLAGS_dynamic_point_sigma;  //! Isotropic noise used for the smart
                                  //! projection factor (mono) on dynamic points

  //! RGBD/Stereo
  bool use_robust_kernals_ = true;
  double k_huber_3d_points_ =
      0.0001;  //! Huber constant used for robust kernal on dynamic points
  double static_point_noise_sigma_ =
      FLAGS_static_point_noise_sigma;  //! Isotropic noise used on
                                       //! PoseToPointFactor for static points
  // TODO: make param!!! and really should come from covariance on image plane
  // and then projection!!!
  double dynamic_point_noise_sigma_ =
      FLAGS_dynamic_point_noise_sigma;  //! Isotropic noise used on
                                        //! PoseToPointFactor for dynamic points
                                        //! //0.0125

  double odometry_rotation_sigma_ =
      FLAGS_odometry_rotation_sigma;  //! sigma used to construct the noise
                                      //! model on the rotation component of the
                                      //! odomety (between factor)
  double odometry_translation_sigma_ =
      FLAGS_odometry_translation_sigma;  //! sigma used to construct the noise
                                         //! model on the translation component
                                         //! of the odomety (between factor)

  double constant_object_motion_rotation_sigma_ =
      FLAGS_constant_object_motion_rotation_sigma;
  double constant_object_motion_translation_sigma_ =
      FLAGS_constant_object_motion_translation_sigma;

  double motion_ternary_factor_noise_sigma_ =
      FLAGS_motion_ternary_factor_noise_sigma;

  bool use_logger_ = true;  // TODO: make param!?
  bool use_use_full_batch_opt = true;

  bool use_smoothing_factor = FLAGS_use_smoothing_factor;

  size_t min_static_obs_ = 2u;
  size_t min_dynamic_obs_ = 3u;

  static BackendParams fromYaml(const std::string& file_path);

  BackendParams& useLogger(bool value) {
    this->use_logger_ = value;
    return *this;
  }

  // for testing only
  mutable int full_batch_frame = -1;
};

}  // namespace dyno
