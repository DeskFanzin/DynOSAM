/*
 *   Copyright (c) 2025 ACFR-RPG, University of Sydney, Jesse Morris
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

#include <functional>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/utils/Timing.hpp"

namespace dyno {

/**
 * @brief Defines a traits interface to any incremental smoother.
 * This must be defined for each type
 *
 * @tparam SMOOTHER
 */
template <typename SMOOTHER>
struct iOptimizationTraits {
  // Template for deriving
  // typedef MySmoother Smoother;
  // typedef MyResult ResultType; ///< This is the type returned by the
  // equivalent of SMOOTHER::update typedef MyUpdateArguments UpdateArguments;
  // ///< Generic arguments to the SMOOTHER::update, this should encapsulate all
  // arguments
  ///< (as they change depending on SMOOTHER) but must include gtsam::Values
  ///< new_values, gtsam::NonlinearFactorGraph new_factors
  // static ResultType update(Smoother& smoother, const UpdateArguments&
  // update_arguments); ///< Performs SMOOTHER::update given the full set of
  // arguments
};

struct UpdateArguments {
  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;
};

namespace internal {
/**
 * @brief iOptimizationTraits for something that looks exactly like ISAM2
 * Really only exists becuase we have a custom dyno::ISAM2 class
 *
 * @tparam _Smoother smoother type e.g. ISAM2
 * @tparam _Result Type returned by Smoother::update
 * @tparam _Update ISAM2UpdateParams type
 */
template <typename _Smoother, typename _Result, typename _Update>
struct isam2_like_traits {
  typedef isam2_like_traits<_Smoother, _Result, _Update> This;
  typedef _Smoother Smoother;
  typedef _Result ResultType;

  struct ISAM2UpdateArguments : public UpdateArguments {
    _Update update_params;
  };

  //! Must contain a gtsam::Values new_values and a gtsam::NonlinearFactorGraph
  //! new_factors
  typedef ISAM2UpdateArguments UpdateArguments;

  // Functional way to fll the update arguments from some external source before
  // update
  using FillArguments = std::function<void(const Smoother&, UpdateArguments&)>;

  static ResultType update(Smoother& smoother,
                           const UpdateArguments& update_arguments) {
    return smoother.template update(update_arguments.new_factors,
                                    update_arguments.new_values,
                                    update_arguments.update_params);
  }

  static ResultType update(Smoother& smoother,
                           const FillArguments& update_arguments_filler) {
    UpdateArguments arguments;
    update_arguments_filler(smoother, arguments);

    return This::update(smoother, arguments);
  }

  static gtsam::NonlinearFactorGraph getFactors(const Smoother& smoother) {
    return smoother.template getFactorsUnsafe();
  }

  static gtsam::Values calculateEstimate(const Smoother& smoother) {
    return smoother.template calculateBestEstimate();
  }

  static gtsam::Values getLinearizationPoint(const Smoother& smoother) {
    return smoother.template getLinearizationPoint();
  }
};
}  // namespace internal

template <>
struct iOptimizationTraits<gtsam::ISAM2>
    : public internal::isam2_like_traits<gtsam::ISAM2, gtsam::ISAM2Result,
                                         gtsam::ISAM2UpdateParams> {};

struct ErrorHandlingHooks {
  ErrorHandlingHooks() {}

  /**
   * @brief IndeterminateLinearSystem (ILS) result that will be used by the
   * IncrementalInterface to try and handle any ILS errors that occur during
   * smoother update
   *
   */
  struct HandleILSResult {
    //! New factors to try and add to handle the the ILS. Usually priors on
    //! undetermined values
    gtsam::NonlinearFactorGraph new_factors;
    //! Indication of objects which had a undetermined variable
    //! If non-empty, used in conjunction with OnFailedObject
    std::vector<std::pair<FrameId, ObjectId>> failed_objects;
  };
  /// @brief Alias to a function call that takes the current set of smoother
  /// values and the nearby gtsam::Key which threw the
  /// IndeterminantLinearSystemException
  using OnIndeterminateLinearSystem =
      std::function<HandleILSResult(const gtsam::Values&, gtsam::Key)>;
  using OnFailedObject =
      std::function<void(const std::pair<FrameId, ObjectId>&)>;

  //! Called when smoother update throws a
  //! gtsam::IndeterminantLinearSystemException Usually used to add additional
  //! priors to the set of existing factors in order to handle the error
  OnIndeterminateLinearSystem handle_ils_exception;
  //! Called if HandleILSResult::failed_objects has entries and AFTER the system
  //! has been attempted to be stabilised with new priors from the
  //! handle_ils_exception
  OnFailedObject handle_failed_object;
};

template <typename SMOOTHER>
class IncrementalInterface {
 public:
  typedef iOptimizationTraits<SMOOTHER> SmootherTraitsType;
  typedef typename SmootherTraitsType::Smoother Smoother;
  typedef typename SmootherTraitsType::UpdateArguments UpdateArguments;
  typedef typename SmootherTraitsType::ResultType ResultType;
  typedef typename SmootherTraitsType::FillArguments FillArguments;

  IncrementalInterface(Smoother* smoother)
      : smoother_(CHECK_NOTNULL(smoother)) {}

  bool optimize(ResultType* result,
                const FillArguments& update_arguments_filler,
                const ErrorHandlingHooks& error_hooks = {}) {
    CHECK_NOTNULL(result);
    auto tic = utils::Timer::tic();

    bool is_smoother_ok =
        updateSmoother(result, update_arguments_filler, error_hooks);

    if (is_smoother_ok) {
      // use dummy isam result when running optimize without new values/factors
      // as we want to use the result to determine which values were
      // changed/marked
      // TODO: maybe we actually need to append results together?
      static ResultType dummy_result;
      static UpdateArguments empty_arguments;
      VLOG(30) << "Doing extra iteration nr: " << max_extra_iterations_;
      for (size_t n_iter = 0; n_iter < max_extra_iterations_ && is_smoother_ok;
           ++n_iter) {
        is_smoother_ok &=
            updateSmoother(&dummy_result, empty_arguments, error_hooks);
      }
    }

    auto toc = utils::Timer::toc<std::chrono::nanoseconds>(tic);
    timing_ =
        std::chrono::duration_cast<std::chrono::milliseconds>(toc).count();
    was_smoother_ok_ = is_smoother_ok;
    result_ = *result;

    return is_smoother_ok;
  }

  Smoother* smoother() const { return smoother_; }
  int64_t timing() const { return timing_; }
  bool wasSmootherOk() const { return was_smoother_ok_; }
  const ResultType& result() const { return result_; }

  IncrementalInterface& setMaxExtraIterations(size_t max_extra_iterations) {
    max_extra_iterations_ = max_extra_iterations;
    return *this;
  };

 protected:
  bool updateSmoother(ResultType* result,
                      const FillArguments& update_arguments_filler,
                      const ErrorHandlingHooks& error_hooks) {
    // pre-fill arguments for this iteration
    UpdateArguments smoother_arguments;
    update_arguments_filler(*smoother_, smoother_arguments);
    return updateSmoother(result, smoother_arguments, error_hooks);
  }

  bool updateSmoother(ResultType* result,
                      const UpdateArguments& smoother_arguments,
                      const ErrorHandlingHooks& error_hooks) {
    // This is not doing a full deep copy: it is keeping same shared_ptrs for
    // factors but copying the isam result.
    Smoother smoother_backup(*smoother_);

    gtsam::Values new_values = smoother_arguments.new_values;
    gtsam::NonlinearFactorGraph new_factors = smoother_arguments.new_factors;

    try {
      *result = SmootherTraitsType::update(*smoother_, smoother_arguments);

    } catch (gtsam::IndeterminantLinearSystemException& e) {
      const gtsam::Key& var = e.nearbyVariable();
      LOG(ERROR) << "gtsam::IndeterminantLinearSystemException with variable "
                 << DynoLikeKeyFormatter(var);

      if (!error_hooks.handle_ils_exception) {
        throw e;
      }
      const gtsam::Values values =
          SmootherTraitsType::calculateEstimate(*smoother_);
      ErrorHandlingHooks::HandleILSResult ils_handle_result =
          error_hooks.handle_ils_exception(values, var);

      // New prior factors that will be used update smoother
      const gtsam::NonlinearFactorGraph& nfg = ils_handle_result.new_factors;

      if (nfg.size() == 0) {
        LOG(WARNING) << DynoLikeKeyFormatter(var)
                     << " not recognised in indeterminant exception handling";
        return false;
      }

      gtsam::NonlinearFactorGraph new_factors_mutable;
      new_factors_mutable.push_back(new_factors.begin(), new_factors.end());
      new_factors_mutable.push_back(nfg.begin(), nfg.end());

      // Update with graph and GN optimized values
      try {
        // Update smoother
        LOG(ERROR) << "Attempting to update smoother with added prior factors";
        // update smoother_arguments with new factors containing the priors
        // this should be the same as the original EXCEPT for the new factors
        UpdateArguments smoother_arguments_copy;
        smoother_arguments_copy.new_factors = new_factors_mutable;
        *smoother_ = smoother_backup;  // reset smoother to backup
        *result =
            SmootherTraitsType::update(*smoother_, smoother_arguments_copy);
      } catch (...) {
        // Catch the rest of exceptions.
        LOG(WARNING)
            << "Smoother recovery failed. Most likely, the additional "
               "prior factors were insufficient to keep the system from "
               "becoming indeterminant.";
        return false;
      }

      // if successful handle any failed objects
      if (error_hooks.handle_failed_object) {
        for (const auto& failed_obj_pair : ils_handle_result.failed_objects) {
          const auto [frame_id, object_id] = failed_obj_pair;
          VLOG(10) << "Handling object with indeterminant variables "
                   << info_string(frame_id, object_id);
          error_hooks.handle_failed_object(failed_obj_pair);
        }
      }

    } catch (gtsam::ValuesKeyDoesNotExist& e) {
      LOG(FATAL) << "gtsam::ValuesKeyDoesNotExist with variable "
                 << DynoLikeKeyFormatter(e.key());
    }
    return true;
  }

 protected:
  Smoother* smoother_;
  size_t max_extra_iterations_ = 3u;

  //! state variables indicating result of last call to optimize
  //! Time in ms for last call to optimize
  int64_t timing_;
  //! Result of last call to optimize
  ResultType result_;
  //! If last call to optimize succeeded
  bool was_smoother_ok_;
};

}  // namespace dyno
