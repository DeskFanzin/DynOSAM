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

#include "dynosam/pipeline/PipelineManager.hpp"

#include <glog/logging.h>

#include "dynosam/backend/ParallelHybridBackendModule.hpp"
#include "dynosam/backend/RegularBackendModule.hpp"
#include "dynosam/common/Map.hpp"
#include "dynosam/frontend/RGBDInstanceFrontendModule.hpp"
#include "dynosam/logger/Logger.hpp"
#include "dynosam/utils/TimingStats.hpp"

DEFINE_bool(use_backend, false, "If any backend should be initalised");

namespace dyno {

DynoPipelineManager::DynoPipelineManager(
    const DynoParams& params, DataProvider::Ptr data_loader,
    FrontendDisplay::Ptr frontend_display, BackendDisplay::Ptr backend_display,
    const ExternalHooks::Ptr external_hooks)
    : params_(params),
      use_offline_frontend_(FLAGS_frontend_from_file),
      data_loader_(std::move(data_loader)),
      displayer_(&display_queue_, params.parallelRun())

{
  LOG(INFO) << "Starting DynoPipelineManager";

  CHECK(data_loader_);
  CHECK(frontend_display);

  data_interface_ =
      std::make_unique<DataInterfacePipeline>(params_.parallelRun());
  data_loader_->registerImageContainerCallback(
      std::bind(&dyno::DataInterfacePipeline::fillImageContainerQueue,
                data_interface_.get(), std::placeholders::_1));

  // if an external hook exists to update the time, add callback in
  // datainterface that will be triggered when new data is added to the output
  // queue (from the DataInterfacePipeline) this is basically used to alert ROS
  // to a new timestamp which we then publish to /clock
  if (external_hooks && external_hooks->update_time) {
    LOG(INFO) << "Added pre-queue callback to register new Timestampd data "
                 "with an external module";
    data_interface_->registerPreQueueContainerCallback(
        [external_hooks](const ImageContainer::Ptr image_container) -> void {
          external_hooks->update_time(image_container->timestamp());
        });
  }

  // ground truth
  data_loader_->registerGroundTruthPacketCallback(
      std::bind(&dyno::DataInterfacePipeline::addGroundTruthPacket,
                data_interface_.get(), std::placeholders::_1));

  // register single and multi IMU callbacks to the data loader
  data_loader_->registerImuSingleCallback(std::bind(
      static_cast<void (DataInterfacePipeline::*)(const ImuMeasurement&)>(
          &dyno::DataInterfacePipeline::fillImuQueue),
      data_interface_.get(), std::placeholders::_1));

  data_loader_->registerImuMultiCallback(std::bind(
      static_cast<void (DataInterfacePipeline::*)(const ImuMeasurements&)>(
          &dyno::DataInterfacePipeline::fillImuQueue),
      data_interface_.get(), std::placeholders::_1));

  // preprocessing
  data_interface_->registerImageContainerPreprocessor(
      std::bind(&dyno::DataProvider::imageContainerPreprocessor,
                data_loader_.get(), std::placeholders::_1));

  // push data from the data interface to the frontend module
  data_interface_->registerOutputQueue(&frontend_input_queue_);

  CameraParams camera_params;
  if (params_.preferDataProviderCameraParams() &&
      data_loader_->getCameraParams().has_value()) {
    LOG(INFO) << "Using camera params from DataProvider, not the config in the "
                 "CameraParams.yaml!";
    camera_params = *data_loader_->getCameraParams();
  } else {
    LOG(INFO) << "Using camera params specified in CameraParams.yaml!";
    camera_params = params_.camera_params_;
  }
  /// NOTE: no need to update the camera params like the imu params as we parse
  /// the camera params into the loadPipeline functions separately!

  ImuParams imu_params;
  if (params_.preferDataProviderImuParams() &&
      data_loader_->getImuParams().has_value()) {
    LOG(INFO) << "Using imu params from DataProvider, not the config in the "
                 "ImuParams.yaml!";
    imu_params = *data_loader_->getImuParams();
  } else {
    LOG(INFO) << "Using imu params specified in ImuParams.yaml!";
    imu_params = params_.imu_params_;
  }

  // update the imu params that will actually get sent to the frontend
  params_.frontend_params_.imu_params = imu_params;

  loadPipelines(camera_params, frontend_display, backend_display);
  launchSpinners();
}

DynoPipelineManager::~DynoPipelineManager() {
  shutdownPipelines();
  shutdownSpinners();

  // TODO: make shutdown hook!
  writeStatisticsSamplesToFile("statistics_samples.csv");
  writeStatisticsModuleSummariesToFile();
}

void DynoPipelineManager::shutdownSpinners() {
  if (frontend_pipeline_spinner_) frontend_pipeline_spinner_->shutdown();

  if (backend_pipeline_spinner_) backend_pipeline_spinner_->shutdown();

  if (data_provider_spinner_) data_provider_spinner_->shutdown();

  if (frontend_viz_pipeline_spinner_)
    frontend_viz_pipeline_spinner_->shutdown();

  if (backend_viz_pipeline_spinner_) backend_viz_pipeline_spinner_->shutdown();
}

void DynoPipelineManager::shutdownPipelines() {
  display_queue_.shutdown();
  frontend_pipeline_->shutdown();

  if (backend_pipeline_) backend_pipeline_->shutdown();

  data_interface_->shutdown();

  if (frontend_viz_pipeline_) frontend_viz_pipeline_->shutdown();
  if (backend_viz_pipeline_) backend_viz_pipeline_->shutdown();
}

bool DynoPipelineManager::spin() {
  std::function<bool()> spin_func;

  if (use_offline_frontend_) {
    // if we have an offline frontend only spiun the frontend pipeline
    // and no need to spin the viz (TODO: right now this is only images and not
    // the actual pipelines...)
    spin_func = [=]() -> bool {
      if (frontend_pipeline_->isWorking()) {
        if (!params_.parallelRun()) {
          frontend_pipeline_->spinOnce();
          if (backend_pipeline_) backend_pipeline_->spinOnce();
        }
        return true;
      }
      return false;
    };
  } else {
    // regular spinner....
    spin_func = [=]() -> bool {
      if (data_loader_->spin() || frontend_pipeline_->isWorking() ||
          (backend_pipeline_ && backend_pipeline_->isWorking())) {
        if (!params_.parallelRun()) {
          frontend_pipeline_->spinOnce();
          if (backend_pipeline_) backend_pipeline_->spinOnce();
        }
        spinViz();  // for now
        // a later problem!
        return true;
      }
      return false;
    };
  }

  utils::TimingStatsCollector timer("pipeline_spin");
  return spin_func();
}

bool DynoPipelineManager::spinViz() {
  // if()
  displayer_
      .process();  // when enabled this gives a segafault when the process ends.
                   // when commented out the program just waits at thee end
  return true;
}

void DynoPipelineManager::launchSpinners() {
  LOG(INFO) << "Running PipelineManager with parallel_run="
            << params_.parallelRun();

  if (params_.parallelRun()) {
    frontend_pipeline_spinner_ = std::make_unique<dyno::Spinner>(
        std::bind(&dyno::FrontendPipeline::spin, frontend_pipeline_.get()),
        "frontend-pipeline-spinner");

    if (backend_pipeline_)
      backend_pipeline_spinner_ = std::make_unique<dyno::Spinner>(
          std::bind(&dyno::BackendPipeline::spin, backend_pipeline_.get()),
          "backend-pipeline-spinner");
  }

  data_provider_spinner_ = std::make_unique<dyno::Spinner>(
      std::bind(&dyno::DataInterfacePipeline::spin, data_interface_.get()),
      "data-interface-spinner");

  if (frontend_viz_pipeline_)
    frontend_viz_pipeline_spinner_ = std::make_unique<dyno::Spinner>(
        std::bind(&dyno::FrontendVizPipeline::spin,
                  frontend_viz_pipeline_.get()),
        "frontend-display-spinner");

  if (backend_viz_pipeline_)
    backend_viz_pipeline_spinner_ = std::make_unique<dyno::Spinner>(
        std::bind(&dyno::BackendVizPipeline::spin, backend_viz_pipeline_.get()),
        "backend-display-spinner");
}

void DynoPipelineManager::loadPipelines(const CameraParams& camera_params,
                                        FrontendDisplay::Ptr frontend_display,
                                        BackendDisplay::Ptr backend_display) {
  BackendModule::Ptr backend = nullptr;
  // the registra for the frontend pipeline
  // this is agnostic to the actual pipeline type so we can add/register
  // a new queue to it regardless of the derived type (as long as it is at least
  // a MIMO, which it should be as this is the lowest type of actual pipeline
  // with any functionality)
  typename FrontendPipeline::OutputRegistra::Ptr frontend_output_registra =
      nullptr;
  const auto parallel_run = params_.parallelRun();

  switch (params_.frontend_type_) {
    case FrontendType::kRGBD: {
      LOG(INFO) << "Making RGBDInstance frontend";
      FrontendModule::Ptr frontend = nullptr;

      Camera::Ptr camera = std::make_shared<Camera>(camera_params);
      CHECK_NOTNULL(camera);

      if (use_offline_frontend_) {
        LOG(INFO) << "Offline RGBD frontend";
        using OfflineFrontend =
            FrontendOfflinePipeline<RegularBackendModule::ModuleTraits>;
        const std::string file_path =
            getOutputFilePath(kRgbdFrontendOutputJsonFile);
        LOG(INFO) << "Loading RGBD frontend output packets from " << file_path;

        OfflineFrontend::UniquePtr offline_frontend =
            std::make_unique<OfflineFrontend>("offline-rgbdfrontend",
                                              file_path);
        // make registra so we can register queues with this pipeline
        frontend_output_registra = offline_frontend->getOutputRegistra();
        // raw ptr type becuase we cannot copy the unique ptr!! This is only
        // becuase we need it in the lambda function which is a temporary
        // solution
        OfflineFrontend* offline_frontend_ptr = offline_frontend.get();
        // set get dataset size function (bit of a hack for now, and only for
        // the batch optimizer so it knows when to optimize!!)
        get_dataset_size_ = [offline_frontend_ptr]() -> FrameId {
          // get frame id of the final frame saved
          return CHECK_NOTNULL(offline_frontend_ptr)
              ->getFrontendOutputPackets()
              .rbegin()
              ->first;
        };
        // convert pipeline to base type
        frontend_pipeline_ = std::move(offline_frontend);
      } else {
        frontend = std::make_shared<RGBDInstanceFrontendModule>(
            params_.frontend_params_, camera, &display_queue_);
        LOG(INFO) << "Made RGBDInstanceFrontendModule";
        // need to make the derived pipeline so we can set parallel run etc
        // the manager takes a pointer to the base MIMO so we can have different
        // types of pipelines
        FrontendPipeline::UniquePtr frontend_pipeline_derived =
            std::make_unique<FrontendPipeline>(
                "frontend-pipeline", &frontend_input_queue_, frontend);
        // make registra so we can register queues with this pipeline
        frontend_output_registra =
            frontend_pipeline_derived->getOutputRegistra();
        frontend_pipeline_derived->parallelRun(parallel_run);
        // conver pipeline to base type
        frontend_pipeline_ = std::move(frontend_pipeline_derived);

        get_dataset_size_ = [=]() -> FrameId {
          CHECK(data_loader_) << "Data Loader is null when accessing "
                                 "get_last_frame_ in BatchOptimizerParams";
          return data_loader_->datasetSize();
        };
      }

      // right now depends on the get_dataset_size_ function being det before
      // the optimzier is created!!!

      if (FLAGS_use_backend) {
        LOG(INFO) << "Construcing RGBD backend";

        // TODO: make better params and hhow they are used in each backend!
        // right now they affect which backend is used AND the formulation in
        // that backend
        auto updater_type =
            static_cast<RGBDFormulationType>(FLAGS_backend_updater_enum);

        if (updater_type == RGBDFormulationType::PARALLEL_HYBRID) {
          backend = std::make_shared<ParallelHybridBackendModule>(
              params_.backend_params_, camera, &display_queue_);
        } else {
          params_.backend_params_.full_batch_frame = (int)get_dataset_size_();

          backend = std::make_shared<RegularBackendModule>(
              params_.backend_params_, camera, updater_type, &display_queue_);
        }

        // if(frontend && backend) {
        //   backend->registerMapUpdater(std::bind(&FrontendModule::mapUpdate,
        //   frontend.get(), std::placeholders::_1)); LOG(INFO) << "Bound map
        //   update between frontend and backend";
        // }

      } else if (use_offline_frontend_) {
        LOG(WARNING)
            << "FLAGS_use_backend is false but use_offline_frontend "
               "(FLAGS_frontend_from_file) us true. "
            << " Pipeline will load data from frontend but send it nowhere!!";
      }

    } break;
    case FrontendType::kMono: {
      LOG(FATAL) << "MONO Not implemented!";
    } break;

    default: {
      LOG(FATAL) << "Not implemented!";
    } break;
  }

  CHECK_NOTNULL(frontend_pipeline_);
  CHECK_NOTNULL(frontend_output_registra);
  // register output queue to send the front-end output to the viz
  frontend_output_registra->registerQueue(&frontend_viz_input_queue_);

  if (backend) {
    backend_pipeline_ = std::make_unique<BackendPipeline>(
        "backend-pipeline", &backend_input_queue_, backend);
    backend_pipeline_->parallelRun(parallel_run);
    // also register connection between front and back
    frontend_output_registra->registerQueue(&backend_input_queue_);

    backend_pipeline_->registerOutputQueue(&backend_output_queue_);
  }

  // right now we cannot use the viz when we load from file as do not load
  // certain data values (e.g. camera and debug info) so these will be null -
  // the viz's try and access these causing a seg fault. Just need to add checks
  if (!use_offline_frontend_) {
    if (backend && backend_display) {
      backend_viz_pipeline_ = std::make_unique<BackendVizPipeline>(
          "backend-viz-pipeline", &backend_output_queue_, backend_display);
    }
    frontend_viz_pipeline_ = std::make_unique<FrontendVizPipeline>(
        "frontend-viz-pipeline", &frontend_viz_input_queue_, frontend_display);
  }
}

// PipelineBase::UniquePtr DynoPipelineManager::makeFrontendPipeline(Camera::Ptr
// camera, bool offline_frontend) {

// }
// BackendPipeline::UniquePtr DynoPipelineManager::makeBackendPipeline() {

// }

}  // namespace dyno
