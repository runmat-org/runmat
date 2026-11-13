//! Authenticated, parent-supervised numerical steps. No worker publication.
use super::fea_run_context::LocalFeaRunContext;
use crate::runtime::{config::SessionConfig, gpu};
use crate::wire::errors::js_error;
use runmat_runtime::analysis::{self, fea_run_admission::FeaRunOutcome};
use serde::{Deserialize, Serialize};
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OwnedFeaStudyStepPayload {
    pub request: analysis::AnalysisStudyStepRequest,
    pub study_identity: runmat_execution::Digest,
    pub program_revision: runmat_execution::ProgramRevision,
    pub runtime: runmat_config::runtime::RunMatRuntimeConfig,
}

#[wasm_bindgen(js_name=executeOwnedFeaStudyStep)]
pub async fn execute_owned_fea_study_step(
    json: String,
    memory_bytes: f64,
    worker_context: JsValue,
    worker_admission: JsValue,
) -> Result<String, JsValue> {
    super::fea_execution::validate_numerical_input(json.len(), memory_bytes)?;
    let payload: OwnedFeaStudyStepPayload =
        serde_json::from_str(&json).map_err(|error| js_error(&error.to_string()))?;
    let context = Rc::new(
        LocalFeaRunContext::for_worker(
            worker_context,
            worker_admission,
            &json,
            payload.study_identity,
        )
        .await?,
    );
    let mut execution_cancelled = false;
    let result = async {
        let config = SessionConfig::from_closed_runtime(payload.runtime)
            .map_err(|error| js_error(&error))?;
        gpu::admit_browser_provider(&config).map_err(|error| js_error(&error))?;
        if config.enable_gpu {
            let initialized = match gpu::validate_webgpu_runtime() {
                Ok(()) => gpu::initialize_gpu_provider(&config).await,
                Err(error) => Err(js_error(&error)),
            };
            if let Err(error) = initialized {
                if !config.runtime.accelerate.allow_inprocess_fallback {
                    return Err(error);
                }
                gpu::install_cpu_provider(&config)?;
            }
        } else {
            gpu::install_cpu_provider(&config)?;
        }
        crate::runtime::settings::apply_resources(&config.runtime)
            .map_err(|error| js_error(&error))?;
        runmat_runtime::builtins::wasm_registry::register_all();
        let mut session =
            runmat_core::RunMatSession::with_options(config.enable_jit, config.verbose)
                .map_err(|error| js_error(&error.to_string()))?;
        session.install_fea_run_admission_service(Rc::new(
            super::fea_async_host::BrowserWorkerFeaAdmission {
                context: Rc::downgrade(&context),
            },
        ));
        session
            .apply_runtime_config(&config.runtime)
            .map_err(|error| js_error(&error.to_string()))?;
        runmat_core::RunMatSession::configure_gc_with_resource_limits(
            &config.runtime,
            memory_bytes as usize,
            1,
        )
        .map_err(|error| js_error(&error.to_string()))?;
        let runtime = session.execution_runtime_context(payload.program_revision);
        let _entered = runtime.enter();
        let capacity = super::fea_async_host::BrowserNumericalCapacity::new(&context);
        let operation_context = payload.request.context.clone();
        let checked = context
            .scope()
            .run_checked_result_future(async {
                analysis::analysis_execute_study_step_with_capacity(
                    payload.request,
                    Some(&capacity),
                )
                .await
                .and_then(|result| {
                    analysis::analysis_validate_study_step_transfer(&result)?;
                    Ok(result)
                })
            })
            .await;
        execution_cancelled = matches!(
            &checked,
            Err(analysis::fea_run_admission::FeaRunExecutionError::Cancelled)
        ) || matches!(
            &checked,
            Ok(Err(error)) if error.error_type == runmat_runtime::operations::OperationErrorType::Cancelled
        );
        let result = checked.map_err(|error| match error {
                analysis::fea_run_admission::FeaRunExecutionError::Denied(denied) => {
                    super::fea_lease::denied_to_js(denied)
                }
                analysis::fea_run_admission::FeaRunExecutionError::Cancelled => {
                    js_error("Simulation cancelled.")
                }
            })?;
        let result = analysis::analysis_merge_study_step_admission(
            result,
            context.scope().check().err(),
            &operation_context,
        );
        let successful = result.is_ok();
        let json = serde_json::to_string(&result).map_err(|error| js_error(&error.to_string()))?;
        super::fea_execution::validate_numerical_input(json.len(), memory_bytes)?;
        Ok((json, successful))
    }
    .await;
    let outcome = if execution_cancelled || runmat_runtime::interrupt::is_cancelled() {
        FeaRunOutcome::Cancelled
    } else if result.as_ref().is_ok_and(|(_, successful)| *successful) {
        FeaRunOutcome::Completed
    } else {
        FeaRunOutcome::Failed
    };
    let context = Rc::try_unwrap(context)
        .map_err(|_| js_error("Owned worker admission remained active after execution."))?;
    let settled = context.settle(outcome).await;
    super::fea_lease::merge_js_lease_error(result.map(|(json, _)| json), settled.err())
}
