//! Authenticated, parent-supervised numerical steps. No worker publication.
use super::fea_run_context::LocalFeaRunContext;
use crate::runtime::{config::SessionConfig, gpu};
use crate::wire::errors::js_error;
use runmat_runtime::analysis::{self, fea_run_admission::FeaRunOutcome};
use serde::{Deserialize, Serialize};
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
    let context = LocalFeaRunContext::for_worker(
        worker_context,
        worker_admission,
        &json,
        payload.study_identity,
    )
    .await?;
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
        let result = context
            .scope()
            .run_checked_sync(|| {
                analysis::analysis_execute_study_step(payload.request).and_then(|result| {
                    analysis::analysis_validate_study_step_transfer(&result)?;
                    Ok(result)
                })
            })
            .map_err(super::fea_lease::denied_to_js)?;
        let successful = result.is_ok();
        let json = serde_json::to_string(&result).map_err(|error| js_error(&error.to_string()))?;
        super::fea_execution::validate_numerical_input(json.len(), memory_bytes)?;
        Ok((json, successful))
    }
    .await;
    let outcome = if runmat_runtime::interrupt::is_cancelled() {
        FeaRunOutcome::Cancelled
    } else if result.as_ref().is_ok_and(|(_, successful)| *successful) {
        FeaRunOutcome::Completed
    } else {
        FeaRunOutcome::Failed
    };
    let settled = context.settle(outcome).await;
    super::fea_lease::merge_js_lease_error(result.map(|(json, _)| json), settled.err())
}
