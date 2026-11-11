//! Browser placement preserves the existing study dispatcher and parent publication.
use super::{fea_run_context::ActiveFeaLease, fea_step_worker::OwnedFeaStudyStepPayload};
use crate::runtime::config::SessionConfig;
use runmat_runtime::{
    analysis::*,
    context::{RuntimeAnalysisService, RuntimeServiceFuture},
    operations::*,
};
use std::{cell::RefCell, rc::Rc};
use wasm_bindgen::prelude::*;

#[derive(Clone)]
pub(super) struct BrowserAnalysisService {
    pub host: Rc<RefCell<Option<JsValue>>>,
    pub active: Rc<RefCell<Option<ActiveFeaLease>>>,
    pub config: Rc<RefCell<SessionConfig>>,
}
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(catch, js_name=__runmatExecuteOwnedFeaStudyStep)]
    async fn execute_step(
        json: &str,
        memory_bytes: f64,
        delegation: JsValue,
    ) -> Result<JsValue, JsValue>;
}
impl AnalysisStudyStepExecutor for BrowserAnalysisService {
    fn allocated_cpu_millicores(&self) -> u64 {
        1000
    }
    fn execute_step<'a>(
        &'a self,
        request: AnalysisStudyStepRequest,
    ) -> AnalysisStudyStepFuture<'a> {
        Box::pin(async move {
            let context = request.context.clone();
            let work = async {
                let active = self.active.borrow().clone().ok_or_else(|| {
                    super::fea_lease::denied_to_js(fea_run_admission::FeaRunDenied::SignInRequired)
                })?;
                let host = self.host.borrow().clone().ok_or_else(|| {
                    super::fea_lease::denied_to_js(fea_run_admission::FeaRunDenied::SignInRequired)
                })?;
                let program_revision = runmat_runtime::context::legacy::active()
                    .and_then(|runtime| runtime.program_revision().cloned())
                    .ok_or_else(|| {
                        crate::wire::errors::js_error(
                            "Owned solving requires the original program revision.",
                        )
                    })?;
                let payload = OwnedFeaStudyStepPayload {
                    request,
                    study_identity: active.study_identity()?,
                    program_revision,
                    runtime: self.config.borrow().runtime.clone(),
                };
                let json = serde_json::to_string(&payload)
                    .map_err(|error| crate::wire::errors::js_error(&error.to_string()))?;
                super::fea_execution::validate_numerical_input(json.len(), (1u64 << 30) as f64)?;
                let delegation = active
                    .worker_context(
                        &host,
                        &json,
                        super::fea_run_context::WorkerWorkClass::Numerical,
                    )
                    .await?;
                let value = execute_step(&json, (1u64 << 30) as f64, delegation).await?;
                active.retire_finished(&host).await?;
                active
                    .control
                    .check()
                    .map_err(super::fea_lease::denied_to_js)?;
                let json = value.as_string().ok_or_else(|| {
                    crate::wire::errors::js_error("Invalid owned study-step response.")
                })?;
                super::fea_execution::validate_numerical_input(json.len(), (1u64 << 30) as f64)?;
                serde_json::from_str::<Result<AnalysisStudyStepResult, OperationErrorEnvelope>>(
                    &json,
                )
                .map_err(|error| crate::wire::errors::js_error(&error.to_string()))
            }
            .await;
            let result = work.map_err(|error| {
                let denied = super::fea_lease::denial_from_js(error);
                denied.into_operation_error("fea.run_study", "fea.run_study/v1", &context)
            })??;
            analysis_validate_study_step_transfer(&result)?;
            Ok(result)
        })
    }
}
impl RuntimeAnalysisService for BrowserAnalysisService {
    fn generate_mesh(
        &self,
        request: runmat_runtime::meshing::MeshingExecutionRequest,
    ) -> Option<
        RuntimeServiceFuture<
            Result<
                runmat_runtime::meshing::MeshingExecutionOutput,
                runmat_runtime::meshing::MeshingExecutionError,
            >,
        >,
    > {
        Some(Box::pin(super::fea_meshing::execute(
            request,
            self.host.clone(),
            self.active.clone(),
        )))
    }
    fn materialize_signal(
        &self,
        _request: runmat_runtime::context::RuntimeSignalMaterializationRequest,
    ) -> RuntimeServiceFuture<
        Result<runmat_analysis_core::PhysicalSignal, runmat_runtime::RuntimeError>,
    > {
        Box::pin(async {
            Err(runmat_runtime::build_runtime_error(
                "Generated physical signals require an execution host.",
            )
            .build())
        })
    }
    fn execute_study(
        &self,
        study: AnalysisStudySpec,
        context: OperationContext,
    ) -> RuntimeServiceFuture<Result<OperationEnvelope<AnalysisStudyRunData>, OperationErrorEnvelope>>
    {
        let executor = self.clone();
        Box::pin(async move {
            analysis_execute_study_with_step_executor(&study, context, &executor).await
        })
    }
    fn execute_sweep(
        &self,
        sweep: AnalysisStudySweepSpec,
        context: OperationContext,
    ) -> RuntimeServiceFuture<
        Result<OperationEnvelope<AnalysisStudySweepData>, OperationErrorEnvelope>,
    > {
        let executor = self.clone();
        Box::pin(async move {
            analysis_execute_study_sweep_with_step_executor(&sweep, context, &executor).await
        })
    }
}
