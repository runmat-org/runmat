//! Browser adapter for checked-source serial execution.
use super::fea_run_context::LocalFeaRunContext;
use super::session::RunMatWasm;
use crate::wire::errors::{js_error, js_value_to_string};
use runmat_analysis_execution::{
    GeometryProviderProfile, PreparedFeaGeometry, PreparedGeometryMaterializationProgram,
    PreparedGeometrySource, SerialAnalysisHost,
};
use runmat_core::{FeaGeometryExecutionHost, PreparedSerialFeaStudy};
use runmat_execution::Digest;
use runmat_runtime::analysis::fea_run_admission::{FeaRunKind, FeaRunOutcome};
use runmat_runtime::analysis::{
    AnalysisExecutionCandidate, AnalysisExecutionChoices, AnalysisExecutionMode,
    AnalysisExecutionReadiness,
};
use std::path::Path;
use wasm_bindgen::{prelude::*, JsCast};

const BROWSER_MEMORY_BYTES: u64 = 1 << 30;
const ARTIFACT_BYTES: u64 = 128 << 20;
const PUBLICATION_HOST_REASON: &str = "Result storage is unavailable in this browser workspace.";

// Capability metadata does not open or materialize storage while showing choices.
fn supports_result_publication(provider: &dyn runmat_filesystem::FsProvider) -> bool {
    provider.supports_workspace_transactions()
}
pub(crate) struct BrowserFeaSelection {
    id: String,
    path: String,
    checked_revision: [u8; 32],
}
struct BrowserGeometryHost<'a> {
    context: &'a LocalFeaRunContext,
}
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(catch,js_name=__runmatOcctExecute)]
    async fn exact_execute(json: &str, delegation: JsValue) -> Result<JsValue, JsValue>;
    #[wasm_bindgen(catch,js_name=__runmatExecutePreparedFeaStudy)]
    async fn execute_prepared(
        json: &str,
        memory_bytes: f64,
        delegation: JsValue,
    ) -> Result<JsValue, JsValue>;
    #[wasm_bindgen(catch,js_name=__runmatOcctResidentBytes)]
    fn exact_resident_bytes() -> Result<f64, JsValue>;
}
impl FeaGeometryExecutionHost for BrowserGeometryHost<'_> {
    async fn provider_profile(&self) -> Result<GeometryProviderProfile, String> {
        serde_json::from_str(
            &exact_execute("{\"operation\":\"profile\"}", JsValue::UNDEFINED)
                .await
                .map_err(js_value_to_string)?
                .as_string()
                .ok_or("Invalid geometry profile response.")?,
        )
        .map_err(|e| e.to_string())
    }
    async fn materialize_and_mesh(
        &self,
        program: PreparedGeometryMaterializationProgram,
        source: PreparedGeometrySource,
        settings: runmat_meshing_core::MeshingRequestSettings,
    ) -> Result<PreparedFeaGeometry, runmat_core::FeaStudyPreparationError> {
        let mut objects = source.objects;
        objects.push(program.descriptor);
        let json=serde_json::to_string(&serde_json::json!({"operation":"materialize_and_mesh","program":program.program,"objects":objects,"settings":settings})).map_err(|e|e.to_string())?;
        serde_json::from_str(
            &exact_execute(
                &json,
                self.context
                    .worker_context(&json)
                    .map_err(preparation_error_from_js)?,
            )
            .await
            .map_err(preparation_error_from_js)?
            .as_string()
            .ok_or("Invalid geometry execution response.")?,
        )
        .map_err(|e| e.to_string().into())
    }
}
fn preparation_error_from_js(error: JsValue) -> runmat_core::FeaStudyPreparationError {
    if let Ok(value) = js_sys::Reflect::get(&error, &JsValue::from_str("meshingFailure")) {
        if let Ok(failure) =
            serde_wasm_bindgen::from_value::<runmat_meshing_core::MeshingFailure>(value)
        {
            if failure.validate().is_ok() {
                return runmat_core::FeaStudyPreparationError::Meshing(Box::new(failure));
            }
        }
    }
    js_value_to_string(error).into()
}
fn resident_bytes() -> u64 {
    let memory = wasm_bindgen::memory().unchecked_into::<js_sys::WebAssembly::Memory>();
    let rust = js_sys::Reflect::get(&memory.buffer(), &JsValue::from_str("byteLength"))
        .ok()
        .and_then(|x| x.as_f64())
        .unwrap_or(BROWSER_MEMORY_BYTES as f64);
    let exact = exact_resident_bytes().unwrap_or(BROWSER_MEMORY_BYTES as f64);
    if !rust.is_finite() || !exact.is_finite() || rust < 0.0 || exact < 0.0 {
        return BROWSER_MEMORY_BYTES;
    }
    (rust as u64).saturating_add(exact as u64)
}
#[wasm_bindgen]
impl RunMatWasm {
    #[wasm_bindgen(js_name=planFeaSource)]
    pub async fn plan_fea_source_js(
        &self,
        path: String,
        inputs: JsValue,
    ) -> Result<JsValue, JsValue> {
        let result_provider = runmat_filesystem::current_provider();
        let _filesystem = pin_inputs(inputs)?;
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let _interrupt =
            super::session::install_wasm_interrupt(&self.active_interrupt, cancellation.clone());
        let mut session = self.take_session()?;
        self.fea_selection.borrow_mut().take();
        let source = session
            .load_and_check_fea_source_path(Path::new(&path))
            .await
            .map_err(|e| js_error(&e))?;
        ensure_not_cancelled(&cancellation)?;
        let preflight = source.fea_execution_preflight().map_err(|e| js_error(&e))?;
        let id = uuid::Uuid::new_v4().to_string();
        let revision = source.check_result().revision.0;
        let publication_supported = supports_result_publication(result_provider.as_ref());
        let resident = resident_bytes();
        let preparation_fits =
            resident.saturating_add((512 + 128 + 64) << 20) <= BROWSER_MEMORY_BYTES;
        let ready = preflight.browser_supported && publication_supported && preparation_fits;
        let available = BROWSER_MEMORY_BYTES
            .saturating_sub(resident)
            .saturating_sub(ARTIFACT_BYTES)
            .saturating_sub(64 << 20);
        let reason = if !preflight.browser_supported {
            preflight.desktop_reason
        } else if !publication_supported {
            Some(PUBLICATION_HOST_REASON)
        } else if !preparation_fits {
            Some("This session needs more memory for geometry preparation. Continue in the desktop app.")
        } else if preflight
            .approximate_memory_bytes
            .is_some_and(|bytes| bytes > available)
        {
            Some("Desktop recommended for memory")
        } else {
            Some("Runs in this browser")
        };
        let duration = ready
            .then(|| {
                session
                    .fea_execution_timing_profile()
                    .for_source(
                        &source,
                        runmat_core::FeaTimingRoute::BrowserCheckedSource,
                        1,
                    )
                    .ok()
                    .and_then(|key| key.estimate())
            })
            .flatten();
        let plan = AnalysisExecutionChoices {
            schema_version: 1,
            plan_id: id.clone(),
            revision,
            candidates: vec![
                AnalysisExecutionCandidate {
                    mode: AnalysisExecutionMode::SingleCore,
                    cores: Some(1),
                    memory_bytes: preflight.approximate_memory_bytes,
                    estimated_millis: duration.as_ref().map(|value| value.0),
                    duration_evidence: duration.map(|value| value.1),
                    readiness: if ready {
                        AnalysisExecutionReadiness::Ready
                    } else {
                        AnalysisExecutionReadiness::DesktopRequired
                    },
                    reason: reason.map(str::to_owned),
                },
                AnalysisExecutionCandidate {
                    mode: AnalysisExecutionMode::MultiCore,
                    cores: None,
                    memory_bytes: None,
                    estimated_millis: None,
                    duration_evidence: None,
                    readiness: AnalysisExecutionReadiness::DesktopRequired,
                    reason: Some("Multi-core simulation runs in the desktop app.".into()),
                },
            ],
        };
        // A choice is bound to checked inputs, not to a prematurely generated mesh.
        *self.fea_selection.borrow_mut() = ready.then_some(BrowserFeaSelection {
            id,
            path,
            checked_revision: revision,
        });
        crate::wire::json_contract_to_js(&plan).map_err(|e| js_error(&e.to_string()))
    }
    #[wasm_bindgen(js_name=runFeaSource)]
    pub async fn run_fea_source_js(
        &self,
        path: String,
        plan_id: String,
        inputs: JsValue,
    ) -> Result<JsValue, JsValue> {
        if !supports_result_publication(runmat_filesystem::current_provider().as_ref()) {
            return Err(js_error(PUBLICATION_HOST_REASON));
        }
        let _filesystem = pin_inputs(inputs)?;
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let _interrupt =
            super::session::install_wasm_interrupt(&self.active_interrupt, cancellation.clone());
        let _progress = runmat_analysis_fea::replace_fea_progress_context(
            runmat_analysis_fea::FeaProgressContext {
                handler: Some(std::sync::Arc::new(forward_progress)),
                cancellation: runmat_analysis_fea::FeaCancellationToken::from_token(
                    cancellation.token(),
                ),
                ..Default::default()
            },
        );
        let mut session = self.take_session()?;
        let fresh = session
            .load_and_check_fea_source_path(Path::new(&path))
            .await
            .map_err(|e| js_error(&e))?;
        ensure_not_cancelled(&cancellation)?;
        let checked = fresh.check_result();
        {
            let slot = self.fea_selection.borrow();
            let pending = slot
                .as_ref()
                .ok_or_else(|| js_error("Prepare this simulation before running."))?;
            if pending.id != plan_id
                || pending.path != path
                || pending.checked_revision != checked.revision.0
            {
                return Err(js_error(
                    "The simulation changed. Refresh its execution options before running.",
                ));
            }
        }
        self.fea_selection
            .borrow_mut()
            .take()
            .ok_or_else(|| js_error("The selected execution options are no longer available."))?;
        // Local work has no billing gate; retain one allocation identity and
        // usage scope through preparation, numerical work and publication.
        let context =
            LocalFeaRunContext::new(FeaRunKind::MeshAndSolve, checked_study_identity(&fresh)?)?;
        let scope = context.scope()?;
        ensure_not_cancelled(&cancellation)?;
        let observation = session
            .fea_execution_timing_profile()
            .for_source(&fresh, runmat_core::FeaTimingRoute::BrowserCheckedSource, 1)
            .ok()
            .map(|key| key.start());
        let outcome = scope.run_future(async {
            // First mesh only after explicit Run. The existing preparation and
            // exact resource planner remain the sole allocation authority.
            if resident_bytes().saturating_add((512 + 128 + 64) << 20) > BROWSER_MEMORY_BYTES {
                return Err(js_error("Not enough browser memory to prepare this simulation. Continue in the desktop app."));
            }
            let study = fresh.prepare_serial_fea_study(&BrowserGeometryHost { context: &context }, &format!("browser-fea:{plan_id}"))
                .await.map_err(|error| crate::wire::json_contract_to_js(&error.diagnostic()).unwrap_or_else(|_| js_error(&error.to_string())))?;
            ensure_not_cancelled(&cancellation)?;
            let available_memory = BROWSER_MEMORY_BYTES.saturating_sub(resident_bytes()).saturating_sub(ARTIFACT_BYTES).saturating_sub(64 << 20);
            let transfer_bytes = (serde_json::to_vec(&study).map_err(|e| js_error(&e.to_string()))?.len() as u64).saturating_mul(4);
            if study.resource_plan.request.memory_bytes.max(transfer_bytes) > available_memory || study.resource_plan.request.scratch_bytes > 256 << 20 {
                return Err(js_error("The prepared simulation exceeds browser memory. Continue in the desktop app."));
            }
            let request =
                serde_json::to_string(&study).map_err(|e| js_error(&e.to_string()))?;
            let json = execute_prepared(
                &request,
                available_memory as f64,
                context.worker_context(&request)?,
            )
            .await?
            .as_string()
            .ok_or_else(|| js_error("Invalid numerical execution response."))?;
            ensure_not_cancelled(&cancellation)?;
            if json.len() > 128 << 20 {
                return Err(js_error(
                    "The numerical result exceeds the browser allowance.",
                ));
            }
            let numerical: runmat_runtime::analysis::StructuralStudySequenceNumericalResult =
                serde_json::from_str(&json).map_err(|e| js_error(&e.to_string()))?;
            let result = runmat_runtime::analysis::StructuralStudyResult::new(
                study.compiled,
                study.mesh,
                study.evidence,
                study.workloads,
                runmat_runtime::analysis::StructuralStudyNumericalResult::LinearSequence(numerical),
            )
            .map_err(|e| js_error(&e))?;
            // Frozen source inputs have finished their job. Result I/O uses the
            // original provider and its canonical workspace transaction authority.
            drop(_filesystem);
            let publication_identity = runmat_runtime::analysis::AnalysisStudyCompletionIdentity::new(format!("browser-fea:{plan_id}")).map_err(|e| js_error(&e))?;
            let payload = runmat_runtime::analysis::storage::with_artifact_access(true, |_| {
                let publication = runmat_runtime::analysis::publish_structural_study_result(
                    &result,
                    runmat_runtime::operations::OperationContext::new(None, None),
                    publication_identity,
                ).map_err(|error| js_error(&error.message))?;
                runmat_runtime::analysis::analysis_document_study_result(path, publication.data, Vec::new())
                    .map_err(|error| js_error(&error))
            }).await.map_err(|error| js_error(&error.to_string()))??;
            let payload =
                crate::wire::json_contract_to_js(&payload).map_err(|e| js_error(&e.to_string()))?;
            Ok(payload)
        })
        .await;
        if let Some(observation) = observation {
            if !cancellation.token().is_cancelled() {
                observation.complete(outcome.is_ok());
            }
        }
        scope.settle(if cancellation.token().is_cancelled() {
            FeaRunOutcome::Cancelled
        } else if outcome.is_ok() {
            FeaRunOutcome::Completed
        } else {
            FeaRunOutcome::Failed
        });
        outcome
    }
}

/// Reuse the authored study's canonical digest, before materializing geometry.
fn checked_study_identity(
    source: &runmat_core::CheckedFeaSessionSource,
) -> Result<Digest, JsValue> {
    let runmat_runtime::analysis::ResolvedFeaRoot::Study(resolved) = source
        .resolve_root(None)
        .map_err(|error| js_error(&error.to_string()))?
    else {
        return Err(js_error("Run this simulation group in the desktop app."));
    };
    let realized = resolved
        .study
        .realize_with_models(&resolved.models, &resolved.geometry, &[])
        .map_err(|error| js_error(&error.to_string()))?;
    runmat_analysis_fea::study::structural_study_source_digest(
        &realized,
        &resolved.material_catalog,
    )
    .map_err(|error| js_error(&error.to_string()))
}

fn pin_inputs(inputs: JsValue) -> Result<Option<runmat_filesystem::ProviderGuard>, JsValue> {
    if inputs.is_null() || inputs.is_undefined() {
        return Ok(None);
    }
    let manifest: runmat_filesystem::pinned::InputManifest = serde_wasm_bindgen::from_value(inputs)
        .map_err(|e| js_error(&format!("Invalid simulation inputs: {e}")))?;
    let provider = runmat_filesystem::pinned::PinnedFsProvider::new(
        runmat_filesystem::current_provider(),
        manifest,
    )
    .map_err(|e| js_error(&e.to_string()))?;
    Ok(Some(runmat_filesystem::replace_provider(
        std::sync::Arc::new(provider),
    )))
}
fn forward_progress(event: runmat_analysis_fea::FeaProgressEvent) {
    if let Ok(callback) =
        js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("__runmatFeaProgress"))
    {
        if let Some(callback) = callback.dyn_ref::<js_sys::Function>() {
            if let Ok(payload) = serde_json::to_string(&event) {
                let _ = callback.call1(&JsValue::UNDEFINED, &JsValue::from_str(&payload));
            }
        }
    }
}

/// Runs inside an owned worker. Its scheduler, receipts and result validation
/// are the same as the in-process serial host; termination belongs to the host.
#[wasm_bindgen(js_name=executePreparedFeaStudy)]
pub async fn execute_prepared_fea_study(
    json: String,
    memory_bytes: f64,
    worker_context: JsValue,
) -> Result<String, JsValue> {
    // Local execution is unmetered, but an owned worker still checks the exact
    // parent payload and canonical source identity before numerical execution.
    validate_numerical_input(json.len(), memory_bytes)?;
    let study: PreparedSerialFeaStudy =
        serde_json::from_str(&json).map_err(|error| js_error(&error.to_string()))?;
    let context =
        LocalFeaRunContext::for_worker(worker_context, &json, study.authored_study_digest)?;
    let scope = context.scope()?;
    let result = scope.run_sync(|| execute_prepared_fea_study_local(study, memory_bytes));
    let outcome = if runmat_runtime::interrupt::is_cancelled() {
        FeaRunOutcome::Cancelled
    } else if result.is_ok() {
        FeaRunOutcome::Completed
    } else {
        FeaRunOutcome::Failed
    };
    scope.settle(outcome);
    result
}

fn validate_numerical_input(json_len: usize, memory_bytes: f64) -> Result<(), JsValue> {
    if json_len > 128 << 20
        || !memory_bytes.is_finite()
        || memory_bytes < 1.0
        || memory_bytes > BROWSER_MEMORY_BYTES as f64
        || memory_bytes.fract() != 0.0
    {
        return Err(js_error("Invalid browser numerical execution allowance."));
    }
    let available_memory = memory_bytes as u64;
    if (json_len as u64).saturating_mul(4) > available_memory {
        return Err(js_error(
            "Preparing this simulation requires a larger execution budget. Use the desktop app.",
        ));
    }
    Ok(())
}

fn execute_prepared_fea_study_local(
    study: PreparedSerialFeaStudy,
    memory_bytes: f64,
) -> Result<String, JsValue> {
    let available_memory = memory_bytes as u64;
    runmat_runtime::builtins::wasm_registry::register_all();
    let _progress = runmat_analysis_fea::replace_fea_progress_context(
        runmat_analysis_fea::FeaProgressContext {
            handler: Some(std::sync::Arc::new(forward_progress)),
            ..Default::default()
        },
    );
    let mut resources = runmat_execution::resource::ResourceInventory {
        cpu_millicores: 1000,
        memory_bytes: available_memory,
        scratch_bytes: 256 << 20,
        accelerators: Vec::new(),
        capabilities: runmat_analysis_execution::analysis_worker_capabilities(
            std::iter::empty(),
            std::iter::empty(),
            false,
        ),
    };
    resources
        .capabilities
        .insert(runmat_execution::resource::Capability::BrowserWorker);
    let inventory = crate::runtime::execution::controlled::inventory()
        .map_err(|e| js_error(&e))?
        .host;
    let mut host =
        SerialAnalysisHost::new(inventory, resources, Digest::sha256(b"owned-browser-fea"))
            .map_err(|e| js_error(&e.to_string()))?;
    let mut store = runmat_execution_artifact::cache::MemoryObjectStore::new(
        runmat_execution_artifact::object::ObjectInventoryLimits {
            max_objects: 100_000,
            max_object_bytes: 64 << 20,
            max_total_bytes: ARTIFACT_BYTES,
        },
    )
    .map_err(|e| js_error(&e.to_string()))?;
    let result = study
        .execute(
            &mut host,
            &mut store,
            &|| js_sys::Date::now() as u64,
            &runmat_runtime::interrupt::is_cancelled,
        )
        .map_err(|e| js_error(&e))?;

    let runmat_runtime::analysis::StructuralStudyNumericalResult::LinearSequence(numerical) =
        result.into_parts().4
    else {
        return Err(js_error(
            "Unexpected numerical procedure returned by serial host.",
        ));
    };
    serde_json::to_string(&numerical).map_err(|e| js_error(&e.to_string()))
}

fn ensure_not_cancelled(
    source: &runmat_execution::cancellation::CancellationSource,
) -> Result<(), JsValue> {
    if source.token().is_cancelled() {
        Err(js_error("Simulation cancelled."))
    } else {
        Ok(())
    }
}
