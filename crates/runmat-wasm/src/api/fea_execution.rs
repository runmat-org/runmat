//! Browser adapter for checked-source serial execution.
use super::fea_run_context::LocalFeaRunContext;
use super::session::RunMatWasm;
use crate::runtime::filesystem::captured::CapturedFilesystem;
use crate::wire::errors::{js_error, js_value_to_string};
use runmat_analysis_execution::{
    GeometryProviderProfile, PreparedFeaGeometry, PreparedGeometryMaterializationProgram,
    PreparedGeometrySource, SerialAnalysisHost,
};
use runmat_core::{FeaGeometryExecutionHost, PreparedSerialFeaStudy};
use runmat_execution::Digest;
use runmat_runtime::analysis::fea_run_admission::{FeaRunKind, FeaRunOutcome, FeaRunRequest};
use runmat_runtime::analysis::AnalysisExecutionReadiness;
use std::path::Path;
use wasm_bindgen::{prelude::*, JsCast};

mod plan;
pub(super) use plan::browser_fea_plan;

const BROWSER_MEMORY_BYTES: u64 = 1 << 30;
const ARTIFACT_BYTES: u64 = 128 << 20;
const PUBLICATION_HOST_REASON: &str = "Result storage is unavailable in this browser workspace.";

// Capability metadata does not open or materialize storage while showing choices.
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
    pub(super) async fn exact_execute(json: &str, delegation: JsValue) -> Result<JsValue, JsValue>;
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
        let response = exact_execute(
            &json,
            self.context
                .worker_context(
                    &json,
                    super::fea_run_context::WorkerWorkClass::OcctPreparation,
                )
                .await
                .map_err(preparation_error_from_js)?,
        )
        .await
        .map_err(preparation_error_from_js)?;
        self.context
            .retire_finished()
            .await
            .map_err(preparation_error_from_js)?;
        serde_json::from_str(
            &response
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
    #[wasm_bindgen(js_name=setFeaRunAdmission)]
    pub fn set_fea_run_admission_js(&self, admission: JsValue) -> Result<(), JsValue> {
        if self.active_fea_lease.borrow().is_some() || self.pending_fea_admission.get() {
            return Err(js_error(
                "Finish the active FEA run before replacing its admission host.",
            ));
        }
        *self.fea_admission.borrow_mut() = if admission.is_null() || admission.is_undefined() {
            None
        } else {
            Some(admission)
        };
        Ok(())
    }

    #[wasm_bindgen(js_name=renewFeaRunLease)]
    pub fn renew_fea_run_lease_js(&self, run_id: String, lease: JsValue) -> Result<(), JsValue> {
        let lease: super::fea_lease::Lease =
            serde_wasm_bindgen::from_value(lease).map_err(|_| {
                super::fea_lease::denied_to_js(
                    runmat_runtime::analysis::fea_run_admission::FeaRunDenied::LicenseInvalid,
                )
            })?;
        let slot = self.active_fea_lease.borrow();
        let active = slot
            .as_ref()
            .filter(|active| active.run_id == run_id && active.lease_id == lease.lease_id)
            .ok_or_else(|| {
                super::fea_lease::denied_to_js(
                    runmat_runtime::analysis::fea_run_admission::FeaRunDenied::LicenseInvalid,
                )
            })?;
        active
            .control
            .renew(lease.budget()?)
            .map_err(super::fea_lease::denied_to_js)
    }
    #[wasm_bindgen(js_name=delegateFeaWorker)]
    pub fn delegate_fea_worker_js(&self, context: JsValue) -> Result<JsValue, JsValue> {
        let slot = self.active_fea_lease.borrow();
        slot.as_ref()
            .ok_or_else(|| {
                super::fea_lease::denied_to_js(
                    runmat_runtime::analysis::fea_run_admission::FeaRunDenied::SignInRequired,
                )
            })?
            .delegate_worker(context)
    }
    #[wasm_bindgen(js_name=reportFeaWorker)]
    pub fn report_fea_worker_js(
        &self,
        context: JsValue,
        usage: JsValue,
        final_report: bool,
    ) -> Result<JsValue, JsValue> {
        let slot = self.active_fea_lease.borrow();
        slot.as_ref()
            .ok_or_else(|| {
                super::fea_lease::denied_to_js(
                    runmat_runtime::analysis::fea_run_admission::FeaRunDenied::SignInRequired,
                )
            })?
            .report_worker(context, usage, final_report)
    }
    #[wasm_bindgen(js_name=startFeaWorker)]
    pub fn start_fea_worker_js(&self, context: JsValue) -> Result<(), JsValue> {
        let slot = self.active_fea_lease.borrow();
        slot.as_ref()
            .ok_or_else(|| {
                super::fea_lease::denied_to_js(
                    runmat_runtime::analysis::fea_run_admission::FeaRunDenied::SignInRequired,
                )
            })?
            .start_worker(context)
    }
    #[wasm_bindgen(js_name=discardUndispatchedFeaWorker)]
    pub fn discard_undispatched_fea_worker_js(
        &self,
        context: JsValue,
        receipt: JsValue,
    ) -> Result<(), JsValue> {
        let slot = self.active_fea_lease.borrow();
        slot.as_ref()
            .ok_or_else(|| {
                super::fea_lease::denied_to_js(
                    runmat_runtime::analysis::fea_run_admission::FeaRunDenied::SignInRequired,
                )
            })?
            .discard_undispatched_worker(context, receipt)
    }
    #[wasm_bindgen(js_name=acquireFeaWorkerGrant)]
    pub async fn acquire_fea_worker_grant_js(
        &self,
        context: JsValue,
        required_ops: JsValue,
    ) -> Result<JsValue, JsValue> {
        let active = self
            .active_fea_lease
            .borrow()
            .as_ref()
            .cloned()
            .ok_or_else(|| {
                super::fea_lease::denied_to_js(
                    runmat_runtime::analysis::fea_run_admission::FeaRunDenied::SignInRequired,
                )
            })?;
        let host = self
            .fea_admission
            .borrow()
            .as_ref()
            .cloned()
            .ok_or_else(|| {
                super::fea_lease::denied_to_js(
                    runmat_runtime::analysis::fea_run_admission::FeaRunDenied::SignInRequired,
                )
            })?;
        active.acquire_worker(context, required_ops, host).await
    }
    #[wasm_bindgen(js_name=retireFeaWorker)]
    pub fn retire_fea_worker_js(&self, context: JsValue, receipt: JsValue) -> Result<(), JsValue> {
        let slot = self.active_fea_lease.borrow();
        slot.as_ref()
            .ok_or_else(|| {
                super::fea_lease::denied_to_js(
                    runmat_runtime::analysis::fea_run_admission::FeaRunDenied::SignInRequired,
                )
            })?
            .retire_worker(context, receipt)
    }
    #[wasm_bindgen(js_name=planFeaSource)]
    pub async fn plan_fea_source_js(
        &self,
        path: String,
        inputs: JsValue,
    ) -> Result<JsValue, JsValue> {
        let (filesystem, path) = CapturedFilesystem::simulation(&path, inputs)?;
        self.plan_fea_source_in_scope(path, &filesystem).await
    }

    #[wasm_bindgen(js_name=runFeaSource)]
    pub async fn run_fea_source_js(
        &self,
        path: String,
        plan_id: String,
        inputs: JsValue,
    ) -> Result<JsValue, JsValue> {
        let (filesystem, path) = CapturedFilesystem::simulation(&path, inputs)?;
        self.run_fea_source_in_scope(path, plan_id, &filesystem)
            .await
    }
}

impl RunMatWasm {
    pub(super) async fn plan_fea_source_in_scope(
        &self,
        path: String,
        filesystem: &CapturedFilesystem,
    ) -> Result<JsValue, JsValue> {
        let mut session = self.take_session()?;
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let _interrupt =
            super::session::install_wasm_interrupt(&self.active_interrupt, cancellation.clone());
        let publication_supported = super::session::await_wasm_preparation(
            &cancellation,
            filesystem.publication_supported(),
        )
        .await?;
        super::session::await_wasm_preparation(&cancellation, async {
            filesystem
                .read
                .run(async {
                    self.fea_selection.borrow_mut().take();
                    let source = session
                        .load_and_check_fea_source_path(Path::new(&path))
                        .await
                        .map_err(|e| js_error(&e))?;
                    ensure_not_cancelled(&cancellation)?;
                    let plan = browser_fea_plan(&session, &source, publication_supported)?;
                    let ready = plan
                        .candidates
                        .iter()
                        .any(|candidate| candidate.readiness == AnalysisExecutionReadiness::Ready);
                    // A choice is bound to checked inputs, not to a prematurely generated mesh.
                    *self.fea_selection.borrow_mut() = ready.then_some(BrowserFeaSelection {
                        id: plan.plan_id.clone(),
                        path,
                        checked_revision: plan.revision,
                    });
                    crate::wire::json_contract_to_js(&plan).map_err(|e| js_error(&e.to_string()))
                })
                .await
                .map_err(|error| js_error(&error.to_string()))?
        })
        .await
    }
    pub(super) async fn run_fea_source_in_scope(
        &self,
        path: String,
        plan_id: String,
        filesystem: &CapturedFilesystem,
    ) -> Result<JsValue, JsValue> {
        let mut session = self.take_session()?;
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let _interrupt =
            super::session::install_wasm_interrupt(&self.active_interrupt, cancellation.clone());
        if !super::session::await_wasm_preparation(
            &cancellation,
            filesystem.publication_supported(),
        )
        .await?
        {
            return Err(js_error(PUBLICATION_HOST_REASON));
        }
        let _progress = runmat_analysis_fea::replace_fea_progress_context(
            runmat_analysis_fea::FeaProgressContext {
                handler: Some(std::sync::Arc::new(forward_progress)),
                cancellation: runmat_analysis_fea::FeaCancellationToken::from_token(
                    cancellation.token(),
                ),
                ..Default::default()
            },
        );
        let fresh = super::session::await_wasm_preparation(&cancellation, async {
            filesystem
                .read
                .run(session.load_and_check_fea_source_path(Path::new(&path)))
                .await
                .map_err(|error| js_error(&error.to_string()))?
                .map_err(|e| js_error(&e))
        })
        .await?;
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
        // Await the installed host before the first mesh/solve effect.
        let request = FeaRunRequest::local(
            FeaRunKind::MeshAndSolve,
            1000,
            checked_study_identity(&fresh)?,
        )
        .map_err(super::fea_lease::denied_to_js)?;
        let context = super::fea_async_host::start(
            self.fea_admission.clone(),
            self.active_fea_lease.clone(),
            self.pending_fea_admission.clone(),
            request,
        )
        .await
        .map_err(super::fea_lease::denied_to_js)?;
        let observation = session
            .fea_execution_timing_profile()
            .for_source(&fresh, runmat_core::FeaTimingRoute::BrowserCheckedSource, 1)
            .ok()
            .map(|key| key.start());
        let outcome = context.scope().run_checked_result_future(async {
            ensure_not_cancelled(&cancellation)?;
            context.check().await?;
            let result = filesystem.run(async {
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
            let response = execute_prepared(
                &request,
                available_memory as f64,
                context.worker_context(&request, super::fea_run_context::WorkerWorkClass::Numerical).await?,
            )
            .await?;
            context.retire_finished().await?;
            let json = response
            .as_string()
            .ok_or_else(|| js_error("Invalid numerical execution response."))?;
            context.check().await?;
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
            Ok::<_, JsValue>(result)
            }).await??;
            context.check().await?;
            filesystem.run_original(async {
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
            }).await?
        })
        .await.map_err(super::fea_lease::execution_error_to_js).and_then(|result| result);
        if let Some(observation) = observation {
            if !cancellation.token().is_cancelled() {
                observation.complete(outcome.is_ok());
            }
        }
        let admitted = context.check().await;
        let settled = context
            .finish(if cancellation.token().is_cancelled() {
                FeaRunOutcome::Cancelled
            } else if outcome.is_ok() {
                FeaRunOutcome::Completed
            } else {
                FeaRunOutcome::Failed
            })
            .await;
        let outcome = super::fea_lease::merge_js_lease_error(outcome, admitted.err());
        super::fea_lease::merge_js_lease_error(outcome, settled.err())
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
    worker_admission: JsValue,
) -> Result<String, JsValue> {
    // Content JSON is not authority: the trusted parent revalidates its cached
    // admitted root and exact delegation before this worker can solve.
    validate_numerical_input(json.len(), memory_bytes)?;
    let study: PreparedSerialFeaStudy =
        serde_json::from_str(&json).map_err(|error| js_error(&error.to_string()))?;
    let context = std::rc::Rc::new(
        LocalFeaRunContext::for_worker(
            worker_context,
            worker_admission,
            &json,
            study.authored_study_digest,
        )
        .await?,
    );
    let capacity = super::fea_async_host::BrowserNumericalCapacity::new(&context);
    let checked = context
        .scope()
        .run_checked_result_future(execute_prepared_fea_study_local(
            study,
            memory_bytes,
            &capacity,
        ))
        .await;
    let execution_cancelled = matches!(
        &checked,
        Err(runmat_runtime::analysis::fea_run_admission::FeaRunExecutionError::Cancelled)
    ) || (matches!(&checked, Ok(Err(_)))
        && context.scope().check().is_ok()
        && context.scope().cancellation_token().is_cancelled());
    let result = checked
        .map_err(|error| match error {
            runmat_runtime::analysis::fea_run_admission::FeaRunExecutionError::Denied(denied) => {
                super::fea_lease::denied_to_js(denied)
            }
            runmat_runtime::analysis::fea_run_admission::FeaRunExecutionError::Cancelled => {
                js_error("Simulation cancelled.")
            }
        })
        .and_then(|result| result);
    let result = super::fea_lease::merge_js_lease_error(
        result,
        context
            .scope()
            .check()
            .err()
            .map(super::fea_lease::denied_to_js),
    );
    let outcome = if execution_cancelled || runmat_runtime::interrupt::is_cancelled() {
        FeaRunOutcome::Cancelled
    } else if result.is_ok() {
        FeaRunOutcome::Completed
    } else {
        FeaRunOutcome::Failed
    };
    drop(capacity);
    let context = std::rc::Rc::try_unwrap(context)
        .map_err(|_| js_error("Owned worker admission remained active after execution."))?;
    let settled = context.settle(outcome).await;
    super::fea_lease::merge_js_lease_error(result, settled.err())
}

pub(super) fn validate_numerical_input(json_len: usize, memory_bytes: f64) -> Result<(), JsValue> {
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

async fn execute_prepared_fea_study_local(
    study: PreparedSerialFeaStudy,
    memory_bytes: f64,
    capacity: &dyn runmat_analysis_execution::AsyncCapacityOwner,
) -> Result<String, JsValue> {
    let progress = runmat_analysis_fea::FeaProgressContext {
        handler: Some(std::sync::Arc::new(forward_progress)),
        cancellation: runmat_runtime::interrupt::current_interrupt()
            .map(runmat_analysis_fea::FeaCancellationToken::from_token)
            .unwrap_or_default(),
        ..Default::default()
    };
    let future = async {
        let available_memory = memory_bytes as u64;
        runmat_runtime::builtins::wasm_registry::register_all();
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
            .execute_with_capacity(
                &mut host,
                &mut store,
                &|| js_sys::Date::now() as u64,
                &runmat_runtime::interrupt::is_cancelled,
                Some(capacity),
            )
            .await
            .map_err(|error| {
                if let runmat_core::PreparedSerialFeaExecutionError::Bridge(bridge) = &error {
                    if let Some(cause) = bridge.accounting_cause() {
                        return super::fea_lease::denied_to_js(
                            runmat_runtime::analysis::fea_run_admission::FeaRunDenied::Accounting(
                                cause.clone(),
                            ),
                        );
                    }
                }
                js_error(&error.to_string())
            })?;

        let runmat_runtime::analysis::StructuralStudyNumericalResult::LinearSequence(numerical) =
            result.into_parts().4
        else {
            return Err(js_error(
                "Unexpected numerical procedure returned by serial host.",
            ));
        };
        serde_json::to_string(&numerical).map_err(|e| js_error(&e.to_string()))
    };
    let mut future = std::pin::pin!(future);
    std::future::poll_fn(|cx| {
        let _progress = runmat_analysis_fea::replace_fea_progress_context(progress.clone());
        std::future::Future::poll(future.as_mut(), cx)
    })
    .await
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
