//! Admitted root lease and exact owned-worker content binding.
use super::fea_lease::{call_host, denied_to_js, ExternalPermit, Lease, LeaseState};
use crate::wire::errors::js_error;
use runmat_execution::Digest;
use runmat_runtime::analysis::fea_run_admission::{
    FeaRunDenied, FeaRunKind, FeaRunOutcome, FeaRunRequest, FeaRunScope, FeaRunSite,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use runmat_simulation_usage::{CompletedUsage, DelegatedReservation, LedgerError, OperationCount};
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UsageReport {
    completed_ops: runmat_simulation_usage::CompletedUsage,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
enum WorkerPhase {
    Start,
    Finish,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WorkerReport {
    phase: WorkerPhase,
    usage: UsageReport,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WorkerGrantRequest {
    required_ops: OperationCount,
    usage: UsageReport,
}
#[derive(Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) enum WorkerWorkClass {
    Numerical,
    OcctPreparation,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WorkerAssignment {
    prepared_study_identity: String,
    work_class: WorkerWorkClass,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct WorkerRetirementReceipt {
    run_id: String,
    delegation_id: String,
    worker_id: Option<String>,
    retired: bool,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SettlementReport {
    outcome: FeaRunOutcome,
    usage: UsageReport,
}

#[derive(Clone)]
pub(crate) struct ActiveFeaLease {
    pub run_id: String,
    pub lease_id: String,
    pub control: runmat_runtime::analysis::fea_run_admission::FeaRunControl,
    request: FeaRunRequest,
    root_started_at_unix_millis: u64,
    worker: Arc<Mutex<Option<OwnedWorkerGrant>>>,
    last_retired_worker: Arc<Mutex<Option<(String, String)>>>,
}

struct OwnedWorkerGrant {
    delegation_id: String,
    prepared_study_identity: String,
    work_class: WorkerWorkClass,
    reservation: DelegatedReservation,
    started: bool,
    final_reported: bool,
}

pub(crate) struct LocalFeaRunContext {
    request: FeaRunRequest,
    lease: Lease,
    root_started_at_unix_millis: u64,
    scope: FeaRunScope,
    host: Option<JsValue>,
    worker_authority: Option<(JsValue, JsValue)>,
    worker: Arc<Mutex<Option<OwnedWorkerGrant>>>,
    last_retired_worker: Arc<Mutex<Option<(String, String)>>>,
}

/// Content binding for an owned worker, not a lease or authorization token.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct WorkerContext {
    request: FeaRunRequest,
    prepared_study_identity: String,
    lease: Lease,
    root_started_at_unix_millis: u64,
    delegation_id: String,
    work_class: WorkerWorkClass,
}

impl LocalFeaRunContext {
    pub(super) async fn admit_request(
        request: FeaRunRequest,
        host: Option<JsValue>,
    ) -> Result<Self, JsValue> {
        let host = host.ok_or_else(|| denied_to_js(FeaRunDenied::SignInRequired))?;
        let request_js = crate::wire::json_contract_to_js(&request)
            .map_err(|error| js_error(&error.to_string()))?;
        let state: LeaseState = serde_wasm_bindgen::from_value(
            call_host(&host, "start", &request_js, &JsValue::UNDEFINED).await?,
        )
        .map_err(|_| denied_to_js(FeaRunDenied::LicenseInvalid))?;
        let admitted = state.lease.budget().and_then(|budget| {
            FeaRunScope::from_admitted_permit(request.clone(), Box::new(ExternalPermit(budget)))
                .map_err(denied_to_js)
        });
        let scope = match admitted {
            Ok(scope) => scope,
            Err(error) => {
                // Admission failed before numerical execution began.
                if let Ok(report) = crate::wire::json_contract_to_js(&SettlementReport {
                    outcome: FeaRunOutcome::Failed,
                    usage: UsageReport {
                        completed_ops: Default::default(),
                    },
                }) {
                    let _ = call_host(&host, "settle", &request_js, &report).await;
                }
                return Err(error);
            }
        };
        Ok(Self {
            request,
            lease: state.lease,
            root_started_at_unix_millis: state.root_started_at_unix_millis,
            scope,
            host: Some(host),
            worker_authority: None,
            worker: Arc::new(Mutex::new(None)),
            last_retired_worker: Arc::new(Mutex::new(None)),
        })
    }

    pub(crate) async fn worker_context(&self, prepared_json: &str, work_class: WorkerWorkClass) -> Result<JsValue, JsValue> {
        let host = self
            .host
            .as_ref()
            .ok_or_else(|| denied_to_js(FeaRunDenied::SignInRequired))?;
        self.active_lease()
            .worker_context(host, prepared_json, work_class)
            .await
    }

    pub(crate) async fn for_worker(
        context: JsValue,
        authority: JsValue,
        prepared_json: &str,
        study_identity: Digest,
    ) -> Result<Self, JsValue> {
        if context.is_undefined() || context.is_null() {
            return Err(denied_to_js(FeaRunDenied::SignInRequired));
        }
        let original_context = context.clone();
        let context: WorkerContext = serde_wasm_bindgen::from_value(context)
            .map_err(|_| denied_to_js(FeaRunDenied::LicenseInvalid))?;
        Self::validate_worker(&context, prepared_json, study_identity)
            .map_err(|_| denied_to_js(FeaRunDenied::LicenseInvalid))?;
        // JSON binds content only. A trusted parent callback must find this
        // exact request and payload in its already-admitted root cache.
        let state: LeaseState = serde_wasm_bindgen::from_value(
            call_host(
                &authority,
                "validateWorker",
                &original_context,
                &crate::wire::json_contract_to_js(&WorkerReport {
                    phase: WorkerPhase::Start,
                    usage: UsageReport {
                        completed_ops: Default::default(),
                    },
                })
                .map_err(|error| js_error(&error.to_string()))?,
            )
            .await?,
        )
        .map_err(|_| denied_to_js(FeaRunDenied::LicenseInvalid))?;
        if context.lease.lease_id != state.lease.lease_id
            || context.root_started_at_unix_millis != state.root_started_at_unix_millis
        {
            return Err(denied_to_js(FeaRunDenied::LicenseInvalid));
        }
        let now = js_sys::Date::now();
        if context.root_started_at_unix_millis == 0
            || context.root_started_at_unix_millis as f64 > now
        {
            return Err(denied_to_js(FeaRunDenied::LicenseInvalid));
        }
        let elapsed = (now - context.root_started_at_unix_millis as f64).ceil() as u64;
        let budget = state.lease.budget()?;
        let scope = FeaRunScope::from_parent_supervised_delegation(
            context.request.clone(),
            budget,
            elapsed,
        )
        .map_err(denied_to_js)?;
        Ok(Self {
            request: context.request,
            lease: state.lease,
            root_started_at_unix_millis: state.root_started_at_unix_millis,
            scope,
            host: None,
            worker_authority: Some((authority, original_context)),
            worker: Arc::new(Mutex::new(None)),
            last_retired_worker: Arc::new(Mutex::new(None)),
        })
    }

    fn validate_worker(
        context: &WorkerContext,
        prepared_json: &str,
        study_identity: Digest,
    ) -> Result<(), &'static str> {
        if context.prepared_study_identity != Digest::sha256(prepared_json.as_bytes()).to_string()
            || context.work_class != WorkerWorkClass::Numerical
            || context.delegation_id.is_empty()
            || context.delegation_id.len() > 256
            || context.request.site != FeaRunSite::Local
            || context.request.allocated_cpu_millicores != 1000
            || context.request.kind == FeaRunKind::Meshing
            || context.request.study_identity != study_identity.to_string()
        {
            return Err("Local FEA worker context does not match the exact prepared study and serial allocation.");
        }
        Ok(())
    }

    pub(crate) fn scope(&self) -> &FeaRunScope {
        &self.scope
    }
    pub(super) fn request(&self) -> &FeaRunRequest { &self.request }
    pub(super) fn active_lease(&self) -> ActiveFeaLease {
        ActiveFeaLease {
            run_id: self.request.run_id.clone(),
            lease_id: self.lease.lease_id.clone(),
            control: self.scope.control(),
            request: self.request.clone(),
            root_started_at_unix_millis: self.root_started_at_unix_millis,
            worker: self.worker.clone(),
            last_retired_worker: self.last_retired_worker.clone(),
        }
    }
    pub(crate) async fn check(&self) -> Result<(), JsValue> {
        let report = crate::wire::json_contract_to_js(&UsageReport {
            completed_ops: self.scope.usage().map_err(denied_to_js)?.completed_ops,
        })
        .map_err(|error| js_error(&error.to_string()))?;
        self.refresh("check", &report).await.map(|_| ())
    }
    pub(crate) async fn retire_finished(&self) -> Result<(), JsValue> {
        let host = self.host.as_ref().ok_or_else(|| denied_to_js(FeaRunDenied::LicenseInvalid))?;
        self.active_lease().retire_finished(host).await
    }
    /// Worker-side cooperative boundary. The caller awaits this before it
    /// reserves the next synchronous numerical batch.
    pub(crate) async fn acquire_worker_grant(&self, required: OperationCount) -> Result<runmat_runtime::analysis::fea_run_admission::FeaRunBudget, JsValue> {
        let (authority, original) = self.worker_authority.as_ref()
            .ok_or_else(|| denied_to_js(FeaRunDenied::LicenseInvalid))?;
        let required_js = crate::wire::json_contract_to_js(&WorkerGrantRequest {
            required_ops: required,
            usage: UsageReport { completed_ops: self.scope.usage().map_err(denied_to_js)?.completed_ops },
        })
            .map_err(|error| js_error(&error.to_string()))?;
        let state: LeaseState = serde_wasm_bindgen::from_value(
            call_host(authority, "acquireWorkerGrant", original, &required_js).await?,
        ).map_err(|_| denied_to_js(FeaRunDenied::LicenseInvalid))?;
        if state.lease.lease_id != self.lease.lease_id
            || state.root_started_at_unix_millis != self.root_started_at_unix_millis
        { return Err(denied_to_js(FeaRunDenied::LicenseInvalid)); }
        let budget = state.lease.budget()?;
        if budget.granted_ops < required { return Err(denied_to_js(FeaRunDenied::LicenseInvalid)); }
        self.scope.renew_parent_supervised(budget).map_err(denied_to_js)?;
        Ok(budget)
    }
    async fn refresh(&self, method: &str, extra: &JsValue) -> Result<LeaseState, JsValue> {
        let host = self
            .host
            .as_ref()
            .ok_or_else(|| denied_to_js(FeaRunDenied::SignInRequired))?;
        let request = crate::wire::json_contract_to_js(&self.request)
            .map_err(|error| js_error(&error.to_string()))?;
        let state: LeaseState =
            serde_wasm_bindgen::from_value(call_host(host, method, &request, extra).await?)
                .map_err(|_| denied_to_js(FeaRunDenied::LicenseInvalid))?;
        if state.lease.lease_id != self.lease.lease_id
            || state.root_started_at_unix_millis != self.root_started_at_unix_millis
        {
            return Err(denied_to_js(FeaRunDenied::LicenseInvalid));
        }
        self.scope
            .renew(state.lease.budget()?)
            .map_err(denied_to_js)?;
        Ok(state)
    }
    pub(crate) async fn settle(self, outcome: FeaRunOutcome) -> Result<(), JsValue> {
        if let Some(host) = &self.host {
            self.active_lease().retire_finished(host).await?;
            let undispatched = self.worker.lock().unwrap_or_else(|error| error.into_inner())
                .as_ref().is_some_and(|worker| !worker.started);
            if undispatched {
                let request = crate::wire::json_contract_to_js(&self.request)
                    .map_err(|error| js_error(&error.to_string()))?;
                call_host(host, "retireUndispatched", &request, &JsValue::UNDEFINED).await?;
            }
        }
        let control = self.scope.control();
        let checked = self.scope.settle(outcome).map_err(denied_to_js);
        // A live child or numerical reservation cannot be presented to the
        // host as final usage. Preserve the hold instead of settling a guess.
        let usage = control.settled_usage().map_err(denied_to_js)?;
        if let Some((authority, original)) = &self.worker_authority {
            // The worker has only computed a serialized numerical result.
            // Revalidate the current authoritative UI hold before any result
            // bytes leave it; it never publishes artifacts or settles a lease.
            let state: LeaseState = serde_wasm_bindgen::from_value(
                call_host(
                    authority,
                    "validateWorker",
                    original,
                    &crate::wire::json_contract_to_js(&WorkerReport {
                        phase: WorkerPhase::Finish,
                        usage: UsageReport {
                            completed_ops: usage.completed_ops,
                        },
                    })
                    .map_err(|error| js_error(&error.to_string()))?,
                )
                .await?,
            )
            .map_err(|_| denied_to_js(FeaRunDenied::LicenseInvalid))?;
            if state.lease.lease_id != self.lease.lease_id
                || state.root_started_at_unix_millis != self.root_started_at_unix_millis
            {
                return Err(denied_to_js(FeaRunDenied::LicenseInvalid));
            }
            state.lease.budget()?;
        }
        if let Some(host) = self.host {
            let request = crate::wire::json_contract_to_js(&self.request)
                .map_err(|error| js_error(&error.to_string()))?;
            let outcome = crate::wire::json_contract_to_js(&SettlementReport {
                outcome: if checked.is_err() && outcome == FeaRunOutcome::Completed {
                    FeaRunOutcome::Failed
                } else {
                    outcome
                },
                usage: UsageReport {
                    completed_ops: usage.completed_ops,
                },
            })
            .map_err(|error| js_error(&error.to_string()))?;
            let settled = call_host(&host, "settle", &request, &outcome)
                .await
                .map(|_| ());
            let retired = settled
                .as_ref()
                .err()
                .is_none_or(super::fea_lease::workers_retired);
            let result = super::fea_lease::merge_js_lease_error(checked, settled.err());
            if retired {
                if let Err(error) = &result {
                    let _ = js_sys::Reflect::set(error, &"workersRetired".into(), &JsValue::TRUE);
                }
            }
            result
        } else {
            checked
        }
    }
}

impl ActiveFeaLease {
    pub(super) async fn retire_finished(&self, host: &JsValue) -> Result<(), JsValue> {
        let request = crate::wire::json_contract_to_js(&self.request)
            .map_err(|error| js_error(&error.to_string()))?;
        call_host(host, "retireFinished", &request, &JsValue::UNDEFINED).await.map(|_| ())
    }
    pub(super) fn root_started_at_unix_millis(&self) -> u64 {
        self.root_started_at_unix_millis
    }
    fn owned_worker_context(&self, value: JsValue) -> Result<WorkerContext, JsValue> {
        let context: WorkerContext = serde_wasm_bindgen::from_value(value)
            .map_err(|_| denied_to_js(FeaRunDenied::LicenseInvalid))?;
        if context.request != self.request
            || context.lease.lease_id != self.lease_id
            || context.lease.granted_ops != OperationCount::ZERO
            || context.root_started_at_unix_millis != self.root_started_at_unix_millis
            || context.delegation_id.is_empty()
            || context.delegation_id.len() > 256
            || context.prepared_study_identity.len() != 71
            || !context.prepared_study_identity.starts_with("sha256:")
            || !context.prepared_study_identity.as_bytes()[7..].iter()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
        {
            return Err(denied_to_js(FeaRunDenied::LicenseInvalid));
        }
        Ok(context)
    }
    pub(super) fn delegate_worker(&self, value: JsValue) -> Result<JsValue, JsValue> {
        let context = self.owned_worker_context(value)?;
        let child_budget = context.lease.budget()?;
        let mut worker = self.worker.lock().unwrap_or_else(|error| error.into_inner());
        if worker.is_some() { return Err(denied_to_js(FeaRunDenied::LicenseInvalid)); }
        let mut grant = self.control.delegate(OperationCount::ZERO).map_err(denied_to_js)?;
        if child_budget.budget_ops != grant.delegation.budget.budget_ops
            || child_budget.heartbeat_millis != grant.delegation.budget.heartbeat_millis
            || child_budget.expires_at_unix_millis != grant.delegation.budget.expires_at_unix_millis
        {
            if let Err(error) = grant.reservation.finish(CompletedUsage::default()) {
                *worker = Some(OwnedWorkerGrant {
                    delegation_id: context.delegation_id,
                    prepared_study_identity: context.prepared_study_identity,
                    work_class: context.work_class,
                    reservation: grant.reservation,
                    started: false,
                    final_reported: false,
                });
                return Err(denied_to_js(error.into()));
            }
            return Err(denied_to_js(FeaRunDenied::LicenseInvalid));
        }
        *worker = Some(OwnedWorkerGrant {
            delegation_id: context.delegation_id,
            prepared_study_identity: context.prepared_study_identity,
            work_class: context.work_class,
            reservation: grant.reservation,
            started: false,
            final_reported: false,
        });
        crate::wire::json_contract_to_js(&LeaseState {
            lease: context.lease,
            root_started_at_unix_millis: self.root_started_at_unix_millis,
            delegation_id: None,
        }).map_err(|error| js_error(&error.to_string()))
    }
    pub(super) fn report_worker(&self, value: JsValue, usage: JsValue, final_report: bool) -> Result<JsValue, JsValue> {
        let context = self.owned_worker_context(value)?;
        let usage: UsageReport = serde_wasm_bindgen::from_value(usage)
            .map_err(|_| denied_to_js(FeaRunDenied::LicenseInvalid))?;
        let mut worker = self.worker.lock().unwrap_or_else(|error| error.into_inner());
        let worker = worker.as_mut().filter(|worker| worker.delegation_id == context.delegation_id
            && worker.prepared_study_identity == context.prepared_study_identity
            && worker.work_class == context.work_class)
            .ok_or_else(|| denied_to_js(FeaRunDenied::LicenseInvalid))?;
        if !worker.started { return Err(denied_to_js(FeaRunDenied::LicenseInvalid)); }
        if worker.work_class == WorkerWorkClass::OcctPreparation && usage.completed_ops != CompletedUsage::default() {
            return Err(denied_to_js(FeaRunDenied::LicenseInvalid));
        }
        if worker.final_reported && (!final_report || worker.reservation.completed() != usage.completed_ops) {
            return Err(denied_to_js(FeaRunDenied::LicenseInvalid));
        }
        worker.reservation.report(usage.completed_ops).map_err(|error| denied_to_js(error.into()))?;
        if final_report { worker.final_reported = true; }
        crate::wire::json_contract_to_js(&UsageReport {
            completed_ops: self.control.usage().map_err(denied_to_js)?.completed_ops,
        }).map_err(|error| js_error(&error.to_string()))
    }
    pub(super) fn start_worker(&self, value: JsValue) -> Result<(), JsValue> {
        let context = self.owned_worker_context(value)?;
        let mut slot = self.worker.lock().unwrap_or_else(|error| error.into_inner());
        let worker = slot.as_mut().filter(|worker| worker.delegation_id == context.delegation_id
            && worker.prepared_study_identity == context.prepared_study_identity
            && worker.work_class == context.work_class && !worker.started)
            .ok_or_else(|| denied_to_js(FeaRunDenied::LicenseInvalid))?;
        worker.started = true;
        Ok(())
    }
    /// The UI has proved this assignment never entered numerical execution.
    pub(super) fn discard_undispatched_worker(&self, value: JsValue, receipt: JsValue) -> Result<(), JsValue> {
        let context = self.owned_worker_context(value)?;
        let receipt: WorkerRetirementReceipt = serde_wasm_bindgen::from_value(receipt)
            .map_err(|_| denied_to_js(FeaRunDenied::LicenseInvalid))?;
        self.validate_retirement(&context, &receipt)?;
        let mut slot = self.worker.lock().unwrap_or_else(|error| error.into_inner());
        if slot.is_none() { return Ok(()); }
        let worker = slot.as_mut().filter(|worker| worker.delegation_id == context.delegation_id
            && worker.prepared_study_identity == context.prepared_study_identity
            && worker.work_class == context.work_class && !worker.started
            && worker.reservation.capacity() == OperationCount::ZERO)
            .ok_or_else(|| denied_to_js(FeaRunDenied::LicenseInvalid))?;
        worker.reservation.finish(CompletedUsage::default()).map_err(|error| denied_to_js(error.into()))?;
        slot.take();
        Ok(())
    }
    pub(super) async fn acquire_worker(
        &self,
        context_value: JsValue,
        required_value: JsValue,
        host: JsValue,
    ) -> Result<JsValue, JsValue> {
        let context = self.owned_worker_context(context_value)?;
        let maximum: OperationCount = serde_wasm_bindgen::from_value(required_value)
            .map_err(|_| denied_to_js(FeaRunDenied::LicenseInvalid))?;
        {
            let slot = self.worker.lock().unwrap_or_else(|error| error.into_inner());
            let worker = slot.as_ref().filter(|worker| worker.delegation_id == context.delegation_id
                && worker.prepared_study_identity == context.prepared_study_identity
                && worker.work_class == WorkerWorkClass::Numerical
                && context.work_class == WorkerWorkClass::Numerical && worker.started && !worker.final_reported)
                .ok_or_else(|| denied_to_js(FeaRunDenied::LicenseInvalid))?;
            let _ = worker;
        }
        let request = crate::wire::json_contract_to_js(&self.request)
            .map_err(|error| js_error(&error.to_string()))?;
        let usage = crate::wire::json_contract_to_js(&UsageReport {
            completed_ops: self.control.usage().map_err(denied_to_js)?.completed_ops,
        }).map_err(|error| js_error(&error.to_string()))?;
        let state: LeaseState = serde_wasm_bindgen::from_value(
            call_host(&host, "check", &request, &usage).await?,
        ).map_err(|_| denied_to_js(FeaRunDenied::LicenseInvalid))?;
        if state.lease.lease_id != self.lease_id
            || state.root_started_at_unix_millis != self.root_started_at_unix_millis
        { return Err(denied_to_js(FeaRunDenied::LicenseInvalid)); }
        self.control.renew(state.lease.budget()?).map_err(denied_to_js)?;
        let mut lease = state.lease;
        for _ in 0..3 {
            let acquisition = {
                let mut slot = self.worker.lock().unwrap_or_else(|error| error.into_inner());
                let worker = slot.as_mut().filter(|worker| worker.delegation_id == context.delegation_id
                    && worker.prepared_study_identity == context.prepared_study_identity
                    && worker.work_class == WorkerWorkClass::Numerical
                    && context.work_class == WorkerWorkClass::Numerical && worker.started && !worker.final_reported)
                    .ok_or_else(|| denied_to_js(FeaRunDenied::LicenseInvalid))?;
                // Grow with headroom so per-element reserves in the worker do
                // not each wait on a host call.
                worker.reservation.grow_capacity(maximum).map(|()| worker.reservation.capacity())
            };
            match acquisition {
                Ok(capacity) => {
                    lease.granted_ops = capacity;
                    return crate::wire::json_contract_to_js(&LeaseState {
                        lease,
                        root_started_at_unix_millis: self.root_started_at_unix_millis,
                        delegation_id: None,
                    }).map_err(|error| js_error(&error.to_string()));
                }
                Err(LedgerError::MoreGrantRequired { required, granted }) => {
                    // Ask the host ahead by the shared doubling law so
                    // per-element reserves do not each call the host.
                    let target = runmat_simulation_usage::grant_target(
                        required,
                        granted,
                        lease.budget_ops,
                    );
                    let count = crate::wire::json_contract_to_js(&target)
                        .map_err(|error| js_error(&error.to_string()))?;
                    let state: LeaseState = serde_wasm_bindgen::from_value(
                        call_host(&host, "grant", &request, &count).await?,
                    ).map_err(|_| denied_to_js(FeaRunDenied::LicenseInvalid))?;
                    if state.lease.lease_id != self.lease_id
                        || state.root_started_at_unix_millis != self.root_started_at_unix_millis
                    { return Err(denied_to_js(FeaRunDenied::LicenseInvalid)); }
                    self.control.renew(state.lease.budget()?).map_err(denied_to_js)?;
                    lease = state.lease;
                }
                Err(error) => return Err(denied_to_js(error.into())),
            }
        }
        Err(denied_to_js(FeaRunDenied::Accounting(LedgerError::GrantUnavailable)))
    }
    /// The UI coordinator invokes this only after stopOwnedWork has resolved.
    /// A missing final report must keep the hold instead of closing the root.
    pub(super) fn retire_worker(&self, value: JsValue, receipt: JsValue) -> Result<(), JsValue> {
        let context = self.owned_worker_context(value)?;
        let receipt: WorkerRetirementReceipt = serde_wasm_bindgen::from_value(receipt)
            .map_err(|_| denied_to_js(FeaRunDenied::LicenseInvalid))?;
        self.validate_retirement(&context, &receipt)?;
        let mut worker = self.worker.lock().unwrap_or_else(|error| error.into_inner());
        if worker.is_none()
            && self.last_retired_worker.lock().unwrap_or_else(|error| error.into_inner()).as_ref()
                .is_some_and(|(id, worker_id)| id == &context.delegation_id && Some(worker_id) == receipt.worker_id.as_ref())
        { return Ok(()); }
        let owned = worker.as_mut().filter(|worker| worker.delegation_id == context.delegation_id
            && worker.prepared_study_identity == context.prepared_study_identity
            && worker.work_class == context.work_class)
            .ok_or_else(|| denied_to_js(FeaRunDenied::LicenseInvalid))?;
        if receipt.worker_id.is_none() { return Err(denied_to_js(FeaRunDenied::LicenseInvalid)); }
        if !owned.final_reported { return Err(denied_to_js(FeaRunDenied::Accounting(LedgerError::OutstandingWork))); }
        let completed: CompletedUsage = owned.reservation.completed();
        owned.reservation.finish(completed).map_err(|error| denied_to_js(error.into()))?;
        worker.take();
        *self.last_retired_worker.lock().unwrap_or_else(|error| error.into_inner()) = Some((
            context.delegation_id,
            receipt.worker_id.expect("validated physical worker id"),
        ));
        Ok(())
    }
    fn validate_retirement(&self, context: &WorkerContext, receipt: &WorkerRetirementReceipt) -> Result<(), JsValue> {
        if !receipt.retired || receipt.run_id != self.run_id || receipt.delegation_id != context.delegation_id
            || receipt.worker_id.as_ref().is_some_and(|id| id.is_empty() || id.len() > 256)
        { return Err(denied_to_js(FeaRunDenied::LicenseInvalid)); }
        Ok(())
    }
    pub(super) fn study_identity(&self) -> Result<Digest, JsValue> {
        self.request
            .study_identity
            .parse()
            .map_err(|_| denied_to_js(FeaRunDenied::LicenseInvalid))
    }
    pub(super) async fn worker_context(
        &self,
        host: &JsValue,
        prepared_json: &str,
        work_class: WorkerWorkClass,
    ) -> Result<JsValue, JsValue> {
        let digest = Digest::sha256(prepared_json.as_bytes()).to_string();
        let request = crate::wire::json_contract_to_js(&self.request)
            .map_err(|error| js_error(&error.to_string()))?;
        let assignment = crate::wire::json_contract_to_js(&WorkerAssignment {
            prepared_study_identity: digest.clone(),
            work_class,
        }).map_err(|error| js_error(&error.to_string()))?;
        let state: LeaseState = serde_wasm_bindgen::from_value(
            call_host(host, "delegate", &request, &assignment).await?,
        )
        .map_err(|_| denied_to_js(FeaRunDenied::LicenseInvalid))?;
        if state.lease.lease_id != self.lease_id
            || state.root_started_at_unix_millis != self.root_started_at_unix_millis
        {
            return Err(denied_to_js(FeaRunDenied::LicenseInvalid));
        }
        // The returned lease carries only this child's zero-initial grant.
        // Root renewals arrive separately through renewFeaRunLease.
        state.lease.budget()?;
        self.control.check().map_err(denied_to_js)?;
        crate::wire::json_contract_to_js(&WorkerContext {
            request: self.request.clone(),
            prepared_study_identity: digest,
            lease: state.lease,
            root_started_at_unix_millis: self.root_started_at_unix_millis,
            delegation_id: state
                .delegation_id
                .ok_or_else(|| denied_to_js(FeaRunDenied::LicenseInvalid))?,
            work_class,
        })
        .map_err(|error| js_error(&error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn worker_context_binds_payload_study_site_and_allocation() {
        let identity = Digest::sha256(b"authored-study");
        let request = FeaRunRequest::local(FeaRunKind::MeshAndSolve, 1000, identity).unwrap();
        let context = || WorkerContext {
            request: request.clone(),
            prepared_study_identity: Digest::sha256(b"prepared").to_string(),
            lease: Lease {
                lease_id: "fealease-original".into(),
                state: super::super::fea_lease::AdmissionState::Active,
                budget_ops: runmat_simulation_usage::OperationCount::from_operations(120_000),
                granted_ops: runmat_simulation_usage::OperationCount::from_operations(120_000),
                expires_at: "2099-01-01T00:00:00Z".into(),
                heartbeat_seconds: 30,
            },
            root_started_at_unix_millis: 1,
            delegation_id: "run-original:worker:1".into(),
            work_class: WorkerWorkClass::Numerical,
        };
        assert!(LocalFeaRunContext::validate_worker(&context(), "prepared", identity).is_ok());
        assert!(LocalFeaRunContext::validate_worker(&context(), "changed", identity).is_err());
        assert!(LocalFeaRunContext::validate_worker(
            &context(),
            "prepared",
            Digest::sha256(b"changed")
        )
        .is_err());
        let mut dispatched = context();
        dispatched.request.site = FeaRunSite::Dispatched;
        assert!(LocalFeaRunContext::validate_worker(&dispatched, "prepared", identity).is_err());
        let mut oversized = context();
        oversized.request.allocated_cpu_millicores = 2000;
        assert!(LocalFeaRunContext::validate_worker(&oversized, "prepared", identity).is_err());
        let mut excluded = context();
        excluded.work_class = WorkerWorkClass::OcctPreparation;
        assert!(LocalFeaRunContext::validate_worker(&excluded, "prepared", identity).is_err());
    }
}
