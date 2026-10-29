//! Browser-local admission. JS callbacks deliberately never implement the native Send port.
use crate::wire::errors::js_error;
use js_sys::{Function, Promise, Reflect};
use serde::{Deserialize, Serialize};
use std::{cell::Cell, rc::Rc};
use wasm_bindgen::{prelude::*, JsCast};
use wasm_bindgen_futures::JsFuture;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BrowserFeaRunRequest {
    pub run_id: String,
    pub kind: runmat_runtime::analysis::fea_run_admission::FeaRunKind,
    pub allocated_cpu_millicores: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_wall_seconds: Option<u64>,
    pub study_identity: String,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Usage {
    cpu_core_millis: u64,
    wall_millis: u64,
}

fn usage(allocated: u64, wall_millis: u64) -> Result<Usage, &'static str> {
    let cpu_core_millis = (allocated as u128 * wall_millis as u128).div_ceil(1000);
    if cpu_core_millis > 9_007_199_254_740_991 || wall_millis > 9_007_199_254_740_991 {
        return Err("Browser FEA usage exceeds exact JavaScript integer accounting; the host must reconcile this lease.");
    }
    Ok(Usage {
        cpu_core_millis: cpu_core_millis as u64,
        wall_millis,
    })
}

fn usage_value(allocated: u64, wall_millis: u64) -> Result<JsValue, JsValue> {
    let usage = usage(allocated, wall_millis).map_err(|message| {
        let error = js_sys::Error::new(message);
        let _ = Reflect::set(
            &error,
            &"identifier".into(),
            &"RM.FEA.AccountingInvalid".into(),
        );
        // Never round/clamp a report into a different charge. Leave the lease
        // pending for host reconciliation rather than submit inexact usage.
        web_sys::console::error_1(&error);
        JsValue::from(error)
    })?;
    serde_wasm_bindgen::to_value(&usage).map_err(|e| js_error(&e.to_string()))
}

pub(crate) fn new_request(
    kind: runmat_runtime::analysis::fea_run_admission::FeaRunKind,
    study_identity: runmat_execution::Digest,
) -> Result<BrowserFeaRunRequest, JsValue> {
    let request = runmat_runtime::analysis::fea_run_admission::FeaRunRequest::local(
        kind,
        1000,
        study_identity,
    )
    .map_err(|_| lease_invalid("Unable to create a secure FEA run identity."))?;
    Ok(BrowserFeaRunRequest {
        run_id: request.run_id,
        kind: request.kind,
        allocated_cpu_millicores: request.allocated_cpu_millicores,
        estimated_wall_seconds: request.estimated_wall_seconds,
        study_identity: request.study_identity,
    })
}

fn now() -> f64 {
    // performance.now is monotonic in both Window and Worker realms.
    Reflect::get(&js_sys::global(), &"performance".into())
        .ok()
        .and_then(|performance| {
            let clock = Reflect::get(&performance, &"now".into()).ok()?;
            clock
                .dyn_ref::<Function>()?
                .call0(&performance)
                .ok()?
                .as_f64()
        })
        .unwrap_or_else(js_sys::Date::now)
}

pub(crate) fn validate_admission(admission: &JsValue) -> Result<(), JsValue> {
    for name in ["admit", "settle"] {
        if !Reflect::get(admission, &name.into())?.is_function() {
            return Err(js_error(&format!("FEA admission requires {name}().")));
        }
    }
    let heartbeat = Reflect::get(admission, &"heartbeat".into())?;
    if !heartbeat.is_undefined() && !heartbeat.is_function() {
        return Err(js_error("FEA admission heartbeat must be a function."));
    }
    Ok(())
}

fn call(admission: &JsValue, name: &str, args: &js_sys::Array) -> Result<JsValue, JsValue> {
    let method = Reflect::get(admission, &name.into())?
        .dyn_into::<Function>()
        .map_err(|_| js_error("Invalid FEA admission callback."))?;
    method.apply(admission, args)
}

/// Reporting failures must not stop an admitted run. The host journals/retries them.
fn report(admission: &JsValue, name: &str, args: &js_sys::Array) {
    if let Ok(result) = call(admission, name, args) {
        wasm_bindgen_futures::spawn_local(async move {
            let _ = JsFuture::from(Promise::resolve(&result)).await;
        });
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Admitted {
    permit_id: String,
}

fn denial(error: &JsValue) -> JsValue {
    let reason = Reflect::get(error, &"reason".into())
        .ok()
        .and_then(|v| v.as_string());
    let (identifier, message) = match reason.as_deref() {
        Some("signInRequired") => ("RM.FEA.SignInRequired", "Meshing and solving need a signed-in RunMat account. Run `runmat login`, then run this study again."),
        Some("allowanceUsed") => ("RM.FEA.AllowanceUsed", "You've used this week's core-hours. To keep going now, add a core-hour pack or move to Scale: https://runmat.com/pricing"),
        Some("leaseExpired" | "licenseInvalid") => ("RM.FEA.LeaseInvalid", "Your RunMat sign-in needs refreshing. Run `runmat login`, then run this study again."),
        _ => ("RM.FEA.ServerUnreachable", "Meshing and solving need a connection to RunMat Server to start. Check the connection, then run this study again."),
    };
    let dated_message = if reason.as_deref() == Some("allowanceUsed") {
        Reflect::get(error, &"refillsAt".into()).ok().and_then(|value| value.as_string())
            .map(|date| format!("You've used this week's core-hours. They refill on {date}. To keep going now, add a core-hour pack or move to Scale: https://runmat.com/pricing"))
    } else {
        None
    };
    let result = js_sys::Error::new(dated_message.as_deref().unwrap_or(message));
    let _ = Reflect::set(&result, &"identifier".into(), &identifier.into());
    let _ = Reflect::set(&result, &"denied".into(), error);
    result.into()
}

pub(crate) fn sign_in_required() -> JsValue {
    denial(
        &crate::wire::json_contract_to_js(&serde_json::json!({"reason":"signInRequired"})).unwrap(),
    )
}

pub(crate) fn lease_invalid(detail: &str) -> JsValue {
    let result = denial(
        &crate::wire::json_contract_to_js(&serde_json::json!({"reason":"licenseInvalid"})).unwrap(),
    );
    let _ = Reflect::set(&result, &"detail".into(), &detail.into());
    result
}

pub(crate) struct BrowserFeaPermit {
    admission: JsValue,
    permit_id: String,
    request: BrowserFeaRunRequest,
    started: f64,
    finished: Rc<Cell<bool>>,
    last_heartbeat: Rc<Cell<f64>>,
    timer: Option<(JsValue, Closure<dyn FnMut()>)>,
    owns_reporting: bool,
}

impl BrowserFeaPermit {
    pub(crate) async fn admit(
        admission: Option<JsValue>,
        request: BrowserFeaRunRequest,
        owns_reporting: bool,
    ) -> Result<Self, JsValue> {
        let admission = admission.ok_or_else(sign_in_required)?;
        validate_admission(&admission)
            .map_err(|_| lease_invalid("Invalid FEA admission adapter."))?;
        let args = js_sys::Array::new();
        args.push(&serde_wasm_bindgen::to_value(&request).map_err(|e| js_error(&e.to_string()))?);
        let pending = call(&admission, "admit", &args).map_err(|_| denial(&JsValue::UNDEFINED))?;
        let response = JsFuture::from(Promise::resolve(&pending))
            .await
            .map_err(|_| denial(&JsValue::UNDEFINED))?;
        let denied = Reflect::get(&response, &"denied".into())
            .map_err(|_| lease_invalid("Invalid FEA permit response."))?;
        if !denied.is_undefined() {
            return Err(denial(&denied));
        }
        let admitted: Admitted = serde_wasm_bindgen::from_value(response)
            .map_err(|_| lease_invalid("Invalid FEA permit response."))?;
        if admitted.permit_id.is_empty() {
            return Err(lease_invalid("Empty FEA permit identifier."));
        }
        let mut permit = Self {
            admission,
            permit_id: admitted.permit_id,
            request,
            started: now(),
            finished: Rc::new(Cell::new(false)),
            last_heartbeat: Rc::new(Cell::new(now())),
            timer: None,
            owns_reporting,
        };
        if owns_reporting {
            permit.start_heartbeats();
        }
        Ok(permit)
    }

    pub(crate) fn delegation(&self, prepared_json: &str) -> Result<JsValue, JsValue> {
        crate::wire::json_contract_to_js(&serde_json::json!({
            "request": self.request, "permitId": self.permit_id,
            "preparedStudyIdentity": runmat_execution::Digest::sha256(prepared_json.as_bytes()).to_string(),
        })).map_err(|e| js_error(&e.to_string()))
    }

    pub(crate) fn permit_id(&self) -> &str {
        &self.permit_id
    }

    /// Establish native nested-run identity only after JS admission succeeds.
    /// This bridge contains no JS values and never acquires/reports another lease.
    pub(crate) fn native_scope(
        &self,
    ) -> Result<runmat_runtime::analysis::fea_run_admission::FeaRunScope, JsValue> {
        use runmat_runtime::analysis::fea_run_admission::{FeaRunRequest, FeaRunScope, FeaRunSite};
        let request = FeaRunRequest {
            run_id: self.request.run_id.clone(),
            kind: self.request.kind.clone(),
            site: FeaRunSite::Local,
            allocated_cpu_millicores: self.request.allocated_cpu_millicores,
            estimated_wall_seconds: self.request.estimated_wall_seconds,
            study_identity: self.request.study_identity.clone(),
        };
        let bridge = AuthorizedBrowserRun {
            request: serde_json::to_vec(&request).map_err(|e| js_error(&e.to_string()))?,
        };
        FeaRunScope::admit_with(request, &bridge)
            .map_err(|_| lease_invalid("Invalid browser allocation."))
    }

    pub(crate) fn reject_delegation(mut self) {
        self.owns_reporting = true;
        self.finish(runmat_runtime::analysis::fea_run_admission::FeaRunOutcome::Failed);
    }

    fn current_usage(&self) -> Result<JsValue, JsValue> {
        usage_value(
            self.request.allocated_cpu_millicores,
            (now() - self.started).max(0.0) as u64,
        )
    }

    /// Synchronous workers cannot service timers while solving. Scheduler clock
    /// checkpoints also report cumulative usage, without blocking on transport.
    pub(crate) fn heartbeat_if_due(&self) {
        let time = now();
        if !self.owns_reporting
            || self.finished.get()
            || time - self.last_heartbeat.get() < 30_000.0
        {
            return;
        }
        self.last_heartbeat.set(time);
        let args = js_sys::Array::new();
        args.push(&self.permit_id.clone().into());
        if let Ok(usage) = self.current_usage() {
            args.push(&usage);
            report(&self.admission, "heartbeat", &args);
        }
    }

    fn start_heartbeats(&mut self) {
        let admission = self.admission.clone();
        let permit_id = self.permit_id.clone();
        let allocated = self.request.allocated_cpu_millicores;
        let started = self.started;
        let finished = self.finished.clone();
        let last_heartbeat = self.last_heartbeat.clone();
        let callback = Closure::wrap(Box::new(move || {
            if finished.get() {
                return;
            }
            last_heartbeat.set(now());
            let args = js_sys::Array::new();
            args.push(&permit_id.clone().into());
            if let Ok(usage) = usage_value(allocated, (now() - started).max(0.0) as u64) {
                args.push(&usage);
                report(&admission, "heartbeat", &args);
            }
        }) as Box<dyn FnMut()>);
        if let Ok(set_interval) = Reflect::get(&js_sys::global(), &"setInterval".into()) {
            if let Some(function) = set_interval.dyn_ref::<Function>() {
                if let Ok(id) = function.call2(&js_sys::global(), callback.as_ref(), &30_000.into())
                {
                    self.timer = Some((id, callback));
                }
            }
        }
    }

    pub(crate) fn settle(
        mut self,
        outcome: runmat_runtime::analysis::fea_run_admission::FeaRunOutcome,
    ) {
        self.finish(outcome);
    }

    fn finish(&mut self, outcome: runmat_runtime::analysis::fea_run_admission::FeaRunOutcome) {
        if self.finished.replace(true) {
            return;
        }
        if let Some((id, _callback)) = self.timer.take() {
            if let Ok(clear) = Reflect::get(&js_sys::global(), &"clearInterval".into()) {
                if let Some(clear) = clear.dyn_ref::<Function>() {
                    let _ = clear.call1(&js_sys::global(), &id);
                }
            }
        }
        if self.owns_reporting {
            let args = js_sys::Array::new();
            args.push(&self.permit_id.clone().into());
            if let (Ok(outcome), Ok(usage)) =
                (serde_wasm_bindgen::to_value(&outcome), self.current_usage())
            {
                args.push(&outcome);
                args.push(&usage);
                report(&self.admission, "settle", &args);
            }
        }
    }
}

struct AuthorizedBrowserRun {
    request: Vec<u8>,
}
struct BrowserMarkerPermit;
impl runmat_runtime::analysis::fea_run_admission::FeaRunPermit for BrowserMarkerPermit {
    fn settle(
        self: Box<Self>,
        _outcome: runmat_runtime::analysis::fea_run_admission::FeaRunOutcome,
        _usage: &runmat_execution::resource::ResourceUsage,
    ) {
    }
}
impl runmat_runtime::analysis::fea_run_admission::FeaRunAdmission for AuthorizedBrowserRun {
    fn admit(
        &self,
        request: &runmat_runtime::analysis::fea_run_admission::FeaRunRequest,
    ) -> Result<
        Box<dyn runmat_runtime::analysis::fea_run_admission::FeaRunPermit>,
        runmat_runtime::analysis::fea_run_admission::FeaRunDenied,
    > {
        use runmat_runtime::analysis::fea_run_admission::FeaRunDenied;
        if serde_json::to_vec(request).map_err(|_| FeaRunDenied::LicenseInvalid)? != self.request {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        Ok(Box::new(BrowserMarkerPermit))
    }
}

impl Drop for BrowserFeaPermit {
    fn drop(&mut self) {
        self.finish(runmat_runtime::analysis::fea_run_admission::FeaRunOutcome::Failed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocation_not_cpu_time() {
        let report = usage(2000, 1234).unwrap();
        assert_eq!(report.cpu_core_millis, 2468);
        assert_eq!(report.wall_millis, 1234);
    }
    #[test]
    fn fractional_cores_round_up_and_inexact_reports_are_rejected() {
        assert_eq!(usage(1500, 3).unwrap().cpu_core_millis, 5);
        assert!(usage(u64::MAX, u64::MAX).is_err());
        assert!(usage(1000, 9_007_199_254_740_992).is_err());
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
mod browser_tests {
    use super::*;
    use runmat_runtime::analysis::fea_run_admission::{FeaRunKind, FeaRunOutcome};
    use wasm_bindgen_test::wasm_bindgen_test;

    fn adapter() -> JsValue {
        Function::new_no_args("return { admitted:0, outcomes:[], admit:async function(request) { this.admitted++; this.request=request; return {permitId:'test-lease'}; }, settle:async function(id,outcome,usage) { this.outcomes.push({id,outcome,usage}); } };")
            .call0(&JsValue::UNDEFINED).unwrap()
    }
    fn outcomes(adapter: &JsValue) -> js_sys::Array {
        Reflect::get(adapter, &"outcomes".into())
            .unwrap()
            .unchecked_into()
    }
    fn request() -> BrowserFeaRunRequest {
        new_request(
            FeaRunKind::MeshAndSolve,
            runmat_execution::Digest::sha256(b"test-study"),
        )
        .unwrap()
    }

    #[wasm_bindgen_test]
    fn synthesized_denials_are_plain_objects_with_stable_identifiers() {
        for (error, identifier, reason) in [
            (
                sign_in_required(),
                "RM.FEA.SignInRequired",
                "signInRequired",
            ),
            (
                lease_invalid("fixture"),
                "RM.FEA.LeaseInvalid",
                "licenseInvalid",
            ),
        ] {
            assert_eq!(
                Reflect::get(&error, &"identifier".into())
                    .unwrap()
                    .as_string()
                    .as_deref(),
                Some(identifier)
            );
            let denied = Reflect::get(&error, &"denied".into()).unwrap();
            assert!(!denied.is_instance_of::<js_sys::Map>());
            assert_eq!(
                Reflect::get(&denied, &"reason".into())
                    .unwrap()
                    .as_string()
                    .as_deref(),
                Some(reason)
            );
        }
    }

    #[wasm_bindgen_test]
    async fn delegation_is_a_plain_object_for_the_owned_javascript_host() {
        let adapter = adapter();
        let permit = BrowserFeaPermit::admit(Some(adapter), request(), true)
            .await
            .unwrap();
        let delegation = permit.delegation("exact-job").unwrap();
        assert!(!delegation.is_instance_of::<js_sys::Map>());
        assert_eq!(
            Reflect::get(&delegation, &"permitId".into())
                .unwrap()
                .as_string()
                .as_deref(),
            Some("test-lease")
        );
        let request = Reflect::get(&delegation, &"request".into()).unwrap();
        assert!(!request.is_instance_of::<js_sys::Map>());
        assert_eq!(
            Reflect::get(&request, &"allocatedCpuMillicores".into())
                .unwrap()
                .as_f64(),
            Some(1000.0)
        );
        assert_eq!(
            Reflect::get(&request, &"kind".into())
                .unwrap()
                .as_string()
                .as_deref(),
            Some("meshAndSolve")
        );
        permit.settle(FeaRunOutcome::Completed);
    }

    #[wasm_bindgen_test]
    async fn unset_admission_denies_before_worker_json_or_solver() {
        let result = crate::api::fea_execution::execute_prepared_fea_study(
            "not-json".into(),
            1024.0,
            JsValue::UNDEFINED,
        )
        .await;
        let error = result.unwrap_err();
        assert_eq!(
            Reflect::get(&error, &"identifier".into())
                .unwrap()
                .as_string()
                .as_deref(),
            Some("RM.FEA.SignInRequired")
        );
    }

    #[wasm_bindgen_test]
    async fn denied_admission_has_no_permit_or_settlement() {
        let adapter = adapter();
        Reflect::set(&adapter, &"admit".into(), &Function::new_no_args("return Promise.resolve({denied:{reason:'allowanceUsed',refillsAt:'2026-11-02T09:14:05Z'}});" )).unwrap();
        assert!(
            BrowserFeaPermit::admit(Some(adapter.clone()), request(), true)
                .await
                .is_err()
        );
        assert_eq!(outcomes(&adapter).length(), 0);
    }

    #[wasm_bindgen_test]
    async fn completed_cancelled_and_failed_settle_exactly_once() {
        for outcome in [
            FeaRunOutcome::Completed,
            FeaRunOutcome::Cancelled,
            FeaRunOutcome::Failed,
        ] {
            let adapter = adapter();
            let permit = BrowserFeaPermit::admit(Some(adapter.clone()), request(), true)
                .await
                .unwrap();
            permit.settle(outcome);
            assert_eq!(outcomes(&adapter).length(), 1);
            let value = outcomes(&adapter).get(0);
            let report: serde_json::Value = serde_wasm_bindgen::from_value(value).unwrap();
            assert_eq!(report["outcome"], serde_json::to_value(outcome).unwrap());
            assert_eq!(
                report["usage"]["cpuCoreMillis"],
                report["usage"]["wallMillis"]
            );
        }
    }

    #[wasm_bindgen_test]
    async fn unfinished_drop_releases_as_failed() {
        let adapter = adapter();
        drop(
            BrowserFeaPermit::admit(Some(adapter.clone()), request(), true)
                .await
                .unwrap(),
        );
        assert_eq!(outcomes(&adapter).length(), 1);
        assert_eq!(
            Reflect::get(&outcomes(&adapter).get(0), &"outcome".into())
                .unwrap()
                .as_string()
                .as_deref(),
            Some("failed")
        );
    }

    #[wasm_bindgen_test]
    async fn authorized_delegation_does_not_report_or_settle_a_second_lease() {
        let adapter = adapter();
        let permit = BrowserFeaPermit::admit(Some(adapter.clone()), request(), false)
            .await
            .unwrap();
        assert_eq!(permit.permit_id(), "test-lease");
        permit.settle(FeaRunOutcome::Completed);
        assert_eq!(
            Reflect::get(&adapter, &"admitted".into()).unwrap().as_f64(),
            Some(1.0)
        );
        assert_eq!(outcomes(&adapter).length(), 0);
    }

    #[wasm_bindgen_test]
    async fn forged_job_digest_cannot_reach_worker_authorization() {
        let adapter = adapter();
        let authorization = js_sys::Object::new();
        Reflect::set(&authorization, &"admission".into(), &adapter).unwrap();
        let delegation = serde_wasm_bindgen::to_value(&serde_json::json!({
            "request": request(), "permitId":"test-lease", "preparedStudyIdentity":"forged"
        }))
        .unwrap();
        Reflect::set(&authorization, &"delegation".into(), &delegation).unwrap();
        assert!(crate::api::fea_execution::execute_prepared_fea_study(
            "not-json".into(),
            1024.0,
            authorization.into()
        )
        .await
        .is_err());
        assert_eq!(
            Reflect::get(&adapter, &"admitted".into()).unwrap().as_f64(),
            Some(0.0)
        );
    }
}
