//! Local JS transport stays outside the runtime's Send permit and control ports.
use runmat_runtime::analysis::fea_run_admission::{
    FeaRunBudget, FeaRunDenied, FeaRunOutcome, FeaRunPermit, FeaRunUsage,
};
use runmat_simulation_usage::{LedgerError, OperationCount};
use serde::{Deserialize, Serialize};
use wasm_bindgen::{prelude::*, JsCast};

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Lease {
    pub lease_id: String,
    pub state: AdmissionState,
    pub budget_ops: OperationCount,
    pub granted_ops: OperationCount,
    pub expires_at: String,
    pub heartbeat_seconds: u64,
}
#[derive(Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) enum AdmissionState {
    Active,
    Settled,
    Expired,
}
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct LeaseState {
    pub lease: Lease,
    pub root_started_at_unix_millis: u64,
    #[serde(default)]
    pub delegation_id: Option<String>,
}
impl Lease {
    pub fn budget(&self) -> Result<FeaRunBudget, JsValue> {
        let expiry = js_sys::Date::parse(&self.expires_at);
        if self.state != AdmissionState::Active
            || self.lease_id.is_empty()
            || self.lease_id.len() > 256
            || !expiry.is_finite()
            || expiry < 0.0
            || expiry > 9_007_199_254_740_991.0
        {
            return Err(denied_to_js(FeaRunDenied::LicenseInvalid));
        }
        let budget = FeaRunBudget {
            budget_ops: self.budget_ops,
            granted_ops: self.granted_ops,
            expires_at_unix_millis: expiry as u64,
            heartbeat_millis: self
                .heartbeat_seconds
                .checked_mul(1000)
                .ok_or_else(|| denied_to_js(FeaRunDenied::LicenseInvalid))?,
        };
        budget.validate().map_err(denied_to_js)?;
        Ok(budget)
    }
}
/// Created only after the installed asynchronous host admits the root. JS owns
/// transport and sends renewals through the pure control; root async code sends
/// the single actual settlement. This value contains no JS reference.
pub(super) struct ExternalPermit(pub FeaRunBudget);
impl FeaRunPermit for ExternalPermit {
    fn budget(&self) -> FeaRunBudget {
        self.0
    }
    fn heartbeat(&self, _: &FeaRunUsage) -> Result<FeaRunBudget, FeaRunDenied> {
        Ok(self.0)
    }
    fn settle(self: Box<Self>, _: FeaRunOutcome, _: &FeaRunUsage) -> Result<(), FeaRunDenied> {
        Ok(())
    }
}
pub(super) async fn call_host(
    host: &JsValue,
    method: &str,
    argument: &JsValue,
    extra: &JsValue,
) -> Result<JsValue, JsValue> {
    let callback = js_sys::Reflect::get(host, &JsValue::from_str(method))
        .ok()
        .and_then(|value| value.dyn_into::<js_sys::Function>().ok())
        .ok_or_else(|| denied_to_js(FeaRunDenied::SignInRequired))?;
    let promise = callback
        .call2(host, argument, extra)
        .map_err(normalize_denial)?;
    wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(&promise))
        .await
        .map_err(normalize_denial)
}
pub(super) fn denied_to_js(denied: FeaRunDenied) -> JsValue {
    let error = js_sys::Error::new(&denied.to_string());
    let reason = match denied {
        FeaRunDenied::SignInRequired => "signInRequired",
        FeaRunDenied::AllowanceUsed { .. } => "allowanceUsed",
        FeaRunDenied::LeaseExpired => "leaseExpired",
        FeaRunDenied::LicenseInvalid => "licenseInvalid",
        FeaRunDenied::ServerUnreachable => "serverUnreachable",
        FeaRunDenied::AllowanceUnpublished => "allowanceUnpublished",
        FeaRunDenied::BudgetUnavailable { .. } => "budgetUnavailable",
        FeaRunDenied::Accounting(_) => "accountingInvalid",
    };
    let payload = js_sys::Object::new();
    let _ = js_sys::Reflect::set(&payload, &"reason".into(), &reason.into());
    if let FeaRunDenied::Accounting(cause) = &denied {
        if let Ok(cause) = crate::wire::json_contract_to_js(cause) {
            let _ = js_sys::Reflect::set(&payload, &"cause".into(), &cause);
        }
    }
    if let FeaRunDenied::BudgetUnavailable {
        available_ops: Some(available),
    } = &denied
    {
        let _ = js_sys::Reflect::set(
            &payload,
            &"availableOps".into(),
            &available.to_string().into(),
        );
    }
    if let FeaRunDenied::AllowanceUsed {
        refills_at_unix_seconds: Some(seconds),
    } = &denied
    {
        let _ = js_sys::Reflect::set(
            &payload,
            &"refillsAtUnixSeconds".into(),
            &JsValue::from_f64(*seconds as f64),
        );
        let date = js_sys::Date::new(&JsValue::from_f64(*seconds as f64 * 1000.0));
        if date.get_time().is_finite() {
            let _ =
                js_sys::Reflect::set(&payload, &"refillsAt".into(), &date.to_iso_string().into());
        }
    }
    let _ = js_sys::Reflect::set(&error, &"code".into(), &denied.error_code().into());
    let _ = js_sys::Reflect::set(&error, &"denied".into(), &payload);
    error.into()
}
pub(super) fn execution_error_to_js(
    error: runmat_runtime::analysis::fea_run_admission::FeaRunExecutionError,
) -> JsValue {
    match error {
        runmat_runtime::analysis::fea_run_admission::FeaRunExecutionError::Denied(denied) => {
            denied_to_js(denied)
        }
        runmat_runtime::analysis::fea_run_admission::FeaRunExecutionError::Cancelled => {
            let error = js_sys::Error::new("FEA run cancelled.");
            let _ = js_sys::Reflect::set(&error, &"code".into(), &"RM.FEA.Cancelled".into());
            error.into()
        }
    }
}
fn field(value: &JsValue, key: &str) -> JsValue {
    js_sys::Reflect::get(value, &key.into()).unwrap_or(JsValue::UNDEFINED)
}
pub(super) fn normalize_denial(error: JsValue) -> JsValue {
    let retired = field(&error, "workersRetired").as_bool() == Some(true);
    let result = denied_to_js(denial_from_js(error.clone()));
    // Retain the host error as the cause of the normalized admission error.
    let _ = js_sys::Reflect::set(&result, &"cause".into(), &error);
    if retired {
        let _ = js_sys::Reflect::set(&result, &"workersRetired".into(), &JsValue::TRUE);
    }
    result
}
pub(super) fn workers_retired(error: &JsValue) -> bool {
    field(error, "workersRetired").as_bool() == Some(true)
}
pub(super) fn denial_from_js(error: JsValue) -> FeaRunDenied {
    let denied = field(&error, "denied");
    let reason = field(&denied, "reason").as_string();
    let status = field(&error, "status")
        .as_f64()
        .or_else(|| field(&error, "statusCode").as_f64());
    let body = field(&error, "body");
    let detail = field(if body.is_undefined() { &error } else { &body }, "error");
    let allowance = reason.as_deref() == Some("allowanceUsed")
        || (status == Some(409.0)
            && field(&detail, "code").as_string().as_deref() == Some("simulation.allowance_used"));
    if allowance {
        let date = field(&denied, "refillsAt")
            .as_string()
            .or_else(|| field(&field(&detail, "details"), "refillsAt").as_string());
        FeaRunDenied::AllowanceUsed {
            refills_at_unix_seconds: field(&denied, "refillsAtUnixSeconds")
                .as_f64()
                .map(|seconds| seconds as i64)
                .or_else(|| {
                    date.and_then(|date| {
                        let ms = js_sys::Date::parse(&date);
                        ms.is_finite().then_some((ms / 1000.0) as i64)
                    })
                }),
        }
    } else if reason.as_deref() == Some("accountingInvalid") {
        FeaRunDenied::Accounting(
            serde_wasm_bindgen::from_value::<LedgerError>(field(&denied, "cause"))
                .unwrap_or(LedgerError::InvalidState),
        )
    } else if reason.as_deref() == Some("allowanceUnpublished")
        || field(&detail, "code").as_string().as_deref() == Some("simulation.allowance_unpublished")
    {
        FeaRunDenied::AllowanceUnpublished
    } else if reason.as_deref() == Some("budgetUnavailable")
        || field(&detail, "code").as_string().as_deref()
            == Some("simulation.budget_exceeds_available")
    {
        let available = field(&denied, "availableOps");
        let available = if available.is_undefined() {
            field(&field(&detail, "details"), "availableOps")
        } else {
            available
        };
        FeaRunDenied::BudgetUnavailable {
            available_ops: serde_wasm_bindgen::from_value(available).ok(),
        }
    } else if reason.as_deref() == Some("signInRequired") || status == Some(401.0) {
        FeaRunDenied::SignInRequired
    } else if reason.as_deref() == Some("leaseExpired") || status == Some(404.0) {
        FeaRunDenied::LeaseExpired
    } else if reason.as_deref() == Some("licenseInvalid")
        || matches!(status, Some(403.0 | 409.0 | 422.0))
    {
        FeaRunDenied::LicenseInvalid
    } else {
        FeaRunDenied::ServerUnreachable
    }
}
/// Keep the original domain error and attach the typed lease failure once.
pub(super) fn merge_js_lease_error<T>(
    result: Result<T, JsValue>,
    lease: Option<JsValue>,
) -> Result<T, JsValue> {
    match (result, lease) {
        (result, None) => result,
        (Ok(_), Some(lease)) => Err(lease),
        (Err(primary), Some(lease)) => {
            let key = JsValue::from_str("feaLeaseError");
            if primary.is_object() && js_sys::Reflect::has(&primary, &key).unwrap_or(false) {
                return Err(primary);
            }
            if primary.is_object() && js_sys::Reflect::set(&primary, &key, &lease).unwrap_or(false)
            {
                return Err(primary);
            }
            let wrapper = js_sys::Error::new("FEA execution failed; see the original cause.");
            let _ = js_sys::Reflect::set(&wrapper, &JsValue::from_str("cause"), &primary);
            let _ = js_sys::Reflect::set(&wrapper, &key, &lease);
            Err(wrapper.into())
        }
    }
}
