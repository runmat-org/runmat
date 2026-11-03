//! Admitted root lease and exact owned-worker content binding.
use super::fea_lease::{call_host, denied_to_js, ExternalPermit, Lease, LeaseState};
use crate::wire::errors::js_error;
use runmat_execution::Digest;
use runmat_runtime::analysis::fea_run_admission::{
    FeaRunDenied, FeaRunKind, FeaRunOutcome, FeaRunRequest, FeaRunScope, FeaRunSite,
};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Clone)]
pub(crate) struct ActiveFeaLease {
    pub run_id: String,
    pub lease_id: String,
    pub control: runmat_runtime::analysis::fea_run_admission::FeaRunControl,
    request: FeaRunRequest,
    root_started_at_unix_millis: u64,
}

pub(crate) struct LocalFeaRunContext {
    request: FeaRunRequest,
    lease: Lease,
    root_started_at_unix_millis: u64,
    scope: FeaRunScope,
    host: Option<JsValue>,
    worker_authority: Option<(JsValue, JsValue)>,
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
                let _ = call_host(&host, "settle", &request_js, &JsValue::from_str("failed")).await;
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
        })
    }

    pub(crate) async fn worker_context(&self, prepared_json: &str) -> Result<JsValue, JsValue> {
        let host = self
            .host
            .as_ref()
            .ok_or_else(|| denied_to_js(FeaRunDenied::SignInRequired))?;
        self.active_lease()
            .worker_context(host, prepared_json)
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
                &JsValue::from_str("start"),
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
        state.lease.budget()?;
        let scope =
            FeaRunScope::from_parent_supervised_delegation(context.request.clone(), elapsed)
                .map_err(denied_to_js)?;
        Ok(Self {
            request: context.request,
            lease: state.lease,
            root_started_at_unix_millis: state.root_started_at_unix_millis,
            scope,
            host: None,
            worker_authority: Some((authority, original_context)),
        })
    }

    fn validate_worker(
        context: &WorkerContext,
        prepared_json: &str,
        study_identity: Digest,
    ) -> Result<(), &'static str> {
        if context.prepared_study_identity != Digest::sha256(prepared_json.as_bytes()).to_string()
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
    pub(super) fn active_lease(&self) -> ActiveFeaLease {
        ActiveFeaLease {
            run_id: self.request.run_id.clone(),
            lease_id: self.lease.lease_id.clone(),
            control: self.scope.control(),
            request: self.request.clone(),
            root_started_at_unix_millis: self.root_started_at_unix_millis,
        }
    }
    pub(crate) async fn check(&self) -> Result<(), JsValue> {
        self.refresh("check", &JsValue::UNDEFINED).await.map(|_| ())
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
        if let Some((authority, original)) = &self.worker_authority {
            // The worker has only computed a serialized numerical result.
            // Revalidate the current authoritative UI hold before any result
            // bytes leave it; it never publishes artifacts or settles a lease.
            let state: LeaseState = serde_wasm_bindgen::from_value(
                call_host(
                    authority,
                    "validateWorker",
                    original,
                    &JsValue::from_str("finish"),
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
        let checked = self.scope.settle(outcome).map_err(denied_to_js);
        if let Some(host) = self.host {
            let request = crate::wire::json_contract_to_js(&self.request)
                .map_err(|error| js_error(&error.to_string()))?;
            let outcome = crate::wire::json_contract_to_js(&outcome)
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
    ) -> Result<JsValue, JsValue> {
        let digest = Digest::sha256(prepared_json.as_bytes()).to_string();
        let request = crate::wire::json_contract_to_js(&self.request)
            .map_err(|error| js_error(&error.to_string()))?;
        let state: LeaseState = serde_wasm_bindgen::from_value(
            call_host(host, "delegate", &request, &JsValue::from_str(&digest)).await?,
        )
        .map_err(|_| denied_to_js(FeaRunDenied::LicenseInvalid))?;
        if state.lease.lease_id != self.lease_id
            || state.root_started_at_unix_millis != self.root_started_at_unix_millis
        {
            return Err(denied_to_js(FeaRunDenied::LicenseInvalid));
        }
        self.control
            .renew(state.lease.budget()?)
            .map_err(denied_to_js)?;
        crate::wire::json_contract_to_js(&WorkerContext {
            request: self.request.clone(),
            prepared_study_identity: digest,
            lease: state.lease,
            root_started_at_unix_millis: self.root_started_at_unix_millis,
            delegation_id: state
                .delegation_id
                .ok_or_else(|| denied_to_js(FeaRunDenied::LicenseInvalid))?,
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
                held_core_millis: 120_000,
                expires_at: "2099-01-01T00:00:00Z".into(),
                heartbeat_seconds: 30,
            },
            root_started_at_unix_millis: 1,
            delegation_id: "run-original:worker:1".into(),
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
    }
}
