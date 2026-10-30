//! Local run identity and owned-worker payload binding. No billing callbacks.
use crate::wire::errors::js_error;
use runmat_execution::Digest;
use runmat_runtime::analysis::fea_run_admission::{
    FeaRunKind, FeaRunRequest, FeaRunScope, FeaRunSite,
};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

pub(crate) struct LocalFeaRunContext {
    request: FeaRunRequest,
}

/// Content binding for an owned worker, not a lease or authorization token.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct WorkerContext {
    request: FeaRunRequest,
    prepared_study_identity: String,
}

impl LocalFeaRunContext {
    pub(crate) fn new(kind: FeaRunKind, study_identity: Digest) -> Result<Self, JsValue> {
        Ok(Self {
            request: FeaRunRequest::local(kind, 1000, study_identity)
                .map_err(|_| js_error("Unable to create a secure local FEA run identity."))?,
        })
    }

    pub(crate) fn worker_context(&self, prepared_json: &str) -> Result<JsValue, JsValue> {
        crate::wire::json_contract_to_js(&WorkerContext {
            request: self.request.clone(),
            prepared_study_identity: Digest::sha256(prepared_json.as_bytes()).to_string(),
        })
        .map_err(|error| js_error(&error.to_string()))
    }

    pub(crate) fn for_worker(
        context: JsValue,
        prepared_json: &str,
        study_identity: Digest,
    ) -> Result<Self, JsValue> {
        if context.is_undefined() || context.is_null() {
            return Self::new(FeaRunKind::Solve, study_identity);
        }
        let context: WorkerContext = serde_wasm_bindgen::from_value(context)
            .map_err(|_| js_error("Invalid local FEA worker context."))?;
        Self::validate_worker(context, prepared_json, study_identity).map_err(js_error)
    }

    fn validate_worker(
        context: WorkerContext,
        prepared_json: &str,
        study_identity: Digest,
    ) -> Result<Self, &'static str> {
        if context.prepared_study_identity != Digest::sha256(prepared_json.as_bytes()).to_string()
            || context.request.site != FeaRunSite::Local
            || context.request.allocated_cpu_millicores != 1000
            || context.request.kind == FeaRunKind::Meshing
            || context.request.study_identity != study_identity.to_string()
        {
            return Err("Local FEA worker context does not match the exact prepared study and serial allocation.");
        }
        Ok(Self {
            request: context.request,
        })
    }

    pub(crate) fn scope(&self) -> Result<FeaRunScope, JsValue> {
        FeaRunScope::admit(self.request.clone())
            .map_err(|_| js_error("Invalid local FEA run context."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn worker_context_binds_payload_study_site_and_allocation_without_billing() {
        let identity = Digest::sha256(b"authored-study");
        let request = FeaRunRequest::local(FeaRunKind::MeshAndSolve, 1000, identity).unwrap();
        let context = || WorkerContext {
            request: request.clone(),
            prepared_study_identity: Digest::sha256(b"prepared").to_string(),
        };
        assert!(LocalFeaRunContext::validate_worker(context(), "prepared", identity).is_ok());
        assert!(LocalFeaRunContext::validate_worker(context(), "changed", identity).is_err());
        assert!(LocalFeaRunContext::validate_worker(
            context(),
            "prepared",
            Digest::sha256(b"changed")
        )
        .is_err());
        let mut dispatched = context();
        dispatched.request.site = FeaRunSite::Dispatched;
        assert!(LocalFeaRunContext::validate_worker(dispatched, "prepared", identity).is_err());
        let mut oversized = context();
        oversized.request.allocated_cpu_millicores = 2000;
        assert!(LocalFeaRunContext::validate_worker(oversized, "prepared", identity).is_err());
    }
}
