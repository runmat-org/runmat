//! Async imported-asset meshing; the owned sidecar returns data, never persists it.
use super::fea_lease::denial_from_js;
use super::fea_run_context::ActiveFeaLease;
use crate::wire::errors::js_value_to_string;
use runmat_analysis_execution::{
    ImportedMeshContactSelectors, ImportedSourceMeshingRequest, ImportedSourceMeshingResult,
};
use runmat_execution::Digest;
use runmat_execution_artifact::{LogicalObject, ObjectNamespace};
use runmat_runtime::meshing::{
    MeshingExecutionError, MeshingExecutionOutput, MeshingExecutionRequest,
};
use std::{cell::RefCell, rc::Rc};
use wasm_bindgen::JsValue;

pub(super) async fn execute(
    request: MeshingExecutionRequest,
    host: Rc<RefCell<Option<JsValue>>>,
    active: Rc<RefCell<Option<ActiveFeaLease>>>,
) -> Result<MeshingExecutionOutput, MeshingExecutionError> {
    let host = host
        .borrow()
        .clone()
        .ok_or(runmat_runtime::analysis::fea_run_admission::FeaRunDenied::SignInRequired)?;
    let active = active
        .borrow()
        .clone()
        .ok_or(runmat_runtime::analysis::fea_run_admission::FeaRunDenied::SignInRequired)?;
    active.control.check()?;
    let bytes = runmat_filesystem::read(&request.source_path)
        .map_err(|error| MeshingExecutionError::Provider(error.to_string().into()))?;
    let expected = Digest::from_bytes(request.expected_source_sha256);
    if Digest::sha256(&bytes) != expected {
        return Err(MeshingExecutionError::Provider(
            "Geometry source changed after the original asset was loaded.".into(),
        ));
    }
    let revision = runmat_runtime::context::legacy::active()
        .and_then(|runtime| runtime.program_revision().cloned())
        .ok_or_else(|| {
            MeshingExecutionError::Provider(
                "Owned meshing requires the original program revision.".into(),
            )
        })?;
    let source_name = request
        .source_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            MeshingExecutionError::Provider("Geometry source name is not valid UTF-8.".into())
        })?
        .to_owned();
    let source = LogicalObject::new(
        ObjectNamespace::InputValue,
        "input/exact-cad-source",
        "application/octet-stream",
        bytes,
    )
    .map_err(|error| MeshingExecutionError::Provider(error.to_string().into()))?;
    let imported = ImportedSourceMeshingRequest {
        source_name,
        source,
        expected_source_digest: expected,
        import_options: runmat_geometry_io::ExactCadImportOptions {
            source_units: request.source_units,
            ..Default::default()
        },
        settings: request.settings,
        contacts: request
            .contact_definitions
            .into_iter()
            .map(|contact| ImportedMeshContactSelectors {
                side_a_face_selectors: contact.side_a_face_selectors,
                side_b_face_selectors: contact.side_b_face_selectors,
            })
            .collect(),
        program_revision: revision,
        artifact_access: runmat_analysis_execution::checked_fea_artifact_access(&active.run_id),
    };
    #[derive(serde::Serialize)]
    #[serde(tag = "operation", rename_all = "snake_case")]
    enum Operation {
        MeshImportedSource {
            request: ImportedSourceMeshingRequest,
        },
    }
    let json = serde_json::to_string(&Operation::MeshImportedSource { request: imported })
        .map_err(|error| MeshingExecutionError::Provider(error.to_string().into()))?;
    if json.len() > 64 << 20 {
        return Err(MeshingExecutionError::Provider(
            "Geometry preparation exceeds this host's input allowance.".into(),
        ));
    }
    let delegation = active
        .worker_context(
            &host,
            &json,
            super::fea_run_context::WorkerWorkClass::OcctPreparation,
        )
        .await
        .map_err(|error| MeshingExecutionError::Admission(denial_from_js(error)))?;
    let result = super::fea_execution::exact_execute(&json, delegation)
        .await
        .map_err(host_error)?;
    active.retire_finished(&host).await.map_err(host_error)?;
    let result: ImportedSourceMeshingResult =
        serde_json::from_str(&result.as_string().ok_or_else(|| {
            MeshingExecutionError::Provider("Invalid owned meshing response.".into())
        })?)
        .map_err(|error| MeshingExecutionError::Provider(error.to_string().into()))?;
    result
        .evidence
        .validate(&result.artifact)
        .map_err(|error| MeshingExecutionError::Provider(error.to_string().into()))?;
    active.control.check()?;
    Ok(MeshingExecutionOutput {
        artifact: result.artifact,
        evidence: result.evidence,
    })
}

fn host_error(error: JsValue) -> MeshingExecutionError {
    let denied =
        js_sys::Reflect::get(&error, &JsValue::from_str("denied")).unwrap_or(JsValue::UNDEFINED);
    if !denied.is_undefined() && !denied.is_null() {
        MeshingExecutionError::Admission(denial_from_js(error))
    } else {
        MeshingExecutionError::Provider(js_value_to_string(error).into())
    }
}
