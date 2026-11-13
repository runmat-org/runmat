//! Emscripten ABI for the same validated OCCT import used by desktop, the FEA meshing
//! execution, and the CAD design runtime on OCCT (`runmat_cad_call`, see [`cad`]).
//! Runs in a worker, never on the UI thread.
use base64::{engine::general_purpose::STANDARD, Engine};
use runmat_geometry_io::import::GeometryImportError;
use runmat_geometry_io::{
    detect_geometry_format, import_exact_cad, ExactCadImportOptions, GeometryImportContext,
};
use serde::Deserialize;
use std::ffi::{c_char, CStr, CString};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    path: String,
    bytes_base64: String,
    mode: Mode,
    options: String,
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Exact,
    Preview,
}
fn import(request: &str) -> Result<String, GeometryImportError> {
    let request: Request = serde_json::from_str(request)
        .map_err(|e| GeometryImportError::InvalidOptions(e.to_string()))?;
    let bytes = STANDARD
        .decode(request.bytes_base64)
        .map_err(|e| GeometryImportError::ParseFailed(e.to_string()))?;
    let context = GeometryImportContext::default();
    match request.mode {
        Mode::Exact => {
            let options: ExactCadImportOptions = serde_json::from_str(&request.options)
                .map_err(|e| GeometryImportError::InvalidOptions(e.to_string()))?;
            let format = detect_geometry_format(&request.path, &bytes);
            let imported = import_exact_cad(&request.path, &bytes, format, &options, &context)?;
            imported
                .build_closure()
                .map_err(|e| GeometryImportError::InvalidGeometry(e.to_string()))?;
            serde_json::to_string(&imported)
                .map_err(|e| GeometryImportError::ParseFailed(e.to_string()))
        }
        Mode::Preview => {
            let options: runmat_geometry_io::GeometryImportOptions =
                serde_json::from_str(&request.options)
                    .map_err(|e| GeometryImportError::InvalidOptions(e.to_string()))?;
            let imported = runmat_geometry_io::import::import_geometry_with_context(
                &request.path,
                &bytes,
                options,
                &context,
            )?;
            serde_json::to_string(&imported)
                .map_err(|e| GeometryImportError::ParseFailed(e.to_string()))
        }
    }
}
/// Caller owns a NUL-terminated UTF-8 request until this call returns.
///
/// # Safety
/// A non-null request must point to readable bytes through its terminating NUL,
/// within one allocation, and remain valid and unmodified for this call. UTF-8
/// and request contents are validated. Free the returned response exactly once
/// using `runmat_occt_exact_free` from this same module instance.
#[no_mangle]
pub unsafe extern "C" fn runmat_occt_exact_import(request: *const c_char) -> *mut c_char {
    let result = if request.is_null() {
        Err(GeometryImportError::InvalidOptions(
            "Missing exact CAD request".into(),
        ))
    } else {
        CStr::from_ptr(request)
            .to_str()
            .map_err(|e| GeometryImportError::InvalidOptions(e.to_string()))
            .and_then(import)
    };
    let value = match result {
        Ok(imported) => serde_json::json!({"imported": imported}),
        Err(error) => serde_json::json!({"error": {"message": error.to_string(), "cause": error}}),
    };
    CString::new(value.to_string())
        .expect("JSON contains no NUL bytes")
        .into_raw()
}
/// Release a response returned by the exact CAD import or execution ABI.
///
/// # Safety
/// A non-null value must be a still-live pointer returned by
/// `runmat_occt_exact_import` or `runmat_occt_exact_execute` from this module
/// instance. Its original terminating NUL and allocation must be intact; it
/// must not have been freed, and no other access may overlap this release.
#[no_mangle]
pub unsafe extern "C" fn runmat_occt_exact_free(value: *mut c_char) {
    if !value.is_null() {
        drop(CString::from_raw(value));
    }
}

pub mod cad;
mod execution;
/// Caller supplies bounded NUL-terminated UTF-8 JSON. The response is freed with
/// runmat_occt_exact_free, just like the import ABI.
///
/// # Safety
/// A non-null request must point to readable bytes through its terminating NUL,
/// within one allocation, and remain valid and unmodified for this call. UTF-8
/// and JSON are validated. Free the returned response exactly once using
/// `runmat_occt_exact_free` from this same module instance.
#[no_mangle]
pub unsafe extern "C" fn runmat_occt_exact_execute(request: *const c_char) -> *mut c_char {
    let result = if request.is_null() {
        Err("Missing exact execution request".into())
    } else {
        CStr::from_ptr(request)
            .to_str()
            .map_err(|e| runmat_analysis_execution::SerialGeometryMeshingError::from(e.to_string()))
            .and_then(execution::execute)
    };
    let response = match result {
        Ok(value) => serde_json::json!({"result":value}),
        Err(error) => {
            serde_json::json!({"error":{"message":error.to_string(),"meshingFailure":error.meshing_failure()}})
        }
    };
    CString::new(response.to_string())
        .expect("JSON contains no NUL bytes")
        .into_raw()
}
