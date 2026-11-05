//! The CAD design runtime on OCCT: `.part`, `.assembly` and `.drawing` compile and the canvas's
//! catalog edits (`runmat-cad-wasm`), with every part regenerated, measured and tessellated on
//! [`OcctKernel`], the kernel the K1 corpus checks.
//!
//! The browser and Desktop worker call [`runmat_cad_call`] with one JSON request,
//! `{ "fn": "dependencies" | "compile" | "edit", "args": [..strings] }`, whose arguments are
//! the ones `runmat-cad-wasm`'s wasm-bindgen entry points take, and get back the same JSON text
//! those entry points return. Native callers use [`call`] (the same function, for host-parity
//! tests).

use runmat_cad_wasm::{edit, kernel, runtime};
use runmat_geometry_io::OcctKernel;
use serde::Deserialize;
use std::ffi::{c_char, CStr, CString};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    #[serde(rename = "fn")]
    function: String,
    args: Vec<String>,
}

/// Installs OCCT as the runtime's kernel (once per process; later calls do nothing).
pub fn install_occt() {
    kernel::install("occt", || Box::new(OcctKernel::new()));
}

fn arg(args: &[String], i: usize) -> &str {
    args.get(i).map_or("", String::as_str)
}

fn error(message: &str) -> String {
    serde_json::json!({ "error": { "message": message } }).to_string()
}

/// Runs one request on OCCT and returns the runtime's JSON answer.
pub fn call(request: &str) -> String {
    install_occt();
    let request: Request = match serde_json::from_str(request) {
        Ok(r) => r,
        Err(e) => return error(&format!("malformed CAD request: {e}")),
    };
    let a = &request.args;
    match request.function.as_str() {
        "dependencies" => {
            let deps = runtime::dependencies(arg(a, 0), arg(a, 1));
            serde_json::to_string(&deps).unwrap_or_else(|_| "[]".into())
        }
        "compile" => {
            let (path, text) = (arg(a, 0), arg(a, 1));
            match runtime::parse_files(arg(a, 2)) {
                Ok(files) => runtime::compile_json(path, text, &files),
                Err(message) => runtime::error_json(path, &message),
            }
        }
        "edit" => {
            let (path, text, request) = (arg(a, 0), arg(a, 1), arg(a, 3));
            match runtime::parse_files(arg(a, 2)) {
                Ok(files) => edit::edit_json(path, text, &files, request),
                Err(message) => {
                    serde_json::json!({ "ok": false, "message": message, "diagnostics": [] })
                        .to_string()
                }
            }
        }
        other => error(&format!("unknown CAD function {other:?}")),
    }
}

/// Caller supplies a NUL-terminated UTF-8 JSON request; free the response with
/// `runmat_occt_exact_free`, like the import and execution ABIs.
///
/// # Safety
/// A non-null request must point to readable bytes through its terminating NUL, within one
/// allocation, and remain valid and unmodified for this call. Free the returned response
/// exactly once using `runmat_occt_exact_free` from this same module instance.
#[no_mangle]
pub unsafe extern "C" fn runmat_cad_call(request: *const c_char) -> *mut c_char {
    let response = if request.is_null() {
        error("missing CAD request")
    } else {
        match CStr::from_ptr(request).to_str() {
            Ok(r) => call(r),
            Err(e) => error(&e.to_string()),
        }
    };
    CString::new(response)
        .unwrap_or_else(|_| CString::new(error("response contains NUL")).expect("no NUL"))
        .into_raw()
}
