//! The CAD design runtime on OCCT: `.part`, `.assembly` and `.drawing` compile and the canvas's
//! catalog edits (`runmat-cad-wasm`), with every part regenerated, measured and tessellated on
//! [`OcctKernel`], the kernel the K1 corpus checks.
//!
//! The browser and Desktop worker call [`runmat_cad_call`] with one JSON request,
//! `{ "fn": "dependencies" | "compile" | "edit" | "analyze" | "export", "args": [..strings] }`, whose arguments are
//! the ones `runmat-cad-wasm`'s wasm-bindgen entry points take, and get back the same JSON text
//! those entry points return. `designTool` runs one call of the design tool layer
//! (`runmat_design_tools::call_json`) here, for the agent's tools on parts, assemblies and drawings. Native callers use [`call`] (the same function, for host-parity
//! tests).
//!
//! The per-part cache travels too (`runmat_cad_wasm::runtime::part_cache`): `probe` is `compile`
//! with each part the cache doesn't hold left out (its fifth argument, a JSON list of digests,
//! names those it may regenerate), and `cacheMisses` lists what the last probe left out.
//! [`call_framed`]'s `cacheFresh` returns the entries regenerated since the last take, and
//! [`runmat_cad_seed`] reads stored entries back in.

use runmat_cad_wasm::{analyze, edit, kernel, runtime};
use runmat_geometry_io::ImportSource;
use runmat_geometry_io::OcctKernel;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::ffi::{c_char, CStr, CString};
use std::sync::{Arc, Mutex};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    #[serde(rename = "fn")]
    function: String,
    args: Vec<String>,
}

/// The STEP files of the request being served, by the names an `import(source = …)` may give
/// them (the workspace path and the bare file name). The browser host has no project folder, so
/// the files travel in the request and the kernel reads their text from here.
static IMPORTS: Mutex<BTreeMap<String, ImportSource>> = Mutex::new(BTreeMap::new());

/// The digests the last `probe` left out.
static MISSES: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn set_imports(files: &runtime::Files) {
    let mut map = BTreeMap::new();
    for (path, text) in files {
        let lower = path.to_ascii_lowercase();
        if !(lower.ends_with(".step") || lower.ends_with(".stp")) {
            continue;
        }
        let source = ImportSource::StepText(Arc::from(text.as_str()));
        // Every tail of the path: `housing.step`, `supplier/housing.step`, … and the whole.
        map.insert(path.clone(), source.clone());
        for (i, _) in path.match_indices('/') {
            map.entry(path[i + 1..].to_string())
                .or_insert_with(|| source.clone());
        }
    }
    *IMPORTS.lock().unwrap_or_else(|e| e.into_inner()) = map;
}

/// Installs OCCT as the runtime's kernel (once per process; later calls do nothing). Each
/// kernel it makes reads imported STEP files from the current request's files.
pub fn install_occt() {
    kernel::install("occt", || {
        let mut k = OcctKernel::new();
        k.imports = IMPORTS.lock().unwrap_or_else(|e| e.into_inner()).clone();
        Box::new(k)
    });
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
                Ok(files) => {
                    set_imports(&files);
                    // `.m` values compile in two phases: `pending_values` lists the calls to run.
                    runtime::values_host::compile_json_with_values(path, text, &files, arg(a, 3))
                }
                Err(message) => runtime::error_json(path, &message),
            }
        }
        "probe" => {
            let (path, text) = (arg(a, 0), arg(a, 1));
            let allow: std::collections::HashSet<String> =
                serde_json::from_str(arg(a, 4)).unwrap_or_default();
            match runtime::parse_files(arg(a, 2)) {
                Ok(files) => {
                    set_imports(&files);
                    let (out, misses) = runtime::part_cache::probe(allow, || {
                        runtime::values_host::compile_json_with_values(
                            path,
                            text,
                            &files,
                            arg(a, 3),
                        )
                    });
                    *MISSES.lock().unwrap_or_else(|e| e.into_inner()) = misses;
                    out
                }
                Err(message) => runtime::error_json(path, &message),
            }
        }
        "cacheMisses" => serde_json::to_string(&*MISSES.lock().unwrap_or_else(|e| e.into_inner()))
            .unwrap_or_else(|_| "[]".into()),
        "edit" => {
            let (path, text, request) = (arg(a, 0), arg(a, 1), arg(a, 3));
            match runtime::parse_files(arg(a, 2)) {
                Ok(files) => {
                    set_imports(&files);
                    edit::edit_json(path, text, &files, request)
                }
                Err(message) => {
                    serde_json::json!({ "ok": false, "message": message, "diagnostics": [] })
                        .to_string()
                }
            }
        }
        "analyze" => {
            let (path, text, request) = (arg(a, 0), arg(a, 1), arg(a, 3));
            match runtime::parse_files(arg(a, 2)) {
                Ok(files) => {
                    set_imports(&files);
                    analyze::analyze_json(path, text, &files, request)
                }
                Err(message) => serde_json::json!({ "ok": false, "message": message }).to_string(),
            }
        }
        "export" => {
            // A part or assembly as STEP AP242 or IGES (`runmat_cad_wasm::exchange::export`), or as
            // STL, 3MF, glTF or a face's DXF (`runmat_design_tools::exchange::host`).
            let (path, text, request) = (arg(a, 0), arg(a, 1), arg(a, 3));
            match runtime::parse_files(arg(a, 2)) {
                Ok(files) => {
                    set_imports(&files);
                    runmat_design_tools::exchange::host::export_json(path, text, &files, request)
                }
                Err(message) => serde_json::json!({ "ok": false, "message": message }).to_string(),
            }
        }
        "designTool" => {
            // One call of the design tool layer (`design.overview`, `check`, `inspect`,
            // `propose_edit`, `export`, …) on a part, assembly or drawing: the agent's design
            // tools in the browser, regenerating on this host's OCCT and its part cache, so they
            // answer as the editor and the server do. Arguments: the tool's name, its input and
            // the files (JSON); the answer is `{ data }` or `{ error: { code, message } }`.
            let (name, input, files) = (arg(a, 0), arg(a, 1), arg(a, 2));
            // An overview reads the linked parts, assemblies and drawings without regenerating
            // each one: in the browser this host has one bounded heap, shared with the editor's
            // compiles, and a product's links (the drone's 20-odd parts and sub-assemblies)
            // regenerated one after another exhausted it. Their status says they're checked when
            // they open in the editor; the file itself is still checked.
            runmat_design_tools::overview::check_linked_models(false);
            if let Ok(parsed) = runtime::parse_files(files) {
                set_imports(&parsed);
            }
            runmat_design_tools::call_json(name, input, files).to_string()
        }
        other => error(&format!("unknown CAD function {other:?}")),
    }
}

/// [`call`] with the display meshes written as bytes beside the JSON
/// ([`runtime::display_wire`]): the JSON text and the bytes its mesh references point into.
pub fn call_framed(request: &str) -> (String, Vec<u8>) {
    if let Ok(Request { function, .. }) = serde_json::from_str::<Request>(request) {
        if function == "cacheFresh" {
            return fresh_entries();
        }
    }
    runtime::display_wire::collect(|| call(request))
}

/// The entries regenerated since the last take: a JSON list of `{digest, start, end}` and the
/// concatenated entry frames those byte ranges cut out (`part_cache::entry_frame`).
fn fresh_entries() -> (String, Vec<u8>) {
    let mut index = Vec::new();
    let mut bytes = Vec::new();
    for (digest, frame) in runtime::part_cache::take_fresh() {
        let start = bytes.len();
        bytes.extend_from_slice(&frame);
        index.push(serde_json::json!({ "digest": digest, "start": start, "end": bytes.len() }));
    }
    (serde_json::Value::Array(index).to_string(), bytes)
}

/// Reads stored cache entries (concatenated entry frames, `len` bytes at `frames`) into the
/// per-part cache; returns how many it took.
///
/// # Safety
/// `frames` is null or points to `len` readable bytes that stay valid for this call.
#[no_mangle]
pub unsafe extern "C" fn runmat_cad_seed(frames: *const u8, len: usize) -> u32 {
    if frames.is_null() {
        return 0;
    }
    runtime::part_cache::seed(std::slice::from_raw_parts(frames, len)) as u32
}

/// One frame holding a framed answer: the JSON's length and the bytes' length (`u32`,
/// little-endian), the JSON, zero padding to a multiple of 4, then the bytes.
///
/// The frame is built in the JSON text's own buffer: it grows by the header, padding and bytes
/// and the text moves up 8 bytes, so an answer of tens of megabytes isn't held twice.
pub fn frame(json: String, bytes: &[u8]) -> Vec<u8> {
    let n = json.len();
    let padded = n.div_ceil(4) * 4;
    let mut out = json.into_bytes();
    out.reserve_exact((8 + padded + bytes.len()).saturating_sub(out.len()));
    out.resize(8 + padded, 0);
    out.copy_within(0..n, 8);
    out[n + 8..8 + padded].fill(0);
    out[..4].copy_from_slice(&(n as u32).to_le_bytes());
    out[4..8].copy_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
    out
}

/// Like [`runmat_cad_call`], but the display meshes come back as bytes in one [`frame`]; free
/// it with [`runmat_cad_free_framed`].
///
/// # Safety
/// As [`runmat_cad_call`]. Free the returned frame exactly once using `runmat_cad_free_framed`
/// from this same module instance.
#[no_mangle]
pub unsafe extern "C" fn runmat_cad_call_framed(request: *const c_char) -> *mut u8 {
    let (json, bytes) = if request.is_null() {
        (error("missing CAD request"), Vec::new())
    } else {
        match CStr::from_ptr(request).to_str() {
            Ok(r) => call_framed(r),
            Err(e) => (error(&e.to_string()), Vec::new()),
        }
    };
    Box::into_raw(frame(json, &bytes).into_boxed_slice()) as *mut u8
}

/// Frees a frame [`runmat_cad_call_framed`] returned.
///
/// # Safety
/// `frame` is null or a pointer `runmat_cad_call_framed` returned from this module instance
/// that has not been freed.
#[no_mangle]
pub unsafe extern "C" fn runmat_cad_free_framed(frame: *mut u8) {
    if frame.is_null() {
        return;
    }
    let header = std::slice::from_raw_parts(frame, 8);
    let json = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let bytes = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    let len = 8 + json.div_ceil(4) * 4 + bytes;
    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
        frame, len,
    )));
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
