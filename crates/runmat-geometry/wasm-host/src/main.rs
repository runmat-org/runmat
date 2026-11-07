// Emscripten emits an ES module factory and its WASM binary. Keep the C ABI
// reachable under Rust LTO; EXPORTED_FUNCTIONS names the browser entry points.
#[cfg(target_os = "emscripten")]
fn main() {
    std::hint::black_box(runmat_geometry_wasm_host::runmat_occt_exact_import as *const ());
    std::hint::black_box(runmat_geometry_wasm_host::runmat_occt_exact_free as *const ());
    std::hint::black_box(runmat_geometry_wasm_host::runmat_occt_exact_execute as *const ());
    std::hint::black_box(runmat_geometry_wasm_host::cad::runmat_cad_call as *const ());
    std::hint::black_box(runmat_geometry_wasm_host::cad::runmat_cad_call_framed as *const ());
    std::hint::black_box(runmat_geometry_wasm_host::cad::runmat_cad_free_framed as *const ());
}

// The identical JSON ABI can be exercised natively for host-parity checks.
#[cfg(not(target_os = "emscripten"))]
fn main() {
    use std::{
        ffi::{CStr, CString},
        io::Read,
    };
    let mut request = String::new();
    std::io::stdin()
        .read_to_string(&mut request)
        .expect("read import request");
    let request = CString::new(request).expect("JSON request must not contain NUL");
    unsafe {
        let response = if std::env::args().any(|arg| arg == "--cad") {
            runmat_geometry_wasm_host::cad::runmat_cad_call(request.as_ptr())
        } else if std::env::args().any(|arg| arg == "--execute") {
            runmat_geometry_wasm_host::runmat_occt_exact_execute(request.as_ptr())
        } else {
            runmat_geometry_wasm_host::runmat_occt_exact_import(request.as_ptr())
        };
        println!(
            "{}",
            CStr::from_ptr(response)
                .to_str()
                .expect("UTF-8 import response")
        );
        runmat_geometry_wasm_host::runmat_occt_exact_free(response);
    }
}
