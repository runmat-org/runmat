//! The display-mesh suite on OCCT, through `runmat_cad_call`: the entry the Desktop and browser
//! worker call in the emscripten build of this host (here its native build, the same code).

fn entry(path: &str, text: &str) -> String {
    let request = serde_json::json!({ "fn": "compile", "args": [path, text, "{}", "{}"] });
    let out = runmat_geometry_wasm_host::cad::call(&request.to_string());
    let v: serde_json::Value = serde_json::from_str(&out).expect("json");
    assert_eq!(v["kernel"], "occt", "the host compiles on OCCT: {out:.400}");
    out
}

#[path = "../../../runmat-cad/wasm/tests/display_mesh/suite.rs"]
mod suite;
