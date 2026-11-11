//! The CAD host's Export call (`export(path, text, files, request)`, `request` `{ command, values }`)
//! for a part or an assembly. STEP AP242 goes to `runmat_cad_wasm::exchange::export`; STL, 3MF,
//! glTF and USD are the same files `design.export` writes ([`super::model_export`]), from the host kernel's
//! exact tessellation; DXF of a face is [`super::face_dxf`]. The answer is `{ ok: true, name, mime, text }` for a text file,
//! `{ ok: true, name, mime, base64 }` for a binary one, or `{ ok: false, message }`.

use super::{model_export, stem, ExportFormat};
use crate::Files;
use runmat_cad::regen::kernel::TessellationOptions;
use serde_json::{json, Value};

/// The exported file for `request`, as JSON text.
pub fn export_json(path: &str, text: &str, files: &Files, request: &str) -> String {
    let parsed: Value = serde_json::from_str(request).unwrap_or(Value::Null);
    let values = &parsed["values"];
    let format = match values["format"].as_str().unwrap_or("step_ap242") {
        "stl" => ExportFormat::Stl,
        "3mf" => ExportFormat::ThreeMf,
        "gltf" => ExportFormat::Gltf,
        "usd" => ExportFormat::Usd,
        "dxf_face" => return face_export(path, text, values),
        _ => return runmat_cad_wasm::exchange::export::export_json(path, text, files, request),
    };
    let configuration = values["configuration"].as_str().filter(|c| !c.is_empty() && *c != "default");
    let mut files = files.clone();
    files.insert(path.to_owned(), text.to_owned());
    let options = match resolution(values["resolution"].as_str().unwrap_or("standard")) {
        Ok(options) => options,
        Err(message) => return json!({ "ok": false, "message": message }).to_string(),
    };
    let mut notes = Vec::new();
    match model_export(path, &files, format, configuration, &stem(path), &mut notes, &options) {
        Ok(out) => match out.into_iter().next() {
            Some(file) if file.encoding == "base64" => {
                json!({ "ok": true, "name": file.name, "mime": file.mime_type, "base64": file.content }).to_string()
            }
            Some(file) => json!({ "ok": true, "name": file.name, "mime": file.mime_type, "text": file.content }).to_string(),
            None => json!({ "ok": false, "message": "the export wrote no file" }).to_string(),
        },
        Err(e) => json!({ "ok": false, "message": e.message }).to_string(),
    }
}

/// The tessellation a mesh export's resolution asks for. Standard is the kernel's display
/// tessellation (0.1 mm chord, 8° between neighbouring normals); fine has a quarter of its chord
/// and half its angle, coarse four times its chord and twice its angle.
pub fn resolution(name: &str) -> Result<TessellationOptions, String> {
    let standard = TessellationOptions::default();
    let scale = match name {
        "fine" => 0.25,
        "standard" => 1.0,
        "coarse" => 4.0,
        other => return Err(format!("the mesh resolution is fine, standard or coarse, not `{other}`")),
    };
    Ok(TessellationOptions {
        chord_tolerance: standard.chord_tolerance * scale,
        angle_tolerance: standard.angle_tolerance * scale.sqrt(),
        size_fraction: standard.size_fraction * scale,
        chord_max: standard.chord_max,
    })
}

fn face_export(path: &str, text: &str, values: &Value) -> String {
    let pick = match &values["face"] {
        Value::Array(a) => a.first().and_then(Value::as_str),
        v => v.as_str(),
    };
    let Some(pick) = pick else {
        return json!({ "ok": false, "message": "DXF of a face needs a face" }).to_string();
    };
    let configuration = values["configuration"].as_str().filter(|c| !c.is_empty() && *c != "default");
    match super::face_dxf::face_dxf(path, text, pick, configuration) {
        Ok(dxf) => json!({ "ok": true, "name": format!("{}-face.dxf", stem(path)), "mime": "image/vnd.dxf", "text": dxf }).to_string(),
        Err(e) => json!({ "ok": false, "message": e.message }).to_string(),
    }
}
