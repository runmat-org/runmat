//! The Export commands (`manage.export` on a part, `assembly.export` on an assembly) as one
//! host call: `export(path, text, files, request)` with `request` `{ command, values }` →
//! `{ ok: true, name, mime, text }` (the file's suggested name beside the source, its media type
//! and its contents) or `{ ok: false, message }`.
//!
//! STEP AP242 carries the exact B-rep, canonical face names and the part's PMI ([`super`]); IGES
//! carries a part's exact B-rep and names. STL, 3MF, glTF and a face's DXF are written by the
//! design tools' host export (`runmat_design_tools::exchange::host`), which falls back here. The
//! active configuration travels in `values.configuration`.

use crate::runtime::Files;
use serde_json::{json, Value};

fn stem(path: &str) -> &str {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.split('.').next().unwrap_or(name)
}

fn failed(message: impl Into<String>) -> String {
    json!({ "ok": false, "message": message.into() }).to_string()
}

/// The exported file for `request`, as JSON text.
pub fn export_json(path: &str, text: &str, files: &Files, request: &str) -> String {
    let request: Value = match serde_json::from_str(request) {
        Ok(v) => v,
        Err(e) => return failed(format!("malformed export request: {e}")),
    };
    let values = &request["values"];
    let format = values["format"].as_str().unwrap_or("step_ap242");
    let configuration = values["configuration"].as_str().filter(|c| !c.is_empty() && *c != "default");
    if format == "iges" {
        if !path.ends_with(".part") {
            return failed("IGES export writes a part");
        }
        return match super::part_iges_in(path, text, configuration) {
            Ok(Some(bytes)) => match String::from_utf8(bytes) {
                Ok(text) => json!({ "ok": true, "name": format!("{}.igs", stem(path)), "mime": "model/iges", "text": text }).to_string(),
                Err(_) => failed("the kernel's IGES file is not text"),
            },
            Ok(None) => failed("this build's geometry kernel doesn't write IGES"),
            Err(message) => failed(message),
        };
    }
    if !matches!(format, "step_ap242" | "step") {
        return failed(format!("this export writes STEP AP242 and IGES, not `{format}`"));
    }
    let step = if path.ends_with(".part") {
        super::part_step_in(path, text, configuration)
    } else if path.ends_with(".assembly") {
        super::assembly_step_in(path, text, files, configuration)
    } else {
        return failed("export writes STEP from a part or an assembly");
    };
    match step {
        Ok(Some(bytes)) => match String::from_utf8(bytes) {
            Ok(text) => json!({ "ok": true, "name": format!("{}.step", stem(path)), "mime": "model/step", "text": text }).to_string(),
            Err(_) => failed("the kernel's STEP file is not text"),
        },
        Ok(None) => failed("this build's geometry kernel doesn't write STEP"),
        Err(message) => failed(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn a_format_other_than_step_says_so_and_a_drawing_is_refused() {
        let out: Value = serde_json::from_str(&export_json("/p/a.part", "", &Files::new(), r#"{"values":{"format":"stl"}}"#)).unwrap();
        assert_eq!(out["ok"], false);
        assert!(out["message"].as_str().unwrap().contains("STEP AP242"), "{out}");
        let out: Value = serde_json::from_str(&export_json("/p/a.drawing", "", &Files::new(), r#"{"values":{}}"#)).unwrap();
        assert_eq!(out["ok"], false, "{out}");
    }
}
