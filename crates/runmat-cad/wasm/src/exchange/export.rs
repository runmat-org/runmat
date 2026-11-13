//! The Export commands (`manage.export` on a part, `assembly.export` on an assembly, `board.export`
//! on a circuit or its layout) as one
//! host call: `export(path, text, files, request)` with `request` `{ command, values }` →
//! `{ ok: true, name, mime, text }` (the file's suggested name beside the source, its media type
//! and its contents) or `{ ok: false, message }`.
//!
//! STEP AP242 carries the exact B-rep, canonical face names and the part's PMI ([`super`]); IGES
//! carries a part's exact B-rep and names. STL, 3MF, glTF and a face's DXF are written by the
//! design tools' host export (`runmat_design_tools::exchange::host`), which falls back here. The
//! active configuration travels in `values.configuration`. A board's STEP ([`super::board`]) takes
//! `values.copper` and `values.mask`, counts its components written from STEP models and as body
//! boxes in `components`, and lists what it couldn't write in `notes`.

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
    let configuration = values["configuration"]
        .as_str()
        .filter(|c| !c.is_empty() && *c != "default");
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
        return failed(format!(
            "this export writes STEP AP242 and IGES, not `{format}`"
        ));
    }
    if path.ends_with(".circuit") || path.ends_with(".circuit.layout") {
        let flag = |id: &str, default: bool| values[id].as_bool().unwrap_or(default);
        let defaults = super::board::BoardStepOptions::default();
        let options = super::board::BoardStepOptions {
            copper: flag("copper", defaults.copper),
            mask: flag("mask", defaults.mask),
        };
        let mut files = files.clone();
        files.insert(path.to_owned(), text.to_owned());
        let name = stem(path);
        return match super::board::board_step(path, &files, options) {
            Ok(Some(out)) => match String::from_utf8(out.step) {
                Ok(text) => json!({ "ok": true, "name": format!("{name}.step"), "mime": "model/step", "text": text, "notes": out.notes, "components": { "models": out.models, "boxes": out.boxes } }).to_string(),
                Err(_) => failed("the kernel's STEP file is not text"),
            },
            Ok(None) => failed("this build's geometry kernel doesn't write STEP"),
            Err(message) => failed(message),
        };
    }
    let step = if path.ends_with(".part") {
        super::part_step_in(path, text, configuration)
    } else if path.ends_with(".assembly") {
        super::assembly_step_in(path, text, files, configuration)
    } else {
        return failed("export writes STEP from a part, an assembly or a board");
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
        let out: Value = serde_json::from_str(&export_json(
            "/p/a.part",
            "",
            &Files::new(),
            r#"{"values":{"format":"stl"}}"#,
        ))
        .unwrap();
        assert_eq!(out["ok"], false);
        assert!(
            out["message"].as_str().unwrap().contains("STEP AP242"),
            "{out}"
        );
        let out: Value = serde_json::from_str(&export_json(
            "/p/a.drawing",
            "",
            &Files::new(),
            r#"{"values":{}}"#,
        ))
        .unwrap();
        assert_eq!(out["ok"], false, "{out}");
    }
}
