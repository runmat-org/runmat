//! Wrapped notes leave the sheet as the lines the layout set: the DXF writes one TEXT entity per
//! line and the PDF one text run per line, at the layout's positions. `design.check` reports a
//! note the layout had to move, so a drawing whose notes overprint never checks clean.

use base64::Engine;
use runmat_design_tools::exchange::dxf;
use runmat_design_tools::formats::sheet::Shape;
use runmat_design_tools::{call, Files};
use serde_json::{json, Value};

const PART: &str = include_str!("../../runmat-cad/tests/fixtures/notes/l_bracket.part");
const AGENT: &str = include_str!("../../runmat-cad/tests/fixtures/notes/l_bracket.drawing");
const LISTED: &str = include_str!("../../runmat-cad/tests/fixtures/notes/l_bracket_notes.drawing");

/// The L bracket and `drawing` as `/p/<name>.drawing`.
fn files(name: &str, drawing: &str) -> Files {
    let path = format!("/p/{name}.drawing");
    [("/p/l_bracket.part", PART), (path.as_str(), drawing)]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect()
}

/// The lines the layout sets for `id`: (text, x, baseline y).
fn layout_lines(drawing: &str, id: &str) -> Vec<(String, f64, f64)> {
    let out: Value = serde_json::from_str(&runmat_cad_wasm::drawing::compile_json(
        "/p/l_bracket_notes.drawing",
        drawing,
        &files("l_bracket_notes", drawing),
    ))
    .unwrap();
    out["projection"]["data"]["layouts"]["1"]["primitives"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|p| p["id"] == id && p["shape"]["type"] == "text")
        .map(|p| {
            let s = &p["shape"];
            (
                s["text"].as_str().unwrap().to_owned(),
                s["x"].as_f64().unwrap(),
                s["y"].as_f64().unwrap(),
            )
        })
        .collect()
}

#[test]
fn wrapped_general_notes_export_as_the_same_lines_to_dxf_and_pdf() {
    let lines = layout_lines(LISTED, "table:notes");
    assert!(lines.len() > 6, "the notes wrap: {lines:?}");
    let f = files("l_bracket_notes", LISTED);
    let out = call(
        "design.export",
        json!({ "path": "/p/l_bracket_notes.drawing", "format": "dxf" }),
        &f,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let sheet = dxf::read(out["files"][0]["content"].as_str().unwrap()).unwrap();
    let texts: Vec<(String, [f64; 2])> = sheet
        .entities
        .iter()
        .filter_map(|e| match &e.shape {
            Shape::Text { text, at, .. } => Some((text.clone(), *at)),
            _ => None,
        })
        .collect();
    for (t, x, y) in &lines {
        assert!(
            texts
                .iter()
                .any(|(u, at)| u == t && (at[0] - x).abs() < 1e-3 && (at[1] - y).abs() < 1e-3),
            "no TEXT {t:?} at [{x}, {y}] in the DXF"
        );
    }
    let pdf = call(
        "design.export",
        json!({ "path": "/p/l_bracket_notes.drawing", "format": "pdf" }),
        &f,
    )
    .unwrap();
    let body = base64::engine::general_purpose::STANDARD
        .decode(pdf["files"][0]["content"].as_str().unwrap())
        .unwrap();
    // With `RUNMAT_NOTES_PDF_OUT` set to a file, the PDF is written there (for review).
    if let Ok(path) = std::env::var("RUNMAT_NOTES_PDF_OUT") {
        std::fs::write(path, &body).unwrap();
    }
    let body = String::from_utf8_lossy(&body);
    // Each line is its own run; lines with symbols the font lacks split around them.
    for (t, _, _) in lines.iter().filter(|(t, _, _)| t.is_ascii()) {
        assert!(
            body.contains(&format!("({t}) Tj")),
            "no line {t:?} in the PDF"
        );
    }
}

#[test]
fn the_check_reports_a_note_that_had_no_room_where_it_was_put() {
    let out = call(
        "design.check",
        json!({ "path": "/p/l_bracket.drawing" }),
        &files("l_bracket", AGENT),
    )
    .unwrap();
    // design.check reports drawing.check's findings: a note the layout moved is an Overlap
    // finding at info severity (the layout settled it), on the note's declaration.
    let moved = |d: &&Value| {
        d["code"] == "drawing.overlap"
            && d["message"]
                .as_str()
                .is_some_and(|m| m.contains("has no room there"))
    };
    let warnings: Vec<&Value> = out["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .filter(moved)
        .collect();
    assert!(!warnings.is_empty(), "{:#}", out["diagnostics"]);
    for w in &warnings {
        assert!(
            w["message"].as_str().unwrap().contains("has no room there"),
            "{w}"
        );
        assert!(
            w["range"].is_object(),
            "the warning points at the note: {w}"
        );
    }
    let clean = call(
        "design.check",
        json!({ "path": "/p/l_bracket_notes.drawing" }),
        &files("l_bracket_notes", LISTED),
    )
    .unwrap();
    // The listed notes keep clear: no Overlap finding (the title block's own fields are
    // another rule's business).
    assert!(
        !clean["diagnostics"]
            .as_array()
            .unwrap()
            .iter()
            .any(|d| d["code"] == "drawing.overlap"),
        "{:#}",
        clean["diagnostics"]
    );
}
