//! What the agent learns about a drawing, on OCCT: `design.check` reports what the Desktop's
//! Check drawing finds (one check), and `design.render` draws the sheet from the layout the
//! Desktop's sheet draws, so the agent can see its work.
//!
//! Run with `--features render,occt`. Set `RUNMAT_DESIGN_RENDER_OUT` to a folder to write the
//! pictures (the sheet and the framed views, light and dark).
#![cfg(all(feature = "render", feature = "occt"))]

use base64::Engine;
use runmat_design_tools::{call, Files};
use serde_json::{json, Value};

const PART: &str =
    include_str!("../../runmat-geometry/wasm-host/tests/fixtures/drawings/rounded_bracket.part");
const DRAWING: &str =
    include_str!("../../runmat-geometry/wasm-host/tests/fixtures/drawings/rounded_bracket.drawing");

fn files(drawing: &str) -> Files {
    [
        ("/p/bracket.part".to_string(), PART.to_string()),
        ("/p/bracket.drawing".to_string(), drawing.to_string()),
    ]
    .into_iter()
    .collect()
}

fn ok(name: &str, input: Value, files: &Files) -> Value {
    match call(name, input, files) {
        Ok(v) => v,
        Err(e) => panic!("{name}: {e}"),
    }
}

/// The agent's check and the Desktop's Check drawing find the same things, with the same
/// severities, for the bracket drawing and for one that overprints a view.
#[test]
fn design_check_reports_what_check_drawing_finds() {
    let overprinted = DRAWING.replace(
        "view side = projected(from = front, direction = right, sheet = \"1\", model = { part = bracket }, at = [205, 185]",
        "view side = projected(from = front, direction = right, sheet = \"1\", model = { part = bracket }, at = [110, 185]",
    );
    for drawing in [DRAWING.to_string(), overprinted] {
        let files = files(&drawing);
        runmat_design_tools::kernel::install();
        let (found, _) = runmat_cad_wasm::analyze::drawing_check::findings(
            "/p/bracket.drawing",
            &drawing,
            &files,
            &serde_json::Map::new(),
        )
        .expect("drawing.check");
        let out = ok(
            "design.check",
            json!({ "path": "/p/bracket.drawing" }),
            &files,
        );
        let diagnostics = out["diagnostics"].as_array().unwrap();
        let said: Vec<(String, String)> = diagnostics
            .iter()
            .map(|d| {
                (
                    d["severity"].as_str().unwrap().to_owned(),
                    d["message"].as_str().unwrap().to_owned(),
                )
            })
            .collect();
        let wanted: Vec<(String, String)> = found
            .iter()
            .map(|f| (f.severity.to_owned(), format!("{}: {}", f.rule, f.message)))
            .collect();
        assert_eq!(said, wanted, "{out:#}");
        let count = |s: &str| found.iter().filter(|f| f.severity == s).count();
        assert_eq!(out["errors"], count("error"));
        assert_eq!(out["warnings"], count("warning"));
        assert_eq!(out["ok"], count("error") == 0);
    }
}

/// The agent renders the sheet and one view framed; the picture is a PNG of the sheet's
/// proportions, its notes place each view and carry Check drawing's findings.
#[test]
fn design_render_draws_a_drawings_sheet_and_a_framed_view() {
    let files = files(DRAWING);
    let out = ok(
        "design.render",
        json!({ "path": "/p/bracket.drawing" }),
        &files,
    );
    let png = base64::engine::general_purpose::STANDARD
        .decode(out["pngBase64"].as_str().unwrap())
        .unwrap();
    assert_eq!(&png[1..4], b"PNG");
    assert_eq!(out["renderer"], "sheet");
    let (w, h) = (
        out["width"].as_f64().unwrap(),
        out["height"].as_f64().unwrap(),
    );
    assert!((w / h - 420.0 / 297.0).abs() < 0.01, "{w} × {h}");
    let notes: Vec<&str> = out["notes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .collect();
    for view in ["front", "top", "side", "iso"] {
        assert!(
            notes
                .iter()
                .any(|n| n.starts_with(&format!("View {view}:"))),
            "{notes:#?}"
        );
    }
    assert!(
        notes.iter().any(|n| n.starts_with("Check drawing")),
        "{notes:#?}"
    );
    // The image is drawn: a fair share of its pixels carry ink.
    assert!(png.len() > 10_000, "{} bytes", png.len());

    let side = ok(
        "design.render",
        json!({ "path": "/p/bracket.drawing", "frame": "side", "theme": "dark" }),
        &files,
    );
    assert!(side["notes"][0]
        .as_str()
        .unwrap()
        .contains("framed on the side view"));
    let err = call(
        "design.render",
        json!({ "path": "/p/bracket.drawing", "frame": "elevation" }),
        &files,
    )
    .unwrap_err();
    assert_eq!(err.code, "invalid_arguments");
    assert!(err.message.contains("front"), "{}", err.message);

    // The PDF export draws the same layout (written beside the pictures for review).
    let pdf = ok(
        "design.export",
        json!({ "path": "/p/bracket.drawing", "format": "pdf" }),
        &files,
    );
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(pdf["files"][0]["content"].as_str().unwrap())
        .unwrap();
    assert_eq!(&bytes[..4], b"%PDF");
    if let Ok(dir) = std::env::var("RUNMAT_DESIGN_RENDER_OUT") {
        std::fs::write(format!("{dir}/bracket-1.pdf"), &bytes).unwrap();
        for theme in ["light", "dark"] {
            for frame in [None, Some("iso"), Some("side")] {
                let mut input = json!({ "path": "/p/bracket.drawing", "theme": theme });
                if let Some(f) = frame {
                    input["frame"] = json!(f);
                }
                let out = ok("design.render", input, &files);
                let png = base64::engine::general_purpose::STANDARD
                    .decode(out["pngBase64"].as_str().unwrap())
                    .unwrap();
                std::fs::write(
                    format!("{dir}/{theme}-bracket-{}.png", frame.unwrap_or("sheet")),
                    &png,
                )
                .unwrap();
            }
        }
    }
}
