//! The design tools end to end over real design files: the bracket corpus part (one broken
//! fillet by design) and the I/O board project.

use base64::Engine;
use runmat_design_tools::{call, dependency_closure, Files};
use serde_json::{json, Value};

const BRACKET: &str = include_str!("../../runmat-cad/tests/corpus/documents/bracket.part");
const IO_CIRCUIT: &str = include_str!("../../runmat-circuit/tests/projects/io/io.circuit");
const IO_LAYOUT: &str = include_str!("../../runmat-circuit/tests/projects/io/io.circuit.layout");
const IO_LOCK: &str = include_str!("../../runmat-circuit/tests/projects/io/io.circuit.lock");

fn part_files() -> Files {
    [("/p/bracket.part".to_owned(), BRACKET.to_owned())].into()
}

fn io_files(start: &str) -> Files {
    let all: Files = [
        ("/b/io.circuit".to_owned(), IO_CIRCUIT.to_owned()),
        ("/b/io.circuit.layout".to_owned(), IO_LAYOUT.to_owned()),
        ("/b/io.circuit.lock".to_owned(), IO_LOCK.to_owned()),
    ]
    .into();
    dependency_closure(start, &all[start], |p| all.get(p).cloned(), 64)
}

fn ok(name: &str, input: Value, files: &Files) -> Value {
    match call(name, input, files) {
        Ok(v) => v,
        Err(e) => panic!("{name}: {e}"),
    }
}

#[test]
fn check_reports_the_root_cause_with_the_features_it_failed_and_where() {
    let out = ok(
        "design.check",
        json!({ "path": "/p/bracket.part" }),
        &part_files(),
    );
    assert_eq!(out["kind"], "part");
    assert!(out["errors"].as_u64().unwrap() >= 1, "{out:#}");
    let root = out["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .find(|d| d["cause"] == "root" && d["severity"] == "error")
        .unwrap_or_else(|| panic!("{out:#}"));
    assert!(
        root["failedFeatures"]
            .as_array()
            .is_some_and(|f| f.iter().any(|f| f == "round_missing")),
        "{root:#}"
    );
    let range = &root["range"];
    let line = range["line"].as_u64().unwrap() as usize;
    let text = BRACKET.lines().nth(line - 1).unwrap();
    assert!(
        text.contains("round_missing")
            || text.contains("s1.l9")
            || BRACKET
                .lines()
                .nth(line)
                .is_some_and(|l| l.contains("s1.l9")),
        "line {line}: {text}"
    );
    let width = out["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["name"] == "width")
        .unwrap();
    assert_eq!(width["display"], "80 mm");
    assert!(out["summary"].as_str().unwrap().contains("error"));
}

#[test]
fn syntax_errors_come_back_as_diagnostics_at_their_line() {
    let files: Files = [(
        "/p/x.part".to_owned(),
        "runmat 0.7\nstd 1\n\npart x {\n  feature ±\n}\n".to_owned(),
    )]
    .into();
    let out = ok("design.check", json!({ "path": "/p/x.part" }), &files);
    assert_eq!(out["ok"], false);
    assert_eq!(out["diagnostics"][0]["range"]["line"], 5, "{out:#}");
}

#[test]
fn inspect_finds_faces_by_name_and_mass_properties() {
    let out = ok(
        "design.inspect",
        json!({ "path": "/p/bracket.part", "filter": "feature.body.face.end", "sections": ["faces", "mass", "regions"] }),
        &part_files(),
    );
    let faces = out["sections"]["faces"].as_array().unwrap();
    assert!(!faces.is_empty(), "{out:#}");
    for f in faces {
        assert!(
            f["reference"]
                .as_str()
                .unwrap()
                .contains("feature.body.face.end")
                || f["names"].to_string().contains("end")
        );
        assert!(f["area_mm2"].as_f64().unwrap() > 0.0);
    }
    // The end face is the top of the 10 mm plate: its normal is +z.
    assert!(
        faces
            .iter()
            .any(|f| (f["normal"][2].as_f64().unwrap() - 1.0).abs() < 1e-3),
        "{faces:#?}"
    );
    let mass = &out["sections"]["mass"];
    let v = mass["volume_mm3"].as_f64().unwrap();
    // 80 × 50 × 10 less the 20 × 10 pocket and the hole.
    assert!(v > 30_000.0 && v < 40_000.0, "{mass:#}");
    assert!(mass["mass_kg"].as_f64().unwrap() > 0.2);
}

#[test]
fn propose_edit_stages_text_with_the_dimension_diff_and_a_patch_that_keeps_and_undoes_it() {
    let files = part_files();
    let next = BRACKET.replace(
        "input width : length = 80 mm",
        "input width : length = 90 mm",
    );
    let out = ok(
        "design.propose_edit",
        json!({ "path": "/p/bracket.part", "text": next }),
        &files,
    );
    let dims = out["diff"]["dimensions"].as_array().unwrap();
    assert!(
        dims.iter()
            .any(|d| d["name"] == "width" && d["before"] == "80 mm" && d["after"] == "90 mm"),
        "{out:#}"
    );
    let keep = &out["keep"][0];
    assert_eq!(keep["tool"], "runmat.apply_patch");
    let edit = &keep["arguments"]["edits"][0];
    let old = edit["oldText"].as_str().unwrap();
    assert_eq!(BRACKET.matches(old).count(), 1);
    let kept = BRACKET.replacen(old, edit["newText"].as_str().unwrap(), 1);
    assert_eq!(kept, next);
    let undo = &out["undo"][0]["arguments"]["edits"][0];
    let back = kept.replacen(
        undo["oldText"].as_str().unwrap(),
        undo["newText"].as_str().unwrap(),
        1,
    );
    assert_eq!(back, BRACKET);
}

#[test]
fn propose_edit_runs_catalog_commands_and_rejects_unknown_ones() {
    let files = part_files();
    let out = ok(
        "design.propose_edit",
        json!({ "path": "/p/bracket.part", "command": "part.suppress", "selection": ["feature:round_missing"] }),
        &files,
    );
    assert_eq!(out["files"][0]["path"], "/p/bracket.part");
    let after = out["files"][0]["text"].as_str().unwrap();
    assert_ne!(after, BRACKET);
    // Suppressing the broken fillet clears the error.
    assert_eq!(out["ok"], true, "{:#}", out["diagnostics"]);
    let err = call(
        "design.propose_edit",
        json!({ "path": "/p/bracket.part", "command": "part.extrudez" }),
        &files,
    )
    .unwrap_err();
    assert_eq!(err.code, "invalid_arguments");
    assert!(err.message.contains("part.extrude"), "{err}");
}

fn bytes(file: &Value) -> Vec<u8> {
    match file["encoding"].as_str() {
        Some("base64") => base64::engine::general_purpose::STANDARD
            .decode(file["content"].as_str().unwrap())
            .unwrap(),
        _ => file["content"].as_str().unwrap().as_bytes().to_vec(),
    }
}

#[test]
fn parts_export_as_step_stl_3mf_and_gltf() {
    let files = part_files();
    let stl = ok(
        "design.export",
        json!({ "path": "/p/bracket.part", "format": "stl" }),
        &files,
    );
    let b = bytes(&stl["files"][0]);
    let n = u32::from_le_bytes([b[80], b[81], b[82], b[83]]) as usize;
    assert!(n > 12);
    assert_eq!(b.len(), 84 + n * 50);
    let step = ok(
        "design.export",
        json!({ "path": "/p/bracket.part", "format": "step" }),
        &files,
    );
    let text = step["files"][0]["content"].as_str().unwrap();
    assert!(text.starts_with("ISO-10303-21;"));
    assert!(text.contains("FACETED_BREP(") && text.contains("SI_UNIT(.MILLI.,.METRE.)"));
    assert!(text.trim_end().ends_with("END-ISO-10303-21;"));
    assert_eq!(step["files"][0]["name"], "bracket.step");
    let mf = ok(
        "design.export",
        json!({ "path": "/p/bracket.part", "format": "3mf" }),
        &files,
    );
    let z = bytes(&mf["files"][0]);
    assert_eq!(&z[..4], b"PK\x03\x04");
    assert!(String::from_utf8_lossy(&z).contains("unit=\"millimeter\""));
    let glb = ok(
        "design.export",
        json!({ "path": "/p/bracket.part", "format": "gltf" }),
        &files,
    );
    let g = bytes(&glb["files"][0]);
    assert_eq!(&g[..4], b"glTF");
    assert_eq!(
        u32::from_le_bytes([g[8], g[9], g[10], g[11]]) as usize,
        g.len()
    );
}

#[test]
fn drc_and_erc_return_structured_violations_on_the_io_board() {
    let files = io_files("/b/io.circuit.layout");
    let drc = ok(
        "design.drc",
        json!({ "path": "/b/io.circuit.layout" }),
        &files,
    );
    assert_eq!(drc["ran"], true, "{drc:#}");
    let v = drc["violations"].as_array().unwrap();
    assert!(
        !v.is_empty(),
        "the ECO scenario board carries seeded violations: {drc:#}"
    );
    for x in v {
        assert!(!x["code"].as_str().unwrap().is_empty());
        assert!(x["objects"].is_array());
        assert!(x["locationMm"].is_array(), "{x:#}");
    }
    assert!(v.iter().any(|x| x["rule"].is_string()));
    let erc = ok(
        "design.erc",
        json!({ "path": "/b/io.circuit" }),
        &io_files("/b/io.circuit"),
    );
    assert_eq!(erc["ran"], true, "{erc:#}");
    assert_eq!(erc["check"], "erc");
    let part = call(
        "design.drc",
        json!({ "path": "/p/bracket.part" }),
        &part_files(),
    )
    .unwrap_err();
    assert_eq!(part.code, "invalid_arguments");
}

#[test]
fn inspect_lists_components_nets_and_rooms() {
    let files = io_files("/b/io.circuit.layout");
    let out = ok(
        "design.inspect",
        json!({ "path": "/b/io.circuit.layout", "sections": ["components", "nets", "rooms"], "filter": "U*" }),
        &files,
    );
    let comps = out["sections"]["components"].as_array().unwrap();
    assert!(!comps.is_empty());
    assert!(
        comps
            .iter()
            .all(|c| c["designator"].as_str().unwrap().starts_with('U')),
        "{comps:#?}"
    );
    assert!(out["totals"]["nets"].as_u64().unwrap() > 0);
}

#[test]
fn circuits_export_to_kicad_and_import_back() {
    let files = io_files("/b/io.circuit");
    let out = ok(
        "design.export",
        json!({ "path": "/b/io.circuit", "format": "kicad" }),
        &files,
    );
    let names: Vec<&str> = out["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"io.kicad_sch"), "{names:?}");
    assert!(names.contains(&"io.kicad_pcb"), "{names:?}");
    let back = ok(
        "design.import",
        json!({
            "format": "kicad",
            "target": "/k/io",
            "files": out["files"].as_array().unwrap().iter().map(|f| json!({ "name": f["name"], "content": f["content"] })).collect::<Vec<_>>(),
        }),
        &Files::new(),
    );
    let paths: Vec<&str> = back["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["path"].as_str().unwrap())
        .collect();
    assert!(paths.contains(&"/k/io.circuit"), "{paths:?}");
    assert!(paths.contains(&"/k/io.circuit.layout"), "{paths:?}");
    assert!(back["keep"]
        .as_array()
        .unwrap()
        .iter()
        .all(|c| c["tool"] == "runmat.write_file"));
}

#[test]
fn step_files_import_as_a_part_that_names_them() {
    let step = "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n";
    let out = ok(
        "design.import",
        json!({ "format": "step", "target": "/p/parts/housing.part", "files": [{ "name": "housing.step", "content": step }] }),
        &Files::new(),
    );
    let files = out["files"].as_array().unwrap();
    assert_eq!(files.len(), 2);
    let part = files
        .iter()
        .find(|f| f["path"] == "/p/parts/housing.part")
        .unwrap();
    assert!(part["text"]
        .as_str()
        .unwrap()
        .contains("import(source = \"housing.step\""));
    assert!(files.iter().any(|f| f["path"] == "/p/parts/housing.step"));
}

#[test]
fn the_catalog_declares_every_tool_with_object_schemas() {
    let names: Vec<String> = runmat_design_tools::catalog()
        .into_iter()
        .map(|t| t.name)
        .collect();
    for n in [
        "design.check",
        "design.render",
        "design.inspect",
        "design.drc",
        "design.erc",
        "design.import",
        "design.export",
        "design.propose_edit",
        "design.overview",
    ] {
        assert!(names.iter().any(|x| x == n), "{n}");
    }
}

#[cfg(feature = "render")]
#[test]
fn render_draws_a_part_view_to_a_png() {
    let out = ok(
        "design.render",
        json!({ "path": "/p/bracket.part", "view": "isometric", "width": 640, "height": 400, "highlight": ["feature.body.face.end"] }),
        &part_files(),
    );
    let png = base64::engine::general_purpose::STANDARD
        .decode(out["pngBase64"].as_str().unwrap())
        .unwrap();
    assert_eq!(&png[1..4], b"PNG");
    assert_eq!(out["width"], 640);
    assert!(out.get("notes").is_none(), "{:#}", out["notes"]);
    assert!(out["triangles"].as_u64().unwrap() > 0);
    if let Ok(dir) = std::env::var("RUNMAT_DESIGN_RENDER_OUT") {
        for view in ["isometric", "top", "front"] {
            for theme in ["light", "dark"] {
                let out = ok(
                    "design.render",
                    json!({ "path": "/p/bracket.part", "view": view, "theme": theme, "highlight": ["feature.body.face.end"] }),
                    &part_files(),
                );
                let png = base64::engine::general_purpose::STANDARD
                    .decode(out["pngBase64"].as_str().unwrap())
                    .unwrap();
                std::fs::write(format!("{dir}/{theme}-render-bracket-{view}.png"), &png).unwrap();
            }
        }
    }
}
