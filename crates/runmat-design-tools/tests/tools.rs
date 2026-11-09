//! The design tools end to end over real design files: the bracket corpus part (one broken
//! fillet by design) and the I/O board project.

use base64::Engine;
use runmat_design_tools::{call, dependencies, dependency_closure, dependency_files, Files};
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
    dependency_closure(
        start,
        &all[start],
        &all.keys().cloned().collect::<Vec<_>>(),
        |p| all.get(p).cloned(),
        64,
    )
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

/// `set` regenerates the part with those inputs for this run: a plate's volume follows
/// w · d · t for each thickness, and the file is unchanged.
#[test]
fn inspect_set_regenerates_with_input_values_for_one_run() {
    let plate = "part plate {\n    input width : length = 80 mm\n    input depth : length = 50 mm\n    input thickness : length = 10 mm\n\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [input.width, 0 mm])\n        l2 = line(start = [input.width, 0 mm], end = [input.width, input.depth])\n        l3 = line(start = [input.width, input.depth], end = [0 mm, input.depth])\n        l4 = line(start = [0 mm, input.depth], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = input.thickness)\n}\n";
    let files: Files = [("/p/plate.part".to_string(), plate.to_string())]
        .into_iter()
        .collect();
    for t in [4.0, 6.5, 10.0] {
        let out = ok(
            "design.inspect",
            json!({ "path": "/p/plate.part", "sections": ["mass"], "set": { "thickness": format!("{t} mm") } }),
            &files,
        );
        let v = out["sections"]["mass"]["volume_mm3"].as_f64().unwrap();
        assert!((v - 80.0 * 50.0 * t).abs() < 1e-6 * v, "t = {t}: {v}");
    }
    assert_eq!(files["/p/plate.part"], plate);
    let e = call(
        "design.inspect",
        json!({ "path": "/p/plate.part", "set": { "length": "1 mm" } }),
        &files,
    )
    .unwrap_err();
    assert_eq!(e.code, "unknown_input");
    assert!(
        e.message.contains("width, depth, thickness"),
        "{}",
        e.message
    );
}

/// `design.export` takes `set` as `design.inspect` does: the exported model regenerates with
/// those inputs (an STL's top sits at each thickness), and the file is unchanged.
#[test]
fn export_set_regenerates_with_input_values_for_one_export() {
    let plate = "part plate {\n    input thickness : length = 10 mm\n\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])\n        l2 = line(start = [80 mm, 0 mm], end = [80 mm, 50 mm])\n        l3 = line(start = [80 mm, 50 mm], end = [0 mm, 50 mm])\n        l4 = line(start = [0 mm, 50 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = input.thickness)\n}\n";
    let files: Files = [("/p/plate.part".to_string(), plate.to_string())]
        .into_iter()
        .collect();
    for t in [4.0_f32, 6.5] {
        let stl = ok(
            "design.export",
            json!({ "path": "/p/plate.part", "format": "stl", "set": { "thickness": format!("{t} mm") } }),
            &files,
        );
        let b = bytes(&stl["files"][0]);
        let n = u32::from_le_bytes([b[80], b[81], b[82], b[83]]) as usize;
        let top = (0..n)
            .flat_map(|i| (0..3).map(move |v| 84 + i * 50 + 12 + v * 12 + 8))
            .map(|at| f32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]))
            .fold(f32::MIN, f32::max);
        assert!((top - t).abs() < 1e-3, "t = {t}: top at {top}");
    }
    assert_eq!(files["/p/plate.part"], plate);
    let e = call(
        "design.export",
        json!({ "path": "/p/plate.part", "format": "stl", "set": { "width": "1 mm" } }),
        &files,
    )
    .unwrap_err();
    assert_eq!(e.code, "unknown_input");
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

/// A design named by symbol needs the file that declares it, whatever that file is called:
/// `dependencies` lists the folder for a host to read, and `dependency_files` names the file.
#[test]
fn a_folder_dependency_resolves_to_the_file_declaring_the_symbol() {
    let part = |name: &str| BRACKET.replace("bracket", name);
    let drawing = r#"runmat 0.7
std 1

drawing sheet {
    format a3 = {
        size = { size = iso, name = A3 },
        title_block = [{ label = Sheet, source = { source = sheet_number } }]
    }

    sheet "1" = { format = a3, scale = [1, 1], projection = third_angle }

    view front = standard(
        orientation = front,
        sheet = "1",
        model = { part = bracket },
        at = [130, 120],
        display = hidden_lines
    )
}
"#;
    assert!(
        BRACKET.contains("part bracket"),
        "the corpus part declares `part bracket`"
    );
    let all: Files = [
        ("/p/d/sheet.drawing".to_owned(), drawing.to_owned()),
        ("/p/d/bracket_v2.part".to_owned(), part("bracket")),
        ("/p/d/other.part".to_owned(), part("other")),
        // The layout's circuit, in a file not named for it.
        ("/b/main.circuit".to_owned(), IO_CIRCUIT.to_owned()),
        ("/b/io.circuit.layout".to_owned(), IO_LAYOUT.to_owned()),
    ]
    .into();
    let project: Vec<String> = all.keys().cloned().collect();
    assert!(dependencies("/p/d/sheet.drawing", drawing).contains(&"/p/d/".to_owned()));
    let files = dependency_closure(
        "/p/d/sheet.drawing",
        drawing,
        &project,
        |p| all.get(p).cloned(),
        64,
    );
    assert!(
        files.contains_key("/p/d/bracket_v2.part") && files.contains_key("/p/d/other.part"),
        "the closure reads the folder: {:?}",
        files.keys()
    );
    let check = call(
        "design.check",
        json!({ "path": "/p/d/sheet.drawing" }),
        &files,
    )
    .expect("check");
    assert_eq!(check["errors"], json!(0), "the drawing compiles: {check}");
    assert_eq!(
        dependency_files("/p/d/sheet.drawing", drawing, &files),
        ["/p/d/bracket_v2.part"]
    );
    let board = dependency_files("/b/io.circuit.layout", IO_LAYOUT, &all);
    assert!(board.contains(&"/b/main.circuit".to_owned()), "{board:?}");
    assert!(board.iter().all(|p| !p.ends_with('/')), "{board:?}");
}

const TUBE: &str = r#"runmat 0.7
std 1

part tube {
    sketch s on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [10 mm, 0 mm])
        l2 = line(start = l1.end, end = [10 mm, 10 mm])
        l3 = line(start = l2.end, end = [0 mm, 10 mm])
        l4 = line(start = l3.end, end = l1.start)
    }
    feature body = extrude(profile = sketch.s, distance = 200 mm)
}
"#;

#[test]
fn propose_edit_creates_a_new_part_and_an_assembly_that_places_it() {
    let files = part_files();
    let out = ok(
        "design.propose_edit",
        json!({ "path": "/p/tube.part", "text": TUBE }),
        &files,
    );
    assert_eq!(out["files"][0]["created"], true, "{out:#}");
    assert_eq!(out["keep"][0]["tool"], "runmat.write_file");
    assert_eq!(out["undo"][0]["tool"], "runmat.delete_file");
    assert_eq!(out["ok"], true, "{:#}", out["diagnostics"]);
    // Every feature of a new part is added.
    let added: Vec<&str> = out["diff"]["features"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|f| f["change"] == "added")
        .filter_map(|f| f["id"].as_str())
        .collect();
    assert!(added.contains(&"body"), "{out:#}");

    // Kept, the tube is a file an assembly can place.
    let mut kept = files.clone();
    kept.insert("/p/tube.part".into(), TUBE.into());
    let assembly = "runmat 0.7\nstd 1\n\nassembly skid {\n    use part tube as left\n    use part tube as right {\n        placement = { placement = offset, offset = { translation = [40 mm, 0 mm, 0 mm] } }\n    }\n}\n";
    let out = ok(
        "design.propose_edit",
        json!({ "path": "/p/skid.assembly", "text": assembly }),
        &kept,
    );
    assert_eq!(out["ok"], true, "{:#}", out["diagnostics"]);
    let occurrences: Vec<&str> = out["diff"]["features"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|f| f["change"] == "added")
        .filter_map(|f| f["id"].as_str())
        .collect();
    assert!(
        occurrences.contains(&"occurrence.left") && occurrences.contains(&"occurrence.right"),
        "{out:#}"
    );

    // A missing file takes its whole text: a replacement or command has nothing to apply to.
    let err = call(
        "design.propose_edit",
        json!({ "path": "/p/clamp.part", "replace": [{ "old": "a", "new": "b" }] }),
        &files,
    )
    .unwrap_err();
    assert_eq!(err.code, "file_missing");
}

#[test]
fn a_call_as_json_answers_data_or_a_coded_error() {
    let files = serde_json::to_string(&part_files()).unwrap();
    let out =
        runmat_design_tools::call_json("design.check", r#"{"path":"/p/bracket.part"}"#, &files);
    assert_eq!(out["data"]["path"], "/p/bracket.part", "{out}");
    assert_eq!(
        runmat_design_tools::call_json("design.check", "{", &files)["error"]["code"],
        "invalid_arguments"
    );
    assert_eq!(
        runmat_design_tools::call_json("design.check", r#"{"path":"/p/none.part"}"#, &files)
            ["error"]["code"],
        "file_missing"
    );
}

#[test]
fn an_assembly_reports_interference_by_name_and_mass_from_its_materials() {
    let plate = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])\n        l2 = line(start = l1.end, end = [80 mm, 50 mm])\n        l3 = line(start = l2.end, end = [0 mm, 50 mm])\n        l4 = line(start = l3.end, end = l1.start)\n    }\n    feature body = extrude(profile = sketch.s, distance = 10 mm)\n    output plate = { body = feature.body.body, material = \"material.aluminum_6061\" }\n}\n";
    // The upper plate sinks 5 mm into the lower one.
    let pair = "runmat 0.7\nstd 1\n\nassembly pair {\n    use part plate as lower\n    use part plate as upper {\n        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 5 mm] } }\n    }\n}\n";
    let files: Files = [("/p/plate.part", plate), ("/p/pair.assembly", pair)]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
    let out = ok(
        "design.inspect",
        json!({ "path": "/p/pair.assembly", "sections": ["interference", "mass"] }),
        &files,
    );
    // The Evaluate tab's check runs on the pair. The grid test kernel has no exact proximity, so
    // it says so per pair rather than reporting a clean result (OCCT finds the overlap: the CAD
    // host's cad_design_tools test).
    let interference = &out["sections"]["interference"];
    assert_eq!(interference["result"]["parts"], 2, "{out:#}");
    assert!(
        interference["problems"][0]
            .as_str()
            .unwrap()
            .contains("lower / upper"),
        "{out:#}"
    );
    // Two 40 cm³ aluminium plates: 108 g each from their material, not steel's 314 g.
    let mass = out["sections"]["mass"]["result"]["parts"]
        .as_array()
        .unwrap();
    assert_eq!(mass.len(), 2, "{out:#}");
    for part in mass {
        assert_eq!(part["density_source"], "material");
        assert!(
            (part["mass"].as_f64().unwrap() - 0.108).abs() < 1e-9,
            "{part}"
        );
    }
    // Unnamed, the costly check doesn't run.
    let all = ok(
        "design.inspect",
        json!({ "path": "/p/pair.assembly" }),
        &files,
    );
    assert!(all["sections"].get("interference").is_none(), "{all:#}");
}

#[test]
fn an_interference_check_limited_to_a_change_checks_only_its_pairs() {
    let plate = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])\n        l2 = line(start = l1.end, end = [80 mm, 50 mm])\n        l3 = line(start = l2.end, end = [0 mm, 50 mm])\n        l4 = line(start = l3.end, end = l1.start)\n    }\n    feature body = extrude(profile = sketch.s, distance = 10 mm)\n    output plate = { body = feature.body.body, material = \"material.aluminum_6061\" }\n}\n";
    // lower and upper already overlap; the change adds `skid`, 500 mm away from both.
    let trio = "runmat 0.7\nstd 1\n\nassembly trio {\n    use part plate as lower\n    use part plate as upper {\n        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 5 mm] } }\n    }\n    use part plate as skid {\n        placement = { placement = offset, offset = { translation = [500 mm, 0 mm, 0 mm] } }\n    }\n}\n";
    let files: Files = [("/p/plate.part", plate), ("/p/trio.assembly", trio)]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
    // The change's part touches nothing: no pair is checked and the existing overlap isn't listed.
    let out = ok(
        "design.inspect",
        json!({ "path": "/p/trio.assembly", "sections": ["interference"], "changed": ["skid"] }),
        &files,
    );
    let interference = &out["sections"]["interference"];
    assert_eq!(interference["result"]["parts"], 3, "{out:#}");
    assert_eq!(interference["result"]["candidates"], 0, "{out:#}");
    assert_eq!(
        interference["result"]["changed"],
        json!(["skid"]),
        "{out:#}"
    );
    assert!(
        interference["problems"].as_array().unwrap().is_empty(),
        "{out:#}"
    );
    assert!(
        out["notes"].as_array().unwrap().iter().any(|n| n
            .as_str()
            .unwrap()
            .contains("only the pairs skid is in were checked")),
        "{out:#}"
    );
    // Naming `upper` checks its pair with `lower` (the grid kernel says it can't measure it).
    let out = ok(
        "design.inspect",
        json!({ "path": "/p/trio.assembly", "sections": ["interference"], "changed": ["upper"] }),
        &files,
    );
    assert_eq!(
        out["sections"]["interference"]["result"]["candidates"], 1,
        "{out:#}"
    );
    assert!(
        out["sections"]["interference"]["problems"][0]
            .as_str()
            .unwrap()
            .contains("lower / upper"),
        "{out:#}"
    );
    // A name that matches no occurrence is an error that lists them.
    let err = runmat_design_tools::call(
        "design.inspect",
        json!({ "path": "/p/trio.assembly", "sections": ["interference"], "changed": ["tube"] }),
        &files,
    )
    .unwrap_err();
    assert!(
        err.message.contains("lower, upper, skid"),
        "{}",
        err.message
    );
}

#[test]
fn the_surface_section_finds_the_face_and_normal_under_a_point() {
    let plate = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])\n        l2 = line(start = l1.end, end = [80 mm, 50 mm])\n        l3 = line(start = l2.end, end = [0 mm, 50 mm])\n        l4 = line(start = l3.end, end = l1.start)\n    }\n    feature body = extrude(profile = sketch.s, distance = 10 mm)\n    output plate = { body = feature.body.body }\n}\n";
    let lifted = "runmat 0.7\nstd 1\n\nassembly lifted {\n    use part plate as base {\n        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 100 mm] } }\n    }\n}\n";
    let files: Files = [("/p/plate.part", plate), ("/p/lifted.assembly", lifted)]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
    // From below, looking up: the plate's underside at z = 100 mm, facing down. Two rays in
    // one call, each answered on its own: the second misses the plate.
    let out = ok(
        "design.inspect",
        json!({ "path": "/p/lifted.assembly", "sections": ["surface"],
        "rays": [{ "from": [40.0, 25.0, 0.0], "direction": [0.0, 0.0, 1.0] }, { "from": [200.0, 25.0, 0.0], "direction": [0.0, 0.0, 1.0] }] }),
        &files,
    );
    let rays = out["sections"]["surface"]["rays"].as_array().unwrap();
    assert_eq!(rays.len(), 2, "{out:#}");
    assert_eq!(rays[0]["from_mm"], json!([40.0, 25.0, 0.0]), "{out:#}");
    let hits = rays[0]["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 2, "{out:#}");
    assert_eq!(hits[0]["occurrence"], "base", "{out:#}");
    assert_eq!(hits[0]["point_mm"], json!([40.0, 25.0, 100.0]), "{out:#}");
    assert_eq!(hits[0]["normal"], json!([0.0, 0.0, -1.0]), "{out:#}");
    assert_eq!(hits[0]["distance_mm"], 100.0, "{out:#}");
    assert_eq!(hits[0]["entering"], true, "{out:#}");
    assert_eq!(hits[1]["point_mm"], json!([40.0, 25.0, 110.0]), "{out:#}");
    assert_eq!(hits[1]["entering"], false, "{out:#}");
    assert!(rays[1]["hits"].as_array().unwrap().is_empty(), "{out:#}");
    // On the part itself, and a ray that misses.
    let out = ok(
        "design.inspect",
        json!({ "path": "/p/plate.part", "sections": ["surface"],
        "rays": [{ "from": [40.0, 25.0, 50.0], "direction": [0.0, 0.0, -1.0] }, { "from": [200.0, 25.0, 50.0], "direction": [0.0, 0.0, -1.0] }] }),
        &files,
    );
    assert_eq!(
        out["sections"]["surface"]["rays"][0]["hits"][0]["point_mm"],
        json!([40.0, 25.0, 10.0]),
        "{out:#}"
    );
    assert!(
        out["sections"]["surface"]["rays"][1]["hits"]
            .as_array()
            .unwrap()
            .is_empty(),
        "{out:#}"
    );
    // The body's own frame: its extent, and its principal axes with the long one (x) first.
    let out = ok(
        "design.inspect",
        json!({ "path": "/p/plate.part", "sections": ["mass"] }),
        &files,
    );
    let mass = &out["sections"]["mass"];
    assert_eq!(
        mass["extent_mm"],
        json!({ "min": [0.0, 0.0, 0.0], "max": [80.0, 50.0, 10.0] }),
        "{out:#}"
    );
    assert_eq!(
        mass["principal_axes"][0]["axis"],
        json!([1.0, 0.0, 0.0]),
        "{out:#}"
    );
    assert_eq!(
        mass["principal_axes"][2]["axis"],
        json!([0.0, 0.0, 1.0]),
        "{out:#}"
    );
}

/// A connector written flat (`at`, `z`, `x` on the connector) places a fastened part: the foot's
/// origin lands on the point, its X toward the given axis (a quarter turn about Z from its start).
#[test]
fn a_flat_connector_seats_a_fastened_part_at_a_point_and_axis() {
    let plate = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])\n        l2 = line(start = l1.end, end = [80 mm, 50 mm])\n        l3 = line(start = l2.end, end = [0 mm, 50 mm])\n        l4 = line(start = l3.end, end = l1.start)\n    }\n    feature body = extrude(profile = sketch.s, distance = 10 mm)\n    output plate = { body = feature.body.body }\n}\n";
    let rig = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part plate as base\n    use part plate as foot {\n        fixed = false\n    }\n    connector mount = { occurrence = [{ name = base }], at = [40 mm, 25 mm, 0 mm], z = [0, 0, 1], x = [0, 1, 0] }\n    connector top = { occurrence = [{ name = foot }] }\n    mate seat = { kind = { mate = fasten }, a = mount, b = top }\n}\n";
    let files: Files = [("/p/plate.part", plate), ("/p/rig.assembly", rig)]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
    let out = ok(
        "design.inspect",
        json!({ "path": "/p/rig.assembly", "sections": ["occurrences"] }),
        &files,
    );
    let foot = out["sections"]["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .find(|o| o["name"] == "foot")
        .cloned()
        .unwrap();
    let world = &foot["world"];
    let t: Vec<f64> = world["translation"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert!(
        (t[0] - 0.040).abs() < 1e-9 && (t[1] - 0.025).abs() < 1e-9 && t[2].abs() < 1e-9,
        "{foot:#}"
    );
    // Columns are the foot's axes in the world: X toward +y, Z up.
    let r = |i: usize, j: usize| world["rotation"][i][j].as_f64().unwrap();
    assert!(
        (r(1, 0) - 1.0).abs() < 1e-6 && (r(2, 2) - 1.0).abs() < 1e-6,
        "{foot:#}"
    );
    assert_eq!(foot["state"], "fully_defined", "{foot:#}");
}
