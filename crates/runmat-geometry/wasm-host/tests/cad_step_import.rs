//! STEP in and out on OCCT through `runmat_cad_call`, the browser worker's entry: a part
//! exported as exact AP242, then imported into another part from the request's files (the
//! browser host has no project folder, so the STEP text travels with the request).

use base64::Engine as _;
use runmat_cad::regen::kernel::{ExchangeDocument, ExchangeFormat, ExchangePart};
use runmat_cad_wasm::runtime::regenerate_part;
use serde_json::Value;

const PLATE: &str = r#"runmat 0.7
std 1

part plate {
    sketch s on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        l2 = line(start = l1.end, end = [80 mm, 50 mm])
        l3 = line(start = l2.end, end = [0 mm, 50 mm])
        l4 = line(start = l3.end, end = l1.start)
    }
    feature body = extrude(profile = sketch.s, distance = 10 mm)
}
"#;

const IMPORTED: &str = r#"runmat 0.7
std 1

part housing {
    feature step = import(source = "plate.step", digest = "unrecorded")
}
"#;

fn call(path: &str, text: &str, files: &Value) -> Value {
    let request =
        serde_json::json!({ "fn": "compile", "args": [path, text, files.to_string(), "{}"] });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

#[test]
fn a_part_exports_exact_ap242_and_imports_back_from_the_requests_files() {
    runmat_geometry_wasm_host::cad::install_occt();
    let part = regenerate_part("/p/plate.part", PLATE).expect("compiles");
    let doc = ExchangeDocument {
        name: "plate".into(),
        parts: vec![ExchangePart {
            name: "plate".into(),
            bodies: part.result.state.bodies.iter().map(|b| &b.handle).collect(),
            color: None,
            material: None,
            face_names: Vec::new(),
            face_colors: Vec::new(),
        }],
        occurrences: Vec::new(),
    };
    let step = part
        .kernel
        .export(&doc, ExchangeFormat::Ap242)
        .expect("OCCT writes STEP");
    let text = String::from_utf8(step).expect("STEP is text");
    let header = &text[..text.find("ENDSEC").unwrap_or(text.len())];
    assert!(
        header.contains("AP242_MANAGED_MODEL_BASED_3D_ENGINEERING"),
        "not AP242: {header}"
    );
    assert!(
        text.contains("ADVANCED_FACE") && !text.contains("FACETED_BREP"),
        "not an exact B-rep"
    );

    let files = serde_json::json!({ "/p/plate.step": text, "/p/housing.part": IMPORTED });
    let v = call("/p/housing.part", IMPORTED, &files);
    let errors: Vec<&Value> = v["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|d| d["severity"] == "error")
        .collect();
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(v["kernel"], "occt");
    let bodies = v["display"]["data"]["bodies"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert_eq!(bodies.len(), 1, "{}", v["display"]["data"]["problems"]);
    let b = &v["display"]["data"]["bounds"];
    let ext = |k: usize| b["max"][k].as_f64().unwrap() - b["min"][k].as_f64().unwrap();
    assert!(
        (ext(0) - 0.08).abs() < 1e-6
            && (ext(1) - 0.05).abs() < 1e-6
            && (ext(2) - 0.01).abs() < 1e-6,
        "bounds {b}"
    );
}

const PAIR: &str = r#"runmat 0.7
std 1

assembly pair {
    use part plate as lower
    use part plate as upper {
        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 10 mm] } }
    }
}
"#;

const STACK: &str = r#"runmat 0.7
std 1

assembly stack {
    use assembly pair as base
    use assembly pair as top {
        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 20 mm] } }
    }
}
"#;

/// An assembly of a sub-assembly placed twice keeps its hierarchy in AP242 (the sub-assembly
/// is one product used twice, holding its two plates), names its faces, and reads back at the
/// size it was written.
#[test]
fn an_assembly_exports_its_hierarchy_and_face_names_as_ap242() {
    runmat_geometry_wasm_host::cad::install_occt();
    let files: std::collections::BTreeMap<String, String> = [
        ("/p/plate.part", PLATE),
        ("/p/pair.assembly", PAIR),
        ("/p/stack.assembly", STACK),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    let step = runmat_cad_wasm::exchange::assembly_step("/p/stack.assembly", STACK, &files)
        .expect("exports")
        .expect("OCCT writes STEP");
    let text = String::from_utf8(step).expect("STEP is text");
    assert!(
        text.contains("AP242_MANAGED_MODEL_BASED_3D_ENGINEERING"),
        "not AP242"
    );
    // stack → base, top (2 uses of pair); pair → lower, upper (2 uses of plate): 4 usages.
    let usages = text.matches("NEXT_ASSEMBLY_USAGE_OCCURRENCE").count();
    assert_eq!(usages, 4, "usages {usages}");
    for product in ["'stack'", "'pair'", "'plate'"] {
        assert!(
            text.contains(&format!("PRODUCT({product}")),
            "no product {product}"
        );
    }
    assert!(
        text.contains("feature.body.face.end"),
        "face names are not written"
    );

    // Read back: the stack is 80 x 50 x 40 mm (four 10 mm plates).
    let mut files = files;
    files.insert("/p/stack.step".into(), text);
    let housing = "runmat 0.7\nstd 1\n\npart housing {\n    feature step = import(source = \"stack.step\", digest = \"unrecorded\")\n}\n";
    files.insert("/p/housing.part".into(), housing.into());
    let v = call(
        "/p/housing.part",
        housing,
        &serde_json::to_value(&files).unwrap(),
    );
    let b = &v["display"]["data"]["bounds"];
    let ext = |k: usize| b["max"][k].as_f64().unwrap() - b["min"][k].as_f64().unwrap();
    assert!(
        (ext(0) - 0.08).abs() < 1e-6
            && (ext(1) - 0.05).abs() < 1e-6
            && (ext(2) - 0.04).abs() < 1e-6,
        "bounds {b}"
    );
}

fn export(path: &str, text: &str, files: &Value, values: Value) -> Value {
    let request = serde_json::json!({ "command": "manage.export", "values": values });
    let call = serde_json::json!({ "fn": "export", "args": [path, text, files.to_string(), request.to_string()] });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&call.to_string())).expect("json")
}

/// The Desktop's Export command: `runmat_cad_call`'s `export` writes a part and an assembly as
/// AP242 with canonical face names, and each file imports back at its size.
#[test]
fn the_export_call_writes_a_part_and_an_assembly_that_import_back() {
    runmat_geometry_wasm_host::cad::install_occt();
    let files = serde_json::json!({ "/p/plate.part": PLATE, "/p/pair.assembly": PAIR });
    let part = export(
        "/p/plate.part",
        PLATE,
        &files,
        serde_json::json!({ "format": "step_ap242" }),
    );
    assert_eq!(part["ok"], true, "{part}");
    assert_eq!(part["name"], "plate.step");
    let step = part["text"].as_str().unwrap();
    assert!(
        step.contains("ADVANCED_FACE('feature.body.face.end'"),
        "canonical face names"
    );
    let pair = export(
        "/p/pair.assembly",
        PAIR,
        &files,
        serde_json::json!({ "format": "step" }),
    );
    assert_eq!(pair["ok"], true, "{pair}");
    assert_eq!(
        pair["text"]
            .as_str()
            .unwrap()
            .matches("NEXT_ASSEMBLY_USAGE_OCCURRENCE")
            .count(),
        2
    );

    for (name, text, height) in [
        ("plate.step", step, 0.01),
        ("pair.step", pair["text"].as_str().unwrap(), 0.02),
    ] {
        let housing = format!("runmat 0.7\nstd 1\n\npart housing {{\n    feature step = import(source = \"{name}\", digest = \"unrecorded\")\n}}\n");
        let files = serde_json::json!({ format!("/p/{name}"): text, "/p/housing.part": housing });
        let v = call("/p/housing.part", &housing, &files);
        let b = &v["display"]["data"]["bounds"];
        let ext = |k: usize| b["max"][k].as_f64().unwrap() - b["min"][k].as_f64().unwrap();
        assert!(
            (ext(0) - 0.08).abs() < 1e-6 && (ext(2) - height).abs() < 1e-6,
            "{name}: bounds {b}"
        );
    }
    let stl = export(
        "/p/plate.part",
        PLATE,
        &files,
        serde_json::json!({ "format": "stl" }),
    );
    assert_eq!(stl["ok"], true, "{stl}");
    assert_eq!(stl["name"], "plate.stl");
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(stl["base64"].as_str().expect("binary STL payload"))
        .expect("valid base64");
    assert!(bytes.len() >= 84, "STL header and triangle count");
    let count = u32::from_le_bytes(bytes[80..84].try_into().unwrap()) as usize;
    assert!(count > 0, "STL contains the plate surface");
    assert_eq!(bytes.len(), 84 + count * 50, "complete STL triangles");
    let unsupported = export(
        "/p/plate.part",
        PLATE,
        &files,
        serde_json::json!({ "format": "unsupported_format" }),
    );
    assert_eq!(unsupported["ok"], false, "{unsupported}");
    assert!(unsupported["message"]
        .as_str()
        .is_some_and(|m| !m.is_empty()));
}

const TILTED: &str = r#"runmat 0.7
std 1

part tilted {
    input depth : length = 10 mm
    input tilt : angle = 0 deg
    sketch s on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        l2 = line(start = l1.end, end = [80 mm, 50 mm])
        l3 = line(start = l2.end, end = [0 mm, 50 mm])
        l4 = line(start = l3.end, end = l1.start)
    }
    feature body = extrude(profile = sketch.s, distance = input.depth)
}
"#;

const TILTED_STUDY: &str = r#"runmat 0.7
std 1

assembly study {
    use part tilted as arm {
        configuration = { select = inputs, values = { depth = 12 mm, tilt = -20 deg } }
    }
}
"#;

/// A part placed in a configuration of its inputs is its own product, named with its input
/// values (`tilted (depth 12 mm, tilt -20°)`). The degree sign must reach the STEP file as
/// valid text and read back: an arm study placed this way failed the FEA import with "data for
/// rust::String is not utf-8".
#[test]
fn a_configured_part_with_an_angle_input_exports_and_imports_back() {
    runmat_geometry_wasm_host::cad::install_occt();
    let files: std::collections::BTreeMap<String, String> = [
        ("/p/tilted.part", TILTED),
        ("/p/study.assembly", TILTED_STUDY),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    let step = runmat_cad_wasm::exchange::assembly_step("/p/study.assembly", TILTED_STUDY, &files)
        .expect("exports")
        .expect("OCCT writes STEP");
    if let Ok(dir) = std::env::var("RUNMAT_STEP_DUMP_DIR") {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(format!("{dir}/study.step"), &step).unwrap();
    }
    let high: Vec<String> = step
        .iter()
        .enumerate()
        .filter(|(_, b)| **b >= 0x80)
        .take(12)
        .map(|(i, b)| format!("{i}:{b:02X}"))
        .collect();
    eprintln!("non-ASCII bytes: {high:?}");
    for line in String::from_utf8_lossy(&step)
        .lines()
        .filter(|l| l.contains("PRODUCT("))
    {
        eprintln!("{line}");
    }
    let text = String::from_utf8(step).expect("STEP is text");

    let mut files = files;
    files.insert("/p/study.step".into(), text);
    let housing = "runmat 0.7\nstd 1\n\npart housing {\n    feature step = import(source = \"study.step\", digest = \"unrecorded\")\n}\n";
    files.insert("/p/housing.part".into(), housing.into());
    let v = call(
        "/p/housing.part",
        housing,
        &serde_json::to_value(&files).unwrap(),
    );
    let errors: Vec<&Value> = v["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|d| d["severity"] == "error")
        .collect();
    assert!(errors.is_empty(), "{errors:?}");
    let b = &v["display"]["data"]["bounds"];
    let ext = |k: usize| b["max"][k].as_f64().unwrap() - b["min"][k].as_f64().unwrap();
    assert!((ext(2) - 0.012).abs() < 1e-6, "bounds {b}");
}

/// A 60 x 40 x 10 mm plate with a 5 mm blind hole whose 118° drill point is a cone with a
/// negative semi-angle, and a 6 mm hole with a 12 mm, 90° countersink.
const CONED: &str = r#"runmat 0.7
std 1

part coned {
    sketch s on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [60 mm, 0 mm])
        l2 = line(start = l1.end, end = [60 mm, 40 mm])
        l3 = line(start = l2.end, end = [0 mm, 40 mm])
        l4 = line(start = l3.end, end = l1.start)
    }
    feature body = extrude(profile = sketch.s, distance = 10 mm)
    feature blind = hole(face = feature.body.face.end, at = { c = point(20 mm, 20 mm, 10 mm) }, diameter = 5 mm, depth = 6 mm, drill_point = 118 deg)
    feature cs = hole(face = feature.body.face.end, at = { a = point(42 mm, 20 mm, 10 mm) }, diameter = 6 mm, depth = 8 mm, form = countersink(diameter = 12 mm, angle = 90 deg))
}
"#;

/// A part whose faces include cones (a drill point with a negative semi-angle, a countersink)
/// exports as AP242 with every face, and reads back as one closed solid whose face names and
/// colours survive. OCCT 8.0's STEP writer dropped the drill point's face and wrote the rest, so
/// the file read back was an open shell.
#[test]
fn a_part_with_a_drill_point_and_a_countersink_exports_a_closed_named_coloured_solid() {
    use runmat_cad::refs::history::SurfaceKind;
    use runmat_cad::regen::kernel::{ExchangeFaceColor, ExchangeFaceName};
    runmat_geometry_wasm_host::cad::install_occt();
    let part = regenerate_part("/p/coned.part", CONED).expect("compiles");
    let bodies = &part.result.state.bodies;
    assert_eq!(bodies.len(), 1);
    let faces = &bodies[0].topology.faces;
    let cones: Vec<u32> = (0..faces.len() as u32)
        .filter(|&i| matches!(faces[i as usize].geom.surface, SurfaceKind::Other))
        .collect();
    assert_eq!(cones.len(), 2, "a drill point and a countersink");
    const RED: [f64; 3] = [1.0, 0.0, 0.0];
    const BLUE: [f64; 3] = [0.0, 0.0, 1.0];
    let doc = ExchangeDocument {
        name: "coned".into(),
        parts: vec![ExchangePart {
            name: "coned".into(),
            bodies: vec![&bodies[0].handle],
            color: None,
            material: None,
            face_names: (0..faces.len() as u32)
                .map(|face| ExchangeFaceName {
                    body: 0,
                    face,
                    name: format!("face {face}"),
                })
                .collect(),
            face_colors: (0..faces.len() as u32)
                .map(|face| ExchangeFaceColor {
                    body: 0,
                    face,
                    color: if cones.contains(&face) { RED } else { BLUE },
                })
                .collect(),
        }],
        occurrences: Vec::new(),
    };
    let step = part
        .kernel
        .export(&doc, ExchangeFormat::Ap242)
        .expect("OCCT writes STEP");
    let text = String::from_utf8(step).expect("STEP is text");
    assert_eq!(
        text.matches("ADVANCED_FACE(").count(),
        faces.len(),
        "every face is written"
    );

    // Closed: the exact import makes one body with no boundary (laminar) edge.
    let dir = std::env::temp_dir().join(format!("runmat-coned-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("coned.step");
    std::fs::write(&file, &text).unwrap();
    let housing = format!(
        "runmat 0.7\nstd 1\n\npart housing {{\n    feature step = import(source = \"{}\", digest = \"unrecorded\")\n}}\n",
        // A Windows temp path has backslashes, which a source string must escape.
        file.display().to_string().replace('\\', "\\\\")
    );
    let back = regenerate_part("/p/housing.part", &housing).expect("compiles");
    let read = &back.result.state.bodies;
    assert_eq!(read.len(), 1, "one body");
    assert_eq!(read[0].topology.faces.len(), faces.len(), "every face");
    assert!(
        !read[0].topology.edges.iter().any(|e| e.laminar),
        "the shell is closed"
    );

    // Names and colours: each face region keeps its name, cones red and the rest blue.
    let imported = runmat_geometry_io::import_geometry(
        "coned.step",
        text.as_bytes(),
        runmat_geometry_io::GeometryImportOptions::default(),
    )
    .expect("imports");
    let _ = std::fs::remove_dir_all(&dir);
    let regions: Vec<_> = imported
        .asset
        .regions
        .iter()
        .filter_map(|r| {
            let owner = r.cad_ownership.as_ref()?;
            owner.face_id?;
            Some((
                r.name.clone(),
                owner
                    .color
                    .as_ref()
                    .map(|c| c.hex_rgba.to_ascii_lowercase()),
            ))
        })
        .collect();
    assert_eq!(regions.len(), faces.len(), "{regions:?}");
    for face in 0..faces.len() as u32 {
        let name = format!("face {face}");
        let (_, color) = regions
            .iter()
            .find(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("no region {name}: {regions:?}"));
        let want = if cones.contains(&face) {
            "ff0000"
        } else {
            "0000ff"
        };
        assert!(
            color
                .as_deref()
                .is_some_and(|c| c.trim_start_matches('#').starts_with(want)),
            "{name}: colour {color:?}, want {want}"
        );
    }
}
