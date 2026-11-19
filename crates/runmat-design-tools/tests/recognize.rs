//! Recognize features on OCCT: a part exported as STEP through the exchange path, imported
//! back, and turned into features that rebuild the same solid.
#![cfg(feature = "occt")]

use runmat_cad_wasm::runtime::Files;
use serde_json::{json, Value};

/// A 60 x 40 x 10 mm plate: two rounded corners (4 mm), one chamfered corner (2 mm), two 6 mm
/// through holes, a 5 mm blind hole 6 mm deep with a 118° drill point, and a 16 x 8 mm pocket
/// 3 mm deep in its top.
const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {
    sketch s1 on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [60 mm, 0 mm])
        l2 = line(start = [60 mm, 0 mm], end = [60 mm, 40 mm])
        l3 = line(start = [60 mm, 40 mm], end = [0 mm, 40 mm])
        l4 = line(start = [0 mm, 40 mm], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.s1, distance = 10 mm)
    feature corners = fillet(edges = [edges_between(feature.body.face.side(s1.l1), feature.body.face.side(s1.l2)), edges_between(feature.body.face.side(s1.l2), feature.body.face.side(s1.l3))], radius = 4 mm)
    feature bevel = chamfer(edges = [edges_between(feature.body.face.side(s1.l3), feature.body.face.side(s1.l4))], distance = 2 mm)
    feature top = plane.offset(base = origin.xy, distance = 10 mm)
    sketch p1 on plane.top {
        m1 = line(start = [22 mm, 6 mm], end = [38 mm, 6 mm])
        m2 = line(start = [38 mm, 6 mm], end = [38 mm, 14 mm])
        m3 = line(start = [38 mm, 14 mm], end = [22 mm, 14 mm])
        m4 = line(start = [22 mm, 14 mm], end = [22 mm, 6 mm])
    }
    feature pocket = extrude(profile = sketch.p1, distance = 3 mm, reverse, mode = cut)
    feature bores = hole(face = feature.body.face.end, at = { a = point(12 mm, 20 mm, 10 mm), b = point(48 mm, 20 mm, 10 mm) }, diameter = 6 mm, through_all)
    feature blind = hole(face = feature.body.face.end, at = { c = point(30 mm, 30 mm, 10 mm) }, diameter = 5 mm, depth = 6 mm, drill_point = 118 deg)
}
";

/// The plate with its top edge along l1 rounded: a cylinder across the plate's axis.
fn with_top_round() -> String {
    PLATE.replace(
        "    feature top = plane",
        "    feature edge = fillet(edges = [edges_between(feature.body.face.end, feature.body.face.side(s1.l1))], radius = 1 mm)\n    feature top = plane",
    )
}

/// The part exported as STEP, written beside a part that imports it.
fn imported(source: &str, name: &str) -> (String, String) {
    runmat_design_tools::kernel::install();
    let mut files = Files::new();
    files.insert("/p/plate.part".into(), source.into());
    let out = runmat_design_tools::call(
        "design.export",
        json!({ "path": "/p/plate.part", "format": "step" }),
        &files,
    )
    .expect("exports");
    let step = out["files"][0]["content"].as_str().expect("STEP text");
    let dir = std::env::temp_dir().join(format!("runmat-recognize-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join(format!("{name}.step"));
    std::fs::write(&file, step).unwrap();
    let path = format!("/p/{name}.part");
    let text = format!(
        "runmat 0.7\nstd 1\n\npart {name} {{\n    feature step = import(source = \"{}\", digest = sha256_test)\n}}\n",
        // A Windows temp path has backslashes, which a source string must escape.
        file.display().to_string().replace('\\', "\\\\")
    );
    (path, text)
}

fn recognize(path: &str, text: &str) -> Value {
    let request = json!({
        "command": "manage.recognize-features",
        "values": { "body": "feature:step", "tolerance": 0.01 }
    });
    serde_json::from_str(&runmat_cad_wasm::edit::edit_json(
        path,
        text,
        &Files::new(),
        &request.to_string(),
    ))
    .unwrap()
}

#[test]
fn an_imported_plate_becomes_an_extrude_holes_a_pocket_rounds_and_a_chamfer() {
    let (path, text) = imported(PLATE, "plate_import");
    let out = recognize(&path, &text);
    assert_eq!(out["ok"], true, "{out}");
    let new = out["files"][0]["text"].as_str().unwrap();
    let summary = out["summary"].as_str().unwrap();
    eprintln!("{new}\n{summary}");
    assert!(!new.contains("import("), "{new}");
    assert!(
        new.contains("feature base = extrude(profile = sketch.outline, distance = 10 mm)"),
        "{new}"
    );
    assert!(
        new.contains("fillet(edges = [edges_between(feature.base.face.side(outline."),
        "{new}"
    );
    assert!(new.contains("radius = 4 mm)"), "{new}");
    assert!(new.contains("chamfer(edges = ["), "{new}");
    assert!(new.contains("distance = 2 mm)"), "{new}");
    assert!(new.contains("diameter = 6 mm, through_all)"), "{new}");
    assert!(
        new.contains("diameter = 5 mm, depth = 6 mm, drill_point = 118 deg)"),
        "{new}"
    );
    assert!(
        new.contains("distance = 3 mm, reverse, mode = cut)"),
        "{new}"
    );
    assert!(summary.contains("2 rounded corners"), "{summary}");
    assert!(summary.contains("1 chamfered corner"), "{summary}");
    assert!(summary.contains("2 through holes of 6 mm"), "{summary}");
    assert!(
        summary.contains("1 blind drilled hole of 5 mm"),
        "{summary}"
    );
    assert!(summary.contains("a pocket 3 mm deep"), "{summary}");
    assert!(
        summary.contains("It matches the import within 0.01 mm"),
        "{summary}"
    );
}

#[test]
fn a_round_along_a_cap_edge_leaves_the_body_imported_and_says_why() {
    let (path, text) = imported(&with_top_round(), "rounded_import");
    let out = recognize(&path, &text);
    assert_eq!(out["ok"], false, "{out}");
    let message = out["message"].as_str().unwrap();
    assert!(message.contains("stays imported"), "{message}");
    assert!(
        message.contains("a cylinder across the z axis"),
        "{message}"
    );
}

#[test]
fn a_body_that_is_not_imported_is_refused() {
    runmat_design_tools::kernel::install();
    let request =
        json!({ "command": "manage.recognize-features", "values": { "body": "feature:body" } });
    let out: Value = serde_json::from_str(&runmat_cad_wasm::edit::edit_json(
        "/p/plate.part",
        PLATE,
        &Files::new(),
        &request.to_string(),
    ))
    .unwrap();
    assert_eq!(out["ok"], false, "{out}");
    assert!(
        out["message"]
            .as_str()
            .unwrap()
            .contains("isn't an imported body"),
        "{out}"
    );
}
