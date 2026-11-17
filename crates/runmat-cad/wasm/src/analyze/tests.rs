//! The Evaluate dispatcher's tests (on the grid test kernel).

use super::*;

#[test]
fn mass_properties_of_a_part_list_its_material_and_mass() {
    let text = "runmat 0.7\nstd 1\n\npart block {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [100 mm, 0 mm])\n        l2 = line(start = [100 mm, 0 mm], end = [100 mm, 10 mm])\n        l3 = line(start = [100 mm, 10 mm], end = [0 mm, 10 mm])\n        l4 = line(start = [0 mm, 10 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    output block = { body = feature.body.body, material = \"material.aluminum\" }\n}\n";
    let out: Value = serde_json::from_str(&analyze_json(
        "/p/block.part",
        text,
        &Files::new(),
        &json!({ "command": "evaluate.mass-properties" }).to_string(),
    ))
    .unwrap();
    assert_eq!(out["ok"], true, "{out}");
    assert!(
        out["summary"].as_str().unwrap().starts_with("27.0 g"),
        "{out}"
    );
    assert_eq!(out["rows"][0]["cells"][2], "Aluminum");
    let refused: Value = serde_json::from_str(&analyze_json(
        "/p/block.part",
        text,
        &Files::new(),
        &json!({ "command": "evaluate.interference" }).to_string(),
    ))
    .unwrap();
    assert_eq!(refused["ok"], false);
}

/// A 200 × 100 mm frame of 20 mm square tube, mitred, written by the printer.
pub(crate) fn frame() -> String {
    use runmat_cad::ids::{FeatureId, SketchEntityId};
    use runmat_cad::model::weldment::*;
    use runmat_cad::model::*;
    use runmat_cad::value::ValueExpr;
    let mut d = PartDocument::new("/p/frame.part");
    let mut s = Sketch {
        plane: PlaneRef::origin_xy(),
        entities: Vec::new(),
        constraints: Vec::new(),
        solved: Default::default(),
    };
    let p = [(0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)];
    let names = ["bottom", "right", "top", "left"];
    for k in 0..4 {
        s.entities.push(SketchEntity {
            id: SketchEntityId::new(names[k]),
            geometry: SketchGeometry::Line {
                start: PointExpr::mm(p[k].0, p[k].1),
                end: PointExpr::mm(p[(k + 1) % 4].0, p[(k + 1) % 4].1),
            },
            construction: false,
            projected: None,
        });
    }
    d.features
        .push(Feature::new("layout", FeatureKind::Sketch(s)));
    d.features.push(Feature::new(
        "members",
        FeatureKind::Weldment(WeldmentFeature::StructuralMember {
            groups: vec![MemberGroup {
                sketch: FeatureId::new("layout"),
                segments: names.iter().map(|n| SketchEntityId::new(*n)).collect(),
                profile: MemberProfile::RectTube {
                    width: ValueExpr::mm(20.0),
                    height: ValueExpr::mm(20.0),
                    wall: ValueExpr::mm(2.0),
                },
                material: Some("material.steel".into()),
                corner: CornerTreatment::Miter,
                rotation: ValueExpr::deg(0.0),
            }],
        }),
    ));
    runmat_cad::syntax::print_part(&d, &[]).unwrap()
}

#[test]
fn draft_analysis_bands_a_blocks_faces_against_the_pull() {
    // A 100 x 10 x 10 mm block pulled along +Z: the top releases, the bottom from the
    // other half, the four walls need draft.
    let text = "runmat 0.7\nstd 1\n\npart block {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [100 mm, 0 mm])\n        l2 = line(start = [100 mm, 0 mm], end = [100 mm, 10 mm])\n        l3 = line(start = [100 mm, 10 mm], end = [0 mm, 10 mm])\n        l4 = line(start = [0 mm, 10 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    output block = { body = feature.body.body }\n}\n";
    let out: Value = serde_json::from_str(&analyze_json(
        "/p/block.part",
        text,
        &Files::new(),
        &json!({ "command": "evaluate.draft", "values": { "direction": ["origin:top"], "angle": 2 } }).to_string(),
    ))
    .unwrap();
    assert_eq!(out["ok"], true, "{out}");
    assert_eq!(
        out["summary"],
        "6 faces · 1 positive · 1 negative · 4 need draft · 0 straddle the parting line",
        "{out}"
    );
    assert_eq!(
        out["settings"],
        json!(["Pull: Top plane (+Z)", "Draft angle: 2°"]),
        "{out}"
    );
    // The pull arrow stands on the block's top, along +Z.
    assert_eq!(out["arrow"]["direction"], json!([0.0, 0.0, 1.0]), "{out}");
    assert!(
        (out["arrow"]["origin"][2].as_f64().unwrap() - 0.01).abs() < 1e-6,
        "{out}"
    );
    // Faces read as labels; the canonical reference rides along for the tooltip and copy.
    let top = out["rows"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["reference"] == "feature.body.face.end")
        .expect("the top face by name");
    assert_eq!(top["cells"][0], "Body end face");
    assert_eq!(top["cells"][1], "90.0°");
    let wall = out["rows"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["reference"] == "feature.body.face.side(s1.l2)")
        .unwrap();
    assert_eq!(wall["cells"][0], "Body side 2");
    assert_eq!(wall["cells"][3], "Needs draft");
    assert_eq!(wall["severity"], "warning");
    // Each row's marker takes its band's legend entry, as the model does.
    for row in out["rows"].as_array().unwrap() {
        let entry = row["legend"].as_u64().unwrap() as usize;
        assert_eq!(
            out["colors"]["legend"][entry]["label"], row["cells"][3],
            "{row}"
        );
    }
    // The model takes the band colors: every face, the top green, with a four-entry legend.
    let colors = &out["colors"];
    assert_eq!(
        colors["legend"].as_array().map(Vec::len),
        Some(4),
        "{colors}"
    );
    assert_eq!(
        colors["faces"].as_array().map(Vec::len),
        Some(6),
        "{colors}"
    );
    let entry = colors["faces"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["item"] == "feature.body.face.end")
        .unwrap()["entry"]
        .as_u64()
        .unwrap();
    assert_eq!(colors["legend"][entry as usize]["label"], "Positive");
    // A face picks the pull too: the top's normal is +Z.
    let by_face: Value = serde_json::from_str(&analyze_json(
        "/p/block.part",
        text,
        &Files::new(),
        &json!({ "command": "evaluate.draft", "values": { "direction": ["feature.body.face.end"], "angle": 2 } }).to_string(),
    ))
    .unwrap();
    assert_eq!(by_face["summary"], out["summary"], "{by_face}");
}

#[test]
fn a_weldment_lists_its_members_by_length_and_end_angle() {
    let text = frame();
    let out: Value = serde_json::from_str(&analyze_json(
        "/p/frame.part",
        &text,
        &Files::new(),
        &json!({ "command": "weldment.cut-list" }).to_string(),
    ))
    .unwrap();
    assert_eq!(out["ok"], true, "{out}\n{text}");
    assert_eq!(out["rows"].as_array().unwrap().len(), 2, "{out}");
    assert_eq!(out["summary"], "2 items · 4 members");
    // The long side: centre line plus half the tube's depth at each mitre.
    assert_eq!(out["rows"][0]["cells"][4], "220.00 mm");
    assert_eq!(out["rows"][0]["cells"][5], "45.0°");
}

fn sketch_check(sketch: &str, values: Value) -> Value {
    let text = format!(
        "runmat 0.7\nstd 1\n\npart p {{\n    sketch s1 on origin.xy {{\n{sketch}    }}\n}}\n"
    );
    let mut values = values;
    values["sketch"] = json!("s1");
    serde_json::from_str(&analyze_json(
        "/p/p.part",
        &text,
        &Files::new(),
        &json!({ "command": "sketch.check", "values": values }).to_string(),
    ))
    .unwrap()
}

fn findings(out: &Value, kind: &str) -> usize {
    out["rows"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["cells"][0] == kind)
        .count()
}

const SQUARE: &str = "        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])\n        l2 = line(start = [20 mm, 0 mm], end = [20 mm, 20 mm])\n        l3 = line(start = [20 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n";

#[test]
fn check_sketch_passes_a_square_for_extrude_and_loft() {
    for for_ in ["extrude", "loft_profile"] {
        let out = sketch_check(SQUARE, json!({ "for": for_ }));
        assert_eq!(out["ok"], true, "{out}");
        assert!(out["rows"].as_array().unwrap().is_empty(), "{out}");
        assert!(
            out["summary"]
                .as_str()
                .unwrap()
                .contains("4 curves · 1 closed loop · ready for"),
            "{out}"
        );
    }
}

#[test]
fn check_sketch_names_an_open_end_and_the_gap_to_the_nearest_end() {
    // The last side stops 0.05 mm short of the start.
    let open = SQUARE.replace("end = [0 mm, 0 mm])", "end = [0 mm, 0.05 mm])");
    let out = sketch_check(&open, json!({}));
    assert_eq!(findings(&out, "Open end"), 2, "{out}");
    let row = &out["rows"][0];
    assert!(
        row["cells"][1]
            .as_str()
            .unwrap()
            .contains("is 0.0500 mm away"),
        "{out}"
    );
    assert!(
        out["summary"]
            .as_str()
            .unwrap()
            .contains("2 problems to fix"),
        "{out}"
    );
}

#[test]
fn check_sketch_finds_crossings_branches_and_overlaps() {
    // A bow tie: the diagonals cross.
    let bow = "        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 20 mm])\n        l2 = line(start = [20 mm, 20 mm], end = [20 mm, 0 mm])\n        l3 = line(start = [20 mm, 0 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n";
    let out = sketch_check(bow, json!({}));
    assert_eq!(findings(&out, "Crossing"), 1, "{out}");
    assert_eq!(out["rows"][0]["cells"][2], "(10.000, 10.000) mm", "{out}");
    // A fifth line from a corner: three ends meet there, and one is open.
    let branch =
        format!("{SQUARE}        l5 = line(start = [20 mm, 20 mm], end = [30 mm, 30 mm])\n");
    let out = sketch_check(&branch, json!({}));
    assert_eq!(findings(&out, "Branch"), 1, "{out}");
    assert_eq!(findings(&out, "Open end"), 1, "{out}");
    // A side drawn twice.
    let twice = format!("{SQUARE}        l5 = line(start = [5 mm, 0 mm], end = [15 mm, 0 mm])\n");
    let out = sketch_check(&twice, json!({}));
    assert!(findings(&out, "Overlap") >= 1, "{out}");
}

#[test]
fn check_sketch_for_revolve_wants_an_axis_beside_the_profile() {
    let out = sketch_check(SQUARE, json!({ "for": "revolve" }));
    assert_eq!(findings(&out, "No axis"), 1, "{out}");
    assert_eq!(out["rows"][0]["severity"], "warning");
    let beside = format!(
        "{SQUARE}        a1 = line(start = [-5 mm, 0 mm], end = [-5 mm, 20 mm], construction)\n"
    );
    let out = sketch_check(&beside, json!({ "for": "revolve" }));
    assert!(out["rows"].as_array().unwrap().is_empty(), "{out}");
    assert!(
        out["settings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s == "Axis: a1"),
        "{out}"
    );
    let through = format!(
        "{SQUARE}        a1 = line(start = [10 mm, -5 mm], end = [10 mm, 25 mm], construction)\n"
    );
    let out = sketch_check(&through, json!({ "for": "revolve" }));
    assert_eq!(findings(&out, "Crosses the axis"), 1, "{out}");
}

#[test]
fn check_sketch_for_a_sweep_path_takes_one_open_chain() {
    let path = "        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])\n        l2 = line(start = [20 mm, 0 mm], end = [20 mm, 20 mm])\n";
    let out = sketch_check(path, json!({ "for": "sweep_path" }));
    assert!(out["rows"].as_array().unwrap().is_empty(), "{out}");
    assert!(
        out["summary"]
            .as_str()
            .unwrap()
            .ends_with("ready for sweep path"),
        "{out}"
    );
    let two = format!("{path}        l3 = line(start = [40 mm, 0 mm], end = [50 mm, 0 mm])\n");
    let out = sketch_check(&two, json!({ "for": "sweep_path" }));
    assert_eq!(findings(&out, "Pieces"), 1, "{out}");
}

const BLOCK_100: &str = "runmat 0.7\nstd 1\n\npart block {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [100 mm, 0 mm])\n        l2 = line(start = [100 mm, 0 mm], end = [100 mm, 10 mm])\n        l3 = line(start = [100 mm, 10 mm], end = [0 mm, 10 mm])\n        l4 = line(start = [0 mm, 10 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    output block = { body = feature.body.body, material = \"material.aluminum\" }\n}\n";

fn run_on(files: &Files, command: &str, values: Value) -> Value {
    serde_json::from_str(&analyze_json(
        "/p/block.part",
        BLOCK_100,
        files,
        &json!({ "command": command, "values": values }).to_string(),
    ))
    .unwrap()
}

#[test]
fn dfm_rows_state_the_rule_the_limit_and_the_measured_value() {
    let out = run_on(
        &Files::new(),
        "evaluate.dfm",
        json!({ "process": "machining" }),
    );
    assert_eq!(out["ok"], true, "{out}");
    assert!(
        out["summary"].as_str().unwrap().starts_with("Machining · "),
        "{out}"
    );
    let wall = out["rows"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["cells"][0] == "Thinnest wall")
        .expect("the wall rule");
    assert_eq!(wall["cells"][2], "≥ 0.8 mm");
    assert_eq!(wall["cells"][3], "10.000 mm", "{out}");
    assert_eq!(wall["cells"][4], "Passes");
    // A company pack in the project runs as well: a 50 mm wall rule fails this 10 mm block.
    let mut files = Files::new();
    files.insert(
        "/p/company/dfm.json".into(),
        json!({ "name": "acme", "version": "1", "checks": [
            { "name": "thick_wall", "select": "part", "attribute": "min_wall", "op": "ge",
              "expected": { "is": "quantity", "text": "50 mm" }, "gate": "release" }
        ] })
        .to_string(),
    );
    let out = run_on(
        &files,
        "evaluate.dfm",
        json!({ "process": "machining", "rules": "company/dfm" }),
    );
    assert_eq!(out["ok"], true, "{out}");
    let first = &out["rows"][0];
    assert_eq!(first["cells"][2], "≥ 50 mm", "{out}");
    assert_eq!(first["cells"][3], "10.000 mm");
    assert_eq!(first["severity"], "error");
    let missing = run_on(
        &Files::new(),
        "evaluate.dfm",
        json!({ "rules": "company/none" }),
    );
    assert_eq!(missing["ok"], false);
}

#[test]
fn cost_lists_each_line_from_the_rates_entered() {
    // 100 × 10 × 10 mm aluminum (2700 kg/m³): 27 g. Machined from its own box: nothing removed.
    let out = run_on(
        &Files::new(),
        "evaluate.cost",
        json!({ "process": "machining", "quantity": 10, "currency": "EUR", "price": 10.0,
                "rate": 60.0, "setup": 50.0, "removal_rate": 5.0, "allowance": 0.0 }),
    );
    assert_eq!(out["ok"], true, "{out}");
    // 0.027 kg × 10 = 0.27; no machining time; 50 ÷ 10 = 5.
    assert_eq!(
        out["summary"], "Machining · 5.27 EUR per part · 52.70 EUR for 10",
        "{out}"
    );
    let settings: Vec<&str> = out["settings"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .collect();
    assert!(
        settings.contains(&"Material price: 10 per kg"),
        "{settings:?}"
    );
    assert!(
        settings.contains(&"Stock: 100.00 × 10.00 × 10.00 mm"),
        "{settings:?}"
    );
    // A rate the process needs, left out, is named.
    let out = run_on(
        &Files::new(),
        "evaluate.cost",
        json!({ "process": "additive", "price": 10.0, "rate": 60.0 }),
    );
    assert_eq!(out["ok"], false);
    assert_eq!(out["message"], "set the build rate");
}

const SMALL: &str = "runmat 0.7\nstd 1\n\npart small {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [10 mm, 0 mm])\n        l2 = line(start = [10 mm, 0 mm], end = [10 mm, 10 mm])\n        l3 = line(start = [10 mm, 10 mm], end = [0 mm, 10 mm])\n        l4 = line(start = [0 mm, 10 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    output small = { body = feature.body.body, material = \"material.aluminum\" }\n}\n";

const PAIR: &str = "runmat 0.7\nstd 1\n\nassembly pair {\n    use part block as big\n    use part small as a {\n        placement = { placement = offset, offset = { translation = [0 mm, 30 mm, 0 mm] } }\n    }\n    use part small as b {\n        placement = { placement = offset, offset = { translation = [20 mm, 30 mm, 0 mm] } }\n    }\n}\n";

#[test]
fn assembly_visualization_sorts_and_colors_components_by_mass_and_count() {
    let files: Files = [("/p/block.part", BLOCK_100), ("/p/small.part", SMALL), ("/p/pair.assembly", PAIR)]
        .map(|(p, t)| (p.to_owned(), t.to_owned()))
        .into_iter()
        .collect();
    let run = |by: &str| -> Value {
        serde_json::from_str(&analyze_json(
            "/p/pair.assembly",
            PAIR,
            &files,
            &json!({ "command": "assembly.visualization", "values": { "by": by } }).to_string(),
        ))
        .unwrap()
    };
    let out = run("mass");
    assert_eq!(out["ok"], true, "{out}");
    let rows = out["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 3, "{out}");
    // The 100 mm block is heaviest: first, in the top band (red).
    assert_eq!(rows[0]["cells"][0], "big", "{out}");
    let legend = out["colors"]["legend"].as_array().unwrap();
    assert_eq!(rows[0]["legend"].as_u64(), Some(legend.len() as u64 - 1), "{out}");
    assert_eq!(out["colors"]["faces"].as_array().unwrap().len(), 3, "{out}");
    assert!(out["colors"]["faces"][0]["item"].as_str().unwrap().starts_with("occurrence:"), "{out}");
    // By count: the two small blocks share a part.
    let out = run("count");
    assert_eq!(out["rows"][0]["cells"][2], "2", "{out}");
    // No component has a cost: every row reads "No value".
    let out = run("cost");
    assert_eq!(out["rows"][0]["cells"][2], "—", "{out}");
    assert_eq!(run("rebuild")["ok"], false);
}
