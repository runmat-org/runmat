//! Evaluate commands that color a part by a value measured on its exact surfaces (thickness,
//! curvature, zebra) and Deviation's angles along edges, on OCCT through `runmat_cad_call`.

use serde_json::{json, Value};

fn call(function: &str, args: &[&str]) -> Value {
    let request = json!({ "fn": function, "args": args });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

fn analyze(path: &str, text: &str, command: &str, values: Value) -> Value {
    let request = json!({ "command": command, "values": values }).to_string();
    let out = call("analyze", &[path, text, "{}", &request]);
    assert_eq!(out["ok"], true, "{out}");
    out
}

/// A part from one sketch extruded 10 mm.
fn extruded(name: &str, sketch: &str) -> String {
    format!("runmat 0.7\nstd 1\n\npart {name} {{\n    sketch s1 on origin.xy {{\n{sketch}    }}\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    output {name} = {{ body = feature.body.body, material = \"material.aluminum\" }}\n}}\n")
}

/// A 100 × 20 mm rectangle (so the block is 100 × 20 × 10 mm).
const BLOCK_SKETCH: &str = "        l1 = line(start = [0 mm, 0 mm], end = [100 mm, 0 mm])\n        l2 = line(start = [100 mm, 0 mm], end = [100 mm, 20 mm])\n        l3 = line(start = [100 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n";

const BAR_SKETCH: &str = "        c1 = circle(center = [0 mm, 0 mm], radius = 5 mm)\n";

fn rows(out: &Value) -> &Vec<Value> {
    out["rows"].as_array().unwrap()
}

fn field_values(out: &Value) -> Vec<f64> {
    out["field"]["faces"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|f| f["values"].as_array().unwrap().iter())
        .filter_map(Value::as_f64)
        .collect()
}

/// Every field face carries a position, a normal and a value per vertex, and whole triangles.
fn assert_field_shape(out: &Value) {
    let faces = out["field"]["faces"].as_array().unwrap();
    assert!(!faces.is_empty(), "{out}");
    for f in faces {
        let n = f["values"].as_array().unwrap().len();
        assert_eq!(f["positions"].as_array().unwrap().len(), 3 * n);
        assert_eq!(f["normals"].as_array().unwrap().len(), 3 * n);
        let t = f["triangles"].as_array().unwrap();
        assert_eq!(t.len() % 3, 0);
        assert!(t.iter().all(|i| (i.as_u64().unwrap() as usize) < n));
        assert!(f["item"].as_str().is_some());
    }
}

#[test]
fn a_10_mm_block_is_10_mm_thin_through_its_top_and_100_mm_end_to_end() {
    let block = extruded("block", BLOCK_SKETCH);
    let out = analyze(
        "/p/block.part",
        &block,
        "evaluate.thickness",
        json!({ "target": 12.0, "show": "thin" }),
    );
    assert_eq!(out["title"], "Thickness");
    assert_field_shape(&out);
    assert_eq!(out["field"]["kind"], "spectrum");
    assert_eq!(out["field"]["unit"], "mm");
    let range = out["field"]["range"].as_array().unwrap();
    assert!((range[0].as_f64().unwrap() - 10.0).abs() < 1e-3, "{out}");
    assert!((range[1].as_f64().unwrap() - 100.0).abs() < 1e-3, "{out}");
    let values = field_values(&out);
    assert!(
        values
            .iter()
            .all(|v| (10.0 - 1e-3..=100.0 + 1e-3).contains(v)),
        "{values:?}"
    );
    // The top and bottom are 10 mm through: thinner than the 12 mm target.
    let thin: Vec<&Value> = rows(&out)
        .iter()
        .filter(|r| r["severity"] == "warning")
        .collect();
    assert_eq!(thin.len(), 2, "{out}");
    assert!(thin.iter().all(|r| r["cells"][1] == "10.000 mm"), "{out}");
    assert!(
        out["summary"].as_str().unwrap().contains(
            "6 faces · thinnest 10.000 mm · thickest 100.000 mm · 2 faces thinner than 12.000 mm"
        ),
        "{out}"
    );
    // Against a 50 mm target for thick regions, only the faces 100 mm apart pass it.
    let thick = analyze(
        "/p/block.part",
        &block,
        "evaluate.thickness",
        json!({ "target": 50.0, "show": "thick" }),
    );
    let flagged = rows(&thick)
        .iter()
        .filter(|r| r["severity"] == "warning")
        .count();
    assert_eq!(flagged, 2, "{thick}");
}

#[test]
fn a_5_mm_bar_is_convex_at_one_over_5_mm_and_a_5_mm_hole_concave() {
    let bar = extruded("bar", BAR_SKETCH);
    let out = analyze("/p/bar.part", &bar, "evaluate.curvature", json!({}));
    assert_field_shape(&out);
    assert!(
        out["summary"]
            .as_str()
            .unwrap()
            .contains("3 faces · 1 curved · smallest radius 5.000 mm"),
        "{out}"
    );
    let side = rows(&out)
        .iter()
        .find(|r| r["cells"][3] == "5.000 mm")
        .expect("the round face");
    assert_eq!(side["cells"][1], "0.2000 1/mm", "{out}");
    assert_eq!(side["cells"][2], "0.2000 1/mm", "{out}");
    // Flat caps measure zero.
    assert_eq!(
        rows(&out)
            .iter()
            .filter(|r| r["cells"][3] == "Flat")
            .count(),
        2
    );
    let range = out["field"]["range"].as_array().unwrap();
    assert!((range[1].as_f64().unwrap() - 0.2).abs() < 1e-6, "{out}");
    assert!((range[0].as_f64().unwrap() + 0.2).abs() < 1e-6, "{out}");

    // A plate with a 5 mm hole: the hole's wall is concave.
    let plate = extruded(
        "plate",
        &format!("{BLOCK_SKETCH}        c1 = circle(center = [50 mm, 10 mm], radius = 5 mm)\n"),
    );
    let out = analyze(
        "/p/plate.part",
        &plate,
        "evaluate.curvature",
        json!({ "measure": "mean" }),
    );
    let hole = rows(&out)
        .iter()
        .find(|r| r["cells"][3] == "5.000 mm")
        .expect("the hole's wall");
    // Mean curvature of a cylinder is half of 1/r.
    assert_eq!(hole["cells"][1], "-0.1000 1/mm", "{out}");
    assert_eq!(out["field"]["label"], "Mean curvature");
}

#[test]
fn zebra_bands_a_block_and_finds_its_12_creases() {
    let block = extruded("block", BLOCK_SKETCH);
    let out = analyze(
        "/p/block.part",
        &block,
        "evaluate.zebra",
        json!({ "stripes": 10, "direction": ["origin.xy"] }),
    );
    assert_field_shape(&out);
    assert_eq!(out["field"]["kind"], "stripes");
    assert!(field_values(&out).iter().all(|v| *v == 0.0 || *v == 1.0));
    assert_eq!(
        out["summary"],
        "12 edges between faces · 12 G0 (crease) · 0 G1 (tangent) · 0 G2 (curvature continuous)",
        "{out}"
    );
    // The top faces the light (band 0); the bottom faces away (the last band of ten, 1).
    let faces = out["field"]["faces"].as_array().unwrap();
    let band_of = |nz: f64| {
        faces
            .iter()
            .find(|f| {
                let n = f["normals"].as_array().unwrap();
                (n[2].as_f64().unwrap() - nz).abs() < 1e-6
            })
            .map(|f| f["values"][0].as_f64().unwrap())
    };
    assert_eq!(band_of(1.0), Some(0.0));
    assert_eq!(band_of(-1.0), Some(1.0));
    // A round bar's side has a seam on itself, which isn't between faces.
    let bar = extruded("bar", BAR_SKETCH);
    let out = analyze("/p/bar.part", &bar, "evaluate.zebra", json!({}));
    assert!(
        out["summary"]
            .as_str()
            .unwrap()
            .starts_with("2 edges between faces · 2 G0"),
        "{out}"
    );
}

#[test]
fn deviation_measures_90_degrees_along_a_blocks_edges() {
    let block = extruded("block", BLOCK_SKETCH);
    let out = analyze("/p/block.part", &block, "evaluate.deviation", json!({}));
    assert_eq!(out["title"], "Deviation");
    assert_eq!(
        out["summary"], "12 edges · 0 within 0.5° (tangent) · 12 over",
        "{out}"
    );
    assert!(
        rows(&out).iter().all(|r| r["cells"][2] == "90.000°"),
        "{out}"
    );
    // One picked edge.
    let edge = rows(&out)[0]["reference"].as_str().unwrap().to_string();
    let one = analyze(
        "/p/block.part",
        &block,
        "evaluate.deviation",
        json!({ "edges": [edge] }),
    );
    assert_eq!(rows(&one).len(), 1, "{one}");
}
