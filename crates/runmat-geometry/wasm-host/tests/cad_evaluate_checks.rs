//! Evaluate commands that inspect a part's exact geometry on OCCT, through `runmat_cad_call`
//! (the Desktop worker's entry): Check geometry and Section properties.

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

/// A part from one closed sketch extruded 10 mm.
fn extruded(name: &str, sketch: &str) -> String {
    format!("runmat 0.7\nstd 1\n\npart {name} {{\n    sketch s1 on origin.xy {{\n{sketch}    }}\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    output {name} = {{ body = feature.body.body, material = \"material.aluminum\" }}\n}}\n")
}

const BLOCK_SKETCH: &str = "        l1 = line(start = [0 mm, 0 mm], end = [100 mm, 0 mm])\n        l2 = line(start = [100 mm, 0 mm], end = [100 mm, 20 mm])\n        l3 = line(start = [100 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n";

fn findings(out: &Value, problem: &str) -> Vec<Value> {
    out["data"]["findings"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|f| f["problem"] == problem)
        .cloned()
        .collect()
}

#[test]
fn a_clean_block_checks_with_no_problems_strict_or_not() {
    let block = extruded("block", BLOCK_SKETCH);
    for strict in [false, true] {
        let out = analyze(
            "/p/block.part",
            &block,
            "evaluate.check-geometry",
            json!({ "strict": strict }),
        );
        assert_eq!(out["title"], "Check geometry");
        assert_eq!(
            out["summary"], "1 body checked · no problems found",
            "{out}"
        );
        assert!(out.get("colors").is_none(), "{out}");
        // Facts only: the largest gaps, on a block with no curved face (no smallest radius).
        assert!(findings(&out, "smallest_radius").is_empty(), "{out}");
        assert_eq!(findings(&out, "largest_edge_gap").len(), 1, "{out}");
        let settings: Vec<&str> = out["settings"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert_eq!(
            settings[0],
            if strict {
                "Check: strict"
            } else {
                "Check: standard"
            }
        );
    }
}

#[test]
fn a_five_micron_step_is_reported_as_short_edges_on_named_edges() {
    let stepped = extruded(
        "stepped",
        "        l1 = line(start = [0 mm, 0 mm], end = [100 mm, 0 mm])\n        l2 = line(start = [100 mm, 0 mm], end = [100 mm, 0.005 mm])\n        l3 = line(start = [100 mm, 0.005 mm], end = [100.005 mm, 0.005 mm])\n        l4 = line(start = [100.005 mm, 0.005 mm], end = [100.005 mm, 20 mm])\n        l5 = line(start = [100.005 mm, 20 mm], end = [0 mm, 20 mm])\n        l6 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n",
    );
    let out = analyze(
        "/p/stepped.part",
        &stepped,
        "evaluate.check-geometry",
        json!({ "short_edge": 0.01 }),
    );
    let short = findings(&out, "short_edge");
    // The step's two 5 µm faces each leave a short edge at the bottom and the top.
    assert_eq!(short.len(), 4, "{out}");
    for f in &short {
        assert!((f["value"].as_f64().unwrap() - 5e-6).abs() < 1e-9, "{f}");
        assert!(
            f["reference"].as_str().is_some_and(|r| !r.is_empty()),
            "an edge without a name: {f}"
        );
    }
    assert!(
        out["summary"]
            .as_str()
            .unwrap()
            .contains("0 errors · 4 warnings"),
        "{out}"
    );
    let row = out["rows"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["cells"][2] == "Short edge")
        .unwrap();
    assert_eq!(row["severity"], "warning");
    assert_eq!(row["cells"][3], "0.00500 mm");
    assert_eq!(row["items"][0], row["reference"]);
    // With short edges off, the same part checks clean.
    let off = analyze(
        "/p/stepped.part",
        &stepped,
        "evaluate.check-geometry",
        json!({ "short_edges": false }),
    );
    assert!(findings(&off, "short_edge").is_empty(), "{off}");
    assert_eq!(off["summary"], "1 body checked · no problems found");
}

#[test]
fn a_round_bar_reports_its_radius_as_the_smallest() {
    let bar = extruded(
        "bar",
        "        c1 = circle(center = [0 mm, 0 mm], radius = 5 mm)\n",
    );
    let out = analyze("/p/bar.part", &bar, "evaluate.check-geometry", json!({}));
    let radius = findings(&out, "smallest_radius");
    assert_eq!(radius.len(), 1, "{out}");
    assert!(
        (radius[0]["value"].as_f64().unwrap() - 0.005).abs() < 1e-7,
        "{out}"
    );
    let row = out["rows"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["cells"][2] == "Smallest radius")
        .unwrap();
    assert_eq!(row["cells"][3], "R 5.000 mm");
    assert_eq!(row["severity"], "info");
    let off = analyze(
        "/p/bar.part",
        &bar,
        "evaluate.check-geometry",
        json!({ "min_radius": false }),
    );
    assert!(findings(&off, "smallest_radius").is_empty(), "{off}");
}

fn section(out: &Value) -> &Value {
    &out["data"]["section"]
}

fn near(v: &Value, want: f64, rel: f64) {
    let got = v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"));
    assert!(
        (got - want).abs() <= rel * want.abs().max(1e-30),
        "got {got}, want {want}"
    );
}

#[test]
fn the_end_face_of_a_100_by_20_block_has_bh3_over_12() {
    let block = extruded("block", BLOCK_SKETCH);
    let out = analyze(
        "/p/block.part",
        &block,
        "evaluate.section-properties",
        json!({ "section": ["feature.body.face.end"] }),
    );
    let s = section(&out);
    near(&s["area"], 2000e-6, 1e-9);
    // x along world X, y along world Y: Ixx = ∫ y² = 100·20³/12 mm⁴, Iyy = 20·100³/12 mm⁴.
    near(&s["ixx"], 100.0 * 20f64.powi(3) / 12.0 * 1e-12, 1e-9);
    near(&s["iyy"], 20.0 * 100f64.powi(3) / 12.0 * 1e-12, 1e-9);
    assert!(s["ixy"].as_f64().unwrap().abs() < 1e-18, "{out}");
    near(&s["centroid"][0], 0.05, 1e-9);
    near(&s["centroid"][2], 0.01, 1e-9);
    assert_eq!(
        out["rows"][0]["cells"],
        json!(["Area", "2000.000 mm²"]),
        "{out}"
    );
    assert_eq!(
        out["colors"]["faces"][0]["item"], "feature.body.face.end",
        "{out}"
    );
    let settings: Vec<&str> = out["settings"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .collect();
    assert!(settings.contains(&"x axis (1, 0, 0)"), "{settings:?}");
}

#[test]
fn a_circle_sketch_has_pi_r4_over_4_and_a_side_face_is_refused() {
    let bar = extruded(
        "bar",
        "        c1 = circle(center = [0 mm, 0 mm], radius = 5 mm)\n",
    );
    let out = analyze(
        "/p/bar.part",
        &bar,
        "evaluate.section-properties",
        json!({ "section": ["sketch.s1"] }),
    );
    let s = section(&out);
    let r = 0.005f64;
    near(&s["area"], std::f64::consts::PI * r * r, 1e-5);
    near(&s["ixx"], std::f64::consts::PI * r.powi(4) / 4.0, 1e-5);
    near(&s["polar"], std::f64::consts::PI * r.powi(4) / 2.0, 1e-5);
    assert!(
        out["summary"]
            .as_str()
            .unwrap()
            .starts_with("Sketch s1 · area 78.540 mm² · J 981.7"),
        "{out}"
    );
    // The bar's round side isn't planar.
    let request = json!({ "command": "evaluate.section-properties", "values": { "section": ["feature.body.face.side(s1.c1)"] } }).to_string();
    let refused = call("analyze", &["/p/bar.part", &bar, "{}", &request]);
    assert_eq!(refused["ok"], false, "{refused}");
    assert!(
        refused["message"].as_str().unwrap().contains("planar"),
        "{refused}"
    );
}
