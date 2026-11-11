//! References the user picks, regenerated on OCCT through `runmat_cad_call`: a fillet or
//! chamfer on a face or a feature rounds its edges, a circular edge or cylindrical face gives
//! its circle's axis, a sketch centerline is a revolve axis, and a hole goes where the face was
//! clicked. Volumes are checked against closed forms.

use serde_json::{json, Value};
use std::f64::consts::PI;

fn call(function: &str, args: &[&str]) -> Value {
    let request = json!({ "fn": function, "args": args });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

const HEAD: &str = "runmat 0.7\nstd 1\n\npart plate {\n";

/// A 40 x 20 x 10 mm plate with 4 mm holes through it at (30, 10) and (20, 10).
fn plate() -> String {
    format!("{HEAD}    sketch s1 on origin.xy {{\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }}\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    feature h = hole(face = feature.body.face.end, at = {{ a = point(30 mm, 10 mm, 10 mm) }}, diameter = 4 mm, through_all)\n    feature g = hole(face = feature.body.face.end, at = {{ a = point(20 mm, 10 mm, 10 mm) }}, diameter = 4 mm, through_all)\n}}\n")
}

/// A 10 x 10 mm square 10 mm from a construction centerline, in one sketch.
fn ring() -> String {
    format!("{HEAD}    sketch s2 on origin.zx {{\n        l1 = line(start = [50 mm, 0 mm], end = [60 mm, 0 mm])\n        l2 = line(start = [60 mm, 0 mm], end = [60 mm, 10 mm])\n        l3 = line(start = [60 mm, 10 mm], end = [50 mm, 10 mm])\n        l4 = line(start = [50 mm, 10 mm], end = [50 mm, 0 mm])\n        ax = line(start = [45 mm, -5 mm], end = [45 mm, 15 mm], construction = true)\n    }}\n}}\n")
}

/// The part after the command: its text and its volume in mm³.
fn after(part: &str, command: &str, values: Value) -> (String, f64) {
    let request = json!({ "command": command, "values": values }).to_string();
    let out = call("edit", &["/p/plate.part", part, "{}", &request]);
    assert_eq!(out["ok"], true, "{command}: {out}");
    let text = out["files"][0]["text"].as_str().unwrap().to_owned();
    let compiled = call("compile", &["/p/plate.part", &text, "{}", "{}"]);
    let errors: Vec<&Value> = compiled["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|d| d["severity"] == "error")
        .collect();
    assert!(errors.is_empty(), "{command}: {errors:?}\n{text}");
    let mass = call(
        "analyze",
        &[
            "/p/plate.part",
            &text,
            "{}",
            &json!({ "command": "evaluate.mass-properties" }).to_string(),
        ],
    );
    assert_eq!(mass["ok"], true, "{mass}");
    (
        text,
        mass["data"]["total"]["volume"].as_f64().unwrap() * 1e9,
    )
}

const HOLE: f64 = PI * 4.0 * 10.0; // one 4 mm hole through 10 mm, mm³
const PLATE: f64 = 8000.0 - 2.0 * HOLE;
const END: &str = "feature.body.face.end";

#[test]
fn a_fillet_or_chamfer_on_a_face_or_feature_rounds_its_edges() {
    // The top face's edges: 120 mm of outer edge and two 4 mm rims, each losing (1 - π/4) r².
    let (text, v) = after(
        &plate(),
        "part.fillet",
        json!({ "type": "constant", "edges": [END], "radius": 1 }),
    );
    assert!(text.contains("edges = [feature.body.face.end]"), "{text}");
    let removed = PLATE - v;
    let want = (1.0 - PI / 4.0) * (120.0 + 2.0 * PI * 4.0);
    assert!(
        (removed - want).abs() < 0.15 * want,
        "removed {removed} mm³, about {want}"
    );
    // A 1 mm chamfer on the top face's outer edges and rims: r²/2 per mm of edge.
    let (_, v) = after(
        &plate(),
        "part.chamfer",
        json!({ "type": "equal", "edges": [END], "distance": 1 }),
    );
    let want = 0.5 * (120.0 + 2.0 * PI * 4.0);
    assert!(
        ((PLATE - v) - want).abs() < 0.15 * want,
        "removed {} mm³, about {want}",
        PLATE - v
    );
    // The plate feature picked: every edge it made, top and bottom and the four corners.
    let (text, v) = after(
        &plate(),
        "part.chamfer",
        json!({ "type": "equal", "edges": ["feature:body"], "distance": 1 }),
    );
    assert!(text.contains("feature.body.body"), "{text}");
    assert!(PLATE - v > 120.0, "removed {} mm³", PLATE - v);
}

#[test]
fn a_circular_edge_or_cylindrical_face_turns_a_pattern_about_its_circle() {
    // Hole g (x = 20) turned half a turn about hole h's axis (x = 30) lands at x = 40: half of it
    // is on the plate's edge.
    let rim = "edges_between(feature.h.face.wall(a), feature.body.face.end)";
    for axis in [rim, "feature.h.face.wall(a)"] {
        let (text, v) = after(
            &plate(),
            "part.pattern",
            json!({ "layout": "circular", "seeds": ["feature:g"], "axis": [axis], "instances": 2, "angle": 360 }),
        );
        let want = PLATE - HOLE / 2.0;
        assert!(
            (v - want).abs() < 0.5,
            "{axis}: {v} mm³, want {want}\n{text}"
        );
    }
}

#[test]
fn a_revolve_turns_about_a_sketch_centerline() {
    // Pappus: 100 mm² at 10 mm from the axis, a quarter turn.
    let (text, v) = after(
        &ring(),
        "part.revolve",
        json!({ "profile": ["sketch:s2"], "axis": ["sketch.s2.ax"], "op": "new", "type": "one", "angle": 90 }),
    );
    assert!(text.contains("along = sketch_line"), "{text}");
    let want = 100.0 * 10.0 * PI / 2.0;
    assert!((v - want).abs() < 0.5, "{v} mm³, want {want}\n{text}");
}

/// A 5 mm cylinder 10 mm tall on the z axis, and a 10 x 10 mm square centred 25 mm off that axis
/// in the ZX plane (25 mm whichever sketch axis is x).
fn post() -> String {
    format!("{HEAD}    sketch s0 on origin.xy {{\n        c = circle(center = [0 mm, 0 mm], radius = 5 mm)\n    }}\n    feature cyl = extrude(profile = sketch.s0, distance = 10 mm)\n    sketch s2 on origin.zx {{\n        l1 = line(start = [20 mm, 20 mm], end = [30 mm, 20 mm])\n        l2 = line(start = [30 mm, 20 mm], end = [30 mm, 30 mm])\n        l3 = line(start = [30 mm, 30 mm], end = [20 mm, 30 mm])\n        l4 = line(start = [20 mm, 30 mm], end = [20 mm, 20 mm])\n    }}\n}}\n")
}

#[test]
fn a_revolve_turns_about_a_circular_edge_on_its_circle_axis() {
    // The cylinder's top rim is a closed circle: its start and end coincide, so a start-to-end
    // axis has no direction. Its circle's axis is z. Pappus: 100 mm² at 25 mm, a quarter turn,
    // plus the cylinder.
    let rim = "edges_between(feature.cyl.face.side(s0.c), feature.cyl.face.end)";
    let (text, v) = after(
        &post(),
        "part.revolve",
        json!({ "profile": ["sketch:s2"], "axis": [rim], "op": "new", "type": "one", "angle": 90 }),
    );
    assert!(text.contains("along = edge"), "{text}");
    let want = PI * 25.0 * 10.0 + 100.0 * 25.0 * PI / 2.0;
    assert!((v - want).abs() < 0.5, "{v} mm³, want {want}\n{text}");
}

#[test]
fn a_hole_goes_where_the_face_was_clicked() {
    let (text, v) = after(
        &plate(),
        "part.hole",
        json!({ "type": "simple", "face": [END], "positions": ["point(8 mm, 6 mm, 10 mm)"], "diameter": 3, "end": "through_all" }),
    );
    assert!(text.contains("point(8 mm, 6 mm, 10 mm)"), "{text}");
    let want = PLATE - PI * 2.25 * 10.0;
    assert!((v - want).abs() < 0.5, "{v} mm³, want {want}\n{text}");
}
