//! The Surfaces tab's create and edit commands on OCCT through `runmat_cad_call`: extruded,
//! revolved, swept, planar, ruled and offset surfaces, extend, delete hole, cut with surface and
//! mid-surface. Each command writes its `surface.*` feature from the form's values, the part
//! regenerates, and the sheet it makes is thickened 1 mm about itself, so the solid's volume in
//! mm³ is the sheet's area in mm².

use serde_json::{json, Value};

fn call(function: &str, args: &[&str]) -> Value {
    runmat_geometry_wasm_host::cad::install_occt();
    let request = json!({ "fn": function, "args": args });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

fn part(body: &str) -> String {
    format!("runmat 0.7\nstd 1\n\npart skin {{\n{body}}}\n")
}

/// A closed 40 x 20 mm rectangle on the XY plane, its lines a (bottom), b, c, e.
const RECT: &str = "    sketch s on origin.xy {\n        a = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        b = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        c = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        e = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n";

/// Runs the command on `text`: the new text and the name of the feature it added.
fn edit(text: &str, command: &str, values: Value) -> (String, String) {
    let request = json!({ "command": command, "values": values }).to_string();
    let out = call("edit", &["/p/skin.part", text, "{}", &request]);
    assert_eq!(out["ok"], true, "{command}: {out}\n{text}");
    let next = out["files"][0]["text"].as_str().unwrap().to_owned();
    let added = out["summary"]
        .as_str()
        .unwrap()
        .trim_start_matches("Added ")
        .to_owned();
    (next, added)
}

/// The solids' volume in mm³ after the text compiles without errors.
fn volume(text: &str) -> f64 {
    let compiled = call("compile", &["/p/skin.part", text, "{}", "{}"]);
    let errors: Vec<&Value> = compiled["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|d| d["severity"] == "error")
        .collect();
    assert!(errors.is_empty(), "{errors:?}\n{text}");
    let mass = call(
        "analyze",
        &[
            "/p/skin.part",
            text,
            "{}",
            &json!({ "command": "evaluate.mass-properties" }).to_string(),
        ],
    );
    assert_eq!(mass["ok"], true, "{mass}");
    mass["data"]["total"]["volume"].as_f64().unwrap() * 1e9
}

/// The volume after thickening `body` 1 mm centred on the sheet: its area in mm², plus any
/// solid already in the part.
fn area(text: &str, body: &str) -> f64 {
    let (text, _) = edit(
        text,
        "surface.thicken",
        json!({ "surface": [body], "thickness": 1, "side": "both" }),
    );
    volume(&text)
}

fn near(got: f64, want: f64, what: &str) {
    assert!(
        (got - want).abs() < 0.02 * want,
        "{what}: {got:.1}, want {want:.1}"
    );
}

#[test]
fn extruded_revolved_swept_and_planar_surfaces_make_sheets_of_their_curves() {
    // An open 40 mm line extruded 20 mm.
    let line = part("    sketch s on origin.xy {\n        a = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n    }\n");
    let (text, f) = edit(
        &line,
        "surface.extrude",
        json!({ "profile": ["sketch:s"], "depth": 20 }),
    );
    assert!(text.contains("surface.extrude("), "{text}");
    near(
        area(&text, &format!("feature.{f}.body")),
        800.0,
        "extruded surface",
    );
    let (text, f) = edit(
        &line,
        "surface.extrude",
        json!({ "profile": ["sketch:s"], "depth": 20, "symmetric": true }),
    );
    near(
        area(&text, &format!("feature.{f}.body")),
        800.0,
        "extruded surface, both sides",
    );
    // A 20 mm line 10 mm off the Y axis, revolved a full turn: a 10 mm radius tube wall.
    let wall = part("    sketch s on origin.xy {\n        a = line(start = [10 mm, 0 mm], end = [10 mm, 20 mm])\n    }\n");
    let (text, f) = edit(
        &wall,
        "surface.revolve",
        json!({ "profile": ["sketch:s"], "axis": ["origin:y"], "angle": 360 }),
    );
    assert!(text.contains("surface.revolve("), "{text}");
    near(
        area(&text, &format!("feature.{f}.body")),
        2.0 * std::f64::consts::PI * 10.0 * 20.0,
        "revolved surface",
    );
    // A 20 mm line on the ZX plane swept 30 mm along Y.
    let swept = part("    sketch p on origin.zx {\n        a = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])\n    }\n    sketch q on origin.xy {\n        a = line(start = [0 mm, 0 mm], end = [0 mm, 30 mm])\n    }\n");
    let (text, f) = edit(
        &swept,
        "surface.sweep",
        json!({ "profile": ["sketch:p"], "path": ["sketch:q"] }),
    );
    assert!(text.contains("surface.sweep("), "{text}");
    near(
        area(&text, &format!("feature.{f}.body")),
        600.0,
        "swept surface",
    );
    // The rectangle filled flat.
    let (text, f) = edit(
        &part(RECT),
        "surface.planar",
        json!({ "profile": ["sketch:s"] }),
    );
    assert!(text.contains("surface.planar("), "{text}");
    near(
        area(&text, &format!("feature.{f}.body")),
        800.0,
        "planar surface",
    );
}

#[test]
fn ruled_offset_and_extended_surfaces_grow_from_a_sheet() {
    let (patch, p) = edit(
        &part(RECT),
        "surface.planar",
        json!({ "profile": ["sketch:s"] }),
    );
    let edge = |e: &str| format!("feature.{p}.edge.boundary(s.{e})");
    // A 10 mm strip off the 40 mm bottom edge, along the patch's normal.
    let (text, f) = edit(
        &patch,
        "surface.ruled",
        json!({ "edges": [edge("a")], "distance": 10, "angle": 0 }),
    );
    assert!(text.contains("surface.ruled("), "{text}");
    near(
        area(&text, &format!("feature.{f}.body")),
        400.0,
        "ruled surface",
    );
    // The patch copied 5 mm along its normal: a second 800 mm² sheet.
    let (text, f) = edit(
        &patch,
        "surface.offset",
        json!({ "faces": [format!("feature.{p}.face.patch")], "distance": 5 }),
    );
    assert!(text.contains("surface.offset("), "{text}");
    near(
        area(&text, &format!("feature.{f}.body")),
        800.0,
        "offset surface",
    );
    // The 20 mm right edge pushed 10 mm out: the sheet becomes 50 x 20 mm.
    let (text, _) = edit(
        &patch,
        "surface.extend",
        json!({ "edges": [edge("b")], "distance": 10 }),
    );
    assert!(text.contains("surface.extend("), "{text}");
    near(
        area(&text, &format!("feature.{p}.body")),
        1000.0,
        "extended surface",
    );
}

#[test]
fn delete_hole_closes_a_sheet_and_cut_with_surface_keeps_one_side() {
    // The rectangle with a 5 mm radius hole, then the hole closed.
    let holed = part(&RECT.replace(
        "    }\n",
        "        h = circle(center = [20 mm, 10 mm], radius = 5 mm)\n    }\n",
    ));
    let (patch, p) = edit(&holed, "surface.planar", json!({ "profile": ["sketch:s"] }));
    near(
        area(&patch, &format!("feature.{p}.body")),
        800.0 - std::f64::consts::PI * 25.0,
        "holed patch",
    );
    let (text, _) = edit(
        &patch,
        "surface.delete-hole",
        json!({ "edges": [format!("feature.{p}.edge.boundary(s.h)")] }),
    );
    assert!(text.contains("surface.delete_hole("), "{text}");
    near(
        area(&text, &format!("feature.{p}.body")),
        800.0,
        "hole deleted",
    );
    // A 40 x 20 x 10 mm block cut by a wall standing across it at y = 5 mm: one side keeps
    // 40 x 5 x 10 mm, the other 40 x 15 x 10 mm.
    let block = part(&format!(
        "{RECT}    feature box = extrude(profile = sketch.s, distance = 10 mm)\n    sketch w on origin.xy {{\n        a = line(start = [-10 mm, 5 mm], end = [50 mm, 5 mm])\n    }}\n    feature wall = surface.extrude(profile = sketch.w, start = -10 mm, end = 20 mm)\n"
    ));
    near(volume(&block), 8000.0, "the block");
    let mut kept = Vec::new();
    for keep in ["positive", "negative"] {
        let (text, _) = edit(
            &block,
            "surface.cut",
            json!({ "surface": ["feature.wall.body"], "body": ["feature.box.body"], "keep": keep }),
        );
        assert!(text.contains("surface.cut_with_surface("), "{text}");
        kept.push(volume(&text));
    }
    kept.sort_by(f64::total_cmp);
    near(kept[0], 2000.0, "the small side");
    near(kept[1], 6000.0, "the large side");
}

#[test]
fn a_mid_surface_lies_between_opposite_faces() {
    // A 4 mm plate: its mid-surface is a 40 x 20 mm sheet.
    let plate = part(&format!(
        "{RECT}    feature plate = extrude(profile = sketch.s, distance = 4 mm)\n"
    ));
    let (text, f) = edit(
        &plate,
        "surface.mid-surface",
        json!({ "pairs": ["feature.plate.face.start", "feature.plate.face.end"] }),
    );
    assert!(text.contains("surface.mid_surface("), "{text}");
    near(
        area(&text, &format!("feature.{f}.body")),
        3200.0 + 800.0,
        "plate and its mid-surface",
    );
    // An odd count of faces isn't a set of pairs.
    let request = json!({ "command": "surface.mid-surface", "values": { "pairs": ["feature.plate.face.start"] } }).to_string();
    let out = call("edit", &["/p/skin.part", &plate, "{}", &request]);
    assert_eq!(out["ok"], false, "{out}");
}
