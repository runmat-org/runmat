//! Convert to sheet metal on OCCT through `runmat_cad_call`: fixed faces with holes and with
//! circular arcs in their outline, rolled walls refused by name, and walls meeting at a concave
//! corner. Volumes and extents check the geometry.

use serde_json::{json, Value};

fn call(function: &str, args: &[&str]) -> Value {
    runmat_geometry_wasm_host::cad::install_occt();
    let request = json!({ "fn": function, "args": args });
    serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).expect("json")
}

fn edit(text: &str, command: &str, values: Value) -> String {
    let request = json!({ "command": command, "values": values }).to_string();
    let out = call("edit", &["/p/sheet.part", text, "{}", &request]);
    assert_eq!(out["ok"], true, "{command}: {out}\n{text}");
    out["files"][0]["text"].as_str().unwrap().to_owned()
}

fn diagnostics(text: &str) -> Vec<String> {
    let compiled = call("compile", &["/p/sheet.part", text, "{}", "{}"]);
    compiled["diagnostics"].as_array().into_iter().flatten().filter(|d| d["severity"] == "error").filter_map(|d| d["message"].as_str().map(str::to_owned)).collect()
}

/// The compiled part, which must have no errors.
fn compile(text: &str) -> Value {
    let compiled = call("compile", &["/p/sheet.part", text, "{}", "{}"]);
    let errors: Vec<&Value> = compiled["diagnostics"].as_array().into_iter().flatten().filter(|d| d["severity"] == "error").collect();
    assert!(errors.is_empty(), "{errors:?}\n{text}");
    compiled
}

fn volume(text: &str) -> f64 {
    compile(text);
    let mass = call("analyze", &["/p/sheet.part", text, "{}", &json!({ "command": "evaluate.mass-properties" }).to_string()]);
    assert_eq!(mass["ok"], true, "{mass}");
    mass["data"]["total"]["volume"].as_f64().unwrap() * 1e9
}

fn extents(text: &str) -> [f64; 3] {
    let b = &compile(text)["display"]["data"]["bounds"];
    [0, 1, 2].map(|k| (b["max"][k].as_f64().unwrap() - b["min"][k].as_f64().unwrap()) * 1e3)
}

fn near(got: f64, want: f64, tol: f64, what: &str) {
    assert!((got - want).abs() <= tol, "{what}: got {got}, want {want}");
}

fn face_names(text: &str) -> Vec<String> {
    let compiled = compile(text);
    compiled["display"]["data"]["bodies"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|b| b["faces"].as_array().cloned().unwrap_or_default())
        .filter_map(|e| e["reference"].as_str().map(str::to_owned))
        .collect()
}

/// The command's text without regenerating it (for parts whose conversion is refused).
fn edit_unchecked(text: &str, command: &str, values: Value) -> String {
    let request = json!({ "command": command, "values": values }).to_string();
    let out = call("edit", &["/p/sheet.part", text, "{}", &request]);
    assert_eq!(out["ok"], true, "{command}: {out}");
    out["files"][0]["text"].as_str().unwrap_or_default().to_owned()
}

const HEAD: &str = "runmat 0.7\nstd 1\n\npart sheet {\n";
const BOX: &str = "    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])\n        l2 = line(start = [80 mm, 0 mm], end = [80 mm, 50 mm])\n        l3 = line(start = [80 mm, 50 mm], end = [0 mm, 50 mm])\n        l4 = line(start = [0 mm, 50 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 25 mm)\n    feature hollow = shell(body = feature.body.body, open_faces = [feature.body.face.end], thickness = 1.5 mm)\n";

/// An 80 x 50 mm rectangle with 8 mm round corners.
const ROUNDED: &str = "    sketch s1 on origin.xy {\n        l1 = line(start = [8 mm, 0 mm], end = [72 mm, 0 mm])\n        a1 = arc(center = [72 mm, 8 mm], start = [72 mm, 0 mm], end = [80 mm, 8 mm])\n        l2 = line(start = [80 mm, 8 mm], end = [80 mm, 42 mm])\n        a2 = arc(center = [72 mm, 42 mm], start = [80 mm, 42 mm], end = [72 mm, 50 mm])\n        l3 = line(start = [72 mm, 50 mm], end = [8 mm, 50 mm])\n        a3 = arc(center = [8 mm, 42 mm], start = [8 mm, 50 mm], end = [0 mm, 42 mm])\n        l4 = line(start = [0 mm, 42 mm], end = [0 mm, 8 mm])\n        a4 = arc(center = [8 mm, 8 mm], start = [0 mm, 8 mm], end = [8 mm, 0 mm])\n    }\n";

fn part(body: &str) -> String {
    format!("{HEAD}{body}}}\n")
}

fn convert(text: &str) -> String {
    edit(text, "sheet-metal.convert", json!({ "fixed": ["feature.body.face.start"], "bends": [], "thickness": 1.5, "radius": 1 }))
}

/// The shelled box with a round hole and a square hole through its floor converts with both
/// holes: the converted part and its blank each lose the holes' area through the sheet, and
/// the flat pattern keeps them.
#[test]
fn a_floor_with_holes_converts_and_unfolds_with_its_holes() {
    let (t, r, k, h) = (1.5, 1.0, 0.44, 25.0);
    let holes = "    sketch h on origin.xy {\n        c = circle(center = [40 mm, 25 mm], radius = 5 mm)\n        q1 = line(start = [10 mm, 10 mm], end = [20 mm, 10 mm])\n        q2 = line(start = [20 mm, 10 mm], end = [20 mm, 18 mm])\n        q3 = line(start = [20 mm, 18 mm], end = [10 mm, 18 mm])\n        q4 = line(start = [10 mm, 18 mm], end = [10 mm, 10 mm])\n    }\n    feature holes = extrude(profile = sketch.h, distance = 10 mm, symmetric, mode = cut)\n";
    let text = convert(&part(&format!("{BOX}{holes}")));
    let setback = r + t;
    let flat = h - setback;
    let spans = 2.0 * (80.0 - 2.0 * setback) + 2.0 * (50.0 - 2.0 * setback);
    let corners = 4.0 * setback * setback;
    let cut = std::f64::consts::PI * 25.0 + 80.0;
    let floor = t * (80.0 * 50.0 - setback * spans - corners - cut);
    let bend = std::f64::consts::FRAC_PI_4 * ((r + t) * (r + t) - r * r);
    near(volume(&text), floor + spans * (bend + t * flat), 0.5, "converted");
    let fixed = face_names(&text).into_iter().find(|n| n.contains("sheet(plate, ordinal = 1)")).expect("the floor's inner face");
    let unfolded = edit(&text, "sheet-metal.unfold", json!({ "fixed": [fixed] }));
    let allowance = std::f64::consts::FRAC_PI_2 * (r + k * t);
    near(volume(&unfolded), t * (80.0 * 50.0 - corners - cut + spans * (allowance - setback + flat)), 0.5, "the blank");
    let folded = edit(&unfolded, "sheet-metal.fold", json!({}));
    near(volume(&folded), floor + spans * (bend + t * flat), 0.5, "folded back");
    let flat_pattern = edit(&text, "sheet-metal.flat-pattern", json!({}));
    compile(&flat_pattern);
}

/// A hole that reaches into a bend zone is refused by its place and the edge it meets.
#[test]
fn a_hole_in_a_bend_zone_is_refused_by_name() {
    let holes = "    sketch h on origin.xy {\n        c = circle(center = [3 mm, 25 mm], radius = 1 mm)\n    }\n    feature holes = extrude(profile = sketch.h, distance = 10 mm, symmetric, mode = cut)\n";
    let text = edit_unchecked(&part(&format!("{BOX}{holes}")), "sheet-metal.convert", json!({ "fixed": ["feature.body.face.start"], "bends": [], "thickness": 1.5, "radius": 1 }));
    let messages = diagnostics(&text);
    assert!(messages.iter().any(|m| m.contains("the hole at (3.00, 25.00") && m.contains("reaches into the bend at outline edge")), "{messages:?}");
}

/// A plate with round corners and a wall on each long side: the arcs stay in the outline, flat
/// and folded, and only the walls bend.
#[test]
fn a_plate_with_round_corners_converts_its_straight_walls() {
    let (t, r, k, h) = (1.5, 1.0, 0.44, 25.0);
    let walls = "    feature body = extrude(profile = sketch.s1, distance = 1.5 mm)\n    sketch w on origin.xy {\n        a1 = line(start = [8 mm, 0 mm], end = [72 mm, 0 mm])\n        a2 = line(start = [72 mm, 0 mm], end = [72 mm, 1.5 mm])\n        a3 = line(start = [72 mm, 1.5 mm], end = [8 mm, 1.5 mm])\n        a4 = line(start = [8 mm, 1.5 mm], end = [8 mm, 0 mm])\n        b1 = line(start = [8 mm, 48.5 mm], end = [72 mm, 48.5 mm])\n        b2 = line(start = [72 mm, 48.5 mm], end = [72 mm, 50 mm])\n        b3 = line(start = [72 mm, 50 mm], end = [8 mm, 50 mm])\n        b4 = line(start = [8 mm, 50 mm], end = [8 mm, 48.5 mm])\n    }\n    feature walls = extrude(profile = sketch.w, distance = 25 mm, mode = add)\n";
    let text = convert(&part(&format!("{ROUNDED}{walls}")));
    let e = extents(&text);
    near(e[0], 80.0, 1e-3, "x");
    near(e[1], 50.0, 1e-3, "y");
    near(e[2], h, 1e-3, "z");
    let setback = r + t;
    let flat = h - setback;
    let span = 64.0;
    let area = 80.0 * 50.0 - (4.0 - std::f64::consts::PI) * 64.0;
    let bend = std::f64::consts::FRAC_PI_4 * ((r + t) * (r + t) - r * r);
    near(volume(&text), t * (area - 2.0 * span * setback) + 2.0 * span * (bend + t * flat), 0.5, "converted");
    let fixed = face_names(&text).into_iter().find(|n| n.contains("sheet(plate, ordinal = 1)")).expect("the plate's inner face");
    let unfolded = edit(&text, "sheet-metal.unfold", json!({ "fixed": [fixed] }));
    let allowance = std::f64::consts::FRAC_PI_2 * (r + k * t);
    near(volume(&unfolded), t * (area + 2.0 * span * (allowance - setback + flat)), 0.5, "the blank");
}

/// A shelled box with round corners has rolled walls there: a wall along an arc has no flat
/// blank, so the conversion names the arc.
#[test]
fn a_rolled_wall_is_refused_by_name() {
    let body = format!("{ROUNDED}    feature body = extrude(profile = sketch.s1, distance = 25 mm)\n    feature hollow = shell(body = feature.body.body, open_faces = [feature.body.face.end], thickness = 1.5 mm)\n");
    let text = edit_unchecked(&part(&body), "sheet-metal.convert", json!({ "fixed": ["feature.body.face.start"], "bends": [], "thickness": 1.5, "radius": 1 }));
    let messages = diagnostics(&text);
    assert!(messages.iter().any(|m| m.contains("is a circular arc (radius 8.00 mm)") && m.contains("outline edge")), "{messages:?}");
}

/// An L-shaped tray: the walls on both sides of its concave corner meet on the corner's
/// bisecting plane, so the converted tray keeps its size; its blank would need the same sheet
/// twice at that corner, and the flat pattern says so.
#[test]
fn walls_at_a_concave_corner_convert_and_their_blank_is_refused() {
    let l = "    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [60 mm, 0 mm])\n        l2 = line(start = [60 mm, 0 mm], end = [60 mm, 30 mm])\n        l3 = line(start = [60 mm, 30 mm], end = [30 mm, 30 mm])\n        l4 = line(start = [30 mm, 30 mm], end = [30 mm, 60 mm])\n        l5 = line(start = [30 mm, 60 mm], end = [0 mm, 60 mm])\n        l6 = line(start = [0 mm, 60 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 20 mm)\n    feature hollow = shell(body = feature.body.body, open_faces = [feature.body.face.end], thickness = 1.5 mm)\n";
    let text = convert(&part(l));
    let e = extents(&text);
    near(e[0], 60.0, 1e-3, "x");
    near(e[1], 60.0, 1e-3, "y");
    near(e[2], 20.0, 1e-3, "z");
    // The floor is the L inset by the setback (2.5 mm). Each wall is a bend and a 17.5 mm flat,
    // stopping a setback short of each of the five convex corners (215 mm of wall in all). At
    // the concave corner both walls run on past it to the bisecting plane: each gains its
    // section's first moment about the outer face, d A, with the flat's 1.5 mm x 17.5 mm and the
    // quarter annulus's centroid 2.5 - 4 (R^3 - r^3) / (3 pi (R^2 - r^2)) in.
    let (t, r, rr) = (1.5, 1.0, 2.5);
    let floor = t * (55.0 * 25.0 + 25.0 * 30.0);
    let bend = std::f64::consts::FRAC_PI_4 * (rr * rr - r * r);
    let wall = bend + t * 17.5;
    let centroid = rr - 4.0 * (rr.powi(3) - r.powi(3)) / (3.0 * std::f64::consts::PI * (rr * rr - r * r));
    let mitre = 2.0 * (17.5 * t * t / 2.0 + bend * centroid);
    near(volume(&text), floor + 215.0 * wall + mitre, 0.5, "converted");
    let flat_pattern = edit_unchecked(&text, "sheet-metal.flat-pattern", json!({}));
    let messages = diagnostics(&flat_pattern);
    assert!(messages.iter().any(|m| m.contains("the flat blank overlaps itself")), "{messages:?}");
}
