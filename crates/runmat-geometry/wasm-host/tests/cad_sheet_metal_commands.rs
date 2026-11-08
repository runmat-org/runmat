//! Sheet Metal commands on a plate from the canvas, on OCCT through `runmat_cad_call`: miter
//! flange, break corner, unfold and fold. Each command writes its `sheet_metal.*` feature from
//! the form's values and the part regenerates; volumes and extents check the geometry.

use serde_json::{json, Value};

/// An 80 x 50 mm rectangle on the XY plane; sketch `m` on the YZ plane is a miter flange's
/// profile: 15 mm up from the top of a 1.5 mm plate at the start of l1.
const SKETCHES: &str = "runmat 0.7\nstd 1\n\npart sheet {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])\n        l2 = line(start = [80 mm, 0 mm], end = [80 mm, 50 mm])\n        l3 = line(start = [80 mm, 50 mm], end = [0 mm, 50 mm])\n        l4 = line(start = [0 mm, 50 mm], end = [0 mm, 0 mm])\n    }\n    sketch m on origin.yz {\n        a = line(start = [0 mm, 1.5 mm], end = [0 mm, 16.5 mm])\n    }\n}\n";

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

/// The folded model's extents in mm (flat patterns left out).
fn extents(text: &str) -> [f64; 3] {
    let b = &compile(text)["display"]["data"]["bounds"];
    [0, 1, 2].map(|k| (b["max"][k].as_f64().unwrap() - b["min"][k].as_f64().unwrap()) * 1e3)
}

/// Every edge reference of the compiled part's bodies.
fn edge_names(text: &str) -> Vec<String> {
    let compiled = compile(text);
    compiled["display"]["data"]["bodies"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|b| b["edges"].as_array().cloned().unwrap_or_default())
        .filter_map(|e| e["reference"].as_str().map(str::to_owned))
        .collect()
}

/// The edge whose reference names every one of `parts`.
fn edge(text: &str, parts: &[&str]) -> String {
    let names = edge_names(text);
    names
        .iter()
        .find(|n| parts.iter().all(|p| n.contains(p)))
        .unwrap_or_else(|| panic!("no edge naming {parts:?} in {names:#?}"))
        .clone()
}

fn plate() -> String {
    edit(SKETCHES, "sheet-metal.base-flange", json!({ "profile": ["sketch:s1"], "thickness": 1.5, "radius": 1, "bend_rule": "k_factor", "k": 0.44 }))
}

fn near(got: f64, want: f64, tol: f64, what: &str) {
    assert!((got - want).abs() <= tol, "{what}: {got:.3}, want {want:.3}");
}

#[test]
fn a_miter_flange_runs_along_the_plates_top_edges() {
    let plate = plate();
    near(volume(&plate), 80.0 * 50.0 * 1.5, 1.0, "the plate");
    let top = |side: &str| edge(&plate, &["sheet(plate, ordinal = 1)", side]);
    let text = edit(&plate, "sheet-metal.miter-flange", json!({ "profile": ["sketch:m"], "edges": [top("sheet_edge(l1)"), top("sheet_edge(l2)")], "gap": 0 }));
    assert!(text.contains("sheet_metal.miter_flange("), "{text}");
    // The wall stands bend radius + thickness out from each side and 15 mm above the plate.
    let e = extents(&text);
    near(e[0], 80.0 + 2.5, 0.2, "x");
    near(e[1], 50.0 + 2.5, 0.2, "y");
    near(e[2], 16.5, 0.2, "z");
}

#[test]
fn a_broken_corner_takes_its_chamfer_off_the_plate() {
    let plate = plate();
    let corner = edge(&plate, &["sheet_edge(l1)", "sheet_edge(l2)"]);
    let text = edit(&plate, "sheet-metal.break-corner", json!({ "edges": [corner], "break": "chamfer", "distance": 4 }));
    assert!(text.contains("sheet_metal.break_corner("), "{text}");
    // A 4 x 4 mm triangle off one corner, through the 1.5 mm sheet.
    near(volume(&text), 6000.0 - 0.5 * 4.0 * 4.0 * 1.5, 0.5, "chamfered");
    let text = edit(&plate, "sheet-metal.break-corner", json!({ "edges": [corner], "break": "fillet", "distance": 4 }));
    near(volume(&text), 6000.0 - (16.0 - std::f64::consts::PI * 4.0) * 1.5, 0.5, "rounded");
}

/// Every face reference of the compiled part's bodies.
fn face_names(text: &str) -> Vec<String> {
    compile(text)["display"]["data"]["bodies"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|b| b["faces"].as_array().cloned().unwrap_or_default())
        .filter_map(|f| f["reference"].as_str().map(str::to_owned))
        .collect()
}

#[test]
fn unfold_flattens_a_bent_sheet_and_fold_bends_it_back() {
    // An L section, 30 mm up and 50 mm along, swept 40 mm: one bend.
    let chain = "runmat 0.7\nstd 1\n\npart sheet {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 30 mm], end = [0 mm, 0 mm])\n        l2 = line(start = [0 mm, 0 mm], end = [50 mm, 0 mm])\n    }\n}\n";
    let bent = edit(chain, "sheet-metal.base-flange", json!({ "profile": ["sketch:s1"], "thickness": 1.5, "radius": 1, "depth": 40, "bend_rule": "k_factor", "k": 0.44 }));
    let before = extents(&bent);
    let faces = face_names(&bent);
    let fixed = faces
        .iter()
        .find(|n| n.contains("sheet(l2"))
        .unwrap_or_else(|| panic!("no face of l2's flat in {faces:#?}"))
        .clone();
    let unfolded = edit(&bent, "sheet-metal.unfold", json!({ "fixed": [fixed] }));
    assert!(unfolded.contains("sheet_metal.unfold("), "{unfolded}");
    let flat = extents(&unfolded);
    // Flat: one extent is the thickness, and the developed length passes the 50 mm flat.
    let thin = flat.iter().cloned().fold(f64::INFINITY, f64::min);
    near(thin, 1.5, 0.01, "flat");
    assert!(flat.iter().any(|e| *e > 70.0), "the 30 mm flat lies out along the 50 mm one: {flat:?}");
    let folded = edit(&unfolded, "sheet-metal.fold", json!({}));
    assert!(folded.contains("sheet_metal.fold("), "{folded}");
    let again = extents(&folded);
    for k in 0..3 {
        near(again[k], before[k], 1e-3, "folded back");
    }
    // The flat pattern of the bent section holds its first flat still and regenerates.
    let flat = edit(&bent, "sheet-metal.flat-pattern", json!({}));
    assert!(flat.contains("sheet_metal.flat_pattern(fixed_face = feature.flange1.face.sheet(l1"), "{flat}");
    compile(&flat);
    // After an unfold and fold, the sheet is bent again and its flat pattern develops it (it
    // was refused: the folded body lost the base flange's body name).
    let lines = |text: &str| -> usize {
        let bodies = compile(text)["display"]["data"]["bodies"].as_array().cloned().unwrap_or_default();
        bodies.iter().map(|b| b["bend_lines"].as_array().map_or(0, Vec::len)).sum()
    };
    let after = edit(&folded, "sheet-metal.flat-pattern", json!({}));
    let (developed, refolded) = (lines(&flat), lines(&after));
    assert!(refolded > 0 && refolded == developed, "the blank carries the bend's lines as before the unfold ({refolded} vs {developed}): {after}");
}

/// Unfolds `text` on the plate's top face, checks it lies flat, folds it and checks it is back
/// where it was; answers the folded text.
fn unfold_and_fold(text: &str) -> String {
    let before = extents(text);
    let faces = face_names(text);
    let fixed = faces
        .iter()
        .find(|n| n.contains("sheet(plate, ordinal = 1)"))
        .unwrap_or_else(|| panic!("no top face of the plate in {faces:#?}"))
        .clone();
    let unfolded = edit(text, "sheet-metal.unfold", json!({ "fixed": [fixed] }));
    let flat = extents(&unfolded);
    let thin = flat.iter().cloned().fold(f64::INFINITY, f64::min);
    near(thin, 1.5, 0.01, "unfolded");
    let folded = edit(&unfolded, "sheet-metal.fold", json!({}));
    let again = extents(&folded);
    for k in 0..3 {
        near(again[k], before[k], 1e-3, "folded back");
    }
    near(volume(&folded), volume(text), 1e-3 * volume(text), "the folded volume");
    folded
}

#[test]
fn unfold_and_fold_a_plate_with_an_edge_flange() {
    let plate = plate();
    let top = edge(&plate, &["sheet(plate, ordinal = 1)", "sheet_edge(l2)"]);
    let text = edit(&plate, "sheet-metal.edge-flange", json!({ "edges": [top], "length": 20, "angle": 90, "position": "bend_outside", "radius": 1 }));
    assert!(text.contains("sheet_metal.edge_flange("), "{text}");
    let folded = unfold_and_fold(&text);
    // Folded again, the flange's faces answer to their names: a second unfold flattens it.
    unfold_and_fold(&folded);
}

/// A one-bend miter flange along two top edges of a plate develops with the plate: each edge
/// is a strip as wide as the bend's allowance and the 14 mm flat, and at the shared corner the
/// strips run on past it by the mitre (the wall's inner face, 1 mm out) and stop short of it by
/// as much. Unfolded, the blank lies flat with those strips; folded, the wall stands again.
#[test]
fn unfold_and_fold_a_plate_with_a_miter_flange() {
    let plate = plate();
    let top = |side: &str| edge(&plate, &["sheet(plate, ordinal = 1)", side]);
    let text = edit(&plate, "sheet-metal.miter-flange", json!({ "profile": ["sketch:m"], "edges": [top("sheet_edge(l1)"), top("sheet_edge(l2)")], "gap": 0 }));
    let fixed = face_names(&text).into_iter().find(|n| n.contains("sheet(plate, ordinal = 1)")).expect("the plate's top face");
    let unfolded = edit(&text, "sheet-metal.unfold", json!({ "fixed": [fixed] }));
    // Bend allowance at k = 0.44: a quarter turn at 1 mm + 0.44 x 1.5 mm.
    let (a, flat, x) = (std::f64::consts::FRAC_PI_2 * (1.0 + 0.44 * 1.5), 14.0, 1.0);
    let mitre = 0.5 * a * x + flat * x;
    near(volume(&unfolded), 1.5 * (80.0 * 50.0 + 130.0 * (a + flat) + 2.0 * mitre), 0.5, "the blank");
    let folded = unfold_and_fold(&text);
    unfold_and_fold(&folded);
    // The flat pattern carries one bend line per edge on each face of the blank; the display
    // lists each line as its two end points, six numbers.
    let flat = edit(&text, "sheet-metal.flat-pattern", json!({}));
    let bodies = compile(&flat)["display"]["data"]["bodies"].as_array().cloned().unwrap_or_default();
    let numbers: usize = bodies.iter().map(|b| b["bend_lines"].as_array().map_or(0, Vec::len)).sum();
    assert_eq!(numbers, 2 * 2 * 6, "two bend lines on both faces of the blank");
}

/// A lofted bend between a straight-cornered profile and an arch 30 mm above it: a sheet of the
/// form's thickness, as tall as the planes are apart, its arch formed in the form's bends.
#[test]
fn a_lofted_bend_joins_two_open_profiles() {
    let text = "runmat 0.7\nstd 1\n\npart sheet {\n    sketch a on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n    }\n    feature top = plane.offset(base = origin.xy, distance = 30 mm)\n    sketch b on plane.top {\n        a1 = arc(center = [20 mm, 0 mm], start = [40 mm, 0 mm], end = [0 mm, 0 mm])\n    }\n}\n";
    let lofted = edit(text, "sheet-metal.lofted-bend", json!({ "profile1": ["sketch:a"], "profile2": ["sketch:b"], "thickness": 1.5, "bends": 6 }));
    assert!(lofted.contains("sheet_metal.lofted_bend("), "{lofted}");
    let e = extents(&lofted);
    near(e[2], 30.0, 0.01, "height");
    // Each level is a thin section 1.5 mm thick whose path runs between the two profiles'
    // (60 mm and 63 mm long at the ends, shorter in between where the rulings cut the corner).
    let v = volume(&lofted);
    assert!(v > 1.5 * 30.0 * 40.0 && v < 1.5 * 30.0 * 63.0, "volume {v:.1} mm³");
    // Fewer bends form a coarser arch: a different solid that still spans the planes.
    let coarse = edit(text, "sheet-metal.lofted-bend", json!({ "profile1": ["sketch:a"], "profile2": ["sketch:b"], "thickness": 1.5, "bends": 2 }));
    near(extents(&coarse)[2], 30.0, 0.01, "height");
    assert!((volume(&coarse) - v).abs() > 1.0, "the bend count changes the arch");
}

/// Two tabs on one plate's end edge reach through a 2 mm wall standing against that end, and
/// the wall gets two slots 0.1 mm clear of them all round.
#[test]
fn tabs_reach_through_slots_in_the_other_sheet() {
    let text = "runmat 0.7\nstd 1\n\npart sheet {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])\n        l2 = line(start = [80 mm, 0 mm], end = [80 mm, 50 mm])\n        l3 = line(start = [80 mm, 50 mm], end = [0 mm, 50 mm])\n        l4 = line(start = [0 mm, 50 mm], end = [0 mm, 0 mm])\n    }\n    feature wall_plane = plane.offset(base = origin.yz, distance = 80 mm)\n    sketch w on plane.wall_plane {\n        k1 = line(start = [0 mm, -10 mm], end = [50 mm, -10 mm])\n        k2 = line(start = [50 mm, -10 mm], end = [50 mm, 20 mm])\n        k3 = line(start = [50 mm, 20 mm], end = [0 mm, 20 mm])\n        k4 = line(start = [0 mm, 20 mm], end = [0 mm, -10 mm])\n    }\n}\n";
    let plate = edit(text, "sheet-metal.base-flange", json!({ "profile": ["sketch:s1"], "thickness": 1.5, "radius": 1, "bend_rule": "k_factor", "k": 0.44 }));
    let both = edit(&plate, "sheet-metal.base-flange", json!({ "profile": ["sketch:w"], "thickness": 2, "radius": 1, "bend_rule": "k_factor", "k": 0.44 }));
    let before = volume(&both);
    near(before, 80.0 * 50.0 * 1.5 + 50.0 * 30.0 * 2.0, 0.5, "the plate and the wall");
    let tab_edge = edge(&both, &["flange1.face.sheet(plate, ordinal = 1)", "flange1.face.sheet_edge(l2)"]);
    let faces = face_names(&both);
    let slot_face = faces.iter().find(|n| n.contains("flange2.face.sheet(plate, ordinal = 0)")).unwrap_or_else(|| panic!("no wall face in {faces:#?}")).clone();
    let text = edit(&both, "sheet-metal.tab-and-slot", json!({ "edge": [tab_edge], "face": [slot_face], "count": 2, "length": 10, "clearance": 0.1 }));
    assert!(text.contains("sheet_metal.tab_and_slot("), "{text}");
    // Two 10 x 1.5 x 2 mm tabs in; two 10.2 x 1.7 x 2 mm slots out.
    near(volume(&text), before + 2.0 * 10.0 * 1.5 * 2.0 - 2.0 * 10.2 * 1.7 * 2.0, 0.05, "tabs and slots");
    // The tabs reach to the wall's far face, 2 mm past the plate's end.
    near(extents(&text)[0], 82.0, 1e-3, "x");
}
