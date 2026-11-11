//! Assembly commands from the canvas: each writes the smallest text change, and the edited
//! assembly compiles, solves and places the components where the command says.
use super::frames::{face_frame, Frame, FrameKind};
use crate::edit::edit_json;
use crate::runtime::{compile_assembly, Files};
use serde_json::{json, Value};

const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])\n        l2 = line(start = [80 mm, 0 mm], end = [80 mm, 50 mm])\n        l3 = line(start = [80 mm, 50 mm], end = [0 mm, 50 mm])\n        l4 = line(start = [0 mm, 50 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    output plate = { body = feature.body.body }\n}\n";

const BLOCK: &str = "runmat 0.7\nstd 1\n\npart block {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])\n        l2 = line(start = [20 mm, 0 mm], end = [20 mm, 20 mm])\n        l3 = line(start = [20 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 15 mm)\n    output block = { body = feature.body.body }\n}\n";

const PIN: &str = "runmat 0.7\nstd 1\n\npart pin {\n    sketch s1 on origin.xy {\n        c1 = circle(center = [0 mm, 0 mm], radius = 5 mm)\n    }\n    feature body = extrude(profile = sketch.s1, distance = 30 mm)\n    output pin = { body = feature.body.body }\n    region rim : edge = { select = [edges_between(feature.body.face.side(s1.c1), feature.body.face.end)], expect = at_least_one }\n    publish rim = region(region = rim)\n}\n";

const RIG: &str = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part plate as plate\n    use part block as block {\n        placement = { placement = offset, offset = { translation = [30 mm, 10 mm, 40 mm] } }\n    }\n    use part pin as pin {\n        placement = { placement = offset, offset = { translation = [60 mm, 30 mm, 25 mm] } }\n    }\n    use part pin as pin2 {\n        placement = { placement = offset, offset = { translation = [10 mm, 70 mm, 0 mm] } }\n    }\n}\n";

fn files() -> Files {
    [
        ("/p/plate.part", PLATE),
        ("/p/block.part", BLOCK),
        ("/p/pin.part", PIN),
    ]
    .map(|(p, t)| (p.to_owned(), t.to_owned()))
    .into_iter()
    .chain(EDITED.with(|e| e.borrow().clone()))
    .collect()
}

thread_local! {
    /// Part files the test's edits changed (a mate publishes the faces it anchors to), read
    /// back by [`files`] so later compiles see them.
    static EDITED: std::cell::RefCell<Files> = const { std::cell::RefCell::new(Files::new()) };
}

fn remember(path: &str, text: &str) {
    if path.ends_with(".part") {
        EDITED.with(|e| e.borrow_mut().insert(path.to_owned(), text.to_owned()));
    }
}

struct Compiled {
    index: Value,
    display: Value,
}

fn compile(text: &str) -> Compiled {
    let (index, display, problems) = compile_assembly("/p/rig.assembly", text, &files())
        .unwrap_or_else(|d| panic!("doesn't compile: {d:?}\n{text}"));
    let errors: Vec<_> = problems.iter().filter(|d| d.severity == "error").collect();
    assert!(errors.is_empty(), "{errors:?}\n{text}");
    Compiled { index, display }
}

impl Compiled {
    fn uuid(&self, path: &str) -> String {
        self.index["occurrences"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["path"] == path)
            .unwrap_or_else(|| panic!("no occurrence {path}"))["uuid"]
            .as_str()
            .unwrap()
            .to_owned()
    }

    fn occurrence(&self, path: &str) -> &Value {
        let uuid = self.uuid(path);
        self.display["occurrences"]
            .as_array()
            .unwrap()
            .iter()
            .find(|o| o["uuid"].as_str() == Some(uuid.as_str()))
            .unwrap()
    }

    fn transform(&self, path: &str) -> Vec<f64> {
        self.occurrence(path)["transform"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    }

    /// The pick ID of the face on `path` whose frame `test` accepts.
    fn face(&self, path: &str, test: impl Fn(&Frame) -> bool) -> String {
        let prototype = self.occurrence(path)["prototype"].as_u64().unwrap() as usize;
        let body = &self.display["prototypes"][prototype]["part"]["bodies"][0];
        let face = body["faces"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| face_frame(f).is_some_and(|frame| test(&frame)))
            .unwrap_or_else(|| panic!("no such face on {path}"));
        format!(
            "occurrence:{}/face:{}:{}",
            self.uuid(path),
            body["index"],
            face["index"]
        )
    }
}

fn run(text: &str, request: Value) -> String {
    let out: Value = serde_json::from_str(&edit_json(
        "/p/rig.assembly",
        text,
        &files(),
        &request.to_string(),
    ))
    .unwrap();
    assert_eq!(out["ok"], true, "{out}");
    for f in out["files"].as_array().unwrap().iter().skip(1) {
        remember(f["path"].as_str().unwrap(), f["text"].as_str().unwrap());
    }
    out["files"][0]["text"].as_str().unwrap().to_owned()
}

fn top(f: &Frame) -> bool {
    f.kind == FrameKind::Plane && f.z[2] > 0.99
}
fn bottom(f: &Frame) -> bool {
    f.kind == FrameKind::Plane && f.z[2] < -0.99
}
fn side(f: &Frame) -> bool {
    f.kind == FrameKind::Plane && f.z[0] > 0.99
}
fn round(f: &Frame) -> bool {
    f.kind == FrameKind::Axis
}

fn mate(text: &str, kind: &str, first: String, second: String, extra: Value) -> String {
    let mut values = json!({ "type": kind, "first": [first], "second": [second] });
    for (k, v) in extra.as_object().unwrap() {
        values[k] = v.clone();
    }
    run(
        text,
        json!({ "command": "assembly.mate", "values": values }),
    )
}

#[test]
fn face_frames_find_planes_and_cylinder_axes() {
    let c = compile(RIG);
    let id = c.face("pin", round);
    assert!(id.contains("/face:"));
    let prototype = c.occurrence("pin")["prototype"].as_u64().unwrap() as usize;
    let faces = c.display["prototypes"][prototype]["part"]["bodies"][0]["faces"]
        .as_array()
        .unwrap();
    let wall = faces
        .iter()
        .filter_map(face_frame)
        .find(|f| f.kind == FrameKind::Axis)
        .unwrap();
    assert!((wall.radius.unwrap() - 0.005).abs() < 1e-5, "{wall:?}");
    assert!(wall.z[2].abs() > 0.999 && wall.origin[0].abs() < 1e-6 && wall.origin[1].abs() < 1e-6);
}

#[test]
fn coincident_puts_the_block_on_the_plate() {
    let c = compile(RIG);
    let text = mate(
        RIG,
        "coincident",
        c.face("plate", top),
        c.face("block", bottom),
        json!({}),
    );
    assert!(text.contains("connector plate_c1"), "{text}");
    assert!(
        text.contains("mate coincident1 = { kind = { mate = planar }"),
        "{text}"
    );
    let after = compile(&text);
    // The block's bottom (its z = 0) sits on the plate's top (z = 10 mm).
    let t = after.transform("block");
    assert!((t[11] - 0.010).abs() < 1e-6, "{t:?}\n{text}");
}

#[test]
fn distance_holds_the_faces_parallel_at_the_offset() {
    let c = compile(RIG);
    let text = mate(
        RIG,
        "distance",
        c.face("plate", top),
        c.face("block", bottom),
        json!({ "offset": 5 }),
    );
    assert!(text.contains("kind = { mate = distance, measure = plane_point, target = { target = exact, value = 5 mm } }") && text.contains("mate distance1_parallel = { kind = { mate = parallel }"), "{text}");
    let t = compile(&text).transform("block");
    assert!((t[11] - 0.015).abs() < 1e-6, "{t:?}\n{text}");
}

#[test]
fn concentric_lines_up_the_pins() {
    let c = compile(RIG);
    let text = mate(
        RIG,
        "concentric",
        c.face("pin", round),
        c.face("pin2", round),
        json!({}),
    );
    assert!(
        text.contains("mate concentric1 = { kind = { mate = cylindrical"),
        "{text}"
    );
    let t = compile(&text).transform("pin2");
    assert!(
        (t[9] - 0.060).abs() < 1e-6 && (t[10] - 0.030).abs() < 1e-6,
        "{t:?}\n{text}"
    );
}

#[test]
fn perpendicular_and_angle_turn_the_block() {
    let c = compile(RIG);
    let text = mate(
        RIG,
        "perpendicular",
        c.face("plate", top),
        c.face("block", top),
        json!({}),
    );
    assert!(text.contains("90 deg"), "{text}");
    let t = compile(&text).transform("block");
    // The block's local Z (third column) now lies flat.
    assert!(t[8].abs() < 1e-6, "{t:?}\n{text}");
    let text = mate(
        RIG,
        "angle",
        c.face("plate", top),
        c.face("block", top),
        json!({ "angle": 30 }),
    );
    let t = compile(&text).transform("block");
    assert!(
        (t[8] - 30f64.to_radians().cos()).abs() < 1e-6,
        "{t:?}\n{text}"
    );
}

#[test]
fn parallel_keeps_the_faces_facing_the_same_way() {
    let c = compile(RIG);
    let text = mate(
        RIG,
        "parallel",
        c.face("plate", side),
        c.face("block", top),
        json!({}),
    );
    assert!(
        text.contains("mate parallel1 = { kind = { mate = parallel }"),
        "{text}"
    );
    let t = compile(&text).transform("block");
    // The block's top normal (local Z) is along the plate's +X side normal, either way.
    assert!(t[2].abs() > 1.0 - 1e-6, "{t:?}\n{text}");
}

#[test]
fn tangent_lays_the_pin_on_the_plate() {
    let c = compile(RIG);
    // Picked cylinder first: the plane still comes first in the mate.
    let text = mate(
        RIG,
        "tangent",
        c.face("pin", round),
        c.face("plate", top),
        json!({}),
    );
    assert!(text.contains("mate tangent1 = {"), "{text}");
    assert!(
        text.contains(
            "kind = { mate = tangent, kind = { tangent = plane_cylinder, radius = 5 mm } }"
        ),
        "{text}"
    );
    let t = compile(&text).transform("pin");
    // The pin's axis lies flat, 5 mm above the plate's top (10 mm).
    assert!(t[8].abs() < 1e-6, "{t:?}\n{text}");
    assert!((t[11] - 0.015).abs() < 1e-6, "{t:?}\n{text}");
}

#[test]
fn gear_screw_rack_and_pin_slot_mates_write_their_values() {
    let c = compile(RIG);
    let (pin, pin2) = (c.face("pin", round), c.face("pin2", round));
    let text = mate(
        RIG,
        "gear",
        pin.clone(),
        pin2.clone(),
        json!({ "ratio": 2 }),
    );
    assert!(text.contains("kind = { mate = gear, ratio = 2 }"), "{text}");
    let text = mate(
        RIG,
        "screw",
        pin.clone(),
        pin2.clone(),
        json!({ "pitch": 1.5 }),
    );
    assert!(
        text.contains("kind = { mate = screw, lead = 1.5 mm }"),
        "{text}"
    );
    let text = mate(
        RIG,
        "rack_pinion",
        pin.clone(),
        pin2.clone(),
        json!({ "pitch": 60 }),
    );
    assert!(
        text.contains("kind = { mate = rack_pinion, travel = 60 mm }"),
        "{text}"
    );
    let text = mate(RIG, "pin_slot", pin, pin2, json!({}));
    assert!(text.contains("kind = { mate = pin_slot }"), "{text}");
    // Each one still compiles.
    compile(&text);
}

/// Every type the Mate command offers applies from the canvas and the result compiles: the list
/// holds no type the applier refuses.
#[test]
fn every_offered_mate_type_applies() {
    use runmat_cad::commands::FieldKind;
    let catalog = runmat_cad::commands::catalog();
    let mate_command = catalog.command("assembly.mate").expect("assembly.mate");
    let FieldKind::Choice { options, .. } = &mate_command
        .inputs
        .iter()
        .find(|f| f.id == "type")
        .unwrap()
        .kind
    else {
        panic!("Mate's type is a choice");
    };
    let c = compile(RIG);
    let (pin, pin2, plate) = (
        c.face("pin", round),
        c.face("pin2", round),
        c.face("plate", top),
    );
    let (under, block_top, block_under) = (
        c.face("plate", bottom),
        c.face("block", top),
        c.face("block", bottom),
    );
    assert_eq!(options.len(), 24, "{options:?}");
    for option in options {
        let kind = option.value.as_str();
        let (first, second) = match kind {
            "tangent" => (plate.clone(), pin2.clone()),
            // The plate's two faces are the groove, the block's the tab.
            "width" => (plate.clone(), block_under.clone()),
            _ => (pin.clone(), pin2.clone()),
        };
        let values = json!({
            "offset": 5, "angle": 30, "ratio": 2, "pitch": 2,
            "first2": [under], "second2": [block_top], "plane": [c.face("plate", side)],
            "profile": "rim", "path": "rim", "joint_point": true,
        });
        let text = mate(RIG, kind, first, second, values);
        compile(&text);
    }
}

/// The advanced mates write their kind and extra connectors, and solve: the block centred
/// across the plate's thickness, pin2 mirrored in the plate's side, pin2 on the pin's rim.
#[test]
fn advanced_mates_write_and_solve() {
    let c = compile(RIG);
    let width = json!({ "first2": [c.face("plate", bottom)], "second2": [c.face("block", top)] });
    let text = mate(
        RIG,
        "width",
        c.face("plate", top),
        c.face("block", bottom),
        width,
    );
    assert!(
        text.contains("mate = width, a2 = plate_c1, b2 = block_c1"),
        "{text}"
    );
    // Plate 0..10 mm, block 15 mm thick: its bottom at -2.5 mm.
    let z = compile(&text).transform("block")[11];
    assert!((z + 0.0025).abs() < 1e-6, "{z}\n{text}");
    let whole = |name: &str| format!("occurrence:{}", c.uuid(name));
    let text = mate(
        RIG,
        "symmetric",
        whole("pin"),
        whole("pin2"),
        json!({ "plane": [c.face("plate", side)] }),
    );
    assert!(
        text.contains("mate = symmetric, plane = plate_c1"),
        "{text}"
    );
    // The pin's origin at x = 60 mm mirrors in the plate's side (x = 80 mm) to 100 mm.
    let x = compile(&text).transform("pin2")[9];
    assert!((x - 0.100).abs() < 1e-6, "{x}\n{text}");
    let text = mate(
        RIG,
        "path",
        c.face("pin", round),
        c.face("pin2", round),
        json!({ "path": "rim" }),
    );
    assert!(text.contains("mate = path, path = rim"), "{text}");
    compile(&text);
    // The groove's other face on another component is refused by name.
    let wrong = json!({ "command": "assembly.mate", "values": { "type": "width", "first": [c.face("plate", top)], "second": [c.face("block", bottom)], "first2": [c.face("block", top)], "second2": [c.face("block", top)] } });
    let out: Value = serde_json::from_str(&edit_json(
        "/p/rig.assembly",
        RIG,
        &files(),
        &wrong.to_string(),
    ))
    .unwrap();
    assert!(
        out["message"]
            .as_str()
            .unwrap_or("")
            .contains("groove's other face on `plate`"),
        "{out}"
    );
}

#[test]
fn joint_limits_are_written_as_ranges() {
    let c = compile(RIG);
    let (pin, pin2) = (c.face("pin", round), c.face("pin2", round));
    let text = mate(
        RIG,
        "revolute",
        pin.clone(),
        pin2.clone(),
        json!({ "angle_limits": true, "min": -20, "max": 60 }),
    );
    assert!(
        text.contains("angle = { target = range, min = -20 deg, max = 60 deg }"),
        "{text}"
    );
    compile(&text);
    let text = mate(
        RIG,
        "slider",
        pin.clone(),
        pin2.clone(),
        json!({ "distance_limits": true, "near": 0, "far": 25 }),
    );
    assert!(
        text.contains("distance = { target = range, min = 0 mm, max = 25 mm }"),
        "{text}"
    );
    compile(&text);
    // Unticked, a revolute turns freely.
    let text = mate(RIG, "revolute", pin, pin2, json!({ "min": -20, "max": 60 }));
    assert!(text.contains("kind = { mate = revolute }"), "{text}");
}

#[test]
fn mates_refuse_picks_that_dont_fit() {
    let c = compile(RIG);
    let request = json!({ "command": "assembly.mate", "values": { "type": "concentric", "first": [c.face("plate", top)], "second": [c.face("pin", round)] } });
    let out: Value = serde_json::from_str(&edit_json(
        "/p/rig.assembly",
        RIG,
        &files(),
        &request.to_string(),
    ))
    .unwrap();
    assert_eq!(out["ok"], false);
    assert!(
        out["message"].as_str().unwrap().contains("concentric"),
        "{out}"
    );
}

#[test]
fn insert_adds_a_floating_component_beside_the_click() {
    let text = run(
        RIG,
        json!({ "command": "assembly.insert", "values": { "source": "part", "file": "block.part", "quantity": 2, "x": 100 } }),
    );
    assert!(text.contains("use part block as block2"), "{text}");
    assert!(text.contains("as block3"), "{text}");
    let c = compile(&text);
    let (t2, t3) = (c.transform("block2"), c.transform("block3"));
    // Display transforms are f32.
    assert!((t2[9] - 0.100).abs() < 1e-6, "{t2:?}");
    // Beside it: the block's 20 mm width and a fifth of it between.
    assert!((t3[9] - t2[9] - 0.024).abs() < 1e-6, "{t3:?}");
}

#[test]
fn move_adds_the_triads_translation_and_rotation() {
    let c = compile(RIG);
    let selection = [format!("occurrence:{}", c.uuid("block"))];
    let text = run(
        RIG,
        json!({ "command": "assembly.move", "selection": selection, "values": { "mode": "triad", "dx": 5, "dz": -40, "rz": 90 } }),
    );
    assert!(
        text.contains("translation = [35 mm, 10 mm, 0 mm]"),
        "{text}"
    );
    let t = compile(&text).transform("block");
    // Turned a quarter about Z: local X now points along +Y.
    assert!(
        (t[3] - 1.0).abs() < 1e-9 && t[0].abs() < 1e-9,
        "{t:?}\n{text}"
    );
    assert!((t[9] - 0.035).abs() < 1e-9 && t[11].abs() < 1e-9, "{t:?}");
}

#[test]
fn linear_and_circular_patterns_repeat_the_component() {
    let c = compile(RIG);
    let selection = [format!("occurrence:{}", c.uuid("block"))];
    let text = run(
        RIG,
        json!({ "command": "assembly.pattern", "selection": selection, "values": { "layout": "linear", "direction": ["origin:y"], "spacing": 25, "instances": 3 } }),
    );
    assert!(text.contains("pattern pattern1 = linear("), "{text}");
    let after = compile(&text);
    let blocks = after.index["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["path"].as_str().unwrap().starts_with("block"))
        .count();
    assert_eq!(blocks, 3, "{}", after.index["occurrences"]);
    let text = run(
        RIG,
        json!({ "command": "assembly.pattern", "selection": selection, "values": { "layout": "circular", "direction": [c.face("pin", round)], "angle": 360, "instances": 4 } }),
    );
    assert!(text.contains("pattern pattern1 = circular("), "{text}");
    compile(&text);
}

#[test]
fn a_broken_mates_repair_suppresses_it() {
    let c = compile(RIG);
    let text = mate(
        RIG,
        "coincident",
        c.face("plate", top),
        c.face("block", bottom),
        json!({}),
    );
    // The mate's second reference names a component the assembly doesn't have.
    let broken: String = text
        .lines()
        .map(|l| {
            if l.contains("mate coincident1") {
                l.replace("block", "ghost")
            } else {
                l.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let (index, _, _) =
        compile_assembly("/p/rig.assembly", &broken, &files()).expect("compiles with an error");
    let diagnostic = index["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .find(|d| {
            d["repairs"]
                .as_array()
                .is_some_and(|r| r.iter().any(|r| r["repair"] == "suppress_mate"))
        })
        .unwrap_or_else(|| panic!("no suppress-mate repair in {}", index["diagnostics"]))
        .clone();
    let at = diagnostic["repairs"]
        .as_array()
        .unwrap()
        .iter()
        .position(|r| r["repair"] == "suppress_mate")
        .unwrap();
    let fixed = run(
        &broken,
        json!({ "command": "repair.repair", "values": { "diagnostic": diagnostic["id"], "repair": at } }),
    );
    let line = fixed
        .lines()
        .find(|l| l.contains("mate coincident1"))
        .unwrap_or("");
    assert!(line.contains("suppressed"), "{fixed}");
    let (index, _, _) = compile_assembly("/p/rig.assembly", &fixed, &files()).expect("compiles");
    let errors: Vec<_> = index["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|d| d["severity"] == "error")
        .collect();
    assert!(errors.is_empty(), "{errors:?}\n{fixed}");
}

#[test]
fn an_override_on_a_missing_component_is_retargeted_or_removed() {
    // `sub` holds a plate and a block; the outer assembly's override names `block2`, which
    // `sub` doesn't have, so the repair offers `block` (a close name) or removing the override.
    let sub = "runmat 0.7\nstd 1\n\nassembly sub {\n    use part plate as plate\n    use part block as block {\n        placement = { placement = offset, offset = { translation = [30 mm, 10 mm, 40 mm] } }\n    }\n}\n";
    let top = "runmat 0.7\nstd 1\n\nassembly top {\n    use assembly sub as s {\n        override block2 {\n            suppressed = true\n        }\n    }\n}\n";
    let mut files = files();
    files.insert("/p/sub.assembly".into(), sub.into());
    let (index, _, _) =
        compile_assembly("/p/top.assembly", top, &files).expect("compiles with an error");
    let diagnostic = index["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .find(|d| {
            d["repairs"]
                .as_array()
                .is_some_and(|r| r.iter().any(|r| r["repair"] == "retarget_override"))
        })
        .unwrap_or_else(|| panic!("no retarget repair in {}", index["diagnostics"]))
        .clone();
    let repairs = diagnostic["repairs"].as_array().unwrap();
    assert_eq!(repairs[0]["occurrence"], "s", "{diagnostic}");
    assert_eq!(
        repairs[0]["candidates"],
        json!([[{ "name": "block" }]]),
        "{diagnostic}"
    );
    let apply = |at: usize| {
        let request = json!({ "command": "repair.repair", "values": { "diagnostic": diagnostic["id"], "repair": at, "candidate": 0 } }).to_string();
        let out: Value =
            serde_json::from_str(&edit_json("/p/top.assembly", top, &files, &request)).unwrap();
        assert_eq!(out["ok"], true, "{out}");
        out["files"][0]["text"].as_str().unwrap().to_owned()
    };
    let retargeted = apply(0);
    assert!(retargeted.contains("override block {"), "{retargeted}");
    let (index, _, _) = compile_assembly("/p/top.assembly", &retargeted, &files).expect("compiles");
    assert!(
        index["diagnostics"]
            .as_array()
            .unwrap()
            .iter()
            .all(|d| d["severity"] != "error"),
        "{}",
        index["diagnostics"]
    );
    let removed = apply(1);
    assert!(!removed.contains("override"), "{removed}");
}

#[test]
fn mirrored_copies_land_where_the_seeds_reflection_lies() {
    let c = compile(RIG);
    let selection = [format!("occurrence:{}", c.uuid("block"))];
    // The block (20 x 20 x 15 mm, symmetric about its mid-planes) sits at y = 10..30 mm; its
    // image about the front plane takes y = -30..-10 mm, the same way up.
    let text = run(
        RIG,
        json!({ "command": "assembly.mirror", "selection": selection, "values": { "plane": ["origin:front"], "result": "copy" } }),
    );
    assert!(text.contains("mirror mirror1"), "{text}");
    let after = compile(&text);
    let rows = after.index["occurrences"].as_array().unwrap();
    let image = rows
        .iter()
        .map(|r| r["path"].as_str().unwrap())
        .find(|p| p.contains("mirror1"))
        .unwrap_or_else(|| panic!("no image in {rows:?}"))
        .to_owned();
    let t = after.transform(&image);
    let expected = [
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.03, -0.03, 0.04,
    ];
    for (k, (got, want)) in t.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-9,
            "transform[{k}] = {got}, want {want}: {t:?}\n{text}"
        );
    }
    // About a face: the plate's top (z = 10 mm) reflects the pin standing on it at z = 25 mm.
    let text = run(
        RIG,
        json!({ "command": "assembly.mirror", "values": { "components": [format!("occurrence:{}", c.uuid("pin"))], "plane": [c.face("plate", top)], "result": "copy" } }),
    );
    compile(&text);
    // An opposite-hand part swaps the file (the pin stands in for the block's other hand).
    let text = run(
        RIG,
        json!({ "command": "assembly.mirror", "selection": selection, "values": { "plane": ["origin:right"], "result": "opposite", "opposite": "pin.part" } }),
    );
    assert!(text.contains("pin"), "{text}");
    compile(&text);
}
