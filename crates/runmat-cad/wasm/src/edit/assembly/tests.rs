//! Assembly commands from the canvas: each writes the smallest text change, and the edited
//! assembly compiles, solves and places the components where the command says.
use super::frames::{face_frame, Frame, FrameKind};
use crate::edit::edit_json;
use crate::runtime::{compile_assembly, Files};
use serde_json::{json, Value};

const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])\n        l2 = line(start = [80 mm, 0 mm], end = [80 mm, 50 mm])\n        l3 = line(start = [80 mm, 50 mm], end = [0 mm, 50 mm])\n        l4 = line(start = [0 mm, 50 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    output plate = { body = feature.body.body }\n}\n";

const BLOCK: &str = "runmat 0.7\nstd 1\n\npart block {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])\n        l2 = line(start = [20 mm, 0 mm], end = [20 mm, 20 mm])\n        l3 = line(start = [20 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 15 mm)\n    output block = { body = feature.body.body }\n}\n";

const PIN: &str = "runmat 0.7\nstd 1\n\npart pin {\n    sketch s1 on origin.xy {\n        c1 = circle(center = [0 mm, 0 mm], radius = 5 mm)\n    }\n    feature body = extrude(profile = sketch.s1, distance = 30 mm)\n    output pin = { body = feature.body.body }\n}\n";

const RIG: &str = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part plate as plate\n    use part block as block {\n        placement = { placement = offset, offset = { translation = [30 mm, 10 mm, 40 mm] } }\n    }\n    use part pin as pin {\n        placement = { placement = offset, offset = { translation = [60 mm, 30 mm, 25 mm] } }\n    }\n    use part pin as pin2 {\n        placement = { placement = offset, offset = { translation = [10 mm, 70 mm, 0 mm] } }\n    }\n}\n";

fn files() -> Files {
    [
        ("/p/plate.part", PLATE),
        ("/p/block.part", BLOCK),
        ("/p/pin.part", PIN),
    ]
    .map(|(p, t)| (p.to_owned(), t.to_owned()))
    .into()
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
    let out: Value =
        serde_json::from_str(&edit_json("/p/rig.assembly", text, &files(), &request.to_string()))
            .unwrap();
    assert_eq!(out["ok"], true, "{out}");
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
    run(text, json!({ "command": "assembly.mate", "values": values }))
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
    let wall = faces.iter().filter_map(face_frame).find(|f| f.kind == FrameKind::Axis).unwrap();
    assert!((wall.radius.unwrap() - 0.005).abs() < 1e-5, "{wall:?}");
    assert!(wall.z[2].abs() > 0.999 && wall.origin[0].abs() < 1e-6 && wall.origin[1].abs() < 1e-6);
}

#[test]
fn coincident_puts_the_block_on_the_plate() {
    let c = compile(RIG);
    let text = mate(RIG, "coincident", c.face("plate", top), c.face("block", bottom), json!({}));
    assert!(text.contains("connector plate_c1"), "{text}");
    assert!(text.contains("mate coincident1 = { kind = { mate = planar }"), "{text}");
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
    let text = mate(RIG, "concentric", c.face("pin", round), c.face("pin2", round), json!({}));
    assert!(text.contains("mate concentric1 = { kind = { mate = cylindrical"), "{text}");
    let t = compile(&text).transform("pin2");
    assert!((t[9] - 0.060).abs() < 1e-6 && (t[10] - 0.030).abs() < 1e-6, "{t:?}\n{text}");
}

#[test]
fn perpendicular_and_angle_turn_the_block() {
    let c = compile(RIG);
    let text = mate(RIG, "perpendicular", c.face("plate", top), c.face("block", top), json!({}));
    assert!(text.contains("90 deg"), "{text}");
    let t = compile(&text).transform("block");
    // The block's local Z (third column) now lies flat.
    assert!(t[8].abs() < 1e-6, "{t:?}\n{text}");
    let text = mate(RIG, "angle", c.face("plate", top), c.face("block", top), json!({ "angle": 30 }));
    let t = compile(&text).transform("block");
    assert!((t[8] - 30f64.to_radians().cos()).abs() < 1e-6, "{t:?}\n{text}");
}

#[test]
fn parallel_keeps_the_faces_facing_the_same_way() {
    let c = compile(RIG);
    let text = mate(RIG, "parallel", c.face("plate", side), c.face("block", top), json!({}));
    assert!(text.contains("mate parallel1 = { kind = { mate = parallel }"), "{text}");
    let t = compile(&text).transform("block");
    // The block's top normal (local Z) is along the plate's +X side normal, either way.
    assert!(t[2].abs() > 1.0 - 1e-6, "{t:?}\n{text}");
}

#[test]
fn tangent_lays_the_pin_on_the_plate() {
    let c = compile(RIG);
    // Picked cylinder first: the plane still comes first in the mate.
    let text = mate(RIG, "tangent", c.face("pin", round), c.face("plate", top), json!({}));
    assert!(text.contains("mate tangent1 = {"), "{text}");
    assert!(text.contains("kind = { mate = tangent, kind = { tangent = plane_cylinder, radius = 5 mm } }"), "{text}");
    let t = compile(&text).transform("pin");
    // The pin's axis lies flat, 5 mm above the plate's top (10 mm).
    assert!(t[8].abs() < 1e-6, "{t:?}\n{text}");
    assert!((t[11] - 0.015).abs() < 1e-6, "{t:?}\n{text}");
}

#[test]
fn mates_refuse_picks_that_dont_fit() {
    let c = compile(RIG);
    let request = json!({ "command": "assembly.mate", "values": { "type": "concentric", "first": [c.face("plate", top)], "second": [c.face("pin", round)] } });
    let out: Value = serde_json::from_str(&edit_json("/p/rig.assembly", RIG, &files(), &request.to_string())).unwrap();
    assert_eq!(out["ok"], false);
    assert!(out["message"].as_str().unwrap().contains("concentric"), "{out}");
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
    assert!(text.contains("translation = [35 mm, 10 mm, 0 mm]"), "{text}");
    let t = compile(&text).transform("block");
    // Turned a quarter about Z: local X now points along +Y.
    assert!((t[3] - 1.0).abs() < 1e-9 && t[0].abs() < 1e-9, "{t:?}\n{text}");
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
