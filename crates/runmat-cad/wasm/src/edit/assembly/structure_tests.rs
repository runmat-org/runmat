//! Flip, edit and smart mates, replace, subassemblies and a new part in context: each edit
//! regenerates, and the components land where the command says.
use super::frames::{Frame, FrameKind};
use crate::edit::edit_json;
use crate::runtime::{compile_assembly, Files};
use serde_json::{json, Value};

const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])\n        l2 = line(start = [80 mm, 0 mm], end = [80 mm, 50 mm])\n        l3 = line(start = [80 mm, 50 mm], end = [0 mm, 50 mm])\n        l4 = line(start = [0 mm, 50 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    output plate = { body = feature.body.body }\n}\n";

const BLOCK: &str = "runmat 0.7\nstd 1\n\npart block {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])\n        l2 = line(start = [20 mm, 0 mm], end = [20 mm, 20 mm])\n        l3 = line(start = [20 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 15 mm)\n    output block = { body = feature.body.body }\n}\n";

const PIN: &str = "runmat 0.7\nstd 1\n\npart pin {\n    sketch s1 on origin.xy {\n        c1 = circle(center = [0 mm, 0 mm], radius = 5 mm)\n    }\n    feature body = extrude(profile = sketch.s1, distance = 30 mm)\n    output pin = { body = feature.body.body }\n}\n";

const RIG: &str = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part plate as plate\n    use part block as block {\n        placement = { placement = offset, offset = { translation = [30 mm, 10 mm, 40 mm] } }\n    }\n    use part pin as pin {\n        placement = { placement = offset, offset = { translation = [60 mm, 30 mm, 25 mm] } }\n    }\n}\n";

const TOP: &str = "/p/rig.assembly";

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

fn compile_with(text: &str, files: &Files) -> Compiled {
    let (index, display, problems) = compile_assembly(TOP, text, files)
        .unwrap_or_else(|d| panic!("doesn't compile: {d:?}\n{text}"));
    let errors: Vec<_> = problems.iter().filter(|d| d.severity == "error").collect();
    assert!(errors.is_empty(), "{errors:?}\n{text}");
    Compiled { index, display }
}

fn compile(text: &str) -> Compiled {
    compile_with(text, &files())
}

impl Compiled {
    fn row(&self, path: &str) -> &Value {
        self.index["occurrences"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["path"] == path)
            .unwrap_or_else(|| panic!("no occurrence {path}: {}", self.index["occurrences"]))
    }
    fn uuid(&self, path: &str) -> String {
        self.row(path)["uuid"].as_str().unwrap().to_owned()
    }
    /// Row-major rotation then translation (m).
    fn world(&self, path: &str) -> Vec<f64> {
        let w = &self.row(path)["world"];
        let mut out: Vec<f64> = w["rotation"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect();
        out.extend(
            w["translation"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap()),
        );
        out
    }
    fn face(&self, path: &str, test: impl Fn(&Frame) -> bool) -> String {
        let uuid = self.uuid(path);
        let occurrence = self.display["occurrences"]
            .as_array()
            .unwrap()
            .iter()
            .find(|o| o["uuid"].as_str() == Some(uuid.as_str()))
            .unwrap();
        let prototype = occurrence["prototype"].as_u64().unwrap() as usize;
        let body = &self.display["prototypes"][prototype]["part"]["bodies"][0];
        let face = body["faces"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| super::frames::face_frame(f).is_some_and(|frame| test(&frame)))
            .unwrap_or_else(|| panic!("no such face on {path}"));
        format!("occurrence:{uuid}/face:{}:{}", body["index"], face["index"])
    }
    fn mate(&self, name: &str) -> &Value {
        self.index["mates"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["name"] == name)
            .unwrap_or_else(|| panic!("no mate {name}"))
    }
}

fn top(f: &Frame) -> bool {
    f.kind == FrameKind::Plane && f.z[2] > 0.99
}
fn bottom(f: &Frame) -> bool {
    f.kind == FrameKind::Plane && f.z[2] < -0.99
}

/// Runs `request` on `text`, answering every changed file.
fn run_with(text: &str, files: &Files, request: Value) -> Vec<(String, String)> {
    let out: Value =
        serde_json::from_str(&edit_json(TOP, text, files, &request.to_string())).unwrap();
    assert_eq!(out["ok"], true, "{out}");
    let changed: Vec<(String, String)> = out["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| {
            (
                f["path"].as_str().unwrap().to_owned(),
                f["text"].as_str().unwrap().to_owned(),
            )
        })
        .collect();
    for (path, text) in changed.iter().skip(1) {
        remember(path, text);
    }
    changed
}

fn run(text: &str, request: Value) -> String {
    run_with(text, &files(), request).remove(0).1
}

fn refused(text: &str, request: Value) -> String {
    let out: Value =
        serde_json::from_str(&edit_json(TOP, text, &files(), &request.to_string())).unwrap();
    assert_eq!(out["ok"], false, "{out}");
    out["message"].as_str().unwrap().to_owned()
}

/// The rig with the block's bottom on the plate's top (a coincident mate, `coincident1`).
fn mated() -> String {
    let c = compile(RIG);
    run(
        RIG,
        json!({ "command": "assembly.mate", "values": { "type": "coincident", "first": [c.face("plate", top)], "second": [c.face("block", bottom)] } }),
    )
}

#[test]
fn flip_turns_the_block_over_on_the_plate() {
    let text = mated();
    assert_eq!(compile(&text).mate("coincident1")["values"]["flip"], false);
    let flipped = run(
        &text,
        json!({ "command": "assembly.flip-mate", "selection": ["mate:/coincident1"] }),
    );
    assert!(flipped.contains("flip = true"), "{flipped}");
    let c = compile(&flipped);
    let w = c.world("block");
    // Upside down (its Z points down), its bottom face still on the plate's top at 10 mm.
    assert!(
        w[8] < -0.99 && (w[11] - 0.010).abs() < 1e-6,
        "{w:?}\n{flipped}"
    );
    assert_eq!(c.mate("coincident1")["values"]["flip"], true);
    // Flipping again turns it back.
    let back = run(
        &flipped,
        json!({ "command": "assembly.flip-mate", "selection": ["mate:/coincident1"] }),
    );
    assert!(compile(&back).world("block")[8] > 0.99);
}

#[test]
fn edit_mate_starts_from_the_mate_and_writes_its_new_values() {
    let c = compile(RIG);
    let text = run(
        RIG,
        json!({ "command": "assembly.mate", "values": { "type": "distance", "first": [c.face("plate", top)], "second": [c.face("block", bottom)], "offset": 5 } }),
    );
    let before = compile(&text);
    let values = before.mate("distance1")["values"].clone();
    assert_eq!(values["type"], "distance");
    assert_eq!(values["offset"]["text"], "5 mm", "{values}");
    // The form's values, the distance changed to 8 mm.
    let mut form = values.clone();
    form["offset"] = json!(8);
    let edited = run(
        &text,
        json!({ "command": "assembly.edit-mate", "selection": ["mate:/distance1"], "values": form }),
    );
    assert!(edited.contains("value = 8 mm"), "{edited}");
    assert!((compile(&edited).world("block")[11] - 0.018).abs() < 1e-6);
    // Changed to a fasten mate, the distance's parallel companion goes with it.
    let fastened = run(
        &text,
        json!({ "command": "assembly.edit-mate", "selection": ["mate:/distance1"], "values": { "type": "fasten", "flip": false } }),
    );
    assert!(
        fastened.contains("mate = fasten") && !fastened.contains("distance1_parallel"),
        "{fastened}"
    );
    compile(&fastened);
    // A revolute with limits writes its range.
    let hinge = run(
        &text,
        json!({ "command": "assembly.edit-mate", "selection": ["mate:/distance1"], "values": { "type": "revolute", "angle_limits": true, "min": -10, "max": 20 } }),
    );
    assert!(
        hinge.contains("target = range") && hinge.contains("-10 deg") && hinge.contains("20 deg"),
        "{hinge}"
    );
    let row = compile(&hinge).mate("distance1")["values"].clone();
    assert_eq!(
        (row["angle_limits"].clone(), row["max"]["text"].clone()),
        (json!(true), json!("20 deg")),
        "{row}"
    );
    // Width needs the other faces it doesn't have yet.
    assert!(refused(&text, json!({ "command": "assembly.edit-mate", "selection": ["mate:/distance1"], "values": { "type": "width" } })).contains("pick the groove's other face"));
}

/// Every type Edit mate offers writes on a mate that exists, given its fields: none needs
/// picks the form doesn't have.
#[test]
fn every_type_edit_mate_offers_applies() {
    use runmat_cad::commands::FieldKind;
    let catalog = runmat_cad::commands::catalog();
    let command = catalog
        .command("assembly.edit-mate")
        .expect("assembly.edit-mate");
    let FieldKind::Choice { options, .. } =
        &command.inputs.iter().find(|f| f.id == "type").unwrap().kind
    else {
        panic!("Edit mate's type is a choice");
    };
    let c = compile(RIG);
    let text = run(
        RIG,
        json!({ "command": "assembly.mate", "values": { "type": "distance", "first": [c.face("plate", top)], "second": [c.face("block", bottom)], "offset": 5 } }),
    );
    for option in options {
        let values = json!({
            "type": option.value, "offset": 5, "angle": 30, "ratio": 2, "pitch": 2, "radius": 5,
            "profile": "lobe", "path": "rail", "joint_point": false, "flip": false,
            "first2": [c.face("plate", bottom)], "second2": [c.face("block", top)], "plane": [c.face("plate", top)],
        });
        let edited = run(
            &text,
            json!({ "command": "assembly.edit-mate", "selection": ["mate:/distance1"], "values": values }),
        );
        assert!(
            edited.contains(&format!("mate = {}", option.value)),
            "{}: {edited}",
            option.value
        );
    }
}

#[test]
fn smart_mate_puts_the_blocks_facing_face_on_the_plate() {
    let c = compile(RIG);
    let text = run(
        RIG,
        json!({ "command": "assembly.smart-mate", "values": { "component": [format!("occurrence:{}", c.uuid("block"))], "target": [c.face("plate", top)] } }),
    );
    assert!(text.contains("mate = planar"), "{text}");
    let w = compile(&text).world("block");
    assert!((w[11] - 0.010).abs() < 1e-6 && w[8] > 0.99, "{w:?}\n{text}");
    // Onto the pin's round side, the block has no axis to line up.
    let pin = c.face("pin", |f| f.kind == FrameKind::Axis);
    assert!(refused(RIG, json!({ "command": "assembly.smart-mate", "values": { "component": [format!("occurrence:{}", c.uuid("block"))], "target": [pin] } })).contains("no round face"));
}

#[test]
fn replace_swaps_the_file_and_keeps_or_drops_the_mates() {
    let text = mated();
    let c = compile(&text);
    let block = format!("occurrence:{}", c.uuid("block"));
    let kept = run(
        &text,
        json!({ "command": "assembly.replace", "values": { "components": [block.clone()], "with": "pin.part", "remate": true } }),
    );
    let after = compile(&kept);
    assert_eq!(
        after.row("block")["definition"]["file"],
        "/p/pin.part",
        "{kept}"
    );
    // The pin's bottom (its z = 0) takes the block's place on the plate.
    assert!((after.world("block")[11] - 0.010).abs() < 1e-6, "{kept}");
    let dropped = run(
        &text,
        json!({ "command": "assembly.replace", "values": { "components": [block.clone()], "with": "pin.part", "remate": false } }),
    );
    assert!(
        !dropped.contains("mate coincident1") && !dropped.contains("connector block_c1"),
        "{dropped}"
    );
    compile(&dropped);
    assert!(refused(&text, json!({ "command": "assembly.replace", "values": { "components": [block], "with": "missing.part" } })).contains("no `missing.part`"));
}

#[test]
fn a_subassembly_forms_dissolves_and_turns_flexible() {
    let text = mated();
    let c = compile(&text);
    let before = (c.world("block"), c.world("pin"));
    let picks = json!([
        format!("occurrence:{}", c.uuid("block")),
        format!("occurrence:{}", c.uuid("pin"))
    ]);
    let changed = run_with(
        &text,
        &files(),
        json!({ "command": "assembly.subassembly", "values": { "action": "form", "components": picks, "name": "hinge.assembly" } }),
    );
    assert_eq!(changed[1].0, "/p/hinge.assembly");
    let (parent, hinge) = (changed[0].1.clone(), changed[1].1.clone());
    // The block's mate to the plate stays here and reaches into the flexible subassembly.
    assert!(
        parent.contains("flexible = true") && parent.contains("mate coincident1"),
        "{parent}"
    );
    assert!(
        hinge.contains("use part block") && hinge.contains("use part pin"),
        "{hinge}"
    );
    let mut with = files();
    with.insert("/p/hinge.assembly".into(), hinge.clone());
    let formed = compile_with(&parent, &with);
    assert_eq!(
        (formed.world("hinge/block"), formed.world("hinge/pin")),
        before.clone(),
        "{parent}\n{hinge}"
    );
    // Rigid, then dissolved: the components come back where they were, with their mate.
    let rigid = run_with(&parent, &with, json!({ "command": "assembly.subassembly", "values": { "action": "rigid", "components": [format!("occurrence:{}", formed.uuid("hinge"))] } })).remove(0).1;
    assert!(!rigid.contains("flexible = true"), "{rigid}");
    let dissolved = run_with(&parent, &with, json!({ "command": "assembly.subassembly", "values": { "action": "dissolve", "components": [format!("occurrence:{}", formed.uuid("hinge"))] } })).remove(0).1;
    assert!(!dissolved.contains("hinge"), "{dissolved}");
    let back = compile(&dissolved);
    for (path, was) in [("block", &before.0), ("pin", &before.1)] {
        let now = back.world(path);
        assert!(
            now.iter()
                .zip(was.iter())
                .all(|(a, b)| (a - b).abs() < 1e-9),
            "{path}: {now:?} vs {was:?}\n{dissolved}"
        );
    }
    let status = back.mate("coincident1")["status"]
        .to_string()
        .to_lowercase();
    assert!(status.contains("satisfied"), "{status}");
}

#[test]
fn a_new_part_in_context_starts_on_the_picked_face() {
    let c = compile(RIG);
    let changed = run_with(
        RIG,
        &files(),
        json!({ "command": "assembly.new-part", "values": { "name": "cover.part", "plane": [c.face("plate", top)] } }),
    );
    assert_eq!(changed[1].0, "/p/cover.part");
    let mut with = files();
    with.insert("/p/cover.part".into(), changed[1].1.clone());
    let after = compile_with(&changed[0].1, &with);
    let w = after.world("cover");
    // Its origin on the plate's top face centre (40, 25, 10 mm), Z up.
    assert!(
        (w[9] - 0.040).abs() < 1e-6
            && (w[10] - 0.025).abs() < 1e-6
            && (w[11] - 0.010).abs() < 1e-6
            && w[8] > 0.99,
        "{w:?}\n{}",
        changed[0].1
    );
    assert!(refused(
        RIG,
        json!({ "command": "assembly.new-part", "values": { "name": "plate.part" } })
    )
    .contains("already exists"));
}
