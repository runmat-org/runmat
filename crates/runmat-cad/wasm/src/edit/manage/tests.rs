//! Manage commands from the canvas: each writes the Source the compiler reads back.

use super::super::edit_json;
use crate::runtime::{compile_json, Files};
use serde_json::{json, Value};

const PATH: &str = "/p/box.part";

/// Two bodies: a 40 x 20 x 10 mm block and a 10 mm cube beside it.
const TWO: &str = "runmat 0.7\nstd 1\n\npart box {\n    input width : length = 40 mm\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [input.width, 0 mm])\n        l2 = line(start = [input.width, 0 mm], end = [input.width, 20 mm])\n        l3 = line(start = [input.width, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature base = extrude(profile = sketch.s1, distance = 10 mm)\n    sketch s2 on origin.xy {\n        m1 = line(start = [60 mm, 0 mm], end = [70 mm, 0 mm])\n        m2 = line(start = [70 mm, 0 mm], end = [70 mm, 10 mm])\n        m3 = line(start = [70 mm, 10 mm], end = [60 mm, 10 mm])\n        m4 = line(start = [60 mm, 10 mm], end = [60 mm, 0 mm])\n    }\n    feature lid = extrude(profile = sketch.s2, distance = 10 mm, mode = new)\n}\n";

/// One body.
fn one() -> String {
    TWO[..TWO.find("    sketch s2").unwrap()].to_owned() + "}\n"
}

fn run_in(path: &str, text: &str, files: &Files, request: Value) -> Value {
    serde_json::from_str(&edit_json(path, text, files, &request.to_string())).unwrap()
}

fn applied(text: &str, request: Value) -> String {
    let out = run_in(PATH, text, &Files::new(), request);
    assert_eq!(out["ok"], true, "{out}");
    out["files"][0]["text"].as_str().unwrap().to_owned()
}

fn refused(text: &str, request: Value) -> String {
    let out = run_in(PATH, text, &Files::new(), request);
    assert_eq!(out["ok"], false, "{out}");
    out["message"].as_str().unwrap().to_owned()
}

/// The compiled part's index, with no error diagnostics.
fn index(path: &str, text: &str, files: &Files) -> Value {
    let compiled: Value = serde_json::from_str(&compile_json(path, text, files)).unwrap();
    let errors: Vec<&Value> = compiled["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|d| d["severity"] == "error")
        .collect();
    assert!(errors.is_empty(), "{errors:?}\n{text}");
    compiled["projection"]["data"]
        .get("index")
        .cloned()
        .unwrap_or_else(|| compiled["projection"]["data"].clone())
}

#[test]
fn a_library_material_lands_on_the_part_or_a_body() {
    let text = applied(
        &one(),
        json!({ "command": "manage.material", "values": { "material": "material.steel_1018" } }),
    );
    assert!(
        text.contains(
            "output box = { body = feature.base.body, material = \"material.steel_1018\" }"
        ),
        "{text}"
    );
    assert_eq!(
        index(PATH, &text, &Files::new())["parts"][0]["material"],
        "material.steel_1018"
    );
    // Again: the output's material changes in place.
    let again = applied(
        &text,
        json!({ "command": "manage.material", "values": { "material": "material.abs" } }),
    );
    assert!(
        again.contains("material = \"material.abs\"") && !again.contains("steel"),
        "{again}"
    );
    // Two bodies, one picked: a material line for that body.
    let lid = applied(
        TWO,
        json!({ "command": "manage.material", "values": { "material": "material.aluminum_6061_t6", "bodies": ["body:1"] } }),
    );
    assert!(
        lid.contains(
            "material = { body = feature.lid.body, material = \"material.aluminum_6061_t6\" }"
        ),
        "{lid}"
    );
    index(PATH, &lid, &Files::new());
    assert!(refused(
        &one(),
        json!({ "command": "manage.material", "values": { "material": "unobtainium" } })
    )
    .contains("library"));
}

#[test]
fn a_color_lands_on_the_part_or_a_body_through_its_output() {
    let text = applied(
        &one(),
        json!({ "command": "manage.appearance", "values": { "color": "2f6db5" } }),
    );
    assert!(text.contains("color = \"#2F6DB5\""), "{text}");
    let compiled: Value = serde_json::from_str(&compile_json(PATH, &text, &Files::new())).unwrap();
    let rgba = compiled["display"]["data"]["appearance"]
        .as_array()
        .unwrap();
    assert!(
        (rgba[2].as_f64().unwrap() - 0xB5 as f64 / 255.0).abs() < 1e-3,
        "{rgba:?}"
    );
    // A body without an output gets one, named after its feature.
    let lid = applied(
        TWO,
        json!({ "command": "manage.appearance", "values": { "color": "#B83A3A", "targets": ["body:1"] } }),
    );
    assert!(
        lid.contains("output lid = { body = feature.lid.body, color = \"#B83A3A\" }"),
        "{lid}"
    );
    index(PATH, &lid, &Files::new());
    assert!(refused(
        &one(),
        json!({ "command": "manage.appearance", "values": { "color": "navy" } })
    )
    .contains("isn't a color"));
    assert!(refused(
        TWO,
        json!({ "command": "manage.appearance", "values": { "color": "#000" } })
    )
    .contains("several bodies"));
}

#[test]
fn equations_set_inputs_and_add_named_values() {
    let text = applied(
        &one(),
        json!({ "command": "manage.equations", "values": { "name": "width", "value": "60 mm" } }),
    );
    assert!(text.contains("input width : length = 60 mm"), "{text}");
    let text = applied(
        &text,
        json!({ "command": "manage.equations", "values": { "name": "half", "value": "input.width / 2" } }),
    );
    assert!(text.contains("let half = input.width / 2"), "{text}");
    let lets = index(PATH, &text, &Files::new())["lets"].clone();
    assert!(lets.to_string().contains("half"), "{lets}");
    let text = applied(
        &text,
        json!({ "command": "manage.equations", "values": { "name": "half", "value": "input.width / 4" } }),
    );
    assert!(
        text.contains("let half = input.width / 4") && !text.contains("/ 2"),
        "{text}"
    );
    assert!(refused(
        &one(),
        json!({ "command": "manage.equations", "values": { "name": "2x", "value": "1" } })
    )
    .contains("isn't a name"));
}

#[test]
fn a_promoted_body_is_its_own_part_output() {
    let text = applied(
        TWO,
        json!({ "command": "part.body-promote", "values": { "body": "body:1", "name": "cover" } }),
    );
    assert!(
        text.contains("output cover = { body = feature.lid.body }"),
        "{text}"
    );
    let parts = index(PATH, &text, &Files::new())["parts"].clone();
    assert_eq!(parts[0]["name"], "cover", "{parts}");
    assert!(refused(
        &text,
        json!({ "command": "part.body-promote", "values": { "body": "body:1", "name": "other" } })
    )
    .contains("already"));
    assert!(refused(
        &text,
        json!({ "command": "part.body-promote", "values": { "body": "body:0", "name": "cover" } })
    )
    .contains("already"));
}

#[test]
fn split_to_files_writes_one_part_per_body_and_an_assembly() {
    let colored = applied(
        TWO,
        json!({ "command": "manage.appearance", "values": { "color": "#B83A3A", "targets": ["body:1"] } }),
    );
    let out = run_in(
        PATH,
        &colored,
        &Files::new(),
        json!({ "command": "manage.split-to-files", "values": {} }),
    );
    assert_eq!(out["ok"], true, "{out}");
    let files = out["files"].as_array().unwrap();
    let paths: Vec<&str> = files.iter().map(|f| f["path"].as_str().unwrap()).collect();
    assert_eq!(
        paths,
        [
            "/p/box_base.part",
            "/p/box_lid.part",
            "/p/box_assembly.assembly"
        ]
    );
    let mut all = Files::new();
    for f in files {
        all.insert(
            f["path"].as_str().unwrap().to_owned(),
            f["text"].as_str().unwrap().to_owned(),
        );
    }
    // Each part keeps its own body and output; the lid keeps its color.
    let lid = &all["/p/box_lid.part"];
    assert!(
        lid.contains("feature split = body_op.delete(bodies = [feature.base.body])")
            && lid.contains("output lid = { body = feature.lid.body, color = \"#B83A3A\" }"),
        "{lid}"
    );
    for path in ["/p/box_base.part", "/p/box_lid.part"] {
        let i = index(path, &all[path], &Files::new());
        assert_eq!(
            i["bodies"].as_array().unwrap().len(),
            1,
            "{path}: {}",
            all[path]
        );
    }
    let assembly = all["/p/box_assembly.assembly"].clone();
    assert!(
        assembly.contains("use part box_base as base")
            && assembly.contains("use part box_lid as lid"),
        "{assembly}"
    );
    let i = index("/p/box_assembly.assembly", &assembly, &all);
    // Row 0 is the assembly itself; the two components follow, one per new part.
    let files: Vec<&str> = i["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["path"] != "")
        .map(|r| r["definition"]["file"].as_str().unwrap())
        .collect();
    assert_eq!(files, ["/p/box_base.part", "/p/box_lid.part"], "{i}");
    // A file that exists is refused.
    let mut taken = Files::new();
    taken.insert("/p/box_lid.part".into(), String::new());
    let again = run_in(
        PATH,
        &colored,
        &taken,
        json!({ "command": "manage.split-to-files", "values": { "bodies": ["body:1"] } }),
    );
    assert!(
        again["message"]
            .as_str()
            .unwrap_or("")
            .contains("already exists"),
        "{again}"
    );
}
