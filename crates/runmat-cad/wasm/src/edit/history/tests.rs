//! Rename, the rollback bar and Parents and children, run through the canvas's dispatchers.

use super::super::edit_json;
use crate::runtime::{compile_json, Files};
use serde_json::{json, Value};

const PATH: &str = "/p/plate.part";

const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    input width : length = 40 mm\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [input.width, 0 mm])\n        l2 = line(start = [input.width, 0 mm], end = [input.width, 20 mm])\n        l3 = line(start = [input.width, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    feature round = fillet(edges = [edges_between(feature.body.face.side(s1.l1), feature.body.face.end)], radius = 1 mm)\n    region top : face = { select = [feature.body.face.end], expect = exactly_one }\n}\n";

fn run(text: &str, files: &Files, request: Value) -> Value {
    serde_json::from_str(&edit_json(PATH, text, files, &request.to_string())).unwrap()
}

fn applied(text: &str, request: Value) -> String {
    let out = run(text, &Files::new(), request);
    assert_eq!(out["ok"], true, "{out}");
    out["files"][0]["text"].as_str().unwrap().to_owned()
}

fn refused(text: &str, request: Value) -> String {
    let out = run(text, &Files::new(), request);
    assert_eq!(out["ok"], false, "{out}");
    out["message"].as_str().unwrap().to_owned()
}

/// Each feature's status after compiling `text`.
fn statuses(text: &str) -> Vec<(String, String)> {
    let out: Value = serde_json::from_str(&compile_json(PATH, text, &Files::new())).unwrap();
    out["projection"]["data"]["index"]["features"]
        .as_array()
        .unwrap_or_else(|| panic!("{out}"))
        .iter()
        .map(|f| {
            (
                f["id"].as_str().unwrap().to_owned(),
                f["status"]["status"].as_str().unwrap().to_owned(),
            )
        })
        .collect()
}

#[test]
fn rename_follows_references_in_the_part_and_its_drawings() {
    let drawing = "runmat 0.7\nstd 1\n\ndrawing plate {\n    format a3 = { size = { size = iso, name = A3 }, title_block = [] }\n    sheet \"1\" = { format = a3, scale = [1, 1], projection = third_angle }\n    view front = standard(orientation = front, sheet = \"1\", model = { part = plate }, at = [100, 100], display = visible_lines)\n    annotation t = dimension(style = linear, references = [feature.body.face.start, feature.body.face.end], view = front)\n}\n";
    let files: Files = [("/p/plate.drawing".to_owned(), drawing.to_owned())].into();
    let out = run(
        PLATE,
        &files,
        json!({ "command": "part.rename", "selection": ["feature:body"], "values": { "name": "slab" } }),
    );
    assert_eq!(out["ok"], true, "{out}");
    assert_eq!(out["summary"], "Renamed body to slab and in 1 drawing");
    let part = out["files"][0]["text"].as_str().unwrap();
    assert!(part.contains("feature slab = extrude("), "{part}");
    assert!(part.contains("feature.slab.face.side(s1.l1)"), "{part}");
    assert!(!part.contains("body"), "{part}");
    assert_eq!(out["files"][1]["path"], "/p/plate.drawing");
    assert!(out["files"][1]["text"]
        .as_str()
        .unwrap()
        .contains("references = [feature.slab.face.start, feature.slab.face.end]"));
    // The renamed part builds every feature, as before.
    assert!(statuses(part).iter().all(|(_, s)| s == "regenerated"), "{:?}", statuses(part));
}

#[test]
fn input_and_region_renames_follow_into_assemblies_and_studies() {
    let part = PLATE.replace(
        "\n}\n",
        "\n    output plate = { body = feature.body.body }\n}\n",
    );
    let assembly = "runmat 0.7\nstd 1\n\nassembly frame {\n    use part plate as left {\n        configuration = { select = inputs, values = { width = 30 mm } }\n    }\n}\n";
    let study = "geometry plate {\n    file = \"../p/plate.part\"\n    units = mm\n    region deck : face { select name \"top\"; expect exactly_one }\n}\n";
    let files: Files = [
        ("/p/frame.assembly".to_owned(), assembly.to_owned()),
        ("/s/plate.fea".to_owned(), study.to_owned()),
    ]
    .into();
    let out = run(
        &part,
        &files,
        json!({ "command": "part.rename", "selection": ["input:width"], "values": { "name": "span" } }),
    );
    assert_eq!(out["ok"], true, "{out}");
    assert_eq!(out["summary"], "Renamed width to span and in 1 assembly");
    assert_eq!(out["files"][1]["path"], "/p/frame.assembly");
    let text = out["files"][1]["text"].as_str().unwrap();
    assert!(text.contains("values = { span = 30 mm }"), "{text}");
    let out = run(
        &part,
        &files,
        json!({ "command": "part.rename", "selection": ["region:top"], "values": { "name": "deck_face" } }),
    );
    assert_eq!(out["ok"], true, "{out}");
    assert_eq!(out["summary"], "Renamed top to deck_face and in 1 study");
    assert!(out["files"][0]["text"].as_str().unwrap().contains("region deck_face : face"), "{out}");
    assert_eq!(out["files"][1]["path"], "/s/plate.fea");
    let text = out["files"][1]["text"].as_str().unwrap();
    assert!(text.contains("region deck : face { select name \"deck_face\";"), "{text}");
}

#[test]
fn properties_name_field_and_rename_share_one_handler() {
    let text = applied(
        PLATE,
        json!({ "command": "part.set-property", "selection": ["feature:round"], "values": { "name": "edge_round" } }),
    );
    assert!(text.contains("feature edge_round = fillet("), "{text}");
    let text = applied(
        PLATE,
        json!({ "command": "part.rename", "selection": ["input:width"], "values": { "name": "span" } }),
    );
    assert!(text.contains("input span : length") && text.contains("[input.span, 0 mm]"), "{text}");
    let clash = refused(
        PLATE,
        json!({ "command": "part.rename", "selection": ["feature:body"], "values": { "name": "round" } }),
    );
    assert!(clash.contains("already has a feature `round`"), "{clash}");
    let bad = refused(
        PLATE,
        json!({ "command": "part.rename", "selection": ["feature:body"], "values": { "name": "two words" } }),
    );
    assert!(bad.contains("isn't a name"), "{bad}");
    let body = refused(
        PLATE,
        json!({ "command": "part.rename", "selection": ["body:0"], "values": { "name": "x" } }),
    );
    assert!(body.contains("rename that feature"), "{body}");
}

#[test]
fn rollback_writes_the_bar_and_the_model_stops_there() {
    let back = applied(
        PLATE,
        json!({ "command": "part.rollback-here", "selection": ["feature:body"] }),
    );
    assert!(back.contains("    rollback_after = body\n"), "{back}");
    assert_eq!(
        statuses(&back),
        [
            ("s1".to_owned(), "regenerated".to_owned()),
            ("body".to_owned(), "regenerated".to_owned()),
            ("round".to_owned(), "rolled_back".to_owned()),
        ]
    );
    // Rolled back already; rolling to the end takes the line out again.
    let again = refused(
        &back,
        json!({ "command": "part.rollback-here", "selection": ["feature:body"] }),
    );
    assert!(again.contains("rolled back to body already"), "{again}");
    let end = applied(&back, json!({ "command": "part.rollback-end" }));
    assert_eq!(end, PLATE);
    assert!(refused(PLATE, json!({ "command": "part.rollback-end" })).contains("at the end already"));
    // Renaming or deleting the feature the bar sits after keeps the bar with it.
    let renamed = applied(
        &back,
        json!({ "command": "part.rename", "selection": ["feature:body"], "values": { "name": "slab" } }),
    );
    assert!(renamed.contains("rollback_after = slab"), "{renamed}");
    let bar_on_round = applied(
        PLATE,
        json!({ "command": "part.rollback-here", "selection": ["feature:body"] }),
    );
    let deleted = applied(
        &bar_on_round,
        json!({ "command": "part.delete", "selection": ["feature:round", "feature:body"] }),
    );
    assert!(deleted.contains("rollback_after = s1"), "{deleted}");
}

#[test]
fn a_feature_added_while_rolled_back_goes_in_at_the_bar() {
    let back = applied(
        PLATE,
        json!({ "command": "part.rollback-here", "selection": ["feature:body"] }),
    );
    let added = applied(
        &back,
        json!({ "command": "part.extrude", "values": { "profile": ["feature:s1"], "op": "new", "end": "blind", "distance": 5 } }),
    );
    let order: Vec<String> = statuses(&added).into_iter().map(|(id, _)| id).collect();
    assert_eq!(order, ["s1", "body", "extrude1", "round"], "{added}");
    assert!(added.contains("rollback_after = extrude1"), "{added}");
}

#[test]
fn parents_and_children_lists_the_dependency_data() {
    let ask = |command: &str, item: &str| -> Value {
        let body = json!({ "command": command, "selection": [item] });
        serde_json::from_str(&crate::analyze::analyze_json(
            PATH,
            PLATE,
            &Files::new(),
            &body.to_string(),
        ))
        .unwrap()
    };
    let out = ask("part.parents-children", "feature:body");
    assert_eq!(out["ok"], true, "{out}");
    let rows: Vec<Vec<String>> = out["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| serde_json::from_value(r["cells"].clone()).unwrap())
        .collect();
    let has = |rel: &str, name: &str| rows.iter().any(|r| r[0] == rel && r[1] == name);
    assert!(has("Parent", "s1") && has("Child", "round") && has("Child", "top"), "{rows:?}");
    assert_eq!(out["data"]["parents"]["features"], json!(["s1"]));
    // The sketch reads the input; Show references on the input lists the sketch.
    let sketch = ask("part.parents-children", "feature:s1");
    assert_eq!(sketch["data"]["parents"]["inputs"], json!(["width"]), "{sketch}");
    let input = ask("part.show-references", "input:width");
    assert_eq!(input["data"]["children"], json!(["s1"]), "{input}");
    let region = ask("part.show-references", "region:top");
    // The fillet shapes the body the region selects from, so it is a parent too.
    assert_eq!(region["data"]["parents"], json!(["body", "round"]), "{region}");
    let none = ask("part.parents-children", "origin:top");
    assert_eq!(none["ok"], false);
}
