//! Exploded views from the canvas: a new view, a step and a radial step written into it, trail
//! lines, and the compile's `exploded` rows the Desktop shows (moved transforms and lines).
use crate::edit::edit_json;
use crate::runtime::{compile_assembly, Files};
use serde_json::{json, Value};

const BLOCK: &str = "runmat 0.7\nstd 1\n\npart block {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])\n        l2 = line(start = [20 mm, 0 mm], end = [20 mm, 20 mm])\n        l3 = line(start = [20 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 15 mm)\n    output block = { body = feature.body.body }\n}\n";

const STACK: &str = "runmat 0.7\nstd 1\n\nassembly stack {\n    use part block as base\n    use part block as lid {\n        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 15 mm] } }\n    }\n    use part block as side {\n        placement = { placement = offset, offset = { translation = [40 mm, 0 mm, 0 mm] } }\n    }\n}\n";

const PATH: &str = "/p/stack.assembly";

fn files() -> Files {
    [("/p/block.part", BLOCK)].map(|(p, t)| (p.to_owned(), t.to_owned())).into()
}

fn index(text: &str) -> Value {
    let (index, _, problems) = compile_assembly(PATH, text, &files()).unwrap_or_else(|d| panic!("{d:?}\n{text}"));
    let errors: Vec<_> = problems.iter().filter(|d| d.severity == "error").collect();
    assert!(errors.is_empty(), "{errors:?}\n{text}");
    index
}

fn uuid(index: &Value, path: &str) -> String {
    index["occurrences"].as_array().unwrap().iter().find(|r| r["path"] == path).unwrap()["uuid"].as_str().unwrap().to_owned()
}

fn run(text: &str, request: Value) -> String {
    let out: Value = serde_json::from_str(&edit_json(PATH, text, &files(), &request.to_string())).unwrap();
    assert_eq!(out["ok"], true, "{out}");
    out["files"][0]["text"].as_str().unwrap().to_owned()
}

fn moved<'a>(view: &'a Value, path: &str) -> Option<&'a Value> {
    view["occurrences"].as_array().unwrap().iter().find(|o| o["path"] == path)
}

#[test]
fn an_assembly_without_exploded_views_projects_none() {
    assert_eq!(index(STACK)["exploded"], json!([]));
}

#[test]
fn a_new_view_takes_a_step_a_radial_step_and_lines_and_compiles_exploded() {
    let text = run(STACK, json!({ "command": "explode.new", "values": { "name": "Service view" } }));
    assert!(text.contains("exploded service_view"), "{text}");
    let i = index(&text);
    assert_eq!(i["exploded"][0]["name"], "service_view");

    // The lid lifts 50 mm along +Z.
    let lid = format!("occurrence:{}", uuid(&i, "lid"));
    let text = run(&text, json!({ "command": "explode.step", "selection": [lid], "values": { "direction": ["origin:z"], "distance": 50 } }));
    let i = index(&text);
    let view = &i["exploded"][0];
    assert_eq!(view["steps"][0]["id"], "step1");
    assert_eq!(view["steps"][0]["motion"], "translate");
    assert_eq!(view["steps"][0]["occurrences"], json!(["lid"]));
    let t = moved(view, "lid").unwrap_or_else(|| panic!("{view}"))["transform"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>();
    assert!((t[11] - 0.065).abs() < 1e-6, "{t:?}");
    assert!(moved(view, "base").is_none());

    // The side spreads 30 mm out from the Z axis, into the view the selection names.
    let side = format!("occurrence:{}", uuid(&i, "side"));
    let text = run(&text, json!({ "command": "explode.radial", "selection": [side, "exploded:service_view"], "values": { "axis": ["origin:z"], "distance": 30 } }));
    let i = index(&text);
    let view = &i["exploded"][0];
    assert_eq!(view["steps"][1]["motion"], "radial");
    let t = moved(view, "side").unwrap()["transform"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>();
    assert!(t[9] > 0.040 + 0.025, "{t:?}");

    // Lines for every moved component; their trails start assembled and end exploded.
    let text = run(&text, json!({ "command": "explode.lines", "values": {} }));
    let i = index(&text);
    let lines = i["exploded"][0]["lines"].as_array().unwrap();
    assert_eq!(lines.len(), 2, "{text}");
    let lid_line = lines.iter().find(|l| l["occurrence"] == "lid").unwrap();
    let points = lid_line["points"].as_array().unwrap();
    let z = |p: &Value| p[2].as_f64().unwrap();
    assert!((z(points.last().unwrap()) - z(&points[0]) - 0.050).abs() < 1e-6, "{points:?}");
    // A second run adds nothing and says so.
    let out: Value = serde_json::from_str(&edit_json(PATH, &text, &files(), &json!({ "command": "explode.lines", "values": {} }).to_string())).unwrap();
    assert_eq!(out["ok"], false, "{out}");
}

#[test]
fn a_step_needs_an_exploded_view() {
    let i = index(STACK);
    let lid = format!("occurrence:{}", uuid(&i, "lid"));
    let out: Value = serde_json::from_str(&edit_json(PATH, STACK, &files(), &json!({ "command": "explode.step", "selection": [lid], "values": { "distance": 50 } }).to_string())).unwrap();
    assert_eq!(out["ok"], false);
    assert!(out["message"].as_str().unwrap().contains("exploded view first"), "{out}");
}
