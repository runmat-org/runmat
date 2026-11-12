//! Exploded views from the canvas: a new view, a step and a radial step written into it, trail
//! lines, and the compile's `exploded` rows the Desktop shows (moved transforms and lines).
use crate::edit::edit_json;
use crate::runtime::{compile_assembly, Files};
use serde_json::{json, Value};

const BLOCK: &str = "runmat 0.7\nstd 1\n\npart block {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])\n        l2 = line(start = [20 mm, 0 mm], end = [20 mm, 20 mm])\n        l3 = line(start = [20 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 15 mm)\n    output block = { body = feature.body.body }\n}\n";

const STACK: &str = "runmat 0.7\nstd 1\n\nassembly stack {\n    use part block as base\n    use part block as lid {\n        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 15 mm] } }\n    }\n    use part block as side {\n        placement = { placement = offset, offset = { translation = [40 mm, 0 mm, 0 mm] } }\n    }\n}\n";

const PATH: &str = "/p/stack.assembly";

fn files() -> Files {
    [("/p/block.part", BLOCK)]
        .map(|(p, t)| (p.to_owned(), t.to_owned()))
        .into()
}

fn index(text: &str) -> Value {
    let (index, _, problems) =
        compile_assembly(PATH, text, &files()).unwrap_or_else(|d| panic!("{d:?}\n{text}"));
    let errors: Vec<_> = problems.iter().filter(|d| d.severity == "error").collect();
    assert!(errors.is_empty(), "{errors:?}\n{text}");
    index
}

fn uuid(index: &Value, path: &str) -> String {
    index["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["path"] == path)
        .unwrap()["uuid"]
        .as_str()
        .unwrap()
        .to_owned()
}

fn run(text: &str, request: Value) -> String {
    let out: Value =
        serde_json::from_str(&edit_json(PATH, text, &files(), &request.to_string())).unwrap();
    assert_eq!(out["ok"], true, "{out}");
    out["files"][0]["text"].as_str().unwrap().to_owned()
}

fn moved<'a>(view: &'a Value, path: &str) -> Option<&'a Value> {
    view["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .find(|o| o["path"] == path)
}

#[test]
fn an_assembly_without_exploded_views_projects_none() {
    assert_eq!(index(STACK)["exploded"], json!([]));
}

#[test]
fn a_new_view_takes_a_step_a_radial_step_and_lines_and_compiles_exploded() {
    let text = run(
        STACK,
        json!({ "command": "explode.new", "values": { "name": "Service view" } }),
    );
    assert!(text.contains("exploded service_view"), "{text}");
    let i = index(&text);
    assert_eq!(i["exploded"][0]["name"], "service_view");

    // The lid lifts 50 mm along +Z.
    let lid = format!("occurrence:{}", uuid(&i, "lid"));
    let text = run(
        &text,
        json!({ "command": "explode.step", "selection": [lid], "values": { "direction": ["origin:z"], "distance": 50 } }),
    );
    let i = index(&text);
    let view = &i["exploded"][0];
    assert_eq!(view["steps"][0]["id"], "step1");
    assert_eq!(view["steps"][0]["motion"], "translate");
    assert_eq!(view["steps"][0]["occurrences"], json!(["lid"]));
    let t = moved(view, "lid").unwrap_or_else(|| panic!("{view}"))["transform"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect::<Vec<_>>();
    assert!((t[11] - 0.065).abs() < 1e-6, "{t:?}");
    assert!(moved(view, "base").is_none());

    // The side spreads 30 mm out from the Z axis, into the view the selection names.
    let side = format!("occurrence:{}", uuid(&i, "side"));
    let text = run(
        &text,
        json!({ "command": "explode.radial", "selection": [side, "exploded:service_view"], "values": { "axis": ["origin:z"], "distance": 30 } }),
    );
    let i = index(&text);
    let view = &i["exploded"][0];
    assert_eq!(view["steps"][1]["motion"], "radial");
    let t = moved(view, "side").unwrap()["transform"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect::<Vec<_>>();
    assert!(t[9] > 0.040 + 0.025, "{t:?}");

    // Lines for every moved component; their trails start assembled and end exploded.
    let text = run(&text, json!({ "command": "explode.lines", "values": {} }));
    let i = index(&text);
    let lines = i["exploded"][0]["lines"].as_array().unwrap();
    assert_eq!(lines.len(), 2, "{text}");
    let lid_line = lines.iter().find(|l| l["occurrence"] == "lid").unwrap();
    let points = lid_line["points"].as_array().unwrap();
    let z = |p: &Value| p[2].as_f64().unwrap();
    assert!(
        (z(points.last().unwrap()) - z(&points[0]) - 0.050).abs() < 1e-6,
        "{points:?}"
    );
    // A second run adds nothing and says so.
    let out: Value = serde_json::from_str(&edit_json(
        PATH,
        &text,
        &files(),
        &json!({ "command": "explode.lines", "values": {} }).to_string(),
    ))
    .unwrap();
    assert_eq!(out["ok"], false, "{out}");
}

#[test]
fn a_step_needs_an_exploded_view() {
    let i = index(STACK);
    let lid = format!("occurrence:{}", uuid(&i, "lid"));
    let out: Value = serde_json::from_str(&edit_json(
        PATH,
        STACK,
        &files(),
        &json!({ "command": "explode.step", "selection": [lid], "values": { "distance": 50 } })
            .to_string(),
    ))
    .unwrap();
    assert_eq!(out["ok"], false);
    assert!(
        out["message"]
            .as_str()
            .unwrap()
            .contains("exploded view first"),
        "{out}"
    );
}

/// A row of three pegs (`pegs`: the seed and two instances) and its mirror image across
/// y = 0 (`pegs_port`), with an exploded view that names each as a whole.
const PEGS: &str = "runmat 0.7\nstd 1\n\nassembly stack {\n    use part block as base\n    use part block as peg {\n        placement = { placement = offset, offset = { translation = [0 mm, 30 mm, 15 mm] } }\n    }\n    pattern pegs = linear(seeds = [peg], direction = [1, 0, 0], spacing = 30 mm, count = 3)\n    mirror pegs_port = {\n        seeds = [pegs],\n        plane = { plane = explicit, point = [0 mm, 0 mm, 0 mm], normal = [0, 1, 0] },\n        hand = symmetric(point = [10 mm, 10 mm, 0 mm], normal = [0, 1, 0])\n    }\n    exploded open = {\n        steps = [\n            STEPS\n        ]\n    }\n}\n";

const UP: &str = "{ id = up, PICK, motion = { motion = \"translate\", direction = { direction = \"vector\", vector = [0, 0, 1] }, distance = 50 mm, spacing = 5 mm } }";

fn pegs(steps: &[&str]) -> String {
    PEGS.replace("STEPS", &steps.join(",\n            "))
}

/// Each moved occurrence's exploded translation (metres), by path.
fn moves(view: &Value) -> std::collections::BTreeMap<String, [f64; 3]> {
    view["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| {
            let t = o["transform"].as_array().unwrap();
            let f = |k: usize| t[k].as_f64().unwrap();
            (o["path"].as_str().unwrap().to_owned(), [f(9), f(10), f(11)])
        })
        .collect()
}

fn paths_with(index: &Value, part: &str) -> Vec<String> {
    index["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|r| r["path"].as_str())
        .filter(|p| p.contains(part))
        .map(str::to_owned)
        .collect()
}

#[test]
fn a_step_naming_a_pattern_moves_its_seed_and_every_instance_as_listing_them_would() {
    let whole = index(&pegs(&[
        &UP.replace("PICK", "patterns = [[{ name = pegs }]]")
    ]));
    let listed = index(&pegs(&[&UP.replace(
        "PICK",
        "occurrences = [[{ name = peg }], [{ name = peg, instance = { by = pegs, key = { instance = linear, step = 1 } } }], [{ name = peg, instance = { by = pegs, key = { instance = linear, step = 2 } } }]]",
    )]));
    let view = &whole["exploded"][0];
    assert_eq!(view["diagnostics"], json!([]), "{view}");
    assert_eq!(view["steps"][0]["patterns"], json!(["pegs"]));
    let m = moves(view);
    assert_eq!(m, moves(&listed["exploded"][0]));
    // The seed and both instances, spaced in their laid-out order; nothing else.
    let mut rows: Vec<_> = m.iter().collect();
    rows.sort_by(|a, b| a.1[2].total_cmp(&b.1[2]));
    assert_eq!(rows.len(), 3, "{m:?}");
    assert_eq!(rows[0].0, "peg");
    for (i, (_, t)) in rows.iter().enumerate() {
        assert!(
            (t[2] - (0.015 + 0.050 + 0.005 * i as f64)).abs() < 1e-9,
            "{rows:?}"
        );
    }
}

#[test]
fn a_step_naming_a_mirror_moves_each_image_and_leaves_the_originals() {
    let text = pegs(&[&UP.replace("PICK", "mirrors = [[{ name = pegs_port }]]")]);
    let i = index(&text);
    let view = &i["exploded"][0];
    assert_eq!(view["diagnostics"], json!([]), "{view}");
    let m = moves(view);
    let images = paths_with(&i, "@pegs_port[");
    assert_eq!(images.len(), 3, "{images:?}");
    assert_eq!(m.keys().cloned().collect::<Vec<_>>(), {
        let mut v = images.clone();
        v.sort();
        v
    });
    // Each image its own occurrence: three distinct ids, three distinct exploded heights.
    let ids: std::collections::BTreeSet<_> = images.iter().map(|p| uuid(&i, p)).collect();
    assert_eq!(ids.len(), 3);
    let mut z: Vec<f64> = m.values().map(|t| t[2]).collect();
    z.sort_by(f64::total_cmp);
    assert!((z[2] - z[0] - 0.010).abs() < 1e-9, "{z:?}");
}

#[test]
fn a_step_naming_an_unknown_pattern_or_an_instance_is_diagnosed() {
    let i = index(&pegs(&[
        &UP.replace("PICK", "patterns = [[{ name = pins }]]"),
        &UP.replace("id = up", "id = down").replace(
            "PICK",
            "mirrors = [[{ name = pegs_port, instance = { by = pegs, key = { instance = linear, step = 1 } } }]]",
        ),
    ]));
    let d = i["exploded"][0]["diagnostics"].to_string();
    assert!(
        d.contains("names pattern `pins`, which `/p/stack.assembly` does not declare"),
        "{d}"
    );
    assert!(d.contains("names an instance; a mirror is named by"), "{d}");
}
