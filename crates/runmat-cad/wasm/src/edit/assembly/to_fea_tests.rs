//! Results to .fea: a dynamic study's joint loads on the arm at a chosen time, turned into the
//! arm's own axes and handed to the Simulation setup guide; nothing is written.
use crate::edit::edit_json;
use crate::runtime::{compile_assembly, Files};
use serde_json::{json, Value};

const BLOCK: &str = "runmat 0.7\nstd 1\n\npart block {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])\n        l2 = line(start = [20 mm, 0 mm], end = [20 mm, 20 mm])\n        l3 = line(start = [20 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 15 mm)\n    output block = { body = feature.body.body, material = \"aluminum_6061\" }\n}\n";

/// The arm on a revolute hinge about Z, released under gravity along −Y, and a driven study.
const HINGE: &str = "runmat 0.7\nstd 1\n\nassembly hinge {\n    use part block as base\n    use part block as arm {\n        fixed = false\n        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 15 mm] } }\n    }\n    connector pivot = { occurrence = [{ name = base }], anchor = { anchor = explicit, offset = { translation = [0 mm, 0 mm, 15 mm] } } }\n    connector root = { occurrence = [{ name = arm }], anchor = { anchor = origin } }\n    motion fall = { kind = dynamic, duration = 0.5 s, gravity = [0 m/s^2, -9.80665 m/s^2, 0 m/s^2] }\n    motion sweep = { duration = 1 s, drives = [{ mate = hinge, from = 0 deg, to = 90 deg }] }\n    mate hinge = { kind = { mate = revolute }, a = pivot, b = root }\n}\n";

const PATH: &str = "/p/hinge.assembly";

fn files() -> Files {
    [("/p/block.part", BLOCK)].map(|(p, t)| (p.to_owned(), t.to_owned())).into()
}

fn edit(request: Value) -> Value {
    serde_json::from_str(&edit_json(PATH, HINGE, &files(), &request.to_string())).unwrap()
}

/// `[a, b, c]` numbers after `label` in `line`.
fn triple(line: &str, label: &str) -> [f64; 3] {
    let rest = &line[line.find(label).unwrap_or_else(|| panic!("{label} in {line}")) + label.len()..];
    let inner = &rest[rest.find('[').unwrap() + 1..rest.find(']').unwrap()];
    let v: Vec<f64> = inner.split(',').map(|x| x.trim().parse().unwrap()).collect();
    [v[0], v[1], v[2]]
}

fn norm(v: [f64; 3]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[test]
fn the_hinge_loads_on_the_arm_go_to_the_guide_in_the_arms_axes() {
    let (index, _, _) = compile_assembly(PATH, HINGE, &files()).unwrap();
    let study = index["motion"].as_array().unwrap().iter().find(|s| s["name"] == "fall").unwrap().clone();
    let times: Vec<f64> = study["times"].as_array().unwrap().iter().map(|t| t.as_f64().unwrap()).collect();

    // At release the arm lies as placed: its axes are the world's, and the hinge is its origin.
    let out = edit(json!({ "command": "motion.to-fea", "values": { "time": 0 }, "selection": ["motion:fall"] }));
    assert_eq!(out["ok"], true, "{out}");
    assert_eq!(out["files"], json!([]), "nothing is written: {out}");
    let summary = out["summary"].as_str().unwrap();
    assert!(summary.starts_with("Set up a structural simulation of block.part under the joint loads it carries at 0 s of motion study fall in hinge.assembly."), "{summary}");
    let line = summary.lines().find(|l| l.starts_with("- mate hinge")).unwrap();
    let world = &study["reactions"][0]["frames"][0][0];
    let f0: Vec<f64> = world["force"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let force = triple(line, "force ");
    for a in 0..3 {
        assert!((force[a] - f0[a]).abs() < 1e-6, "{line} vs {world}");
    }
    assert!(norm(triple(line, " at ")) < 1e-3, "the hinge sits at the arm's origin: {line}");
    assert!(summary.contains("Gravity in block.part's axes: [0, -9.80665, 0] m/s^2"), "{summary}");

    // Later the arm has turned about Z: the same load, in its turned axes.
    let asked = 0.3;
    let k = (0..times.len()).min_by(|&a, &b| (times[a] - asked).abs().total_cmp(&(times[b] - asked).abs())).unwrap();
    let out = edit(json!({ "command": "motion.to-fea", "values": { "study": "fall", "time": "300 ms", "component": "arm" } }));
    assert_eq!(out["ok"], true, "{out}");
    let summary = out["summary"].as_str().unwrap();
    assert!(summary.contains(&format!("at {} s of motion study fall", times[k])), "{summary}");
    let line = summary.lines().find(|l| l.starts_with("- mate hinge")).unwrap();
    let world = &study["reactions"][0]["frames"][k][0];
    let fw: Vec<f64> = world["force"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let local = triple(line, "force ");
    let t: Vec<f64> = study["frames"][k][0]["transform"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    // R local = world, R from the frame's pose (rows).
    for i in 0..3 {
        let back: f64 = (0..3).map(|j| t[i * 3 + j] * local[j]).sum();
        assert!((back - fw[i]).abs() < 1e-5, "{line} vs {world}");
    }
    let g = triple(summary.lines().find(|l| l.starts_with("Gravity")).unwrap(), "axes: ");
    assert!((norm(g) - 9.80665).abs() < 1e-5 && g[0].abs() > 1e-3, "gravity turns with the arm: {g:?}");
    assert!(norm(triple(line, " at ")) < 1e-3, "{line}");
}

#[test]
fn a_driven_study_or_a_fixed_component_is_refused() {
    let out = edit(json!({ "command": "motion.to-fea", "values": { "study": "sweep" } }));
    assert_eq!(out["ok"], false);
    assert!(out["message"].as_str().unwrap().contains("takes a Basic motion (dynamic) study"), "{out}");
    let out = edit(json!({ "command": "motion.to-fea", "values": { "study": "fall", "component": "base" } }));
    assert_eq!(out["ok"], false);
    assert!(out["message"].as_str().unwrap().contains("no mate loads `base`"), "{out}");
    // With one dynamic study, it is the one taken.
    let out = edit(json!({ "command": "motion.to-fea", "values": { "time": 0.1 } }));
    assert_eq!(out["ok"], true, "{out}");
}
