//! A generic airliner, authored as `.part` and `.assembly` text and compiled through the wasm
//! `compile` entry the Desktop calls: fuselage sections of frames, stringers and skin panels,
//! wings of spars, ribs and skins, an empennage, landing gear and engine nacelles, with rivets
//! patterned along the structure. A skeleton part publishes the frames the major assemblies
//! attach to, and the main sizes are inputs.
//!
//! `RUNMAT_AIRCRAFT_DIR=<folder>` writes the files there (the Desktop's shoot scenes load them).

use serde_json::Value;
use std::collections::BTreeMap;

mod aircraft_text;
use aircraft_text::files;

fn compile(path: &str, all: &BTreeMap<String, String>) -> Value {
    let files_json = serde_json::to_string(all).unwrap();
    let out = runmat_cad_wasm::compile(path, &all[path], &files_json, "{}");
    serde_json::from_str(&out).expect("json")
}

fn errors(v: &Value) -> Vec<String> {
    v["diagnostics"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter(|d| d["severity"] == "error")
        .map(|d| d["message"].as_str().unwrap_or("").to_string())
        .collect()
}

#[test]
fn the_aircraft_compiles_part_by_part_and_as_one_assembly() {
    let all: BTreeMap<String, String> = files()
        .into_iter()
        .map(|(n, t)| (format!("/p/aircraft/{n}"), t))
        .collect();
    if let Ok(dir) = std::env::var("RUNMAT_AIRCRAFT_DIR") {
        std::fs::create_dir_all(&dir).unwrap();
        for (path, text) in &all {
            let name = path.rsplit('/').next().unwrap();
            std::fs::write(format!("{dir}/{name}"), text).unwrap();
        }
    }
    let mut failures = Vec::new();
    for path in all.keys() {
        let v = compile(path, &all);
        let errs = errors(&v);
        if !errs.is_empty() {
            failures.push(format!("{path}: {errs:?}"));
            continue;
        }
        if path.ends_with(".part") {
            let data = &v["display"]["data"];
            let problems = data["problems"].as_array().map(|p| p.len()).unwrap_or(0);
            let bodies = data["bodies"].as_array().map(|b| b.len()).unwrap_or(0);
            let failed: Vec<String> = v["projection"]["data"]["index"]["features"]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .iter()
                .filter(|f| {
                    f["status"]
                        .as_str()
                        .is_some_and(|s| s != "ok" && s != "regenerated")
                })
                .map(|f| format!("{} {}", f["id"], f["status"]))
                .collect();
            let skeleton = path.ends_with("skeleton.part");
            if problems > 0 || (bodies == 0 && !skeleton) || !failed.is_empty() {
                failures.push(format!(
                    "{path}: {bodies} bodies, problems {:?}, features {failed:?}",
                    data["problems"]
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));

    let top = compile("/p/aircraft/airliner.assembly", &all);
    eprintln!("airliner diagnostics: {}", top["diagnostics"]);
    let display = &top["display"]["data"];
    let occurrences = display["occurrences"]
        .as_array()
        .map(|o| o.len())
        .unwrap_or(0);
    eprintln!(
        "airliner: {occurrences} occurrences of {} prototypes, bounds {}",
        display["prototypes"]
            .as_array()
            .map(|p| p.len())
            .unwrap_or(0),
        display["bounds"]
    );
    assert!(occurrences >= 2000, "{occurrences} occurrences");
    // True proportions: about 34 m long, a 34 m span, 11 m to the fin tip.
    let b = &display["bounds"];
    let ext = |k: usize| b["max"][k].as_f64().unwrap() - b["min"][k].as_f64().unwrap();
    assert!((ext(0) - 34.4).abs() < 1.0, "length {}", ext(0));
    assert!((ext(1) - 32.4).abs() < 1.0, "span {}", ext(1));
    assert!((ext(2) - 10.3).abs() < 1.0, "height {}", ext(2));
}
