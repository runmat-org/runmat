//! The Desktop's two-phase `.m` compile (runmat_cad::value::m_table): the CAD host compiles,
//! lists the calls it couldn't answer, the host runs them and compiles again with the results,
//! until nothing is pending.

use runmat_cad::value::m_table::{PendingCall, TableResult};
use runmat_cad_wasm::runtime::values_host::compile_json_with_values;
use runmat_cad_wasm::runtime::{dependencies, Files};
use serde_json::{json, Value};
use std::collections::BTreeMap;

const PART: &str = "runmat 0.7\nstd 1\n\npart boss {\n    input lift : length = 8 mm\n    let base = baseRadius(lift = input.lift) as m\n    let outer = withLift(base = base, lift = input.lift) as m\n    sketch s on origin.xy {\n        c = circle(center = [0 mm, 0 mm], radius = outer)\n    }\n    feature disc = extrude(profile = sketch.s, distance = 5 mm)\n}\n";
const BASE: &str = "function rb = baseRadius(lift)\nrb = 2 * lift;\nend\n";
const WITH_LIFT: &str = "function r = withLift(base, lift)\nr = base + lift;\nend\n";

/// What the runtime returns for these two functions.
fn run(call: &PendingCall) -> TableResult {
    match call.function.as_str() {
        "baseRadius" => TableResult::Values(vec![2.0 * call.arguments[0]]),
        "withLift" => TableResult::Values(vec![call.arguments[0] + call.arguments[1]]),
        other => TableResult::Error { error: format!("no {other}") },
    }
}

#[test]
fn a_part_whose_values_call_m_functions_compiles_in_rounds_until_nothing_is_pending() {
    // The part's folder is a dependency, so the host reads its `.m` files.
    assert!(dependencies("/p/boss.part", PART).contains(&"/p/".to_owned()));
    let files: Files = [
        ("/p/baseRadius.m".to_owned(), BASE.to_owned()),
        ("/p/withLift.m".to_owned(), WITH_LIFT.to_owned()),
    ]
    .into();
    let mut values: BTreeMap<String, TableResult> = BTreeMap::new();
    let mut rounds = Vec::new();
    let out = loop {
        let options = json!({ "values": values }).to_string();
        let out: Value = serde_json::from_str(&compile_json_with_values("/p/boss.part", PART, &files, &options)).unwrap();
        let pending: Vec<PendingCall> = serde_json::from_value(out["pending_values"].clone()).unwrap_or_default();
        if pending.is_empty() {
            break out;
        }
        rounds.push(pending.iter().map(|p| p.function.clone()).collect::<Vec<_>>());
        assert!(rounds.len() < 4, "rounds: {rounds:?}");
        for call in &pending {
            assert_eq!(call.sources[0].0, format!("/p/{}.m", call.function), "the function's own file is first");
            values.insert(call.key.clone(), run(call));
        }
    };
    // `outer` takes `base` as an argument, so it waits a round for it.
    assert_eq!(rounds, [vec!["baseRadius".to_owned()], vec!["withLift".to_owned()]]);
    assert!(out["projection"].is_object(), "{}", out["diagnostics"]);
    let errors: Vec<&Value> = out["diagnostics"].as_array().unwrap().iter().filter(|d| d["severity"] == "error").collect();
    assert!(errors.is_empty(), "{errors:?}");
    // Each value names the function that computed it and its file (Properties, hover).
    let lets = &out["projection"]["data"]["index"]["lets"];
    assert_eq!(lets[0]["name"], "base");
    assert!((lets[0]["si"].as_f64().unwrap() - 0.016).abs() < 1e-12, "{lets}");
    assert_eq!(lets[1]["function"], "withLift");
    assert_eq!(lets[1]["file"], "/p/withLift.m");
    // 2 × 8 mm + 8 mm = 24 mm: the disc spans 48 mm (display bounds are in metres).
    let bounds = &out["display"]["data"]["bounds"];
    let span = bounds["max"][0].as_f64().unwrap() - bounds["min"][0].as_f64().unwrap();
    assert!((span - 0.048).abs() < 1e-5, "span {span} from {bounds}");
}

#[test]
fn an_error_the_function_raised_reaches_the_diagnostics_and_a_part_without_calls_lists_none() {
    assert!(!dependencies("/p/plain.part", "runmat 0.7\nstd 1\n\npart plain {\n    input t : length = 5 mm\n}\n").contains(&"/p/".to_owned()));
    let files: Files = [("/p/baseRadius.m".to_owned(), BASE.to_owned()), ("/p/withLift.m".to_owned(), WITH_LIFT.to_owned())].into();
    let first: Value = serde_json::from_str(&compile_json_with_values("/p/boss.part", PART, &files, "{}")).unwrap();
    let pending: Vec<PendingCall> = serde_json::from_value(first["pending_values"].clone()).unwrap();
    let values: BTreeMap<String, TableResult> =
        [(pending[0].key.clone(), TableResult::Error { error: "lift must be positive".into() })].into();
    let out: Value = serde_json::from_str(&compile_json_with_values("/p/boss.part", PART, &files, &json!({ "values": values }).to_string())).unwrap();
    assert!(out.get("pending_values").is_none(), "nothing more runs: {}", out["pending_values"]);
    let text = out["diagnostics"].to_string();
    assert!(text.contains("lift must be positive"), "{text}");
}
