//! Circuit values computed by `.m` functions (PCB-38, report 12 §2.3–2.6): a component's value
//! calls a function of the circuit's folder, the compile lists the call for the host to run,
//! and the next compile takes the result in the unit after `as`.

use runmat_circuit::runtime::values::{fill_values, PENDING_VALUES};
use runmat_circuit::runtime::{self, CompileOptions, Files, Session};
use runmat_circuit::syntax::{parse_circuit, print_circuit};
use runmat_design_syntax::m_calls::TableResult;
use serde_json::Value;

const CIRCUIT: &str = "/p/io.circuit";
const LAYOUT: &str = "/p/io.circuit.layout";
const RC: &str = "function r = rcResistor(fc, c)\n% The resistor that puts an RC low-pass corner at fc.\n    r = 1 / (2 * pi * fc * c);\nend\n";
const CALL: &str = "rcResistor(fc = 160 Hz, c = 100 nF) as ohm";

fn files(function: &str) -> Files {
    let circuit = include_str!("projects/io/io.circuit").replacen(
        "component RS = { part = \"R0402-10K\" }",
        &format!("component RS = {{ part = \"R0402-10K\", value = {CALL} }}"),
        1,
    );
    assert!(circuit.contains(CALL), "the fixture has component RS");
    [
        (CIRCUIT, circuit),
        (LAYOUT, include_str!("projects/io/io.circuit.layout").to_owned()),
        ("/p/io.circuit.lock", include_str!("projects/io/io.circuit.lock").to_owned()),
        ("/p/rcResistor.m", function.to_owned()),
    ]
    .into_iter()
    .map(|(p, t)| (p.to_owned(), t))
    .collect()
}

fn compile(files: &Files, options: &CompileOptions) -> Value {
    serde_json::from_str(&Session::new().compile_json(LAYOUT, &files[LAYOUT], files, options)).unwrap()
}

#[test]
fn a_component_value_from_a_function_waits_for_the_runtime_then_takes_its_result() {
    let files = files(RC);
    // The circuit's folder holds its functions: the host lists it.
    assert!(runtime::dependencies(CIRCUIT, &files[CIRCUIT]).contains(&"/p/".to_owned()));

    // First compile: the call is pending, with its arguments in SI, in declared order.
    let out = compile(&files, &CompileOptions::default());
    assert!(out.get("index").is_none(), "{out}");
    assert_eq!(out["diagnostics"][0]["message"], PENDING_VALUES);
    assert_eq!(out["diagnostics"][0]["severity"], "info");
    let pending = out["pending_values"].as_array().expect("pending values");
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0]["function"], "rcResistor");
    let args: Vec<f64> = pending[0]["arguments"].as_array().unwrap().iter().map(|a| a.as_f64().unwrap()).collect();
    assert_eq!(args.len(), 2);
    assert_eq!(args[0], 160.0);
    assert!((args[1] - 1e-7).abs() < 1e-20, "{args:?}");
    assert_eq!(pending[0]["sources"][0][0], "/p/rcResistor.m");
    let key = pending[0]["key"].as_str().unwrap().to_owned();

    // The host ran it: the next compile has the board and no problem.
    let r = 1.0 / (2.0 * std::f64::consts::PI * 160.0 * 1e-7);
    let options = CompileOptions { values: [(key.clone(), TableResult::Values(vec![r]))].into(), ..Default::default() };
    let out = compile(&files, &options);
    assert!(out.get("index").is_some(), "{}", out["diagnostics"]);
    assert!(out.get("pending_values").is_none());
    assert!(out["diagnostics"].as_array().unwrap().iter().all(|d| d["severity"] != "error"), "{}", out["diagnostics"]);

    // The value reads the result in ohms, and the file prints the call back, not the number.
    let mut design = parse_circuit(CIRCUIT, &files[CIRCUIT]).design.unwrap();
    let filled = fill_values(&mut design, CIRCUIT, &files, &options.values);
    assert!(filled.pending.is_empty() && filled.errors.is_empty());
    let rs = design.sheets.values().flat_map(|s| &s.components).find(|c| c.name == "RS").unwrap();
    assert!((rs.params["value"].as_quantity().unwrap().value - r).abs() < 1e-9);
    assert!(print_circuit(&design).unwrap().contains(&format!("value = {CALL}")));

    // A function that raised an error is an error on the component.
    let failed = CompileOptions { values: [(key, TableResult::Error { error: "c must be positive".into() })].into(), ..Default::default() };
    let out = compile(&files, &failed);
    let message = out["diagnostics"][0]["message"].as_str().unwrap();
    assert_eq!(message, "`rcResistor` raised an error: c must be positive");
    assert_eq!(out["diagnostics"][0]["path"], CIRCUIT);
}

#[test]
fn a_value_function_must_be_deterministic_and_take_the_arguments_it_declares() {
    let random = files("function r = rcResistor(fc, c)\n    r = rand() / (2 * pi * fc * c);\nend\n");
    let out = compile(&random, &CompileOptions::default());
    let message = out["diagnostics"][0]["message"].as_str().unwrap();
    assert!(message.starts_with("`rcResistor` uses `rand` (unseeded randomness)"), "{message}");

    let narrow = files("function r = rcResistor(fc)\n    r = 1 / fc;\nend\n");
    let out = compile(&narrow, &CompileOptions::default());
    assert_eq!(out["diagnostics"][0]["message"], "`rcResistor` takes (fc), not `c`");

    let mut missing = files(RC);
    missing.remove("/p/rcResistor.m");
    let out = compile(&missing, &CompileOptions::default());
    assert_eq!(out["diagnostics"][0]["message"], "no `rcResistor.m` in this folder (or its private/)");
}
