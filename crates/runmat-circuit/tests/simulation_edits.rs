//! Structured edits to a `simulation` section: each changes one statement (or inserts one line)
//! and leaves the rest of the file byte for byte; values print with units.

use runmat_circuit::runtime::{self, CompileOptions, EditRequest, Files};
use runmat_circuit::syntax::parse_circuit;
use runmat_circuit::simulation::SimProbe;
use serde_json::{json, Value};

const PATH: &str = "/p/rc.circuit";

const BASE: &str = "runmat 0.7
std 1

circuit rc {
    sheet main {
        component R1 = { part = \"R0402-1K\" }
        component C1 = { part = \"C0402-1N\" }
        net GND = { global = true }
        connect = [net.IN, \"R1.1\"]
        connect = [net.OUT, \"R1.2\", \"C1.1\"]
        connect = [net.GND, \"C1.2\"]
    }
}
";

fn with_section(section: &str) -> String {
    let end = BASE.rfind('}').unwrap();
    format!("{}{section}{}", &BASE[..end], &BASE[end..])
}

const SECTION: &str = "
    simulation lowpass {
        sheet = main
        source VIN = { p = net.IN, ac = 1 V, pulse = { from = 0 V, to = 1 V, rise = 1 ns } }
        tolerance resistors = { match = \"R*\", value = 1 % }
        analysis step = tran { stop = 8 us, step = 10 ns }
        probe = [net.IN]
        measure rise = { of = step, rise_time = net.OUT }
    }
";

fn edit(text: &str, command: &str, values: Value) -> Result<String, String> {
    edit_with(text, command, values, vec![])
}

fn edit_with(text: &str, command: &str, values: Value, selection: Vec<String>) -> Result<String, String> {
    let mut files = Files::new();
    files.insert(PATH.into(), text.into());
    let req = EditRequest { command: command.into(), values: values.as_object().cloned().unwrap_or_default(), selection, options: CompileOptions::default() };
    let out = runtime::edit(PATH, text, &files, &req).map_err(|e| e.message)?;
    assert_eq!(out.files.len(), 1);
    let f = &out.files[0];
    assert!(!f.reprinted);
    assert_eq!(f.edits.len(), 1, "one text edit");
    assert!(parse_circuit(PATH, &f.text).diagnostics.is_empty());
    Ok(f.text.clone())
}

/// The lines `after` has that `before` doesn't, and the lines it lost.
fn line_diff(before: &str, after: &str) -> (Vec<String>, Vec<String>) {
    let b: Vec<&str> = before.lines().collect();
    let a: Vec<&str> = after.lines().collect();
    let added = a.iter().filter(|l| !b.contains(l)).map(|l| l.to_string()).collect();
    let removed = b.iter().filter(|l| !a.contains(l)).map(|l| l.to_string()).collect();
    (added, removed)
}

#[test]
fn add_probe_extends_the_probe_list_in_place() {
    let text = with_section(SECTION);
    let out = edit(&text, "simulate.probe", json!({ "at": "net:OUT" })).unwrap();
    assert_eq!(line_diff(&text, &out), (vec!["        probe = [net.IN, net.OUT]".into()], vec!["        probe = [net.IN]".into()]));
    assert_eq!(edit(&out, "simulate.probe", json!({ "at": "OUT" })).unwrap_err(), "V(OUT) is already probed");
    // From the net's context menu: the schematic selection names the net.
    let picked = edit_with(&text, "simulate.probe-net", json!({}), vec!["net:OUT".into()]).unwrap();
    assert_eq!(picked, out);
}

#[test]
fn a_pin_probe_adds_the_current_into_the_pin() {
    let text = with_section(SECTION);
    let out = edit(&text, "simulate.probe", json!({ "at": "R1.2" })).unwrap();
    assert_eq!(line_diff(&text, &out).0, vec!["        probe = [net.IN, \"R1.2\"]".to_string()]);
    let sim = &parse_circuit(PATH, &out).design.unwrap().simulations["lowpass"];
    assert_eq!(sim.probes[1], SimProbe::Pin { component: "R1".into(), pin: "2".into() });
    assert_eq!(edit(&out, "simulate.probe", json!({ "at": "R1.2" })).unwrap_err(), "I(R1.2) is already probed");
    // From a pin selected on the schematic.
    let picked = edit_with(&text, "simulate.probe-pin", json!({}), vec!["pin:R1.2".into()]).unwrap();
    assert_eq!(picked, out);
}

#[test]
fn a_first_probe_goes_after_the_analyses() {
    let text = with_section(&SECTION.replace("        probe = [net.IN]\n", ""));
    let out = edit(&text, "simulate.probe", json!({ "at": "net:OUT" })).unwrap();
    let lines: Vec<&str> = out.lines().collect();
    let k = lines.iter().position(|l| l.contains("probe = [net.OUT]")).unwrap();
    assert!(lines[k - 1].contains("analysis step = tran"));
    assert!(lines[k + 1].contains("measure rise"));
    assert_eq!(line_diff(&text, &out).1, Vec::<String>::new());
}

#[test]
fn add_analysis_writes_units_and_replaces_by_name() {
    let text = with_section(SECTION);
    let out = edit(&text, "simulate.ac", json!({ "start": 1000.0, "stop": 1e8, "points": 50 })).unwrap();
    let (added, removed) = line_diff(&text, &out);
    assert_eq!(added, ["        analysis response = ac { from = 1 kHz, to = 100 MHz, points = 50 }"]);
    assert!(removed.is_empty());
    // It sits under the transient, the last analysis.
    assert!(out.contains("analysis step = tran { stop = 8 us, step = 10 ns }\n        analysis response = ac"));
    let again = edit(&out, "simulate.transient", json!({ "name": "step", "stop": 2e-5, "step": 2e-8 })).unwrap();
    assert_eq!(line_diff(&out, &again), (vec!["        analysis step = tran { stop = 20 us, step = 20 ns }".into()], vec!["        analysis step = tran { stop = 8 us, step = 10 ns }".into()]));
    let mc = edit(&again, "simulate.monte-carlo", json!({ "kind": "monte_carlo", "of": "response", "runs": 500, "seed": 3 })).unwrap();
    assert_eq!(line_diff(&again, &mc).0, ["        analysis spread = monte_carlo { of = response, runs = 500, seed = 3 }"]);
    assert_eq!(edit(&again, "simulate.monte-carlo", json!({ "kind": "monte_carlo", "of": "nope" })).unwrap_err(), "the simulation has no analysis `nope` to repeat");
    assert!(edit(&again, "simulate.transient", json!({ "stop": 1e-6, "step": 1e-5 })).unwrap_err().contains("smaller step"));
}

#[test]
fn set_tolerance_replaces_the_pattern_or_adds_one() {
    let text = with_section(SECTION);
    let out = edit(&text, "simulate.set-tolerance", json!({ "match": "R*", "percent": 0.1 })).unwrap();
    assert_eq!(line_diff(&text, &out), (vec!["        tolerance resistors = { match = \"R*\", value = 0.1 % }".into()], vec!["        tolerance resistors = { match = \"R*\", value = 1 % }".into()]));
    let caps = edit(&out, "simulate.set-tolerance", json!({ "match": "C*", "percent": 5.0 })).unwrap();
    assert!(caps.contains("value = 0.1 % }\n        tolerance capacitors = { match = \"C*\", value = 5 % }\n"));
}

#[test]
fn a_design_without_a_section_gets_one() {
    let out = edit(BASE, "simulate.transient", json!({ "stop": 1e-5, "step": 1e-8 })).unwrap();
    assert!(out.contains("\n    simulation main {\n        analysis step = tran { stop = 10 us, step = 10 ns }\n    }\n}\n"), "{out}");
    let design = parse_circuit(PATH, &out).design.unwrap();
    assert!(design.simulations["main"].analyses.contains_key("step"));
    // The rest of the file is as it was.
    assert!(out.starts_with(&BASE[..BASE.rfind('}').unwrap()]));
}

#[test]
fn measure_specs_read_check_their_unit_and_print_back() {
    let text = with_section(&SECTION.replace(
        "measure rise = { of = step, rise_time = net.OUT }",
        "measure rise = { of = step, rise_time = net.OUT, spec = { max = 2.5 us } }\n        measure peak = { of = step, overshoot = net.OUT, spec = { max = 5 % } }",
    ));
    let parsed = parse_circuit(PATH, &text);
    assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
    let sim = &parsed.design.as_ref().unwrap().simulations["lowpass"];
    assert_eq!(sim.measures["rise"].spec_min, None);
    assert!((sim.measures["rise"].spec_max.unwrap() / 2.5e-6 - 1.0).abs() < 1e-12);
    assert_eq!(sim.measures["peak"].spec_max, Some(5.0));
    let printed = runmat_circuit::syntax::print_circuit(parsed.design.as_ref().unwrap()).unwrap();
    assert!(printed.contains("rise_time = net.OUT, spec = { max = 2.5 us } }"), "{printed}");
    assert!(printed.contains("overshoot = net.OUT, spec = { max = 5 % } }"), "{printed}");
    let wrong = with_section(&SECTION.replace("rise_time = net.OUT }", "rise_time = net.OUT, spec = { max = 2 kHz } }"));
    let d = parse_circuit(PATH, &wrong).diagnostics;
    assert!(d.iter().any(|d| d.message.contains("this measure is in s")), "{d:?}");
}

#[test]
fn a_run_gathers_the_model_libraries_its_section_names() {
    let model = "        model OPA171 = { library = \"../models/opa171.lib\", subckt = OPA171, pins = [\"IN+\", \"IN-\", OUT, \"V+\", \"V-\"] }\n";
    let text = with_section(&SECTION.replace("        probe = [net.IN]\n", &format!("{model}        probe = [net.IN]\n")));
    assert!(parse_circuit(PATH, &text).diagnostics.is_empty());
    let deps = runtime::dependencies(PATH, &text);
    assert!(deps.contains(&"/models/opa171.lib".to_string()), "{deps:?}");
}

#[test]
fn noise_and_temperature_sweeps_are_added_with_units() {
    let text = with_section(SECTION);
    let out = edit(&text, "simulate.noise", json!({ "output": "net.OUT", "input": "VIN", "start": 10.0, "stop": 1e7, "points": 20 })).unwrap();
    assert_eq!(line_diff(&text, &out).0, ["        analysis hiss = noise { output = net.OUT, input = VIN, from = 10 Hz, to = 10 MHz, points = 20 }"]);
    let out = edit(&out, "simulate.temperature", json!({ "of": "step", "values": "-40, 25, 85" })).unwrap();
    assert!(out.contains("        analysis corners = temperature { of = step, values = [-40 degC, 25 degC, 85 degC] }"), "{out}");
    assert_eq!(edit(&text, "simulate.noise", json!({ "output": "OUT", "input": "V9", "start": 10.0, "stop": 1e3 })).unwrap_err(), "the simulation has no source `V9` to refer the noise to");
    assert_eq!(edit(&text, "simulate.temperature", json!({ "of": "step", "values": "hot" })).unwrap_err(), "temperatures are numbers in °C (`-40, 25, 85`)");
    assert_eq!(edit(&out, "simulate.temperature", json!({ "of": "corners", "values": [0.0] })).unwrap_err(), "`corners` already reruns another analysis; name that one");
}
