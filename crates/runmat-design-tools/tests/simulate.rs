//! `design.simulate` on the daq showcase with a `simulation` section for its analog channel.

use runmat_design_tools::{call, catalog, Files};
use serde_json::json;

const DAQ: &str = include_str!("../../runmat-circuit/tests/projects/daq/daq.circuit");

/// The daq showcase without its own `simulation` sections, so a test adds the section it checks.
fn bare(base: &str) -> String {
    let mut s = base.to_string();
    while let Some(i) = s.find("\n    simulation ") {
        let close = "\n    }\n";
        let j = i + 1 + s[i + 1..].find(close).expect("a closed section") + close.len();
        s.replace_range(i + 1..j, "");
    }
    s
}

const SECTION: &str = r#"
    simulation front_end {
        sheet = channel
        source VIN = { p = net.IN, ac = 1 V, pulse = { from = 0.25 V, to = 1.75 V, delay = 2 us, rise = 10 ns } }
        source V5 = { p = "net.+5V", dc = 5 V }
        source V3 = { p = "net.+3V3", dc = 3.3 V }
        source VR = { p = net.VREF, dc = 4.096 V }
        model OPA171 = { opamp = { a0 = 1000000, gbw = 3 MHz, sr = 1500000, rout = 10 ohm, vdrop = 0.02 V }, pins = ["IN+", "IN-", OUT, "V+", "V-"] }
        tolerance resistors = { match = "R*", value = 1 % }
        tolerance capacitors = { match = "C*", value = 5 % }
        analysis step = tran { stop = 20 us, step = 10 ns }
        analysis response = ac { from = 100 Hz, to = 100 MHz, points = 50 }
        analysis spread = monte_carlo { of = response, runs = 100, seed = 7 }
        probe = [net.IN, net.AIN]
        measure f3db = { of = response, bandwidth = net.AIN }
        measure f3db_spread = { of = spread, bandwidth = net.AIN }
    }
"#;

fn files() -> Files {
    let daq = bare(DAQ);
    let end = daq.rfind('}').unwrap();
    [("/p/daq.circuit".to_owned(), format!("{}{SECTION}{}", &daq[..end], &daq[end..]))].into()
}

#[test]
fn the_catalog_lists_design_simulate() {
    assert!(catalog().iter().any(|t| t.name == "design.simulate"));
}

#[test]
fn simulate_returns_waveforms_measures_and_operations() {
    let out = call("design.simulate", json!({ "path": "/p/daq.circuit", "maxPoints": 50 }), &files()).unwrap();
    assert_eq!(out["simulation"], "front_end");
    let analyses = out["analyses"].as_array().unwrap();
    let step = analyses.iter().find(|a| a["name"] == "step").unwrap();
    assert_eq!(step["kind"], "tran");
    assert_eq!(step["x"].as_array().unwrap().len(), 50);
    let ain = step["signals"].as_array().unwrap().iter().find(|s| s["probe"] == "AIN" && s["quantity"] == "voltage").unwrap();
    let last = ain["values"].as_array().unwrap().last().unwrap().as_f64().unwrap();
    assert!((last - 3.5).abs() < 0.01, "{last}");
    let response = analyses.iter().find(|a| a["name"] == "response").unwrap();
    assert!(response["signals"][1]["magnitudeDb"].is_array());
    let m = out["measures"].as_array().unwrap();
    let f3 = m.iter().find(|x| x["name"] == "f3db").unwrap()["value"].as_f64().unwrap();
    assert!(f3 > 140e3 && f3 < 160e3, "{f3}");
    let spread = m.iter().find(|x| x["name"] == "f3db_spread").unwrap();
    assert_eq!(spread["runs"], 100);
    assert_eq!(spread["histogramCounts"].as_array().unwrap().iter().map(|c| c.as_u64().unwrap()).sum::<u64>(), 100);
    assert_eq!(out["usage"]["measurementVersion"], "rsua-1");
    assert!(out["usage"]["total"].as_str().unwrap().parse::<u64>().unwrap() > 0);
    assert!(out["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("ADC")));
}

#[test]
fn a_circuit_without_a_section_says_how_to_add_one() {
    let files: Files = [("/p/daq.circuit".to_owned(), bare(DAQ))].into();
    let e = call("design.simulate", json!({ "path": "/p/daq.circuit" }), &files).unwrap_err();
    assert_eq!(e.code, "no_simulation");
}

#[test]
fn the_report_names_units_overlays_and_tolerances() {
    let out = call("design.simulate", json!({ "path": "/p/daq.circuit", "maxPoints": 40 }), &files()).unwrap();
    assert_eq!(out["probes"], json!(["IN", "AIN"]));
    let analyses = out["analyses"].as_array().unwrap();
    let step = analyses.iter().find(|a| a["name"] == "step").unwrap();
    assert_eq!((step["xName"].as_str(), step["xUnit"].as_str()), (Some("time"), Some("s")));
    let spread = analyses.iter().find(|a| a["name"] == "spread").unwrap();
    assert_eq!((spread["kind"].as_str(), spread["of"].as_str(), spread["ofKind"].as_str()), (Some("monte_carlo"), Some("response"), Some("ac")));
    let runs = spread["runs"].as_array().unwrap();
    assert_eq!(runs.len(), 20);
    assert!(runs[0]["signals"][1]["magnitudeDb"].as_array().unwrap().len() == runs[0]["x"].as_array().unwrap().len());
    let m = out["measures"].as_array().unwrap();
    let f3 = m.iter().find(|x| x["name"] == "f3db").unwrap();
    assert_eq!((f3["unit"].as_str(), f3["kind"].as_str(), f3["probe"].as_str()), (Some("Hz"), Some("bandwidth"), Some("AIN")));
    let tol = out["tolerances"].as_array().unwrap();
    assert!(tol.iter().any(|t| t["from"] == "tolerance resistors" && t["plusPercent"] == 1.0));
    assert!(out.get("stopped").is_none());
}

#[test]
fn a_limited_run_stops_under_the_grant_with_what_it_solved() {
    let full = call("design.simulate", json!({ "path": "/p/daq.circuit" }), &files()).unwrap();
    let total: u64 = full["usage"]["total"].as_str().unwrap().parse().unwrap();
    let limit = total / 2;
    let out = call("design.simulate", json!({ "path": "/p/daq.circuit", "limitOps": limit.to_string() }), &files()).unwrap();
    let used: u64 = out["usage"]["total"].as_str().unwrap().parse().unwrap();
    assert!(used <= limit, "{used} > {limit}");
    let stopped = out["stopped"].as_str().unwrap();
    assert!(stopped.contains("granted operations"), "{stopped}");
    let e = call("design.simulate", json!({ "path": "/p/daq.circuit", "limitOps": "lots" }), &files()).unwrap_err();
    assert_eq!(e.code, "invalid_arguments");
}

/// The Desktop's simulation scene (`runmat.simulation.scene`): the daq showcase's own section
/// with a current probe on RO.1 and a measurement of its peak, run through `design.simulate`.
/// The capture script (desktop/scripts/site-captures/simulation-scene) pastes the same text.
/// With RUNMAT_SIMULATION_SCENE_OUT set, the report is written there for the Desktop.
#[test]
fn the_desktop_simulation_scene_reports_the_current_probe() {
    let text = DAQ
        .replace("probe = [net.IN, net.INF, net.AMP, net.AIN]", "probe = [net.IN, net.AMP, net.AIN, \"RO.1\"]")
        .replace("        measure rise = {", "        measure i_ro = { of = step, max = \"RO.1\" }\n        measure rise = {");
    assert!(text.contains("\"RO.1\"]") && text.contains("measure i_ro"), "the daq section changed; update the scene");
    let files: Files = [("/p/daq.circuit".to_owned(), text)].into();
    let out = call("design.simulate", json!({ "path": "/p/daq.circuit", "maxPoints": 160 }), &files).unwrap();
    assert_eq!(out["probes"], json!(["IN", "AMP", "AIN", "RO.1"]));
    let i = out["measures"].as_array().unwrap().iter().find(|m| m["name"] == "i_ro").unwrap();
    assert_eq!((i["quantity"].as_str(), i["unit"].as_str()), (Some("current"), Some("A")));
    // The op-amp slews at 1.5 V/µs into RO and CO (1 nF): while it slews, RO carries
    // C·dV/dt = 1 nF × 1.5 V/µs = 1.5 mA, and that is the peak.
    let peak = i["value"].as_f64().unwrap();
    assert!((peak / 1.5e-3 - 1.0).abs() < 0.03, "peak {peak}");
    if let Ok(path) = std::env::var("RUNMAT_SIMULATION_SCENE_OUT") {
        std::fs::write(path, serde_json::to_string(&out).unwrap() + "\n").unwrap();
    }
}

/// The Desktop's simulation scene on the camera drone: the ESC's current sense, the drone's
/// esc.circuit's second simulation. Every measurement carries a spec, so each row reads pass or
/// fail. With RUNMAT_SIMULATION_SCENE_ESC_OUT set, the report is written there for the Desktop.
#[test]
fn the_desktop_simulation_scene_reads_the_esc_current_sense() {
    const ESC: &str = include_str!("../../runmat-cad/wasm/tests/drone/esc.circuit");
    let files: Files = [("/p/esc.circuit".to_owned(), ESC.to_owned())].into();
    let out = call("design.simulate", json!({ "path": "/p/esc.circuit", "simulation": "current_sense", "maxPoints": 160 }), &files).unwrap();
    assert_eq!(out["probes"], json!(["ISENSE", "SRC"]));
    let m = |name: &str| out["measures"].as_array().unwrap().iter().find(|m| m["name"] == name).unwrap().clone();
    for name in ["f3db", "sense", "rise", "settle"] {
        assert_eq!(m(name)["pass"], json!(true), "{name}: {}", m(name));
    }
    // Closed form: the corner is 1 / (2 pi R C) = 1 / (2 pi x 1 k x 10 nF) = 15.92 kHz; 20 A
    // through 0.5 mOhm reads 10 mV; the 10-90 % rise of one RC is tau ln 9 = 21.97 us.
    let f = m("f3db")["value"].as_f64().unwrap();
    assert!((f / 15_915.5 - 1.0).abs() < 0.01, "f3db {f}");
    let v = m("sense")["value"].as_f64().unwrap();
    assert!((v / 10e-3 - 1.0).abs() < 0.01, "sense {v}");
    let r = m("rise")["value"].as_f64().unwrap();
    assert!((r / (10e-6 * 9f64.ln()) - 1.0).abs() < 0.03, "rise {r}");
    // Settling to 1 % takes tau ln 100 = 46.05 us from the step (at 20 us, the window start).
    let t = m("settle")["value"].as_f64().unwrap();
    assert!(t > 44e-6 && t < 50e-6, "settle {t}");
    let spread = m("f3db_spread");
    let y = spread["yieldPercent"].as_f64().unwrap();
    assert!(y > 95.0 && y <= 100.0, "yield {y}");
    if let Ok(path) = std::env::var("RUNMAT_SIMULATION_SCENE_ESC_OUT") {
        std::fs::write(path, serde_json::to_string(&out).unwrap() + "\n").unwrap();
    }
}
