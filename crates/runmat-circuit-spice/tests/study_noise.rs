//! Noise and temperature analyses and pin-current measurements in a `.circuit` simulation
//! section, on an RC low-pass with closed forms: R = 1 kΩ and C = 1 nF.
//!
//! - The output noise of an RC integrated over all frequencies is √(kT/C), whatever R is.
//! - Below the corner the output density is the resistor's √(4kTR).
//! - Both scale with √T, so a temperature sweep of the noise follows √(T/T₀).
//! - A 1 V step into the RC drives 1 V / 1 kΩ = 1 mA into C1.1 at the edge.

use runmat_circuit::fixture::standard_pool;
use runmat_circuit::syntax::{parse_circuit, print_circuit};
use runmat_circuit_spice::{report, study};

const K_B: f64 = 1.380_649e-23;
const R: f64 = 1e3;
const C: f64 = 1e-9;

const RC: &str = r#"runmat 0.7
std 1

circuit rc {
    sheet main {
        component R1 = { part = "R0402-1K" }
        component C1 = { part = "C0402-1N" }
        net GND = { global = true }
        connect = [net.IN, "R1.1"]
        connect = [net.OUT, "R1.2", "C1.1"]
        connect = [net.GND, "C1.2"]
    }

    simulation lowpass {
        sheet = main
        source VIN = { p = net.IN, ac = 1 V, pulse = { from = 0 V, to = 1 V, delay = 0 s, rise = 1 ns } }
        analysis step = tran { stop = 8 us, step = 10 ns }
        analysis hiss = noise { output = net.OUT, input = VIN, from = 1 Hz, to = 10 GHz, points = 50 }
        analysis hot = temperature { of = hiss, values = [-40 degC, 27 degC, 85 degC] }
        analysis warm = temperature { of = step, values = [0 degC, 50 degC] }
        probe = [net.OUT, "C1.1"]
        measure en = { of = hiss, output_noise = net.OUT }
        measure en_in = { of = hiss, input_noise = net.OUT, from = 1 Hz, to = 1 kHz }
        measure density = { of = hiss, noise_density = net.OUT, at = 100 Hz, spec = { max = 5 nV } }
        measure en_hot = { of = hot, output_noise = net.OUT }
        measure ipk = { of = step, max = "C1.1", spec = { min = 900 uA, max = 1.1 mA } }
        measure ipk_warm = { of = warm, max = "C1.1" }
    }
}
"#;

fn design() -> runmat_circuit::ir::Design {
    let parsed = parse_circuit("/p/rc.circuit", RC);
    assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
    parsed.design.unwrap()
}

fn kelvin(c: f64) -> f64 {
    c + 273.15
}

#[test]
fn the_section_prints_back_and_reads_to_the_same_design() {
    let d = design();
    let text = print_circuit(&d).unwrap();
    for line in [
        "analysis hiss = noise { output = net.OUT, input = VIN, from = 1 Hz, to = 10 GHz, points = 50 }",
        "analysis hot = temperature { of = hiss, values = [-40 degC, 27 degC, 85 degC] }",
        "measure ipk = { of = step, max = \"C1.1\", spec = { min = 900 uA, max = 1.1 mA } }",
        "measure density = { of = hiss, noise_density = net.OUT, at = 100 Hz, spec = { max = 5 nV } }",
    ] {
        assert!(text.contains(line), "missing `{line}` in\n{text}");
    }
    let again = parse_circuit("/p/rc.circuit", &text);
    assert!(again.diagnostics.is_empty(), "{:?}", again.diagnostics);
    // The same section, apart from where it sits in the printed text.
    let mut a = again.design.unwrap().simulations["lowpass"].clone();
    a.source = d.simulations["lowpass"].source.clone();
    assert_eq!(a, d.simulations["lowpass"]);
}

#[test]
fn rc_noise_integrates_to_kt_over_c_and_starts_at_the_resistors_density() {
    let d = design();
    let s = study::run(&d, &standard_pool(), "lowpass", &|_| None, None).unwrap();
    let t = kelvin(27.0);
    let en = s.measures["en"].value.unwrap();
    let want = (K_B * t / C).sqrt();
    assert!(
        (en / want - 1.0).abs() < 0.01,
        "√(kT/C): {en:e} vs {want:e}"
    );
    let density = s.measures["density"].value.unwrap();
    let want = (4.0 * K_B * t * R).sqrt();
    assert!(
        (density / want - 1.0).abs() < 1e-3,
        "√(4kTR): {density:e} vs {want:e}"
    );
    // Below the corner the gain is 1, so the input-referred noise from 1 Hz to 1 kHz is
    // √(4kTR · 999 Hz).
    let en_in = s.measures["en_in"].value.unwrap();
    let want = (4.0 * K_B * t * R * 999.0).sqrt();
    assert!(
        (en_in / want - 1.0).abs() < 1e-3,
        "input-referred: {en_in:e} vs {want:e}"
    );
}

#[test]
fn a_temperature_sweep_reruns_the_noise_at_each_temperature() {
    let d = design();
    let s = study::run(&d, &standard_pool(), "lowpass", &|_| None, None).unwrap();
    let m = &s.measures["en_hot"];
    assert_eq!(m.temperatures, [-40.0, 27.0, 85.0]);
    assert!(m.error.is_none(), "{:?}", m.error);
    for (t, v) in m.temperatures.iter().zip(&m.runs) {
        let want = (K_B * kelvin(*t) / C).sqrt();
        assert!((v / want - 1.0).abs() < 0.01, "{t} °C: {v:e} vs {want:e}");
    }
    let ratio = m.runs[2] / m.runs[0];
    let want = (kelvin(85.0) / kelvin(-40.0)).sqrt();
    assert!(
        (ratio / want - 1.0).abs() < 1e-6,
        "√(T₂/T₁): {ratio} vs {want}"
    );
}

#[test]
fn a_measure_reads_the_current_into_a_pin() {
    let d = design();
    let s = study::run(&d, &standard_pool(), "lowpass", &|_| None, None).unwrap();
    let ipk = s.measures["ipk"].value.unwrap();
    assert!(
        (ipk / (1.0 / R) - 1.0).abs() < 0.02,
        "peak charging current {ipk:e}"
    );
    // A resistor and capacitor with no temperature coefficient charge the same at 0 and 50 °C.
    let warm = &s.measures["ipk_warm"];
    assert_eq!(warm.temperatures, [0.0, 50.0]);
    assert!(
        (warm.runs[0] - ipk).abs() < 1e-9 && (warm.runs[1] - ipk).abs() < 1e-9,
        "{:?}",
        warm.runs
    );
}

#[test]
fn the_report_carries_noise_figures_temperature_runs_and_current_units() {
    let d = design();
    let s = study::run(&d, &standard_pool(), "lowpass", &|_| None, None).unwrap();
    let out = serde_json::to_value(report::report(
        &s,
        &d.simulations["lowpass"],
        &Default::default(),
    ))
    .unwrap();
    let analyses = out["analyses"].as_array().unwrap();
    let find = |n: &str| {
        analyses
            .iter()
            .find(|a| a["name"] == n)
            .unwrap_or_else(|| panic!("no analysis {n}"))
    };

    let hiss = find("hiss");
    assert_eq!(
        (hiss["kind"].as_str(), hiss["xUnit"].as_str()),
        (Some("noise"), Some("Hz"))
    );
    assert_eq!(hiss["signals"][0]["quantity"], "noise");
    let noise = &hiss["noise"];
    assert_eq!(
        (noise["output"].as_str(), noise["input"].as_str()),
        (Some("OUT"), Some("VIN"))
    );
    assert_eq!(
        noise["outputDensity"].as_array().unwrap().len(),
        hiss["x"].as_array().unwrap().len()
    );
    let top = &noise["contributors"][0];
    assert_eq!(top["device"], "r1");
    assert!(
        (top["percent"].as_f64().unwrap() - 100.0).abs() < 0.5,
        "{top}"
    );

    let hot = find("hot");
    assert_eq!(
        (
            hot["kind"].as_str(),
            hot["of"].as_str(),
            hot["ofKind"].as_str()
        ),
        (Some("temperature"), Some("hiss"), Some("noise"))
    );
    let temps: Vec<f64> = hot["runs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["temperature"].as_f64().unwrap())
        .collect();
    assert_eq!(temps, [-40.0, 27.0, 85.0]);
    let warm = find("warm");
    assert_eq!(
        (
            warm["kind"].as_str(),
            warm["ofKind"].as_str(),
            warm["xUnit"].as_str()
        ),
        (Some("temperature"), Some("tran"), Some("s"))
    );
    assert_eq!(warm["runs"][1]["signals"][1]["probe"], "C1.1");

    let measures = out["measures"].as_array().unwrap();
    let m = |n: &str| measures.iter().find(|x| x["name"] == n).unwrap();
    assert_eq!(
        (
            m("ipk")["quantity"].as_str(),
            m("ipk")["unit"].as_str(),
            m("ipk")["pass"].as_bool()
        ),
        (Some("current"), Some("A"), Some(true))
    );
    assert_eq!(
        (
            m("density")["unit"].as_str(),
            m("density")["pass"].as_bool()
        ),
        (Some("V/√Hz"), Some(true))
    );
    let hot = m("en_hot");
    assert_eq!(hot["temperatures"].as_array().unwrap().len(), 3);
    assert_eq!(hot["values"].as_array().unwrap().len(), 3);
    assert!(hot.get("histogramCounts").is_none());
}

#[test]
fn errors_name_what_is_wrong() {
    let bad = |from: &str, to: &str| {
        let text = RC.replace(from, to);
        let parsed = parse_circuit("/p/rc.circuit", &text);
        parsed
            .diagnostics
            .iter()
            .map(|d| d.message.clone())
            .collect::<Vec<_>>()
            .join("; ")
    };
    assert!(bad("output_noise = net.OUT }", "output_noise = \"C1.1\" }").contains("output net"));
    assert!(bad("[-40 degC, 27 degC, 85 degC]", "[-40 V]").contains("degC"));
    assert!(bad("values = [0 degC, 50 degC]", "values = []").contains("list of temperatures"));
    assert!(bad(
        "spec = { min = 900 uA, max = 1.1 mA }",
        "spec = { max = 1 V }"
    )
    .contains("write the limit in A"));
}

#[test]
fn a_noise_run_stopped_at_its_grant_reports_what_it_spent() {
    // Only the noise analysis, with a grant too small for its operating point: the study
    // stops, and the operations the stopped run spent still count.
    let text = RC
        .replace("        analysis step = tran { stop = 8 us, step = 10 ns }\n", "")
        .replace("        analysis hot = temperature { of = hiss, values = [-40 degC, 27 degC, 85 degC] }\n", "")
        .replace("        analysis warm = temperature { of = step, values = [0 degC, 50 degC] }\n", "")
        .replace("        measure en_hot = { of = hot, output_noise = net.OUT }\n", "")
        .replace("        measure ipk = { of = step, max = \"C1.1\", spec = { min = 900 uA, max = 1.1 mA } }\n", "")
        .replace("        measure ipk_warm = { of = warm, max = \"C1.1\" }\n", "");
    let parsed = parse_circuit("/p/rc.circuit", &text);
    assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
    let d = parsed.design.unwrap();
    let limit = 20;
    let s = study::run_with(
        &d,
        &standard_pool(),
        "lowpass",
        &|_| None,
        study::RunOptions {
            threads: None,
            limit: Some(limit),
        },
    )
    .unwrap();
    let stopped = s.stopped.as_deref().expect("the grant stops the run");
    assert!(stopped.contains("noise analysis `hiss`"), "{stopped}");
    let spent = s.usage.total().unwrap().operations();
    assert!(spent > 0 && spent <= limit, "spent {spent} of {limit}");
}
