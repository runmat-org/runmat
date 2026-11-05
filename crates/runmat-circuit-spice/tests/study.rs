//! A `simulation` section on the daq showcase's analog channel, run on the native engine: the
//! source text reads and prints back, and the step response, frequency response and Monte
//! Carlo spread match the channel's closed-form behaviour.

use runmat_circuit::fixture::standard_pool;
use runmat_circuit::syntax::{parse_circuit, print_circuit};
use runmat_circuit_spice::study;

pub const SECTION: &str = r#"
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
        analysis response = ac { from = 100 Hz, to = 100 MHz, points = 100 }
        analysis spread = monte_carlo { of = response, runs = 400, seed = 7 }
        probe = [net.IN, net.INF, net.AMP, net.AIN]
        measure gain = { of = response, gain = net.AIN, at = 1 kHz }
        measure f3db = { of = response, bandwidth = net.AIN }
        measure rise = { of = step, rise_time = net.AIN }
        measure f3db_spread = { of = spread, bandwidth = net.AIN }
    }
"#;

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

pub fn daq_with_section() -> String {
    let base = bare(include_str!("../../runmat-circuit/tests/projects/daq/daq.circuit"));
    let end = base.rfind('}').unwrap();
    format!("{}{SECTION}{}", &base[..end], &base[end..])
}

#[test]
fn section_reads_and_prints_back() {
    let parsed = parse_circuit("/p/daq.circuit", &daq_with_section());
    assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
    let design = parsed.design.unwrap();
    let sim = &design.simulations["front_end"];
    assert_eq!(sim.sheet.as_deref(), Some("channel"));
    assert_eq!(sim.analyses.len(), 3);
    assert_eq!(sim.probes.iter().map(|p| p.name()).collect::<Vec<_>>(), ["IN", "INF", "AMP", "AIN"]);
    let printed = print_circuit(&design).unwrap();
    // Values print with their units, as an engineer writes them.
    for line in [
        "analysis step = tran { stop = 20 us, step = 10 ns }",
        "analysis response = ac { from = 100 Hz, to = 100 MHz, points = 100 }",
        "tolerance capacitors = { match = \"C*\", value = 5 % }",
        "source VR = { p = net.VREF, dc = 4.096 V }",
        "pulse = { from = 250 mV, to = 1.75 V, delay = 2 us, rise = 10 ns, fall = 10 ns }",
        "gbw = 3 MHz",
        "rout = 10 ohm",
    ] {
        assert!(printed.contains(line), "`{line}` not in\n{printed}");
    }
    let again = parse_circuit("/p/daq.circuit", &printed);
    assert!(again.diagnostics.is_empty(), "{:?}\n{printed}", again.diagnostics);
    let strip = |mut m: std::collections::BTreeMap<String, runmat_circuit::simulation::SimulationDef>| {
        m.values_mut().for_each(|s| s.source = Default::default());
        m
    };
    assert_eq!(strip(again.design.unwrap().simulations), strip(design.simulations));
}

#[test]
fn errors_name_the_statement() {
    let bad = daq_with_section().replace("analysis step = tran { stop = 20 us, step = 10 ns }", "analysis step = transient { stop = 20 us }");
    let parsed = parse_circuit("/p/daq.circuit", &bad);
    assert!(parsed.diagnostics.iter().any(|d| d.message.contains("`transient` is not an analysis")), "{:?}", parsed.diagnostics);
}

#[test]
fn channel_step_response_bandwidth_and_spread() {
    let design = parse_circuit("/p/daq.circuit", &daq_with_section()).design.unwrap();
    let pool = standard_pool();
    let s = study::run(&design, &pool, "front_end", &|_| None, None).unwrap();
    assert!(s.warnings.iter().any(|w| w.contains("ADC")), "{:?}", s.warnings);
    // Non-inverting gain 1 + RF/RG = 2 (6.02 dB), less the 1 Ω-level output divider.
    let gain = s.measures["gain"].value.unwrap();
    assert!((gain - 20.0 * 2f64.log10()).abs() < 0.01, "gain {gain} dB");
    // Poles: RS·CF at 159 kHz, the amplifier at GBW/2 = 1.5 MHz, RO·CO at 3.2 MHz.
    let f3 = s.measures["f3db"].value.unwrap();
    assert!(f3 > 140e3 && f3 < 160e3, "f3db {f3}");
    let rise = s.measures["rise"].value.unwrap();
    // A dominant 1 µs pole rises 10–90 % in 2.2 τ.
    assert!(rise > 2.0e-6 && rise < 2.5e-6, "rise {rise}");
    let st = s.measures["f3db_spread"].stats.clone().unwrap();
    assert_eq!(st.n, 400);
    // f ≈ 1/(2π·RS·CF): σ_f/f ≈ √(σ_R² + σ_C²) with σ = tolerance/3.
    let rel = st.std / st.mean;
    let want = ((0.01f64 / 3.0).powi(2) + (0.05f64 / 3.0).powi(2)).sqrt();
    assert!((rel / want - 1.0).abs() < 0.2, "spread {rel} vs {want}");
    assert!(s.usage.total().unwrap().operations() > 0);
    let tran = s.results["step"].real().unwrap();
    for (_, sig) in &s.probes {
        assert!(tran.get(sig).is_some(), "probe {sig}");
    }
    let ain = tran.at(&s.probes[3].1, 19e-6).unwrap();
    assert!((ain - 3.5).abs() < 0.01, "settles to 2 × 1.75 V: {ain}");
}

fn with_section_lines(drop: &[&str], add: &str) -> runmat_circuit::ir::Design {
    let mut text = daq_with_section();
    for d in drop {
        text = text.lines().filter(|l| !l.contains(d)).collect::<Vec<_>>().join("\n");
    }
    let text = text.replace("        probe = [", &format!("{add}\n        probe = ["));
    let parsed = parse_circuit("/p/daq.circuit", &text);
    assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
    parsed.design.unwrap()
}

#[test]
fn monte_carlo_takes_each_components_own_tolerance() {
    // No tolerance statements: the parts' own bands apply (R0402 ±1 %, C0402-1N ±5 %).
    let design = with_section_lines(&["tolerance resistors", "tolerance capacitors"], "");
    let s = study::run(&design, &standard_pool(), "front_end", &|_| None, None).unwrap();
    let rs = s.tolerances.iter().find(|t| t.elements == "rs").expect("RS varies by its part's ±1 %");
    assert_eq!(rs.from, "component");
    assert!((rs.band.0 - 0.99).abs() < 1e-12 && (rs.band.1 - 1.01).abs() < 1e-12);
    let cf = s.tolerances.iter().find(|t| t.elements == "cf").unwrap();
    assert!((cf.band.1 - 1.05).abs() < 1e-12);
    let st = s.measures["f3db_spread"].stats.clone().unwrap();
    let rel = st.std / st.mean;
    let want = ((0.01f64 / 3.0).powi(2) + (0.05f64 / 3.0).powi(2)).sqrt();
    assert!((rel / want - 1.0).abs() < 0.2, "spread {rel} vs {want}");
}

#[test]
fn a_section_statement_replaces_the_component_band_it_matches() {
    // RS and CF held to ±0.1 %: the corner spreads by √2·0.1 %/3 instead.
    let design = with_section_lines(&["tolerance resistors", "tolerance capacitors"], "        tolerance input_rc = { match = \"?[SF]\", value = 0.1 % }");
    let s = study::run(&design, &standard_pool(), "front_end", &|_| None, None).unwrap();
    assert!(!s.tolerances.iter().any(|t| t.elements == "rs" || t.elements == "cf"));
    assert!(s.tolerances.iter().any(|t| t.elements == "rg" && t.from == "component"));
    let st = s.measures["f3db_spread"].stats.clone().unwrap();
    let rel = st.std / st.mean;
    let want = 2f64.sqrt() * 0.001 / 3.0;
    assert!((rel / want - 1.0).abs() < 0.25, "spread {rel} vs {want}");
}

#[test]
fn a_limited_study_stops_and_says_what_did_not_run() {
    let design = parse_circuit("/p/daq.circuit", &daq_with_section()).design.unwrap();
    let pool = standard_pool();
    let full = study::run(&design, &pool, "front_end", &|_| None, None).unwrap();
    let total = full.usage.total().unwrap().operations();
    // Enough for the AC and transient analyses, not all 400 Monte Carlo runs.
    let base = total - full.monte_carlo["spread"].usage.total().unwrap().operations();
    let limit = base + (total - base) / 4;
    let s = study::run_with(&design, &pool, "front_end", &|_| None, study::RunOptions { threads: Some(4), limit: Some(limit) }).unwrap();
    assert!(s.usage.total().unwrap().operations() <= limit);
    let mc = &s.monte_carlo["spread"];
    assert!(mc.stopped && mc.completed > 0 && mc.completed < 400, "{}", mc.completed);
    assert!(s.stopped.as_deref().unwrap().contains("Monte Carlo `spread`"));
    assert_eq!(s.measures["f3db_spread"].stats.as_ref().unwrap().n, mc.completed);
}

#[test]
fn a_pin_probe_reads_the_current_into_the_pin() {
    // RS (1 kΩ) runs from IN to INF: the current into RS.1 is (V(IN) − V(INF))/1 kΩ, and the
    // current into RS.2 is its negative.
    let text = daq_with_section().replace("probe = [net.IN, net.INF, net.AMP, net.AIN]", "probe = [net.IN, net.INF, \"RS.1\", \"RS.2\"]");
    let parsed = parse_circuit("/p/daq.circuit", &text);
    assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
    let design = parsed.design.unwrap();
    assert!(print_circuit(&design).unwrap().contains("probe = [net.IN, net.INF, \"RS.1\", \"RS.2\"]"));
    let s = study::run(&design, &standard_pool(), "front_end", &|_| None, None).unwrap();
    let labels: Vec<String> = s.probes.iter().map(|(p, _)| p.label()).collect();
    assert_eq!(labels, ["V(IN)", "V(INF)", "I(RS.1)", "I(RS.2)"]);
    let tran = s.results["step"].real().unwrap();
    let (vin, vinf, i1, i2) = (tran.get(&s.probes[0].1).unwrap(), tran.get(&s.probes[1].1).unwrap(), tran.get(&s.probes[2].1).unwrap(), tran.get(&s.probes[3].1).unwrap());
    let mut peak = 0f64;
    for k in 0..tran.x.len() {
        let want = (vin[k] - vinf[k]) / 1e3;
        assert!((i1[k] - want).abs() < 1e-9 + 1e-6 * want.abs(), "t = {}: {} vs {want}", tran.x[k], i1[k]);
        // Each pin node converges to the abstol of the Newton step.
        assert!((i1[k] + i2[k]).abs() < 1e-9 + 1e-6 * want.abs(), "KCL through RS: {} vs {}", i1[k], i2[k]);
        peak = peak.max(want.abs());
    }
    // The 1.5 V step charges CF through RS: the current peaks near 1.5 V / 1 kΩ.
    assert!(peak > 1e-3, "peak {peak}");
    let report = runmat_circuit_spice::report::report(&s, &design.simulations["front_end"], &Default::default());
    let json = serde_json::to_value(&report).unwrap();
    let step = json["analyses"].as_array().unwrap().iter().find(|a| a["name"] == "step").unwrap();
    assert_eq!(step["signals"][2]["probe"], "RS.1");
    assert_eq!(step["signals"][2]["quantity"], "current");
}

#[test]
fn a_pin_probe_matches_the_pin_by_name_or_pad_number() {
    // The op-amp's output is pin OUT, pad 1 on its SOT-23-5. Both name the same sense source,
    // and KCL holds at AMP: the current out of U.OUT enters RF.2 and RO.1.
    let text = daq_with_section().replace("probe = [net.IN, net.INF, net.AMP, net.AIN]", "probe = [\"U.OUT\", \"U.1\", \"RF.2\", \"RO.1\", \"U.IN+\"]");
    let design = parse_circuit("/p/daq.circuit", &text).design.unwrap();
    let s = study::run(&design, &standard_pool(), "front_end", &|_| None, None).unwrap();
    assert!(!s.warnings.iter().any(|w| w.starts_with("probe ")), "{:?}", s.warnings);
    let labels: Vec<String> = s.probes.iter().map(|(p, _)| p.label()).collect();
    assert_eq!(labels, ["I(U.OUT)", "I(U.1)", "I(RF.2)", "I(RO.1)", "I(U.IN+)"]);
    assert_eq!(s.probes[0].1, s.probes[1].1, "name and pad number share one sense source");
    let tran = s.results["step"].real().unwrap();
    let (out, rf, ro, inp) = (tran.get(&s.probes[0].1).unwrap(), tran.get(&s.probes[2].1).unwrap(), tran.get(&s.probes[3].1).unwrap(), tran.get(&s.probes[4].1).unwrap());
    let mut peak = 0f64;
    for k in 0..tran.x.len() {
        let sum = out[k] + rf[k] + ro[k];
        assert!(sum.abs() < 1e-9 + 1e-6 * out[k].abs(), "t = {}: KCL at AMP {sum}", tran.x[k]);
        peak = peak.max(out[k].abs());
        // The macromodel's input draws only its bias current.
        assert!(inp[k].abs() < 1e-6, "I(U.IN+) = {}", inp[k]);
    }
    assert!(peak > 1e-5, "the output drives RF and RO: peak {peak}");
}
