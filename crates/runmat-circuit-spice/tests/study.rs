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

pub fn daq_with_section() -> String {
    let base = include_str!("../../runmat-circuit/tests/projects/daq/daq.circuit");
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
    assert_eq!(sim.probes, ["IN", "INF", "AMP", "AIN"]);
    let printed = print_circuit(&design).unwrap();
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
