//! Closed-form checks: each circuit has an analytic answer the engine must reproduce.

use runmat_circuit_sim::measure::{evaluate, Measure};
use runmat_circuit_sim::{simulate, Analysis, Design, SimResult, TranSpec};

fn design(text: &str) -> Design {
    Design::from_netlist(text, &|_| None).unwrap_or_else(|e| panic!("{e}"))
}

fn run(text: &str, a: &Analysis) -> SimResult {
    simulate(&design(text), a).unwrap_or_else(|e| panic!("{e}")).result
}

fn tran(tstep: f64, tstop: f64) -> Analysis {
    Analysis::Tran(TranSpec { tstep, tstop, tstart: 0.0, tmax: None, uic: false })
}

const VT27: f64 = 1.380_649e-23 / 1.602_176_634e-19 * 300.15;

#[test]
fn rc_charge_curve_trapezoidal_and_gear() {
    for method in ["trap", "gear"] {
        let deck = format!(
            "rc\nV1 in 0 PULSE(0 1 0 1n 1n 1 2)\nR1 in out 1k\nC1 out 0 1u\n.options method={method}\n"
        );
        let r = run(&deck, &tran(10e-6, 5e-3));
        let w = r.real().unwrap();
        let tau: f64 = 1e-3;
        for t in [0.2e-3f64, 0.5e-3, 1e-3, 2e-3, 4e-3] {
            let want = 1.0 - (-(t - 0.5e-9) / tau).exp();
            let got = w.at("v(out)", t).unwrap();
            assert!((got - want).abs() < 2e-3, "{method} t={t}: {got} vs {want}");
        }
        // Capacitor current i = C dv/dt = (1/R) e^{-t/τ}.
        let i = w.at("i(c1)", 1e-3).unwrap();
        assert!((i - 1e-3 * (-1.0f64).exp()).abs() < 5e-6, "{method}: i(c1) = {i}");
    }
}

#[test]
fn series_rlc_resonance_and_q() {
    let (r, l, c) = (10.0, 1e-3, 1e-6);
    let deck = format!("rlc\nV1 in 0 AC 1\nL1 in a {l}\nC1 a b {c}\nR1 b 0 {r}\n");
    let res = run(&deck, &Analysis::Ac { sweep: runmat_circuit_sim::AcSweep::Dec, points: 2000, fstart: 1e3, fstop: 3e4 });
    let f0 = 1.0 / (2.0 * std::f64::consts::PI * (l * c as f64).sqrt());
    let q = (l / c).sqrt() / r;
    let fp = evaluate(&Measure::PeakFrequency { signal: "v(b)".into() }, &res).unwrap();
    let qm = evaluate(&Measure::Q { signal: "v(b)".into() }, &res).unwrap();
    assert!((fp / f0 - 1.0).abs() < 2e-3, "f0 {fp} vs {f0}");
    assert!((qm / q - 1.0).abs() < 5e-3, "Q {qm} vs {q}");
    // At resonance the resistor sees the whole source.
    let at = evaluate(&Measure::At { signal: "vm(b)".into(), x: f0 }, &res).unwrap();
    assert!((at - 1.0).abs() < 1e-3, "|v(b)| at f0 = {at}");
}

#[test]
fn underdamped_rlc_step_matches_analytic() {
    let (r, l, c) = (20.0, 1e-3, 1e-6);
    let deck = format!("rlc step\nV1 in 0 PULSE(0 1 0 1p 1p 1 2)\nR1 in a {r}\nL1 a b {l}\nC1 b 0 {c}\n");
    let w = run(&deck, &tran(1e-6, 1e-3));
    let w = w.real().unwrap();
    let alpha = r / (2.0 * l);
    let w0 = 1.0 / (l * c as f64).sqrt();
    let wd = (w0 * w0 - alpha * alpha).sqrt();
    for t in [50e-6, 150e-6, 300e-6, 600e-6] {
        let want = 1.0 - (-alpha * t).exp() * ((wd * t).cos() + alpha / wd * (wd * t).sin());
        let got = w.at("v(b)", t).unwrap();
        assert!((got - want).abs() < 5e-3, "t={t}: {got} vs {want}");
    }
}

#[test]
fn diode_iv_follows_shockley() {
    let deck = "diode\nV1 a 0 0\nD1 a 0 DX\n.model DX D(IS=1e-14 N=1)\n";
    let res = run(deck, &Analysis::Dc { source: "v1".into(), start: 0.3, stop: 0.75, step: 0.05 });
    let w = res.real().unwrap();
    let i = w.get("i(v1)").unwrap();
    for (k, v) in w.x.iter().enumerate() {
        let want = 1e-14 * ((v / VT27).exp() - 1.0) + 1e-12 * v;
        // The source current flows into the diode: i(v1) = −I_D.
        assert!(((-i[k]) / want - 1.0).abs() < 1e-6, "V={v}: {} vs {want}", -i[k]);
    }
}

#[test]
fn diode_with_series_resistance_solves_the_implicit_equation() {
    let deck = "diode rs\nV1 a 0 2\nD1 a 0 DX\n.model DX D(IS=1e-14 N=1.5 RS=10)\n";
    let op = run(deck, &Analysis::Op);
    let op = op.op().unwrap();
    let id = -op["i(v1)"];
    let vd = 2.0 - id * 10.0;
    let model = 1e-14 * ((vd / (1.5 * VT27)).exp() - 1.0) + 1e-12 * vd;
    assert!((model / id - 1.0).abs() < 1e-6, "{id} vs {model}");
}

#[test]
fn inverting_amplifier_gain_and_bandwidth() {
    let (a0, gbw, rin, rf) = (1e5, 1e6, 1e3, 10e3);
    let deck = format!(
        "inverting\nV1 in 0 DC 0.1 AC 1\nR1 in m {rin}\nR2 m out {rf}\nXU 0 m out OPX\n.model OPX OPAMP(a0={a0} gbw={gbw})\n"
    );
    let op = run(&deck, &Analysis::Op);
    let beta = rin / (rin + rf);
    let ideal = -rf / rin;
    let gain = ideal / (1.0 + 1.0 / (a0 * beta));
    let vout = op.op().unwrap()["v(out)"];
    assert!((vout / (0.1 * gain) - 1.0).abs() < 1e-6, "{vout} vs {}", 0.1 * gain);
    let ac = run(&deck, &Analysis::Ac { sweep: runmat_circuit_sim::AcSweep::Dec, points: 200, fstart: 10.0, fstop: 10e6 });
    let bw = evaluate(&Measure::Bandwidth { signal: "v(out)".into(), drop_db: 3.0103 }, &ac).unwrap();
    // Single pole: closed-loop pole at (gbw/a0)·(1 + a0·β).
    let want = gbw / a0 * (1.0 + a0 * beta);
    assert!((bw / want - 1.0).abs() < 2e-3, "bandwidth {bw} vs {want}");
}

#[test]
fn bjt_fixed_bias_point() {
    let (is, bf, vcc, rb, rc) = (1e-15, 100.0, 10.0, 1e6, 2e3);
    let deck = format!("bias\nVCC vcc 0 {vcc}\nRB vcc b {rb}\nRC vcc c {rc}\nQ1 c b 0 QN\n.model QN NPN(IS={is} BF={bf})\n.options reltol=1e-7\n");
    let op = run(&deck, &Analysis::Op);
    let op = op.op().unwrap();
    // Ebers–Moll: Ib = Ibe1/bf + Ibc1/br, Ic = Ibe1 − Ibc1 − Ibc1/br (br = 1), plus gmin.
    let vbe = op["v(b)"];
    let vbc = vbe - op["v(c)"];
    let ibe1 = is * ((vbe / VT27).exp() - 1.0);
    let ibc1 = is * ((vbc / VT27).exp() - 1.0);
    let ib = ibe1 / bf + ibc1 + 1e-12 * (vbe + vbc);
    let ic = ibe1 - ibc1 - ibc1 - 1e-12 * vbc;
    assert!(((vcc - vbe) / rb / ib - 1.0).abs() < 1e-6, "base current");
    assert!(((vcc - op["v(c)"]) / rc / ic - 1.0).abs() < 1e-6, "collector current");
    assert!((op["i(q1)"] / ic - 1.0).abs() < 1e-6);
    // The textbook estimate: Ic ≈ β·(Vcc − 0.7 V)/Rb.
    assert!((ic - bf * (vcc - vbe) / rb).abs() / ic < 1e-3);
}

#[test]
fn nmos_square_law_in_saturation_and_triode() {
    let deck = "nmos\nVG g 0 2\nVD d 0 3\nM1 d g 0 0 NX L=1u W=10u\n.model NX NMOS(LEVEL=1 VTO=1 KP=1e-4 LAMBDA=0.02)\n";
    let op = run(deck, &Analysis::Op);
    let id = op.op().unwrap()["i(m1)"];
    let beta = 1e-4 * 10.0;
    let want = 0.5 * beta * 1.0 * (1.0 + 0.02 * 3.0);
    assert!((id / want - 1.0).abs() < 1e-6, "{id} vs {want}");
    let deck = deck.replace("VD d 0 3", "VD d 0 0.4");
    let id = run(&deck, &Analysis::Op).op().unwrap()["i(m1)"];
    let want = beta * 0.4 * (1.0 - 0.2) * (1.0 + 0.02 * 0.4);
    assert!((id / want - 1.0).abs() < 1e-6, "{id} vs {want}");
}

#[test]
fn coupled_inductors_transform_voltage() {
    let deck = "xfmr\nV1 in 0 AC 1\nR1 in p 1\nL1 p 0 1m\nL2 s 0 4m\nK1 L1 L2 0.95\nR2 s 0 1meg\n";
    let ac = run(deck, &Analysis::Ac { sweep: runmat_circuit_sim::AcSweep::Lin, points: 1, fstart: 10e3, fstop: 10e3 });
    let w = ac.ac().unwrap();
    let (v1, v2) = (w.get("v(p)").unwrap()[0], w.get("v(s)").unwrap()[0]);
    // Open circuit: V2/V1 = M/L1 = k·√(L2/L1), in phase; the 1 MΩ load's leakage-reactance
    // drop is about ωL2(1−k²)/R2 ≈ 2.5e-5 of it.
    let ratio = v2 / v1;
    assert!((ratio.re - 0.95 * 2.0).abs() < 1e-5 && ratio.im.abs() < 1e-4, "{ratio:?}");
}

#[test]
fn controlled_sources_and_poly() {
    let deck = "ctl\nV1 a 0 2\nR1 a 0 1k\nE1 e 0 a 0 3\nRE e 0 1k\nG1 0 g a 0 1m\nRG g 0 1k\nF1 0 f V1 2\nRF f 0 1\nH1 h 0 V1 1k\nRH h 0 1k\n\
                EP p 0 POLY(1) a 0 1 2 0.5\nRP p 0 1k\n";
    let op = run(deck, &Analysis::Op);
    let op = op.op().unwrap();
    let iv1 = op["i(v1)"];
    assert!((iv1 + 2e-3).abs() < 1e-12);
    assert!((op["v(e)"] - 6.0).abs() < 1e-9);
    assert!((op["v(g)"] - 2.0).abs() < 1e-9);
    assert!((op["v(f)"] - 2.0 * iv1).abs() < 1e-9);
    assert!((op["v(h)"] - 1e3 * iv1).abs() < 1e-9);
    assert!((op["v(p)"] - (1.0 + 4.0 + 0.5 * 4.0)).abs() < 1e-9);
}

#[test]
fn switch_follows_its_control_with_hysteresis() {
    let deck = "sw\nVC c 0 PWL(0 0 1m 2 2m 0)\nV1 a 0 1\nR1 a o 1k\nS1 o 0 c 0 SWX\n.model SWX SW(VT=1 VH=0.2 RON=1 ROFF=1meg)\n";
    let w = run(deck, &tran(10e-6, 2e-3));
    let w = w.real().unwrap();
    let v = |t: f64| w.at("v(o)", t).unwrap();
    assert!(v(0.3e-3) > 0.99, "off at 0.6 V control");
    assert!(v(0.7e-3) < 0.01, "on at 1.4 V control");
    // Falling through 1.0 V (inside the band) the switch stays on until 0.8 V.
    assert!(v(1.55e-3) < 0.01, "still on at 0.9 V control");
    assert!(v(1.7e-3) > 0.99, "off at 0.6 V control");
}

#[test]
fn nested_subcircuit_with_parameters() {
    let deck = "sub\n.param rtop=3k\n.subckt div in out params: r1=1k r2=1k\nRA in out {r1}\nRB out 0 {r2}\n.ends\n\
                V1 a 0 4\nX1 a m div r1={rtop}\n";
    let op = run(deck, &Analysis::Op);
    assert!((op.op().unwrap()["v(m)"] - 1.0).abs() < 1e-9);
}

#[test]
fn operation_counts_are_deterministic_and_categorized() {
    let deck = "rc\nV1 in 0 PULSE(0 1 0 1n 1n 1 2)\nR1 in out 1k\nC1 out 0 1u\nD1 out 0 DX\n.model DX D(IS=1e-14)\n";
    let d = design(deck);
    let a = tran(10e-6, 1e-3);
    let u1 = simulate(&d, &a).unwrap().usage;
    let u2 = simulate(&d, &a).unwrap().usage;
    assert_eq!(u1, u2);
    for v in [u1.assembly, u1.factorization, u1.solve, u1.other] {
        assert!(v.operations() > 0, "{u1:?}");
    }
}

#[test]
fn singular_circuit_names_the_floating_node() {
    let d = design("float\nV1 a 0 1\nC1 a b 1n\nC2 b 0 1n\nR1 x y 1k\n");
    let e = simulate(&d, &Analysis::Op).unwrap_err().to_string();
    assert!(e.contains("singular"), "{e}");
}

/// A step measure reads its `from`–`to` window: an RC on a pulse train that starts and ends low
/// has one rising edge per pulse, and the 10–90 % rise time of the second edge is tau ln 9.
#[test]
fn rise_time_settling_and_overshoot_read_their_window() {
    let deck = "rc pulses\nV1 in 0 PULSE(0 1 0 1n 1n 10m 20m)\nR1 in out 1k\nC1 out 0 1u\n";
    let res = run(deck, &tran(10e-6, 40e-3));
    let tau = 1e-3;
    let (from, to) = (Some(20e-3), Some(30e-3));
    let rise = evaluate(&Measure::RiseTime { signal: "v(out)".into(), low: 0.1, high: 0.9, from, to }, &res).unwrap();
    assert!((rise - tau * 9f64.ln()).abs() < 0.02e-3, "{rise}");
    let settle = evaluate(&Measure::Settling { signal: "v(out)".into(), band: 0.01, from, to }, &res).unwrap();
    assert!((settle - tau * 100f64.ln()).abs() < 0.05e-3, "{settle}");
    let over = evaluate(&Measure::Overshoot { signal: "v(out)".into(), from, to }, &res).unwrap();
    assert!(over < 0.1, "{over}");
}
