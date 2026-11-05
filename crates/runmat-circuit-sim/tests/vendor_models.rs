//! Vendor-style netlists: a Boyle op-amp macromodel subcircuit (bipolar input pair, `POLY`
//! controlled sources, clamp diodes), `.lib` sections and `.include`, and analyses read from the
//! deck's control lines.
//!
//! The macromodel follows Boyle, Cohn, Pederson and Solomon's published structure (IEEE JSSC
//! SC-9, 1974) with 741-class element values (about 1 MHz unity-gain bandwidth and 200 V/mV
//! open-loop gain), typed in as a test fixture; it exercises the same netlist features vendor
//! op-amp files use.

use runmat_circuit_sim::measure::{evaluate, Measure};
use runmat_circuit_sim::{simulate, Analysis, Design};

const BOYLE: &str = "\
* Boyle-structure op-amp: in+ in- v+ v- out
.subckt boyle 1 2 3 4 5
c1   11 12 4.664e-12
c2   6 7 30.00e-12
dc   5 53 dx
de   54 5 dx
dlp  90 91 dx
dln  92 90 dx
dp   4 3 dx
egnd 99 0 poly(2) 3 0 4 0 0 .5 .5
fb   7 99 poly(5) vb vc ve vlp vln 0 10.61e6 -10e6 10e6 10e6 -10e6
ga   6 0 11 12 137.7e-6
gcm  0 6 10 99 2.574e-9
iee  10 4 dc 10.16e-6
hlim 90 0 vlim 1k
q1   11 2 13 qx
q2   12 1 14 qx
r2   6 9 100.0e3
rc1  3 11 7.957e3
rc2  3 12 7.957e3
re1  13 10 2.740e3
re2  14 10 2.740e3
ree  10 99 19.69e6
ro1  8 5 150
ro2  7 99 150
rp   3 4 18.11e3
vb   9 0 dc 0
vc   3 53 dc 2.600
ve   54 4 dc 2.600
vlim 7 8 dc 0
vlp  91 0 dc 25
vln  0 92 dc 25
.model dx d(is=800.0e-18)
.model qx npn(is=800.0e-18 bf=62.50)
.ends
";

fn deck(body: &str) -> String {
    format!("boyle test\n.include boyle.lib\n{body}")
}

fn resolver(name: &str) -> Option<String> {
    (name == "boyle.lib").then(|| BOYLE.to_string())
}

#[test]
fn boyle_follower_and_inverting_gain() {
    let text = deck("VCC vcc 0 15\nVEE vee 0 -15\nVIN in 0 DC 0.5 AC 1\nR1 in m 1k\nR2 m out 10k\nX1 0 m vcc vee out boyle\n.op\n.ac dec 50 10 10meg\n");
    let d = Design::from_netlist(&text, &resolver).unwrap();
    let analyses = d.analyses().unwrap();
    assert_eq!(analyses.len(), 2);
    let op = simulate(&d, &analyses[0]).unwrap().result;
    let vout = op.op().unwrap()["v(out)"];
    // −10 × 0.5 V within the loop-gain error and input offset of a 2e5-gain stage.
    assert!((vout + 5.0).abs() < 5e-3, "vout {vout}");
    let ac = simulate(&d, &analyses[1]).unwrap().result;
    let g = evaluate(&Measure::At { signal: "vdb(out)".into(), x: 100.0 }, &ac).unwrap();
    assert!((g - 20.0).abs() < 0.01, "gain {g} dB");
    let bw = evaluate(&Measure::Bandwidth { signal: "v(out)".into(), drop_db: 3.0103 }, &ac).unwrap();
    // Unity-gain bandwidth about 1 MHz with β = 1/11: a closed-loop corner near 90 kHz.
    assert!(bw > 60e3 && bw < 130e3, "bandwidth {bw}");
}

#[test]
fn boyle_output_clamps_near_the_rails_and_slews() {
    let text = deck("VCC vcc 0 15\nVEE vee 0 -15\nVIN in 0 PULSE(-5 5 1u 1n 1n 60u 120u)\nX1 in out vcc vee out boyle\n");
    let d = Design::from_netlist(&text, &resolver).unwrap();
    let a = Analysis::Tran(runmat_circuit_sim::TranSpec { tstep: 0.2e-6, tstop: 80e-6, tstart: 0.0, tmax: None, uic: false });
    let o = simulate(&d, &a).unwrap();
    let w = o.result;
    let vmax = evaluate(&Measure::Max { signal: "v(out)".into(), from: None, to: None }, &w).unwrap();
    assert!((vmax - 5.0).abs() < 0.01, "follower reaches 5 V: {vmax}");
    // Slew limited: a 741-class stage slews at about 0.5 V/µs.
    let tr = evaluate(&Measure::Delay { trig: ("v(out)".into(), -4.0, runmat_circuit_sim::measure::Edge::Rise, 1), targ: ("v(out)".into(), 4.0, runmat_circuit_sim::measure::Edge::Rise, 1) }, &w).unwrap();
    let sr = 8.0 / tr;
    assert!(sr > 0.3e6 && sr < 2e6, "slew {sr} V/s");
}

#[test]
fn lib_sections_select_models() {
    let lib = ".lib fast\n.model dz d(is=1e-12)\n.endl\n.lib slow\n.model dz d(is=1e-16)\n.endl\n";
    let r = move |n: &str| (n == "parts.lib").then(|| lib.to_string());
    let fast = Design::from_netlist("t\n.lib parts.lib fast\nV1 a 0 0.5\nD1 a 0 dz\n", &r).unwrap();
    let slow = Design::from_netlist("t\n.lib parts.lib slow\nV1 a 0 0.5\nD1 a 0 dz\n", &r).unwrap();
    let i = |d: &Design| -simulate(d, &Analysis::Op).unwrap().result.op().unwrap()["i(v1)"];
    assert!((i(&fast) / i(&slow) - 1e4).abs() < 1.0, "{} {}", i(&fast), i(&slow));
}
