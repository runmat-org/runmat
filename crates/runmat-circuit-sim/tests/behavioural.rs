//! Behavioural sources against closed forms: `B` voltage and current sources, `E`/`G` VALUE
//! and TABLE, `.func`, `time`, and expressions inside subcircuits (their nodes, sources,
//! parameters and local functions mapped through each instance).

use runmat_circuit_sim::{simulate, AcSweep, Analysis, Design, SimResult, TranSpec};

fn run(text: &str, a: &Analysis) -> SimResult {
    let d = Design::from_netlist(text, &|_| None).unwrap_or_else(|e| panic!("{e}"));
    simulate(&d, a).unwrap_or_else(|e| panic!("{e}")).result
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!((got - want).abs() <= tol * want.abs().max(1.0), "{what}: {got} vs {want}");
}

#[test]
fn a_b_squarer_follows_its_law_in_dc_and_its_slope_in_ac() {
    let deck = "sq\nV1 a 0 DC 1.5 AC 1\nRA a 0 1k\nB1 o 0 V={V(a)*V(a)}\nRO o 0 1k\nB2 u 0 V=V(a)*2-1\nRU u 0 1k\n";
    let r = run(deck, &Analysis::Dc { source: "v1".into(), start: -2.0, stop: 2.0, step: 0.25 });
    let w = r.real().unwrap();
    for (k, a) in w.x.iter().enumerate() {
        close(w.get("v(o)").unwrap()[k], a * a, 1e-12, "v(o) = V(a)²");
        close(w.get("v(u)").unwrap()[k], 2.0 * a - 1.0, 1e-12, "unbraced V = V(a)*2-1");
    }
    // Small signal about V(a) = 1.5: d(V²)/dV = 3, at every frequency.
    let ac = run(deck, &Analysis::Ac { sweep: AcSweep::Dec, points: 5, fstart: 10.0, fstop: 1e6 });
    let ac = ac.ac().unwrap();
    for m in ac.get("v(o)").unwrap() {
        close(m.abs(), 3.0, 1e-9, "|v(o)| small signal");
    }
}

#[test]
fn a_b_current_source_solves_an_implicit_diode_equation() {
    let (is, vt) = (1e-14, 0.025);
    let deck = format!("bd\nV1 a 0 1\nR1 a d 1k\nB1 d 0 I={{{is}*(exp(V(d)/{vt})-1)}}\n");
    let op = run(&deck, &Analysis::Op);
    let vd = op.op().unwrap()["v(d)"];
    let resistor = (1.0 - vd) / 1e3;
    let law = is * ((vd / vt).exp() - 1.0);
    close(law, resistor, 1e-9, "KCL at d");
    assert!(vd > 0.5 && vd < 0.7, "vd = {vd}");
}

#[test]
fn value_and_table_sources_and_current_controls() {
    let deck = "vt\nV1 a 0 3\nR1 a b 1k\nR2 b 0 2k\n\
                E1 e 0 VALUE={2*V(a,b)+1}\nRE e 0 1k\n\
                G1 0 g VALUE={1m*V(b)}\nRG g 0 1k\n\
                B2 h 0 V={1k*I(V1)}\nRH h 0 1k\n\
                E2 t 0 TABLE {V(a)} = (0, 0) (2, 1) (4, 1.5)\nRT t 0 1k\n";
    let op = run(deck, &Analysis::Op);
    let op = op.op().unwrap();
    let vb = 2.0;
    close(op["v(e)"], 2.0 * (3.0 - vb) + 1.0, 1e-12, "E VALUE");
    close(op["v(g)"], 1e-3 * vb * 1e3, 1e-12, "G VALUE");
    close(op["v(h)"], 1e3 * op["i(v1)"], 1e-12, "B on I(V1)");
    close(op["i(v1)"], -1e-3, 1e-12, "source current");
    close(op["v(t)"], 1.0 + 0.5 * (3.0 - 2.0) / 2.0, 1e-12, "TABLE between points");
}

#[test]
fn expressions_inside_subcircuits_read_the_instance_nodes_sources_and_functions() {
    let deck = "sub\n.func sq(x) {x*x}\n\
                .subckt cube in out params: k=1\n\
                .func half(y) {y/2}\n\
                Rint in mid 1k\nVs mid s 0\nRg s 0 1k\n\
                B1 out 0 V={k*half(sq(V(mid))*V(in))}\nRl out 0 1k\n\
                B2 cur 0 V={1k*I(Vs)}\nRc cur 0 1k\n\
                B3 dif 0 V={V(in, mid)}\nRd dif 0 1k\n\
                .ends\n\
                V1 a 0 2\nX1 a o cube k=3\nX2 a o2 cube\n";
    let op = run(deck, &Analysis::Op);
    let op = op.op().unwrap();
    // mid = in/2 = 1 V in both instances; the instance current is 1 mA.
    close(op["v(o)"], 3.0 * (1.0 * 2.0) / 2.0, 1e-12, "X1 (k = 3)");
    close(op["v(o2)"], 1.0, 1e-12, "X2 (k = 1)");
    close(op["v(x1.cur)"], 1.0, 1e-12, "I(Vs) is the instance's own source");
    close(op["v(x2.dif)"], 1.0, 1e-12, "V(in, mid) across a port and an internal node");
}

#[test]
fn time_drives_a_b_source_in_a_transient() {
    let deck = "t\nB1 o 0 V={sin(2*pi*1k*time)}\nRO o 0 1k\n";
    let r = run(deck, &Analysis::Tran(TranSpec { tstep: 1e-6, tstop: 1e-3, tstart: 0.0, tmax: Some(1e-6), uic: false }));
    let w = r.real().unwrap();
    for t in [0.1e-3, 0.25e-3, 0.6e-3, 0.9e-3] {
        let want = (2.0 * std::f64::consts::PI * 1e3 * t).sin();
        assert!((w.at("v(o)", t).unwrap() - want).abs() < 2e-3, "t = {t}");
    }
}

#[test]
fn bad_expressions_name_the_element() {
    let err = Design::from_netlist("bad\nV1 a 0 1\nB1 o 0 V={nope(V(a))}\nR1 o 0 1\n", &|_| None).err().expect("rejected").to_string();
    assert!(err.contains("b1") && err.contains("unknown function `nope`"), "{err}");
}
