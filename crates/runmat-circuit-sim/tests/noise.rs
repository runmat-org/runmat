//! Noise analysis against closed forms: resistor thermal noise, kT/C, diode shot noise, MOSFET
//! channel noise and flicker noise.

use runmat_circuit_sim::analysis::noise::{NoiseSpec, NoiseWave, K_B, Q_E};
use runmat_circuit_sim::{simulate, simulate_noise, AcSweep, Analysis, Design};

const T: f64 = 300.15;

fn noise(deck: &str, out: &str, input: &str, freqs: Vec<f64>) -> NoiseWave {
    let d = Design::from_netlist(deck, &|_| None).unwrap_or_else(|e| panic!("{e}"));
    let c = d.circuit().unwrap();
    simulate_noise(
        &c,
        &NoiseSpec {
            output: (out.into(), "0".into()),
            input: input.into(),
            freqs,
        },
        None,
    )
    .0
    .unwrap_or_else(|e| panic!("{e}"))
}

fn decades(from: f64, to: f64, per: usize) -> Vec<f64> {
    runmat_circuit_sim::analysis::ac::frequencies(AcSweep::Dec, per, from, to).unwrap()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got / want - 1.0).abs() < rel, "{what}: {got} vs {want}");
}

#[test]
fn a_divider_shows_the_thermal_noise_of_its_parallel_resistance() {
    let w = noise(
        "div\nV1 in 0 AC 1\nR1 in out 1k\nR2 out 0 1k\n",
        "out",
        "v1",
        vec![1e3, 1e6],
    );
    for k in 0..2 {
        close(w.onoise[k], (4.0 * K_B * T * 500.0).sqrt(), 1e-9, "onoise");
        close(w.gain[k], 0.5, 1e-12, "gain");
        close(w.inoise[k], w.onoise[k] / 0.5, 1e-12, "inoise");
    }
    // Each resistor gives half of the output density.
    close(
        w.contributions["r1"][0],
        w.contributions["r2"][0],
        1e-12,
        "equal shares",
    );
}

#[test]
fn an_rc_low_pass_integrates_to_kt_over_c() {
    let c = 1e-9;
    let w = noise(
        &format!("rc\nV1 in 0 AC 1\nR1 in out 1k\nC1 out 0 {c}\n"),
        "out",
        "v1",
        decades(1.0, 1e11, 200),
    );
    close(
        w.total_onoise(),
        (K_B * T / c).sqrt(),
        5e-3,
        "∫ noise = √(kT/C)",
    );
}

#[test]
fn a_forward_diode_shows_its_shot_noise_through_its_dynamic_resistance() {
    let deck = "d\nV1 in 0 DC 5 AC 1\nR1 in a 100k\nD1 a 0 DX\n.model DX D(IS=1e-14 N=1)\n";
    let op = simulate(
        &Design::from_netlist(deck, &|_| None).unwrap(),
        &Analysis::Op,
    )
    .unwrap()
    .result;
    let id = -op.op().unwrap()["i(v1)"];
    let vt = K_B * T / Q_E;
    // gmin (1e-12 S) sits across the junction with the diode's own conductance.
    let rd = 1.0 / (id / vt + 1e-12);
    let r = 1.0 / (1.0 / 100e3 + 1.0 / rd);
    let want = ((2.0 * Q_E * id + 4.0 * K_B * T / 100e3) * r * r).sqrt();
    let w = noise(deck, "a", "v1", vec![1e3]);
    close(w.onoise[0], want, 2e-3, "shot + thermal");
}

#[test]
fn a_common_source_stage_shows_channel_noise_of_its_transconductance() {
    let (rd, beta, vgst) = (10e3, 1e-3, 0.3);
    let deck = format!("cs\nVDD dd 0 5\nVG g 0 DC {} AC 1\nRD dd d {rd}\nM1 d g 0 0 MX L=1u W=10u\n.model MX NMOS(vto=0.7 kp=1e-4)\n", 0.7 + vgst);
    let gm = beta * vgst;
    let w = noise(&deck, "d", "vg", vec![1e3]);
    let want = ((8.0 / 3.0 * K_B * T * gm + 4.0 * K_B * T / rd) * rd * rd).sqrt();
    close(w.onoise[0], want, 1e-3, "channel + load");
    close(w.gain[0], gm * rd, 1e-3, "gain gm·RD");
}

#[test]
fn a_diode_with_kf_adds_flicker_noise_falling_as_one_over_f() {
    let (kf, af) = (1e-16, 1.2);
    let card = |extra: &str| {
        format!("d\nV1 in 0 DC 5 AC 1\nR1 in a 100k\nD1 a 0 DX\n.model DX D(IS=1e-14 N=1{extra})\n")
    };
    let freqs = vec![1.0, 10.0, 100.0, 1e6];
    let white = noise(&card(""), "a", "v1", freqs.clone());
    let pink = noise(
        &card(&format!(" KF={kf} AF={af}")),
        "a",
        "v1",
        freqs.clone(),
    );
    let op = simulate(
        &Design::from_netlist(&card(""), &|_| None).unwrap(),
        &Analysis::Op,
    )
    .unwrap()
    .result;
    let id = -op.op().unwrap()["i(v1)"];
    // The flicker generator sits where the shot generator does, so it reaches the output through
    // the same transfer: |H|² = white² / (shot + thermal density).
    let vt = K_B * T / Q_E;
    let r = 1.0 / (1.0 / 100e3 + id / vt + 1e-12);
    for (k, f) in freqs.iter().enumerate() {
        let added = pink.onoise[k].powi(2) - white.onoise[k].powi(2);
        close(
            added,
            kf * id.powf(af) / f * r * r,
            2e-3,
            &format!("flicker at {f} Hz"),
        );
    }
    // Far above the corner the flicker share is gone.
    close(
        pink.onoise[3],
        white.onoise[3],
        1e-3,
        "white above the corner",
    );
}

#[test]
fn a_mosfet_with_kf_adds_drain_flicker_over_w_l_cox_squared() {
    let (kf, af, tox) = (1e-25, 1.0, 1e-8);
    let card = |extra: &str| {
        format!("cs\nVDD dd 0 5\nVG g 0 DC 1.0 AC 1\nRD dd d 10k\nM1 d g 0 0 MX L=1u W=10u\n.model MX NMOS(vto=0.7 kp=1e-4 tox={tox}{extra})\n")
    };
    let white = noise(&card(""), "d", "vg", vec![10.0]);
    let pink = noise(&card(&format!(" KF={kf} AF={af}")), "d", "vg", vec![10.0]);
    let op = simulate(
        &Design::from_netlist(&card(""), &|_| None).unwrap(),
        &Analysis::Op,
    )
    .unwrap()
    .result;
    let id = -op.op().unwrap()["i(vdd)"];
    let cox = 3.9 * 8.854_187_8e-12 / tox;
    // The drain current's generator reaches the drain through RD in parallel with the output
    // resistance (λ = 0: RD alone).
    let want = kf * id.powf(af) / (10.0 * 10e-6 * 1e-6 * cox * cox) * 10e3 * 10e3;
    close(
        pink.onoise[0].powi(2) - white.onoise[0].powi(2),
        want,
        2e-3,
        "MOSFET flicker",
    );
}
