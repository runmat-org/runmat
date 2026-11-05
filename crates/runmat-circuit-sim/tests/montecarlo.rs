//! Monte Carlo statistics against analytic distributions, and thread-count independence.

use runmat_circuit_sim::measure::Measure;
use runmat_circuit_sim::montecarlo::{run, Distribution, MonteCarlo, Target, Tolerance};
use runmat_circuit_sim::{AcSweep, Analysis, Design};

fn design(t: &str) -> Design {
    Design::from_netlist(t, &|_| None).unwrap()
}

fn mc(analysis: Analysis, tol: Vec<Tolerance>, measure: Measure, runs: usize) -> MonteCarlo {
    MonteCarlo { runs, first: 0, seed: 0x5EED, tolerances: tol, analysis, measures: vec![("m".into(), measure)], keep_results: 0, limit: None }
}

fn elements(pat: &str, dist: Distribution) -> Tolerance {
    Tolerance::new(Target::Elements(pat.into()), dist)
}

#[test]
fn linear_gaussian_mean_and_variance() {
    // V = I·R is linear in R: mean I·R0, standard deviation I·R0·σ exactly.
    let d = design("lin\nI1 0 a DC 1m\nR1 a 0 1k\n");
    let n = 4000;
    let sigma = 0.01;
    let o = run(&d, &mc(Analysis::Op, vec![elements("r1", Distribution::Gaussian { sigma })], Measure::Final { signal: "v(a)".into() }, n), None).unwrap();
    let s = &o.stats["m"];
    assert_eq!(s.n, n);
    let se_mean = sigma / (n as f64).sqrt();
    let se_std = sigma / (2.0 * n as f64).sqrt();
    assert!((s.mean - 1.0).abs() < 4.0 * se_mean, "mean {}", s.mean);
    assert!((s.std - sigma).abs() < 4.0 * se_std, "std {}", s.std);
}

#[test]
fn linear_uniform_variance() {
    let d = design("lin\nI1 0 a DC 1m\nR1 a 0 1k\n");
    let n = 4000;
    let hw = 0.05;
    let o = run(&d, &mc(Analysis::Op, vec![elements("r*", Distribution::Uniform { half_width: hw })], Measure::Final { signal: "v(a)".into() }, n), None).unwrap();
    let s = &o.stats["m"];
    let want = hw / 3f64.sqrt();
    // Uniform: Var(s²) = σ⁴·(2/(n−1) + κ/n) with excess kurtosis κ = −1.2.
    let se_std = want * ((2.0 / (n as f64 - 1.0) - 1.2 / n as f64).sqrt()) / 2.0;
    assert!((s.std - want).abs() < 4.0 * se_std, "std {} vs {want}", s.std);
    assert!(s.min >= 1.0 - hw - 1e-12 && s.max <= 1.0 + hw + 1e-12);
}

#[test]
fn rc_corner_frequency_spread_matches_first_order_propagation() {
    // f = 1/(2πRC): to first order σ_f/f = √(σ_R² + σ_C²).
    let d = design("rc\nV1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1n\n");
    let n = 2000;
    let sigma = 0.01;
    let tol = vec![elements("r1", Distribution::Gaussian { sigma }), elements("c1", Distribution::Gaussian { sigma })];
    let a = Analysis::Ac { sweep: AcSweep::Dec, points: 400, fstart: 1e2, fstop: 1e7 };
    let o = run(&d, &mc(a, tol, Measure::Bandwidth { signal: "v(out)".into(), drop_db: 3.0103 }, n), None).unwrap();
    let s = &o.stats["m"];
    let f0 = 1.0 / (2.0 * std::f64::consts::PI * 1e3 * 1e-9);
    let rel = s.std / s.mean;
    let want = (2.0f64).sqrt() * sigma;
    // E[f] = f0·(1 + σ_R² + σ_C² + …): a 2e-4 bias, plus 3e-4 sampling error.
    assert!((s.mean / f0 - 1.0).abs() < 1.5e-3, "mean {} vs {f0}", s.mean);
    assert!((rel / want - 1.0).abs() < 0.08, "relative spread {rel} vs {want}");
    assert!(o.failures.is_empty());
    assert!(o.usage.total().unwrap().operations() > 0);
}

#[test]
fn results_do_not_depend_on_thread_count() {
    let d = design("rc\nV1 in 0 PULSE(0 1 0 1n 1n 1 2)\nR1 in out 1k\nC1 out 0 1u\n");
    let a = Analysis::Tran(runmat_circuit_sim::TranSpec { tstep: 20e-6, tstop: 2e-3, tstart: 0.0, tmax: None, uic: false });
    let m = mc(a, vec![elements("*", Distribution::Gaussian { sigma: 0.05 })], Measure::At { signal: "v(out)".into(), x: 1e-3 }, 64);
    let one = run(&d, &m, Some(1)).unwrap();
    let many = run(&d, &m, Some(8)).unwrap();
    assert_eq!(one.values, many.values);
    assert_eq!(one.usage, many.usage);
}

#[test]
fn a_monte_carlo_split_into_ranges_gives_the_same_runs() {
    let d = design("rc\nV1 in 0 AC 1\nR1 in out 1k\nC1 out 0 1n\n");
    let a = Analysis::Ac { sweep: AcSweep::Dec, points: 20, fstart: 1e3, fstop: 1e7 };
    let whole = mc(a.clone(), vec![elements("*", Distribution::Gaussian { sigma: 0.02 })], Measure::Bandwidth { signal: "v(out)".into(), drop_db: 3.0103 }, 100);
    let all = run(&d, &whole, Some(2)).unwrap();
    let mut head = run(&d, &MonteCarlo { runs: 40, ..whole.clone() }, Some(2)).unwrap();
    let tail = run(&d, &MonteCarlo { runs: 60, first: 40, ..whole.clone() }, Some(2)).unwrap();
    head.merge(tail);
    assert_eq!(head.values, all.values);
    assert_eq!(head.usage, all.usage);
    assert_eq!(head.completed, 100);
}

#[test]
fn tolerance_globs_take_sets_and_ranges() {
    use runmat_circuit_sim::montecarlo::glob;
    assert!(glob("?[SF]", "rs") && glob("?[SF]", "cf") && !glob("?[SF]", "rg"));
    assert!(glob("r[1-3]", "r2") && !glob("r[1-3]", "r4"));
    assert!(glob("[!r]*", "c5") && !glob("[!r]*", "r5"));
    assert!(glob("xu1.r?", "xu1.r7") && glob("R*", "r10"));
}
