//! Operation limits: a run stops before it would pass the host's granted operations and keeps
//! the points it solved.

use runmat_circuit_sim::measure::Measure;
use runmat_circuit_sim::montecarlo::{run, Distribution, MonteCarlo, Target, Tolerance};
use runmat_circuit_sim::{simulate_circuit, simulate_limited, AcSweep, Analysis, Design, SimError, SimResult, TranSpec};

const RC: &str = "rc\nV1 in 0 PULSE(0 1 0 1n 1n 1 2) AC 1\nR1 in out 1k\nC1 out 0 1u\n";

fn total(o: &runmat_circuit_sim::Outcome) -> u64 {
    o.usage.total().unwrap().operations()
}

#[test]
fn a_transient_stops_under_its_limit_with_the_points_so_far() {
    let c = Design::from_netlist(RC, &|_| None).unwrap().circuit().unwrap();
    let a = Analysis::Tran(TranSpec { tstep: 10e-6, tstop: 5e-3, tstart: 0.0, tmax: None, uic: false });
    let full = simulate_circuit(&c, &a).unwrap();
    assert!(!full.stopped);
    let limit = total(&full) / 3;
    let part = simulate_limited(&c, &a, Some(limit)).unwrap();
    assert!(part.stopped);
    assert!(total(&part) <= limit, "{} > {limit}", total(&part));
    let (SimResult::Tran(p), SimResult::Tran(f)) = (&part.result, &full.result) else { panic!() };
    assert!(p.x.len() > 10 && p.x.len() < f.x.len());
    // The points kept are the full run's points.
    let n = p.x.len();
    assert_eq!(p.x[..], f.x[..n]);
    assert_eq!(p.get("v(out)").unwrap()[..], f.get("v(out)").unwrap()[..n]);
}

#[test]
fn an_ac_sweep_stops_under_its_limit() {
    let c = Design::from_netlist(RC, &|_| None).unwrap().circuit().unwrap();
    let a = Analysis::Ac { sweep: AcSweep::Dec, points: 50, fstart: 1.0, fstop: 1e6 };
    let full = simulate_circuit(&c, &a).unwrap();
    let limit = total(&full) / 2;
    let part = simulate_limited(&c, &a, Some(limit)).unwrap();
    assert!(part.stopped && total(&part) <= limit);
    let (SimResult::Ac(p), SimResult::Ac(f)) = (&part.result, &full.result) else { panic!() };
    assert!(!p.freq.is_empty() && p.freq.len() < f.freq.len());
}

#[test]
fn an_operating_point_that_cannot_finish_is_stopped() {
    let c = Design::from_netlist(RC, &|_| None).unwrap().circuit().unwrap();
    assert_eq!(simulate_limited(&c, &Analysis::Op, Some(0)).unwrap_err(), SimError::Stopped);
    // A generous limit changes nothing.
    let free = simulate_circuit(&c, &Analysis::Op).unwrap();
    let held = simulate_limited(&c, &Analysis::Op, Some(total(&free) * 10)).unwrap();
    assert_eq!(free, held);
}

#[test]
fn a_limited_monte_carlo_keeps_its_first_runs_and_stays_under_the_limit() {
    let d = Design::from_netlist(RC, &|_| None).unwrap();
    let mut mc = MonteCarlo {
        runs: 200,
        first: 0,
        seed: 3,
        tolerances: vec![Tolerance::new(Target::Elements("*".into()), Distribution::Gaussian { sigma: 0.02 })],
        analysis: Analysis::Ac { sweep: AcSweep::Dec, points: 20, fstart: 1.0, fstop: 1e5 },
        measures: vec![("f".into(), Measure::Bandwidth { signal: "v(out)".into(), drop_db: 3.0103 })],
        keep_results: 0,
        limit: None,
    };
    let full = run(&d, &mc, Some(4)).unwrap();
    assert_eq!((full.completed, full.stopped), (200, false));
    let limit = full.usage.total().unwrap().operations() * 2 / 5;
    mc.limit = Some(limit);
    let one = run(&d, &mc, Some(1)).unwrap();
    let many = run(&d, &mc, Some(8)).unwrap();
    assert!(one.stopped && one.completed > 0 && one.completed < 200);
    assert!(one.usage.total().unwrap().operations() <= limit);
    assert_eq!((one.completed, &one.values, one.usage), (many.completed, &many.values, many.usage));
    assert_eq!(one.values["f"][..], full.values["f"][..one.completed]);
}

/// Circuits whose steps differ in cost: junction limiting, a switch that changes state, a
/// rectifier whose pivots change as the diode turns on, a bipolar amplifier.
const RECTIFIER: &str = "rect\nV1 in 0 SIN(0 5 1k) AC 1\nD1 in out DX\nR1 out 0 1k\nC1 out 0 10u\n.model DX D(IS=1e-14 N=1.5 RS=10)\n";
const SWITCHED: &str = "sw\nVC c 0 PWL(0 0 1m 2 2m 0)\nV1 a 0 1\nR1 a o 1k\nS1 o 0 c 0 SWX\nC1 o 0 1u\n.model SWX SW(VT=1 VH=0.2 RON=1 ROFF=1meg)\n";
const AMP: &str = "amp\nVCC vcc 0 12\nVIN in 0 SIN(0 10m 1k) AC 1\nC1 in b 10u\nRB1 vcc b 100k\nRB2 b 0 22k\nRC vcc c 4.7k\nRE e 0 1k\nCE e 0 100u\nQ1 c b e QN\n.model QN NPN(IS=1e-15 BF=150 CJE=2p CJC=1p)\n";
const NMOS: &str = "nmos\nVDD vdd 0 5\nVG g 0 PULSE(0 5 0 10u 10u 100u 200u) AC 1\nRD vdd d 2k\nM1 d g 0 0 NX L=1u W=10u\nCL d 0 10p\n.model NX NMOS(LEVEL=1 VTO=1 KP=1e-4 LAMBDA=0.02)\n";

/// For every limit on a fine grid from zero to past the full cost, the operations a run reports
/// never pass the limit, whatever the analysis stops on: a costlier Newton iteration, a fresh
/// factorization after a pivot change, a switch that rebuilds the pattern, a convergence test.
#[test]
fn no_limit_is_ever_passed() {
    let tran = |tstep: f64, tstop: f64| Analysis::Tran(TranSpec { tstep, tstop, tstart: 0.0, tmax: None, uic: false });
    let cases: Vec<(&str, Analysis)> = vec![
        (RECTIFIER, tran(10e-6, 3e-3)),
        (RECTIFIER, Analysis::Ac { sweep: AcSweep::Dec, points: 10, fstart: 10.0, fstop: 1e6 }),
        (SWITCHED, tran(10e-6, 2e-3)),
        (AMP, tran(10e-6, 2e-3)),
        (AMP, Analysis::Op),
        (AMP, Analysis::Ac { sweep: AcSweep::Dec, points: 10, fstart: 10.0, fstop: 1e8 }),
        (AMP, Analysis::Dc { source: "VCC".into(), start: 0.0, stop: 12.0, step: 0.25 }),
        (NMOS, tran(1e-6, 400e-6)),
        (NMOS, Analysis::Dc { source: "VG".into(), start: 0.0, stop: 5.0, step: 0.05 }),
    ];
    for (deck, a) in &cases {
        let c = Design::from_netlist(deck, &|_| None).unwrap().circuit().unwrap();
        let full = total(&simulate_circuit(&c, a).unwrap());
        let mut stops = 0;
        for k in 0..=240u64 {
            let limit = full * k / 200;
            let (out, usage) = runmat_circuit_sim::simulate_spent(&c, a, Some(limit));
            let spent = usage.total().unwrap().operations();
            assert!(spent <= limit, "{a:?} on `{}`: spent {spent} > limit {limit}", deck.lines().next().unwrap());
            match out {
                Ok(o) => {
                    assert_eq!(total(&o), spent);
                    if o.stopped {
                        stops += 1;
                    } else {
                        assert_eq!(spent, full, "{a:?}: a finished run spends what the unlimited one does");
                    }
                }
                Err(SimError::Stopped) => stops += 1,
                Err(e) => panic!("{a:?} at {limit}: {e}"),
            }
        }
        // Every limit under the full cost stops the run.
        assert!(stops >= 200, "{a:?}: only {stops} limits stopped the run");
        let roomy = simulate_limited(&c, a, Some(full * 2)).unwrap();
        assert!(!roomy.stopped && total(&roomy) == full);
    }
}
