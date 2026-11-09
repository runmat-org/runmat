//! Monte Carlo parallel speedup: the same runs on 1, 2, 4, … threads up to every core.
//!
//! `cargo run --release -p runmat-circuit-sim --example mc_speedup [runs]`

use runmat_circuit_sim::measure::Measure;
use runmat_circuit_sim::montecarlo::{run, Distribution, MonteCarlo, Target, Tolerance};
use runmat_circuit_sim::{Analysis, Design, TranSpec};
use std::time::Instant;

const DECK: &str = "rc ladder with an op-amp buffer
VIN in 0 PULSE(0 1 0 1u 1u 1 2)
R1 in a 1k
C1 a 0 1n
R2 a b 1k
C2 b 0 1n
R3 b c 1k
C3 c 0 1n
XU c out out OPX
.model OPX OPAMP(a0=2e5 gbw=3meg sr=1.5meg)
RL out 0 10k
";

fn main() {
    let runs: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let d = Design::from_netlist(DECK, &|_| None).unwrap();
    let mc = MonteCarlo {
        runs,
        first: 0,
        seed: 7,
        tolerances: vec![
            Tolerance::new(
                Target::Elements("r*".into()),
                Distribution::Gaussian { sigma: 0.01 / 3.0 },
            ),
            Tolerance::new(
                Target::Elements("c*".into()),
                Distribution::Gaussian { sigma: 0.05 / 3.0 },
            ),
        ],
        analysis: Analysis::Tran(TranSpec {
            tstep: 0.1e-6,
            tstop: 20e-6,
            tstart: 0.0,
            tmax: None,
            uic: false,
        }),
        measures: vec![(
            "t50".into(),
            Measure::When {
                signal: "v(out)".into(),
                value: 0.5,
                edge: runmat_circuit_sim::measure::Edge::Rise,
                nth: 1,
            },
        )],
        keep_results: 0,
        limit: None,
    };
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut threads = vec![1];
    while threads.last().unwrap() * 2 < cores {
        threads.push(threads.last().unwrap() * 2);
    }
    if *threads.last().unwrap() != cores {
        threads.push(cores);
    }
    let mut base = 0.0;
    println!("{runs} transient runs, {cores} cores");
    for t in threads {
        let start = Instant::now();
        let o = run(&d, &mc, Some(t)).unwrap();
        let s = start.elapsed().as_secs_f64();
        if t == 1 {
            base = s;
        }
        let st = &o.stats["t50"];
        println!(
            "threads {t:>2}: {s:7.3} s  speedup {:5.2}x  t50 mean {:.4e} s std {:.3e} s  ops {}",
            base / s,
            st.mean,
            st.std,
            o.usage.total().unwrap()
        );
    }
}
