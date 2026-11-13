//! Circuit simulation in the module: a design's `simulation` sections, run one unit at a time
//! (`runmat_circuit_spice::study::Runner`) so the host shows progress, reports usage to its
//! simulation lease, passes the granted operations as each step's limit, and can cancel between
//! units with the exact count.
//!
//! - `simulation_setup(path, text, files)` → `{ simulations: [{ name, probes, analyses }] }`
//! - `simulate_begin(path, text, files, options)` → `{ ok, simulation, progress }`; options
//!   `{ simulation?, maxPoints?, bins? }`
//! - `simulate_step(limitOps)` → `{ ok, done, progress, usage }`; `limitOps` is the decimal
//!   count of operations the whole run may have spent after this step (the lease's grant)
//! - `simulate_end()` → the report (`runmat_circuit_spice::report`); a run not finished is
//!   cancelled first, and the report says what didn't run

use runmat_circuit::runtime::{design_and_pool, CompileOptions, Files};
use runmat_circuit::simulation::SimulationDef;
use runmat_circuit_spice::report::{self, ReportOptions};
use runmat_circuit_spice::study::{Progress, RunOptions, Runner};
use serde::Deserialize;
use serde_json::{json, Value};
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

struct Active {
    runner: Runner,
    def: SimulationDef,
    report: ReportOptions,
}

thread_local! {
    static ACTIVE: RefCell<Option<Active>> = const { RefCell::new(None) };
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct BeginOptions {
    simulation: Option<String>,
    max_points: Option<usize>,
    bins: Option<usize>,
}

fn failed(message: impl Into<String>) -> String {
    json!({ "ok": false, "message": message.into() }).to_string()
}

fn progress(p: &Progress) -> Value {
    json!({ "done": p.done, "total": p.total, "next": p.next })
}

fn usage(u: &runmat_circuit_sim::Usage) -> Value {
    let s = |c: runmat_simulation_usage::OperationCount| c.operations().to_string();
    json!({
        "measurementVersion": runmat_circuit_sim::MEASUREMENT_VERSION,
        "assembly": s(u.assembly),
        "factorization": s(u.factorization),
        "solve": s(u.solve),
        "other": s(u.other),
        "total": u.total().map(s).unwrap_or_else(|_| "overflow".into()),
    })
}

fn load(
    path: &str,
    text: &str,
    files: &str,
) -> Result<
    (
        runmat_circuit::ir::Design,
        runmat_circuit::library::Pool,
        Files,
    ),
    String,
> {
    let files: Files = crate::files(files)?;
    let (design, pool) =
        design_and_pool(path, text, &files, &CompileOptions::default()).map_err(|d| {
            d.iter()
                .map(|x| x.message.clone())
                .collect::<Vec<_>>()
                .join("; ")
        })?;
    Ok((design, pool, files))
}

/// The design's simulation sections, before any run.
#[wasm_bindgen]
pub fn simulation_setup(path: &str, text: &str, files: &str) -> String {
    match load(path, text, files) {
        Err(m) => failed(m),
        Ok((design, _, _)) => {
            let sims: Vec<Value> = design
                .simulations
                .values()
                .map(|s| json!({ "name": s.name, "probes": s.probes.iter().map(|p| p.name()).collect::<Vec<_>>(), "currents": s.probes.iter().filter(|p| p.quantity() == "current").map(|p| p.name()).collect::<Vec<_>>(), "analyses": s.analyses.iter().map(|(k, a)| json!({ "name": k, "kind": a.word() })).collect::<Vec<_>>() }))
                .collect();
            json!({ "ok": true, "simulations": sims }).to_string()
        }
    }
}

/// Starts a run of the design's simulation section (its only one, or `options.simulation`).
#[wasm_bindgen]
pub fn simulate_begin(path: &str, text: &str, files: &str, options: &str) -> String {
    let options: BeginOptions = if options.trim().is_empty() {
        BeginOptions::default()
    } else {
        match serde_json::from_str(options) {
            Ok(o) => o,
            Err(e) => return failed(format!("the options aren't valid: {e}")),
        }
    };
    let (design, pool, files) = match load(path, text, files) {
        Ok(x) => x,
        Err(m) => return failed(m),
    };
    let name = match options.simulation.clone() {
        Some(n) => n,
        None => match design.simulations.keys().collect::<Vec<_>>().as_slice() {
            [one] => (*one).clone(),
            [] => {
                return failed(format!(
                    "{path} has no `simulation` section; add one to the circuit block"
                ))
            }
            many => {
                return failed(format!(
                    "{path} has several simulations ({}); name one",
                    many.iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            }
        },
    };
    // A section's `library = "…"` is relative to the circuit file, as `dependencies` lists it.
    let resolve = |lib: &str| {
        files
            .get(&runmat_circuit::runtime::resolve_path(path, lib))
            .or_else(|| files.get(lib))
            .cloned()
    };
    let runner = match Runner::new(&design, &pool, &name, &resolve, RunOptions::default()) {
        Ok(r) => r,
        Err(m) => return failed(m),
    };
    let p = runner.progress();
    let def = design.simulations[&name].clone();
    let report = ReportOptions {
        max_points: options.max_points.unwrap_or(400),
        bins: options.bins.unwrap_or(24),
        ..Default::default()
    };
    ACTIVE.with(|a| {
        *a.borrow_mut() = Some(Active {
            runner,
            def,
            report,
        })
    });
    json!({ "ok": true, "simulation": name, "progress": progress(&p) }).to_string()
}

/// Runs the next unit of the active run under `limit_ops`.
#[wasm_bindgen]
pub fn simulate_step(limit_ops: &str) -> String {
    let limit = match limit_ops.trim() {
        "" => None,
        t => match t.parse::<u64>() {
            Ok(v) => Some(v),
            Err(_) => {
                return failed(format!(
                    "limitOps is a decimal count of operations, not `{t}`"
                ))
            }
        },
    };
    ACTIVE.with(|a| {
        let mut a = a.borrow_mut();
        let Some(active) = a.as_mut() else { return failed("no simulation is running; begin one first") };
        active.runner.set_limit(limit);
        match active.runner.step() {
            Ok(done) => json!({ "ok": true, "done": done, "progress": progress(&active.runner.progress()), "usage": usage(active.runner.usage()) }).to_string(),
            Err(m) => failed(m),
        }
    })
}

/// Ends the active run (cancelling what's left) and returns its report.
#[wasm_bindgen]
pub fn simulate_end() -> String {
    ACTIVE.with(|a| match a.borrow_mut().take() {
        None => failed("no simulation is running"),
        Some(mut active) => {
            active.runner.cancel();
            let study = active.runner.finish();
            let out = report::report(&study, &active.def, &active.report);
            serde_json::to_string(&out)
                .unwrap_or_else(|e| failed(format!("the report can't be written: {e}")))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECTION: &str = r#"
    simulation front_end {
        sheet = channel
        source VIN = { p = net.IN, ac = 1 V, pulse = { from = 0.25 V, to = 1.75 V, delay = 2 us, rise = 10 ns } }
        source V5 = { p = "net.+5V", dc = 5 V }
        source V3 = { p = "net.+3V3", dc = 3.3 V }
        source VR = { p = net.VREF, dc = 4.096 V }
        model OPA171 = { opamp = { a0 = 1000000, gbw = 3 MHz, sr = 1500000, rout = 10 ohm, vdrop = 0.02 V }, pins = ["IN+", "IN-", OUT, "V+", "V-"] }
        analysis response = ac { from = 100 Hz, to = 100 MHz, points = 20 }
        analysis spread = monte_carlo { of = response, runs = 200, seed = 7 }
        probe = [net.IN, net.AIN]
        measure f3db = { of = response, bandwidth = net.AIN }
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

    fn daq() -> String {
        let base = bare(include_str!("../../tests/projects/daq/daq.circuit"));
        let end = base.rfind('}').unwrap();
        format!("{}{SECTION}{}", &base[..end], &base[end..])
    }

    #[test]
    fn a_run_steps_reports_usage_and_ends_with_the_report() {
        let text = daq();
        let setup: Value =
            serde_json::from_str(&simulation_setup("/p/daq.circuit", &text, "")).unwrap();
        assert_eq!(setup["simulations"][0]["probes"], json!(["IN", "AIN"]));
        let b: Value =
            serde_json::from_str(&simulate_begin("/p/daq.circuit", &text, "", "")).unwrap();
        assert_eq!(b["ok"], true, "{b}");
        // One AC unit, then 200 runs in batches of 64.
        assert_eq!(b["progress"]["total"], 5);
        let mut last = 0u64;
        loop {
            let s: Value = serde_json::from_str(&simulate_step("")).unwrap();
            let total: u64 = s["usage"]["total"].as_str().unwrap().parse().unwrap();
            assert!(total > last);
            last = total;
            if s["done"] == true {
                break;
            }
        }
        let r: Value = serde_json::from_str(&simulate_end()).unwrap();
        assert_eq!(
            r["usage"]["total"]
                .as_str()
                .unwrap()
                .parse::<u64>()
                .unwrap(),
            last
        );
        assert_eq!(r["measures"][1]["runs"], 200);
        assert!(r.get("stopped").is_none());
    }

    #[test]
    fn ending_early_cancels_and_says_what_did_not_finish() {
        let text = daq();
        simulate_begin("/p/daq.circuit", &text, "", "");
        simulate_step("");
        simulate_step("");
        let r: Value = serde_json::from_str(&simulate_end()).unwrap();
        let stopped = r["stopped"].as_str().unwrap();
        assert!(
            stopped.starts_with("cancelled") && stopped.contains("spread"),
            "{stopped}"
        );
        assert_eq!(r["measures"][1]["runs"], 64);
    }
}
