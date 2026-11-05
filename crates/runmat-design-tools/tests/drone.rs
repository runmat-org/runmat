//! The quadcopter showcase's circuit simulations (`runmat-cad/wasm/tests/drone`), run through
//! `design.simulate` as the Desktop and the agent run them: the flight controller's 5 V buck
//! start-up and ripple, its setpoint over tolerance (Monte Carlo), and the ESC's gate-drive
//! switching transient. Every measure with a spec passes. `RUNMAT_DRONE_SRC=<folder>` runs
//! another copy of the project; `RUNMAT_DRONE_SIM_OUT=<folder>` writes each report as JSON.

use runmat_design_tools::{call, Files};
use serde_json::{json, Value};
use std::path::PathBuf;

fn project() -> Files {
    let dir = std::env::var("RUNMAT_DRONE_SRC").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../runmat-cad/wasm/tests/drone")
    });
    let mut files = Files::new();
    for e in std::fs::read_dir(&dir).expect("the drone project is checked in").flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if let Ok(text) = std::fs::read_to_string(e.path()) {
            files.insert(format!("/p/drone/{name}"), text);
        }
    }
    files
}

fn run(files: &Files, path: &str, simulation: &str) -> Value {
    let started = std::time::Instant::now();
    let out = call("design.simulate", json!({ "path": path, "simulation": simulation, "maxPoints": 400 }), files)
        .unwrap_or_else(|e| panic!("{simulation}: {e:?}"));
    eprintln!("{simulation}: {:.2?}, usage {}", started.elapsed(), out["usage"]["total"]);
    if let Ok(dir) = std::env::var("RUNMAT_DRONE_SIM_OUT") {
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(format!("{dir}/{simulation}.json"), serde_json::to_string(&out).unwrap()).ok();
    }
    let mut failed = Vec::new();
    for m in out["measures"].as_array().unwrap() {
        eprintln!(
            "  {} {} = {} {} (mean {}, std {}, yield {}) pass {} {}",
            m["name"], m["kind"], m["value"], m["unit"], m["mean"], m["std"], m["yieldPercent"], m["pass"], m["error"]
        );
        let bad = m["pass"] == false || m["error"].is_string() || m["yieldPercent"].as_f64().is_some_and(|y| y < 99.0);
        if bad {
            failed.push(m["name"].to_string());
        }
    }
    for w in out["warnings"].as_array().unwrap() {
        eprintln!("  warning: {w}");
    }
    assert!(failed.is_empty(), "{simulation}: {failed:?}");
    out
}

fn measure(out: &Value, name: &str) -> f64 {
    let m = out["measures"].as_array().unwrap().iter().find(|m| m["name"] == name).unwrap();
    m["value"].as_f64().or_else(|| m["mean"].as_f64()).unwrap()
}

#[test]
fn the_buck_starts_up_and_regulates_5v() {
    let files = project();
    let out = run(&files, "/p/drone/flight_controller.circuit", "buck_startup");
    // Closed form: FB at 0.8 V through 52.3 k over 10 k gives 0.8 x 6.23 = 4.984 V; peak-to-peak
    // ripple of an LC filter is dI / (8 f C) with dI = Vo (1 - D) / (L f) = 0.776 A, so about
    // 4.4 mV on 44 uF at 500 kHz.
    let v5 = measure(&out, "v5");
    assert!((v5 - 4.984).abs() < 0.05, "{v5}");
}

#[test]
fn the_5v_setpoint_holds_over_tolerance() {
    let files = project();
    let out = run(&files, "/p/drone/flight_controller.circuit", "buck_tolerance");
    let v5 = measure(&out, "v5");
    assert!((v5 - 4.984).abs() < 0.01, "{v5}");
}

#[test]
fn the_esc_phase_switches_cleanly_into_the_motor() {
    let files = project();
    let out = run(&files, "/p/drone/esc.circuit", "gate_drive");
    // Phase A's high side carries the 20 A winding current when on: the phase node swings to the
    // 22.2 V battery less I x R_DS(on) (20 A x 1.6 mOhm = 32 mV), and the bootstrapped gate sits
    // about 10 V above it.
    let gate = measure(&out, "gate_high");
    assert!(gate > 22.2 + 8.0 && gate < 34.0, "{gate}");
    let on = measure(&out, "on_current");
    assert!(on > 10.0, "{on}");
    // The phase node's 10-90 % rise on the high side's turn-on: the 2 Ohm driver charging the
    // FET's 0.25 nF Miller capacitance through the 20 A commutation takes a few nanoseconds.
    let rise = measure(&out, "rise");
    assert!(rise > 2e-9 && rise < 40e-9, "{rise}");
}
