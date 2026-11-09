//! Netlist emission, the fake engine, the power map into the board FEA source, the ngspice
//! loader and the usage seam.

use runmat_circuit::elab::{elaborate, ElabOptions, Netlist};
use runmat_circuit::fixture::{self, standard_pool, LargeParams};
use runmat_circuit::geom::{mm, Point, Rect};
use runmat_circuit::ir::{Design, DesignBuilder};
use runmat_circuit::layout::{Board, Side, Stackup};
use runmat_circuit::library::Pool;
use runmat_circuit_spice::fake::FakeEngine;
use runmat_circuit_spice::ngspice::Ngspice;
use runmat_circuit_spice::*;

/// VIN ─ R1 ─ MID ─ R2 ─ GND, 10 kΩ each.
fn divider() -> (Pool, Netlist) {
    let pool = standard_pool();
    let mut d = Design::new("divider");
    let mut b = DesignBuilder::new(&mut d, "main");
    b.global("GND", None);
    b.component("R1", "R0402-10K");
    b.component("R2", "R0402-10K");
    b.wire("VIN", &[("R1", "1")]);
    b.wire("MID", &[("R1", "2"), ("R2", "1")]);
    b.wire("GND", &[("R2", "2")]);
    let nl = elaborate(&d, &pool, &ElabOptions::default(), None);
    assert!(nl.diagnostics.is_empty(), "{:?}", nl.diagnostics);
    (pool, nl)
}

fn vin(v: f64) -> EmitOptions {
    EmitOptions {
        stimuli: vec![Stimulus::Voltage {
            name: "IN".into(),
            pos: "VIN".into(),
            neg: "GND".into(),
            wave: Waveform::Dc(v),
        }],
        ..Default::default()
    }
}

#[test]
fn emits_subcircuits_primitives_and_senses() {
    let p = LargeParams {
        channels: 2,
        copper_layers: 4,
        route_every: 0,
        pours: false,
    };
    let (pool, _, nl, _) = fixture::large(&p);
    // The FPGA and headers have no model: reported, or skipped on request.
    let errs = emit(&nl, &pool, &EmitOptions::default()).unwrap_err();
    assert!(errs.iter().any(|e| e.designator.starts_with("U9")));
    let deck = emit(
        &nl,
        &pool,
        &EmitOptions {
            skip_unmodelled: true,
            ..Default::default()
        },
    )
    .unwrap();
    let text = deck.text();
    assert!(text.contains(".include \"opa4171.lib\""), "{text}");
    // The op-amp as a subcircuit in its model's pin order, through sense sources.
    let x = deck
        .elements
        .iter()
        .find(|e| e.name.starts_with("XU") && e.value == "OPA4171")
        .unwrap();
    assert_eq!(x.nodes.len(), 5);
    assert!(x.nodes.iter().all(|n| n.starts_with("p_")));
    // Resistors and capacitors as primitives with their values.
    assert!(deck
        .elements
        .iter()
        .any(|e| e.name.starts_with('R') && e.value == "1e4"));
    assert!(deck
        .elements
        .iter()
        .any(|e| e.name.starts_with('C') && e.value == "1e-7"));
    // Ground is node 0; every sense names its component and net.
    assert_eq!(deck.node("GND"), "0");
    assert!(deck.senses.iter().all(|s| !s.designator.is_empty()));
    assert!(text.ends_with(".end\n"));
}

#[test]
fn fake_engine_operating_point_sweep_and_power() {
    let (pool, nl) = divider();
    let deck = emit(&nl, &pool, &vin(5.0)).unwrap();
    let mut eng = FakeEngine::new();
    let op = eng.run(&deck, &Analysis::Op).unwrap();
    let op = op.op().unwrap();
    assert!((op.voltages["MID"] - 2.5).abs() < 1e-12);
    assert!((op.voltages["VIN"] - 5.0).abs() < 1e-12);
    // The source delivers 0.25 mA: its branch current (into +) is negative.
    assert!(
        (op.currents["VIN"] + 0.25e-3).abs() < 1e-15,
        "{:?}",
        op.currents
    );
    let p = power_map(&deck, op);
    assert!((p["R1"] - 0.625e-3).abs() < 1e-12, "{p:?}");
    assert!((p["R2"] - 0.625e-3).abs() < 1e-12);
    // DC sweep of the input.
    let sw = eng
        .run(
            &deck,
            &Analysis::Dc {
                source: "VIN".into(),
                start: 0.0,
                stop: 10.0,
                step: 2.5,
            },
        )
        .unwrap();
    let SimResult::Dc(sw) = sw else { panic!() };
    assert_eq!(sw.x, vec![0.0, 2.5, 5.0, 7.5, 10.0]);
    assert!((sw.voltages["MID"][4] - 5.0).abs() < 1e-12);
    // Analyses the fake does not run say so.
    let e = eng
        .run(
            &deck,
            &Analysis::Tran {
                step: 1e-6,
                stop: 1e-3,
                start: 0.0,
                max_step: None,
            },
        )
        .unwrap_err();
    assert!(matches!(e, SpiceError::Unsupported(_)));
    assert_eq!(
        Analysis::Ac {
            sweep: AcSweep::Dec,
            points: 10,
            fstart: 1.0,
            fstop: 1e6
        }
        .card(),
        "ac dec 10 1e0 1e6"
    );
}

#[test]
fn operating_point_feeds_the_board_power_map() {
    let (pool, nl) = divider();
    let deck = emit(&nl, &pool, &vin(5.0)).unwrap();
    let op = FakeEngine::new().run(&deck, &Analysis::Op).unwrap();
    let watts = power_map(&deck, op.op().unwrap());
    let mut board = Board::new(
        "divider",
        runmat_circuit::zone::rect_outline(Rect::new(Point::ORIGIN, Point::mm(20.0, 10.0))),
        Stackup::standard(2, mm(1.6), 1.0, 1.0),
    );
    for (i, c) in nl.components.iter().enumerate() {
        let pkg = pool.packages[&c.package.unwrap()].clone();
        board.place(
            &c.designator,
            pkg,
            Point::mm(5.0 + 10.0 * i as f64, 5.0),
            0.0,
            Side::Top,
        );
    }
    board.assign_nets(&nl);
    let src = runmat_circuit::fea::pcb_source(&board, Some(&nl), Some(&pool), &watts);
    let r1 = src
        .power_map
        .iter()
        .find(|(k, _)| k.ends_with("/component/R1"))
        .map(|(_, w)| *w)
        .unwrap();
    assert!((r1 - 0.625e-3).abs() < 1e-12);
}

#[test]
fn usage_seam_sees_each_analysis() {
    struct Count(Vec<AnalysisReport>);
    impl UsageSeam for Count {
        fn analysis_finished(&mut self, r: &AnalysisReport) {
            self.0.push(r.clone());
        }
    }
    let (pool, nl) = divider();
    let deck = emit(&nl, &pool, &vin(1.0)).unwrap();
    let mut eng = FakeEngine::new();
    let mut u = Count(vec![]);
    simulate(&mut eng, &deck, &Analysis::Op, Some(&mut u)).unwrap();
    assert_eq!(u.0.len(), 1);
    assert_eq!(u.0[0].analysis, "op");
    assert_eq!(u.0[0].points, 1);
    assert!(u.0[0].elements >= 3);
}

#[test]
fn ngspice_runs_or_explains_why_not() {
    match Ngspice::load() {
        Err(SpiceError::EngineUnavailable { tried, hint }) => {
            assert!(!tried.is_empty());
            assert!(hint.contains("RUNMAT_NGSPICE"));
            let msg = SpiceError::EngineUnavailable { tried, hint }.to_string();
            assert!(msg.contains("ngspice is not available"));
        }
        Err(e) => panic!("{e}"),
        Ok(mut ng) => {
            // With ngspice present it agrees with the exact solver.
            let (pool, nl) = divider();
            let deck = emit(&nl, &pool, &vin(5.0)).unwrap();
            let a = ng.run(&deck, &Analysis::Op).unwrap();
            let b = FakeEngine::new().run(&deck, &Analysis::Op).unwrap();
            let (a, b) = (a.op().unwrap(), b.op().unwrap());
            assert!((a.voltages["MID"] - b.voltages["MID"]).abs() < 1e-6);
            let pa = power_map(&deck, a);
            assert!((pa["R1"] - 0.625e-3).abs() < 1e-7, "{pa:?}");
        }
    }
}
