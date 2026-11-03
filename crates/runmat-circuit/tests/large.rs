//! Large fixture: thousands of nets on 8 layers. Prints timings; checks determinism.
//!
//! Run with `--release --nocapture` to see the timing lines.

use runmat_circuit::board_rules::{props, BItem, BoardGraph, BoardRules, RuleSpec, RuleValue};
use runmat_circuit::drc::Drc;
use runmat_circuit::elab::{elaborate, ElabOptions};
use runmat_circuit::erc::{self, ErcOptions};
use runmat_circuit::fixture::{self, LargeParams};
use runmat_circuit::geom::{mm, Point};
use runmat_circuit::route;
use runmat_circuit::zone::fill_zones;
use std::time::Instant;

fn specs() -> Vec<RuleSpec> {
    vec![
        RuleSpec::pair("power", "copper:net-class(power)", "copper")
            .set(props::CLEARANCE, RuleValue::Len(mm(0.2))),
        RuleSpec::pair("inner", "copper:layer(inner)", "copper:layer(inner)")
            .set(props::CLEARANCE, RuleValue::Len(mm(0.12))),
        RuleSpec::single("power-width", "track:net-class(power)")
            .set(props::TRACK_WIDTH_MIN, RuleValue::Len(mm(0.3))),
        RuleSpec::single("adc", "net[name^=ADC]")
            .set(props::MATCH_GROUP, RuleValue::Text("adc".into()))
            .set(props::SKEW_MAX, RuleValue::Len(mm(25.0))),
    ]
}

#[test]
fn large_board_end_to_end() {
    let params = LargeParams {
        channels: 400,
        copper_layers: 8,
        route_every: 2,
        pours: true,
    };
    let t = Instant::now();
    let pool = fixture::standard_pool();
    let design = fixture::large_design(&params);
    let t_design = t.elapsed();
    let t = Instant::now();
    let nl = elaborate(&design, &pool, &ElabOptions::default(), None);
    let t_elab = t.elapsed();
    assert!(
        nl.diagnostics.is_empty(),
        "{:?}",
        &nl.diagnostics[..nl.diagnostics.len().min(5)]
    );
    assert!(nl.nets.len() >= 2000, "{} nets", nl.nets.len());
    assert!(nl.components.len() >= 2800);
    let t = Instant::now();
    let erc = erc::run(
        &nl,
        &ErcOptions {
            power_flags: ["GND", "+5V", "-5V"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            ..Default::default()
        },
    );
    let t_erc = t.elapsed();
    let t = Instant::now();
    let mut board = fixture::large_board(&pool, &nl, &params);
    // Large pours: skip the min-width opening here (it is exercised on small boards).
    for z in board.zones.values_mut() {
        z.min_width = 0;
    }
    let t_board = t.elapsed();
    let (mut rules, errs) = BoardRules::new(&board.limits, &specs());
    assert!(errs.is_empty());
    let t = Instant::now();
    rules.rebuild(&BoardGraph::new(&board, &[]));
    let t_rules = t.elapsed();
    let t = Instant::now();
    let reps = fill_zones(&mut board, &rules, &[]);
    let t_fill = t.elapsed();
    let fill_vertices: usize = board
        .zones
        .values()
        .flat_map(|z| z.fill.values())
        .map(|f| f.vertex_count())
        .sum();
    let t = Instant::now();
    let mut drc = Drc::run(&board, &rules, &[], &reps);
    let t_drc = t.elapsed();
    let t = Instant::now();
    let rn = route::ratsnest(&board);
    let t_rats = t.elapsed();
    // Incremental edit: move one track, update.
    let tid = *board.tracks.keys().nth(board.tracks.len() / 2).unwrap();
    let t = Instant::now();
    if let runmat_circuit::layout::TrackGeom::Segment { a, .. } =
        &mut board.tracks.get_mut(&tid).unwrap().geom
    {
        *a = *a + Point::mm(0.3, 0.3);
    }
    let g = BoardGraph::new(&board, &[]);
    rules.single.items_changed(&g, &[BItem::Track(tid)], &[]);
    rules.pair.items_changed(&g, &[BItem::Track(tid)], &[]);
    drc.update(&board, &rules, &[BItem::Track(tid)], &[], None);
    let t_inc = t.elapsed();
    println!(
        "large fixture: {} components, {} nets, {} pads, {} tracks, {} vias, {} copper layers",
        nl.components.len(),
        nl.nets.len(),
        board.all_pads().len(),
        board.tracks.len(),
        board.vias.len(),
        board.copper_count()
    );
    println!(
        "timings: design {t_design:?}, elaborate {t_elab:?}, ERC {t_erc:?} ({} findings), place+route {t_board:?}, rules {t_rules:?}",
        erc.len()
    );
    println!(
        "timings: zone fill {t_fill:?} ({} zone layers, {fill_vertices} vertices), DRC full {t_drc:?} ({} findings), ratsnest {t_rats:?} ({} airwires), incremental DRC edit {t_inc:?}",
        reps.len(),
        drc.findings().len(),
        rn.len()
    );
    let mut by_code: std::collections::BTreeMap<&str, usize> = Default::default();
    for f in drc.findings() {
        *by_code.entry(f.code.as_str()).or_default() += 1;
    }
    println!("DRC findings by code: {by_code:?}");
    assert!(!rn.is_empty());
    assert!(reps.iter().all(|r| r.polygons > 0));
}

#[test]
fn large_board_is_deterministic() {
    let params = LargeParams {
        channels: 60,
        copper_layers: 10,
        route_every: 2,
        pours: true,
    };
    let run = || {
        let (_, _, _, mut board) = fixture::large(&params);
        let (mut rules, _) = BoardRules::new(&board.limits, &specs());
        rules.rebuild(&BoardGraph::new(&board, &[]));
        let reps = fill_zones(&mut board, &rules, &[]);
        let drc = Drc::run(&board, &rules, &[], &reps);
        let fills: Vec<_> = board.zones.values().map(|z| z.fill.clone()).collect();
        (drc.digest(), fills, route::ratsnest(&board))
    };
    let a = run();
    let b = run();
    assert_eq!(a.0, b.0);
    assert_eq!(a.1, b.1);
    assert_eq!(a.2, b.2);
}
