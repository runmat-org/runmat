//! Desktop fixture: the real circuit JSON projection of a small mixed-signal board (two analog
//! channels, a third added since the board was last synced, edits of every ECO type, board-side
//! swaps and renumbering, pours, seeded DRC violations), for the Desktop's circuit Outline
//! provider, its ECO review and their tests.
//!
//! The Desktop checks it in at `desktop/src/app/components/circuit/fixtures/io.circuit.json`.
//! Set `RUNMAT_DESKTOP_FIXTURES` to that folder to write it; with it unset, the test compares
//! against the sibling Desktop checkout when there is one.

use runmat_circuit::board_rules::{props, BoardGraph, BoardRules, RuleSpec, RuleValue};
use runmat_circuit::drc::Drc;
use runmat_circuit::eco;
use runmat_circuit::erc::{self, ErcOptions};
use runmat_circuit::fixture;
use runmat_circuit::geom::mm;
use runmat_circuit::index::{add_board, add_eco, build_index, to_json};
use runmat_circuit::review::{self, ReviewOptions};
use runmat_circuit::zone::fill_zones;
use runmat_rules::{SourceKey, SourceOrder};
use std::path::PathBuf;

fn at(spec: RuleSpec, offset: u32) -> RuleSpec {
    spec.at(
        SourceKey::new("io.circuit", offset, offset + 48),
        SourceOrder {
            file_rank: 1,
            offset,
        },
    )
}

fn specs() -> Vec<RuleSpec> {
    vec![
        at(
            RuleSpec::pair("power-clearance", "copper:net-class(power)", "copper")
                .set(props::CLEARANCE, RuleValue::Len(mm(0.2))),
            120,
        ),
        at(
            RuleSpec::single("power-width", "track:net-class(power)")
                .set(props::TRACK_WIDTH, RuleValue::Len(mm(0.4)))
                .set(props::TRACK_WIDTH_MIN, RuleValue::Len(mm(0.3))),
            180,
        ),
        at(
            RuleSpec::single("signal-width", "track")
                .set(props::TRACK_WIDTH, RuleValue::Len(mm(0.15))),
            240,
        ),
    ]
}

#[test]
fn circuit_fixture() {
    // A board synced with two channels and edited on the board side; the schematic since gained a
    // third channel and edits of every ECO type (`fixture::eco_scenario`).
    let sc = fixture::eco_scenario();
    let pool = &sc.pool;
    let design = &sc.design;
    let mut nl = sc.netlist.clone();
    // The FPGA's unused IO carry no-ERC markers, as a designer places them; two are still unmarked.
    if let Some(ci) = nl.components.iter().position(|c| c.designator == "U9000") {
        let unused: Vec<usize> = nl.components[ci]
            .pins
            .iter()
            .enumerate()
            .filter(|(_, p)| p.net.is_none())
            .map(|(pi, _)| pi)
            .collect();
        for pi in unused.into_iter().skip(2) {
            nl.no_erc.insert((ci, pi));
        }
    }
    // +5 V and ground come in on J1 (power flags); the negative rail has no source yet.
    let erc = erc::run(
        &nl,
        &ErcOptions {
            power_flags: ["+5V", "GND"].iter().map(|n| (*n).to_owned()).collect(),
            allow_single: ["SHIELD"].iter().map(|n| (*n).to_owned()).collect(),
            ..ErcOptions::default()
        },
    );
    assert!(
        erc.len() < 12,
        "a realistic ERC list, not noise: {} entries",
        erc.len()
    );
    let findings = review::run_all(&nl, pool, &ReviewOptions::default());
    let mut index = build_index(design, &nl, pool, &erc, &findings);

    let mut board = sc.board.clone();
    fixture::seed_violations(&mut board, 2);
    let (mut rules, errs) = BoardRules::new(&board.limits, &specs());
    assert!(errs.is_empty(), "{errs:?}");
    rules.rebuild(&BoardGraph::new(&board, &[]));
    let fills = fill_zones(&mut board, &rules, &[]);
    let drc = Drc::run(&board, &rules, &[], &fills);
    add_board(&mut index, &board, &rules, &drc);
    let s = eco::Schematic {
        design,
        netlist: &nl,
        pool,
        lock: &sc.lock,
    };
    let forward = eco::compare(&s, &board, &eco::ComparatorOptions::default());
    let back = eco::back_annotate(&s, &board, &eco::ComparatorOptions::default());
    add_eco(&mut index, &forward, &back, &board);

    assert!(index.eco.total > 0, "the third channel waits for the board");
    assert!(
        index.back_annotation.total > 0,
        "board edits wait for the schematic"
    );
    let board_index = index.board.as_ref().unwrap();
    assert!(board_index.layers.iter().any(|l| l.kind == "copper"));
    assert!(!board_index.pads.is_empty() && !board_index.rules.is_empty());
    assert!(board_index
        .zones
        .iter()
        .all(|z| z.text.contains_key("connect") && z.values.contains_key("clearance")));
    assert!(
        !board_index.drc.is_empty(),
        "seeded violations are reported"
    );
    assert!(board_index
        .tracks
        .iter()
        .any(|t| t.explain.values().any(|line| line.contains("from rule"))));
    let json = to_json(&index);
    assert_eq!(json, to_json(&index.clone()));

    let dir = std::env::var("RUNMAT_DESKTOP_FIXTURES")
        .map(PathBuf::from)
        .ok();
    // `RUNMAT_PRIVATE_DIR` names the private checkout (a lane's own worktree), else the
    // `runmat-private-cad` beside this one.
    let sibling = std::env::var_os("RUNMAT_PRIVATE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../runmat-private-cad"))
        .join("desktop/src/app/components/circuit/fixtures");
    if let Some(dir) = dir {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("io.circuit.json"), &json).unwrap();
    } else if sibling.exists() {
        let checked_in =
            std::fs::read_to_string(sibling.join("io.circuit.json")).unwrap_or_default();
        assert!(
            checked_in == json,
            "io.circuit.json is stale; regenerate with RUNMAT_DESKTOP_FIXTURES"
        );
    }
}
