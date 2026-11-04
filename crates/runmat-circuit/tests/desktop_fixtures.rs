//! Desktop fixture: the real circuit JSON projection of a small mixed-signal board (two analog
//! channels, a third added since the board was last synced, pours, seeded DRC violations), for
//! the Desktop's circuit Outline provider and its tests.
//!
//! The Desktop checks it in at `desktop/src/app/components/circuit/fixtures/io.circuit.json`.
//! Set `RUNMAT_DESKTOP_FIXTURES` to that folder to write it; with it unset, the test compares
//! against the sibling Desktop checkout when there is one.

use runmat_circuit::board_rules::{props, BoardGraph, BoardRules, RuleSpec, RuleValue};
use runmat_circuit::drc::Drc;
use runmat_circuit::elab::{elaborate, ElabOptions};
use runmat_circuit::erc::{self, ErcOptions};
use runmat_circuit::fixture::{self, LargeParams};
use runmat_circuit::geom::mm;
use runmat_circuit::ids::{DerivedUuids, LockSection};
use runmat_circuit::index::{add_board, build_index, to_json};
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
    let synced = LargeParams {
        channels: 2,
        copper_layers: 4,
        route_every: 2,
        pours: true,
    };
    let params = LargeParams {
        channels: 3,
        ..synced.clone()
    };
    let pool = fixture::standard_pool();
    let design = fixture::large_design(&params);
    let nl = elaborate(&design, &pool, &ElabOptions::default(), None);
    // The lock as of the last board sync (two channels); the third channel is pending ECO.
    let mut lock = LockSection::default();
    let old = elaborate(
        &fixture::large_design(&synced),
        &pool,
        &ElabOptions::default(),
        None,
    );
    lock.reconcile(&old.lock_items(), &mut DerivedUuids { namespace: 7 });
    let eco = lock.reconcile(&nl.lock_items(), &mut DerivedUuids { namespace: 7 });
    let erc = erc::run(&nl, &ErcOptions::default());
    let findings = review::run_all(&nl, &pool, &ReviewOptions::default());
    let mut index = build_index(&design, &nl, &pool, &erc, &findings, &eco);

    let (mut board, _) = fixture::large_board_report(&pool, &old, &synced);
    fixture::seed_violations(&mut board, 2);
    let (mut rules, errs) = BoardRules::new(&board.limits, &specs());
    assert!(errs.is_empty(), "{errs:?}");
    rules.rebuild(&BoardGraph::new(&board, &[]));
    let fills = fill_zones(&mut board, &rules, &[]);
    let drc = Drc::run(&board, &rules, &[], &fills);
    add_board(&mut index, &board, &rules, &drc);

    assert!(
        !index.eco.is_empty(),
        "the third channel waits for the board"
    );
    let board_index = index.board.as_ref().unwrap();
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
    let sibling = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../runmat-private-cad/desktop/src/app/components/circuit/fixtures");
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
