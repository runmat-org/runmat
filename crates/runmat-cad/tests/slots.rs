//! Body slots across regenerations: lineage signatures stored beside items that name one
//! body, compared on the next regeneration.

mod common;
use common::*;
use runmat_cad::diagnostics::DiagCode;
use runmat_cad::model::*;
use runmat_cad::refs::*;
use runmat_cad::regen::slots::{record_slots, SlotKey, SlotStatus, SlotTable, SLOT_KEEP};
use runmat_cad::regen::*;
use runmat_cad::value::ValueExpr;

/// The block plus a separate tab body with four holes, and a part output on the tab.
fn two_bodies() -> PartDocument {
    let mut d = block();
    d.features.push(sketch(
        "s2",
        rect_names(
            PlaneRef::origin_xy(),
            100.0,
            0.0,
            120.0,
            50.0,
            ["t1", "t2", "t3", "t4"],
        ),
    ));
    d.features
        .push(extrude("tab", "s2", mm(10.0), BodyMode::New));
    let at = |x: f64, y: f64| PointRef::Coords {
        x: mm(x),
        y: mm(y),
        z: mm(10.0),
    };
    d.features.push(Feature::new(
        "holes",
        FeatureKind::Hole(Hole {
            face: Reference::end_cap("tab"),
            positions: vec![
                HolePosition {
                    key: "a".into(),
                    at: at(105.0, 10.0),
                },
                HolePosition {
                    key: "b".into(),
                    at: at(115.0, 10.0),
                },
                HolePosition {
                    key: "c".into(),
                    at: at(105.0, 40.0),
                },
                HolePosition {
                    key: "d".into(),
                    at: at(115.0, 40.0),
                },
            ],
            diameter: mm(4.0),
            depth: None,
            form: HoleForm::Simple,
        }),
    ));
    d.parts.push(PartOutput {
        name: "tab".into(),
        body: Reference::feature_body("tab"),
        part_number: None,
        material: None,
        properties: Vec::new(),
    });
    d
}

fn regen_with(env: &Env, d: &PartDocument, slots: &SlotTable) -> RegenResult {
    let ctx = RegenContext {
        slots: Some(slots),
        ..env.ctx()
    };
    regenerate(
        d,
        &ctx,
        &RegenRequest::default(),
        &mut RegenCache::default(),
    )
}

#[test]
fn slots_are_kept_across_an_unchanged_regeneration() {
    let env = Env::new();
    let d = two_bodies();
    let first = env.regen(&d);
    assert_clean(&first);
    let table = record_slots(&d, &first);
    assert_eq!(table.slots.len(), 2);
    // The table round-trips through JSON.
    let json = serde_json::to_string(&table).unwrap();
    assert_eq!(serde_json::from_str::<SlotTable>(&json).unwrap(), table);
    let again = regen_with(&env, &d, &table);
    assert_clean(&again);
    for row in &again.slots {
        assert_eq!(
            row.status,
            SlotStatus::Kept {
                agreement: runmat_cad::refs::naming::SIGNATURE_LEN as u32
            }
        );
    }
    // A parameter edit keeps every name, so every slot stays full.
    let mut wider = d.clone();
    set_input(&mut wider, "width", mm(90.0));
    let r = regen_with(&env, &wider, &table);
    assert!(r
        .slots
        .iter()
        .all(|s| matches!(s.status, SlotStatus::Kept { .. })));
}

#[test]
fn a_broken_item_is_offered_the_body_carrying_its_lineage() {
    let env = Env::new();
    let d = two_bodies();
    let table = record_slots(&d, &env.regen(&d));
    // A merge brought a part output whose reference names a feature that no longer exists;
    // the tab it was written against is still there, unchanged.
    let mut broken = d.clone();
    broken.parts.last_mut().unwrap().body = Reference::feature_body("tab_old");
    let r = regen_with(&env, &broken, &table);
    let row = r
        .slots
        .iter()
        .find(|s| s.key == SlotKey::Part { part: "tab".into() })
        .unwrap();
    let SlotStatus::Lost { candidates } = &row.status else {
        panic!("{:?}", row.status)
    };
    assert_eq!(candidates.len(), 1, "{candidates:?}");
    assert_eq!(
        candidates[0].1,
        runmat_cad::refs::naming::SIGNATURE_LEN as u32
    );
    assert!(r.state.bodies[candidates[0].0]
        .names
        .body
        .iter()
        .any(|n| n.maker().0 == "tab"));
    let drift = r
        .diagnostics
        .diagnostics
        .iter()
        .find(|x| x.code == DiagCode::RefBodySlotDrift)
        .expect("a drift warning with a repair");
    assert_eq!(drift.repairs.len(), 1);
    // The block's slot is kept.
    let block_row = r
        .slots
        .iter()
        .find(|s| {
            s.key
                == SlotKey::Part {
                    part: "bracket".into(),
                }
        })
        .unwrap();
    assert!(matches!(block_row.status, SlotStatus::Kept { .. }));
}

#[test]
fn a_body_remade_by_other_features_drifts() {
    let env = Env::new();
    let d = two_bodies();
    let table = record_slots(&d, &env.regen(&d));
    // The tab's history is rewritten: its extrude is now a 2 x 6 grid of holes deeper, and
    // the part output points at a new body made from the same sketch.
    let mut redo = d.clone();
    redo.features.retain(|f| f.id.0 != "holes");
    redo.features.push(Feature::new(
        "tab2",
        FeatureKind::Extrude(Extrude {
            profile: ProfileRef::sketch("s2"),
            extent: Extent::Blind {
                distance: ValueExpr::mm(20.0),
            },
            reverse: true,
            mode: BodyMode::New,
        }),
    ));
    redo.parts.last_mut().unwrap().body = Reference::feature_body("tab2");
    let r = regen_with(&env, &redo, &table);
    let row = r
        .slots
        .iter()
        .find(|s| s.key == SlotKey::Part { part: "tab".into() })
        .unwrap();
    let SlotStatus::Drifted { agreement, better } = &row.status else {
        panic!("{:?}", row.status)
    };
    assert!(*agreement < SLOT_KEEP);
    // The original tab is still alive and carries the stored lineage best.
    let (b, a) = better.expect("the original tab is offered");
    assert!(a > *agreement);
    assert!(r.state.bodies[b]
        .names
        .body
        .iter()
        .any(|n| n.maker().0 == "tab"));
    assert!(has(&r, DiagCode::RefBodySlotDrift));
}
