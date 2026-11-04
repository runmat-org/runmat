//! Engineering change orders on a realistic multi-sheet, multi-channel design: every change
//! type, renames through the lock, a channel addition, partial apply, conflicts, back-annotation,
//! agent change sets and determinism.

use runmat_circuit::board_rules::BoardRules;
use runmat_circuit::eco::{
    self, ApplyOptions, ChangeKind, ComparatorOptions, Direction, EcoPlan, Op, Origin, Schematic,
    Status,
};
use runmat_circuit::elab::{elaborate, ElabOptions};
use runmat_circuit::fixture::{eco_scenario, EcoScenario};
use runmat_circuit::ids::DerivedUuids;
use runmat_circuit::index::{eco_index, to_json, CircuitIndex};
use runmat_circuit::ir::{ComponentDecl, Endpoint, Param, PartRef};
use runmat_circuit::review::ModelEdit;
use std::collections::BTreeSet;

fn forward(sc: &EcoScenario) -> EcoPlan {
    eco::compare(&schematic(sc), &sc.board, &ComparatorOptions::default())
}

fn schematic(sc: &EcoScenario) -> Schematic<'_> {
    Schematic {
        design: &sc.design,
        netlist: &sc.netlist,
        pool: &sc.pool,
        lock: &sc.lock,
    }
}

fn of(plan: &EcoPlan, kind: ChangeKind) -> Vec<&eco::Change> {
    plan.changes.iter().filter(|c| c.kind == kind).collect()
}

fn find<'a>(plan: &'a EcoPlan, kind: ChangeKind, item: &str) -> &'a eco::Change {
    plan.changes
        .iter()
        .find(|c| c.kind == kind && c.item == item)
        .unwrap_or_else(|| {
            panic!(
                "no {kind:?} for {item}: {:#?}",
                of(plan, kind).iter().map(|c| &c.item).collect::<Vec<_>>()
            )
        })
}

fn designator(sc: &EcoScenario, path: &str) -> String {
    sc.netlist
        .components
        .iter()
        .find(|c| c.path.to_string() == path)
        .unwrap()
        .designator
        .clone()
}

#[test]
fn every_change_type_with_cause_and_impact() {
    let sc = eco_scenario();
    let plan = forward(&sc);
    assert_eq!(plan.direction, Direction::Forward);

    // The third channel: seven parts staged beside the board, grouped in a new room and class.
    let adds = of(&plan, ChangeKind::AddComponent);
    assert_eq!(
        adds.len(),
        7,
        "{:?}",
        adds.iter().map(|c| &c.item).collect::<Vec<_>>()
    );
    let outline = sc.board.outline.bbox();
    for a in &adds {
        assert_eq!(a.cause.sheet, "ch[2]");
        assert!(a
            .cause
            .source
            .as_deref()
            .is_some_and(|s| s.starts_with("io.circuit:")));
        let p = a.impact.placement.as_ref().unwrap();
        assert!(p.staged && p.room.as_deref() == Some("ch[2]"));
        assert!(p.at.x > outline.max.x, "staged off the board");
        assert!(
            !a.item.contains('/'),
            "items are designators, not paths: {}",
            a.item
        );
    }
    let room = find(&plan, ChangeKind::AddRoom, "ch[2]");
    assert!(
        room.requires
            .iter()
            .all(|r| r.starts_with("add_component:/ch[2]/"))
            && room.requires.len() == 7
    );
    find(&plan, ChangeKind::AddComponentClass, "ch[2]");
    // Its local nets come with it; adds need them.
    let net = find(&plan, ChangeKind::AddNet, "INF (ch[2])");
    assert_eq!(net.cause.sheet, "ch[2]");
    let u3 = adds
        .iter()
        .find(|c| c.id == "add_component:/ch[2]/U")
        .unwrap();
    assert!(u3.requires.contains(&"add_net:/ch[2]/INF".to_owned()));

    // C900 moved to a 0805 10 µF part: footprint, value and parameters.
    let fp = find(&plan, ChangeKind::ChangeFootprint, "C900");
    assert_eq!(
        (fp.from.as_deref(), fp.to.as_deref()),
        (
            Some("C_0402_1005Metric_IPCN"),
            Some("C_0805_2012Metric_IPCN")
        )
    );
    assert!(fp.impact.footprints == ["C900"]);
    let value = find(&plan, ChangeKind::ChangeValue, "C900");
    assert_eq!(value.to.as_deref(), Some("10 \u{b5}F \u{b1}20%"));
    let param = find(&plan, ChangeKind::ChangeParameter, "C900 voltage rating");
    assert_eq!(
        (param.from.as_deref(), param.to.as_deref()),
        (Some("16 V"), Some("25 V"))
    );

    // R900 is gone; its pads' copper is the impact.
    let rm = find(&plan, ChangeKind::RemoveComponent, "R900");
    assert_eq!(rm.from.as_deref(), Some("R_0402_1005Metric_IPCN"));

    // J100 → J1 through the lock alias: a designator change, never remove plus add.
    let j = find(&plan, ChangeKind::ChangeDesignator, "J1");
    assert_eq!(j.from.as_deref(), Some("J100"));
    assert!(of(&plan, ChangeKind::RemoveComponent)
        .iter()
        .all(|c| c.item != "J100"));
    // Applying it moves the footprint's link and sync record to the new path.
    let mut board = sc.board.clone();
    eco::apply(
        &mut board,
        &plan,
        &[j.id.clone()].into(),
        &sc.pool,
        &ApplyOptions::default(),
    )
    .unwrap();
    let f = board.footprint_by_ref("J1").unwrap();
    assert_eq!(f.component.as_deref(), Some("/J1"));
    assert!(
        board
            .sync
            .components
            .get("/J1")
            .is_some_and(|r| r.designator == "J1")
            && !board.sync.components.contains_key("/J100")
    );
    assert!(adds.iter().all(|c| c.item != "J1"));

    // -5V → VNEG keeps its routing; nothing removes it.
    let rn = find(&plan, ChangeKind::RenameNet, "VNEG");
    assert_eq!(rn.from.as_deref(), Some("-5V"));
    assert!(
        rn.impact.tracks.len() + rn.impact.zones.len() > 0
            || rn.impact.nets.contains(&"VNEG".to_owned())
    );
    assert!(of(&plan, ChangeKind::RemoveNet)
        .iter()
        .all(|c| c.item != "-5V"));
    // The new channel's -5V pins join VNEG through the rename.
    let cd2 = designator(&sc, "/ch[2]/CD2");
    let pin = adds.iter().find(|c| c.item == cd2).unwrap();
    assert!(
        pin.requires.contains(&"rename_net:-5V".to_owned()),
        "{:?}",
        pin.requires
    );

    // J1.9 leaves ground for the new SHIELD net.
    let shield = find(&plan, ChangeKind::ChangePinNet, "J1.9");
    assert_eq!(
        (shield.from.as_deref(), shield.to.as_deref()),
        (Some("GND"), Some("SHIELD"))
    );
    assert_eq!(shield.requires, ["add_net:SHIELD"]);
    assert!(
        shield.impact.pads == ["J100.9"],
        "impact names the board's pad: {:?}",
        shield.impact.pads
    );

    // Classes, pairs and rules.
    find(&plan, ChangeKind::AddNetClass, "analog");
    let rule = find(&plan, ChangeKind::ChangeRule, "power track width");
    assert_eq!(
        (rule.from.as_deref(), rule.to.as_deref()),
        (None, Some("0.40 mm"))
    );
    assert!(rule.impact.nets.contains(&"GND".to_owned()));
    let dp = find(&plan, ChangeKind::AddDiffPair, "AIN01");
    assert_eq!(dp.to.as_deref(), Some("AIN0 / AIN1"));

    // The lite variant leaves channel 0's CD2 off.
    let cd2_0 = designator(&sc, "/ch[0]/CD2");
    let fit = find(&plan, ChangeKind::ChangeFitted, &cd2_0);
    assert_eq!(fit.to.as_deref(), Some("not fitted"));
    assert!(fit.cause.reason.contains("lite"));

    // Every line has a stable, unique id and a cause.
    let ids = plan.ids();
    assert_eq!(ids.len(), plan.changes.len());
    assert!(plan
        .changes
        .iter()
        .all(|c| !c.cause.reason.is_empty() && !c.cause.sheet.is_empty()));
    // Group order is the execution order.
    let kinds: Vec<_> = plan.changes.iter().map(|c| c.kind.group()).collect();
    let mut sorted = kinds.clone();
    sorted.sort();
    assert_eq!(kinds, sorted);
}

#[test]
fn board_edits_are_flagged_and_double_edits_conflict() {
    let sc = eco_scenario();
    let plan = forward(&sc);
    // RS of channel 1 was renumbered R40 on the board: reverting it warns.
    let rs = find(
        &plan,
        ChangeKind::ChangeDesignator,
        &designator(&sc, "/ch[1]/RS"),
    );
    assert_eq!(rs.from.as_deref(), Some("R40"));
    assert!(rs
        .issues
        .iter()
        .any(|(s, r)| *s == Status::Warning && r.contains("back-annotate")));
    // RG of channel 0: R50 on the board, R60 in the schematic.
    let rg = find(&plan, ChangeKind::ChangeDesignator, "R60");
    assert!(rg
        .conflict
        .as_deref()
        .is_some_and(|c| c.contains("both sides")));
    let checks = eco::validate(&plan, &sc.board, &plan.ids());
    assert_eq!(checks[&rg.id].status, Status::Error);
    // The gate swap on channel 1's op-amp shows as pin changes that would undo it.
    let u = designator(&sc, "/ch[1]/U");
    let swapped: Vec<_> = of(&plan, ChangeKind::ChangePinNet)
        .into_iter()
        .filter(|c| c.item.starts_with(&format!("{u}.")))
        .collect();
    assert!(!swapped.is_empty());
    assert!(swapped.iter().all(|c| c
        .issues
        .iter()
        .any(|(_, r)| r.contains("swapped on the board"))));
}

#[test]
fn validation_reports_missing_requirements_footprints_and_collisions() {
    let sc = eco_scenario();
    let plan = forward(&sc);
    let shield = find(&plan, ChangeKind::ChangePinNet, "J1.9").id.clone();
    let alone: BTreeSet<String> = [shield.clone()].into();
    let c = eco::validate(&plan, &sc.board, &alone);
    assert_eq!(c[&shield].status, Status::Error);
    assert!(c[&shield]
        .reason()
        .unwrap()
        .contains("needs add net SHIELD (not selected)"));
    // With requirements closed over, it validates.
    let closed = plan.with_requirements(&alone);
    assert!(closed.contains("add_net:SHIELD"));
    assert_ne!(
        eco::validate(&plan, &sc.board, &closed)[&shield].status,
        Status::Error
    );

    // A footprint missing from the library.
    let mut pool = sc.pool.clone();
    let c0805 = pool
        .packages
        .values()
        .find(|p| p.name.starts_with("C_0805"))
        .unwrap()
        .uuid;
    pool.packages.remove(&c0805);
    let s = Schematic {
        pool: &pool,
        ..schematic(&sc)
    };
    let p2 = eco::compare(&s, &sc.board, &ComparatorOptions::default());
    let fp = find(&p2, ChangeKind::ChangeFootprint, "C900");
    let v = eco::validate(&p2, &sc.board, &[fp.id.clone()].into());
    assert!(
        v[&fp.id].status == Status::Error
            && v[&fp.id].reason().unwrap().contains("not in the library")
    );

    // Renaming onto a designator the board still uses.
    let j = find(&plan, ChangeKind::ChangeDesignator, "J1");
    let mut board = sc.board.clone();
    board
        .footprints
        .values_mut()
        .find(|f| f.reference == "R900")
        .unwrap()
        .reference = "J1".into();
    let v = eco::validate(&plan, &board, &[j.id.clone()].into());
    assert!(v[&j.id]
        .reason()
        .unwrap()
        .contains("J1 is already used on the board"));

    // Apply refuses the whole selection when anything fails: nothing changes.
    let mut b = sc.board.clone();
    let err = eco::apply(
        &mut b,
        &plan,
        &plan.ids(),
        &sc.pool,
        &ApplyOptions::default(),
    )
    .unwrap_err();
    assert!(err
        .failures
        .iter()
        .any(|(id, _)| *id == find(&plan, ChangeKind::ChangeDesignator, "R60").id));
    assert_eq!(b, sc.board);
}

#[test]
fn pad_mismatch_is_an_error() {
    let sc = eco_scenario();
    // Put C900 on the 2x5 header's part: its pads 1, 2 exist, so give it a part whose pads don't.
    let mut design = sc.design.clone();
    let main = design.sheets.get_mut("main").unwrap();
    main.components
        .iter_mut()
        .find(|c| c.name == "C900")
        .unwrap()
        .part = PartRef::Part("FPGA-BGA256".into());
    let nl = elaborate(
        &design,
        &sc.pool,
        &ElabOptions {
            variant: Some("lite".into()),
        },
        Some(&sc.lock),
    );
    let s = Schematic {
        design: &design,
        netlist: &nl,
        ..schematic(&sc)
    };
    let plan = eco::compare(&s, &sc.board, &ComparatorOptions::default());
    let fp = find(&plan, ChangeKind::ChangeFootprint, "C900");
    assert!(
        fp.issues
            .iter()
            .any(|(s, r)| *s == Status::Error && r.contains("pads don't match")),
        "{:?}",
        fp.issues
    );
}

#[test]
fn partial_apply_keeps_the_rest_pending_and_preserves_routing() {
    let sc = eco_scenario();
    let plan = forward(&sc);
    // The third channel only, with what it needs.
    let wanted: BTreeSet<String> = plan
        .changes
        .iter()
        .filter(|c| {
            c.cause.sheet == "ch[2]"
                && matches!(
                    c.kind,
                    ChangeKind::AddComponent | ChangeKind::AddRoom | ChangeKind::AddComponentClass
                )
        })
        .map(|c| c.id.clone())
        .collect();
    let selected = plan.with_requirements(&wanted);
    assert!(
        selected.contains("rename_net:-5V"),
        "the new channel's VNEG pins need the rename"
    );
    let before = sc.board.clone();
    let mut board = sc.board.clone();
    let mut rules = BoardRules::new(&board.limits, &[]).0;
    let (report, checks) = eco::apply_and_check(
        &mut board,
        &plan,
        &selected,
        &sc.pool,
        &ApplyOptions::default(),
        &mut rules,
        &[],
    )
    .unwrap();
    assert_eq!(report.applied.len(), selected.len());
    let row = runmat_circuit::index::EcoApplied::forward(&report, &checks);
    assert_eq!(row.applied as usize, selected.len());
    assert!(checks.drc_errors + checks.drc_warnings < 10_000);
    // Untouched placement and routing are exactly as they were.
    for (id, t) in &before.tracks {
        assert_eq!(board.tracks[id].geom, t.geom);
        assert_eq!(board.tracks[id].layer, t.layer);
    }
    for (id, f) in &before.footprints {
        assert_eq!(
            (board.footprints[id].at, board.footprints[id].rotation),
            (f.at, f.rotation)
        );
    }
    assert_eq!(board.footprints.len(), before.footprints.len() + 7);
    assert!(board
        .rooms
        .iter()
        .any(|r| r.name == "ch[2]" && r.members.len() == 7));
    assert!(board.net("VNEG").is_some() && board.net("-5V").is_none());

    // The rest is still pending, under the same ids.
    let after = eco::compare(&schematic(&sc), &board, &ComparatorOptions::default());
    let left = after.ids();
    for id in &selected {
        assert!(!left.contains(id), "{id} was applied");
    }
    for id in plan.ids().difference(&selected) {
        if !id.starts_with("change_pin_net:/ch[2]") {
            assert!(left.contains(id), "{id} is still pending");
        }
    }
    // One undo step restores the board.
    eco::undo(&mut board, report);
    assert_eq!(board, before);
}

#[test]
fn orphaned_copper_is_kept_or_removed_by_choice() {
    let sc = eco_scenario();
    let plan = forward(&sc);
    let rm = find(&plan, ChangeKind::RemoveComponent, "R900");
    let sel: BTreeSet<String> = [rm.id.clone()].into();
    // Give R900's pad a stub so it has copper to orphan.
    let mut board = sc.board.clone();
    let f = board
        .footprints
        .values()
        .find(|f| f.reference == "R900")
        .unwrap()
        .clone();
    let pad = f.pads(board.copper_count()).into_iter().next().unwrap();
    let stub = board.add_track(
        pad.net,
        0,
        runmat_circuit::geom::mm(0.2),
        pad.center,
        pad.center + runmat_circuit::geom::Point::mm(0.0, 3.0),
    );
    let plan = eco::compare(&schematic(&sc), &board, &ComparatorOptions::default());
    let rm = find(&plan, ChangeKind::RemoveComponent, "R900");
    assert!(rm.impact.tracks.contains(&stub.0));
    assert!(rm
        .issues
        .iter()
        .any(|(s, r)| *s == Status::Warning && r.contains("lose their pad")));
    let mut keep = board.clone();
    eco::apply(&mut keep, &plan, &sel, &sc.pool, &ApplyOptions::default()).unwrap();
    assert!(keep.tracks.contains_key(&stub));
    let mut clean = board.clone();
    let r = eco::apply(
        &mut clean,
        &plan,
        &sel,
        &sc.pool,
        &ApplyOptions {
            remove_orphaned_copper: true,
        },
    )
    .unwrap();
    assert!(!clean.tracks.contains_key(&stub) && r.removed_tracks.contains(&stub.0));
}

#[test]
fn removed_nets_renumber_cleanly() {
    let sc = eco_scenario();
    let mut board = sc.board.clone();
    let orphan = board.add_net("SPARE", None);
    let t = board.add_track(
        Some(orphan),
        0,
        runmat_circuit::geom::mm(0.2),
        runmat_circuit::geom::Point::mm(1.0, 1.0),
        runmat_circuit::geom::Point::mm(1.0, 4.0),
    );
    let plan = eco::compare(&schematic(&sc), &board, &ComparatorOptions::default());
    let rm = find(&plan, ChangeKind::RemoveNet, "SPARE");
    assert_eq!(rm.impact.tracks, [t.0]);
    let gnd_before = board.net("GND").unwrap();
    let vneg = board.net("-5V").unwrap();
    assert!(vneg.0 < orphan.0 && gnd_before.0 < orphan.0);
    let r = eco::apply(
        &mut board,
        &plan,
        &[rm.id.clone()].into(),
        &sc.pool,
        &ApplyOptions::default(),
    )
    .unwrap();
    assert!(board.net("SPARE").is_none());
    assert_eq!(r.unnetted, [t.0]);
    assert_eq!(board.tracks[&t].net, None);
    // Every pad still names a real net.
    for f in board.footprints.values() {
        for n in f.pad_nets.values() {
            assert!((n.0 as usize) < board.nets.len());
        }
    }
}

#[test]
fn comparator_options_ignore_what_they_say() {
    let sc = eco_scenario();
    let o = ComparatorOptions {
        ignore_parameters: true,
        ignore_rooms: true,
        ignore_component_classes: true,
        ignore_footprints: true,
        ..ComparatorOptions::default()
    };
    let plan = eco::compare(&schematic(&sc), &sc.board, &o);
    for k in [
        ChangeKind::ChangeParameter,
        ChangeKind::AddRoom,
        ChangeKind::AddComponentClass,
        ChangeKind::ChangeFootprint,
    ] {
        assert!(of(&plan, k).is_empty(), "{k:?} ignored");
    }
    assert!(!of(&plan, ChangeKind::AddComponent).is_empty());
    let named = ComparatorOptions {
        ignored_parameters: ["voltage_rating".to_owned()].into(),
        ..ComparatorOptions::default()
    };
    let plan = eco::compare(&schematic(&sc), &sc.board, &named);
    assert!(of(&plan, ChangeKind::ChangeParameter)
        .iter()
        .all(|c| !c.item.ends_with("voltage rating")));
    // Unlinked footprints marked board-only stay.
    let mut board = sc.board.clone();
    let id = board
        .footprints
        .values()
        .find(|f| f.reference == "R900")
        .unwrap()
        .id;
    board
        .footprints
        .get_mut(&id)
        .unwrap()
        .properties
        .insert("board_only".into(), "true".into());
    let plan = eco::compare(&schematic(&sc), &board, &ComparatorOptions::default());
    assert!(of(&plan, ChangeKind::RemoveComponent).is_empty());
}

#[test]
fn back_annotation_takes_board_edits_into_the_schematic() {
    let mut sc = eco_scenario();
    let plan = eco::back_annotate(&schematic(&sc), &sc.board, &ComparatorOptions::default());
    assert_eq!(plan.direction, Direction::Back);
    assert!(plan.changes.iter().all(|c| c.origin == Origin::Board));
    let rs = find(&plan, ChangeKind::ChangeDesignator, "R40");
    assert_eq!(
        rs.from.as_deref(),
        Some(designator(&sc, "/ch[1]/RS").as_str())
    );
    let u = designator(&sc, "/ch[1]/U");
    let gates = find(
        &plan,
        ChangeKind::SwapGates,
        &format!("{u} gates B \u{2194} C"),
    );
    assert!(gates
        .issues
        .iter()
        .any(|(_, r)| r.contains("every channel")));
    assert!(
        of(&plan, ChangeKind::SwapPins).is_empty(),
        "a whole-gate swap is one line"
    );
    // A rename made in the schematic (J100 → J1) is not the board's to send back.
    assert!(plan.changes.iter().all(|c| c.item != "J100"));
    // The double renumber is a conflict here too.
    assert!(find(&plan, ChangeKind::ChangeDesignator, "R50")
        .conflict
        .is_some());

    // Apply the renumber: the lock takes R40, and the next forward compare is clean for RS.
    let sel: BTreeSet<String> = [rs.id.clone()].into();
    let mut board = sc.board.clone();
    let applied = eco::apply_back(&mut sc.design, &mut sc.lock, &mut board, &plan, &sel).unwrap();
    assert_eq!(applied, [rs.id.clone()]);
    sc.board = board;
    sc.netlist = elaborate(
        &sc.design,
        &sc.pool,
        &ElabOptions {
            variant: Some("lite".into()),
        },
        Some(&sc.lock),
    );
    sc.lock
        .reconcile(&sc.netlist.lock_items(), &mut DerivedUuids { namespace: 7 });
    assert_eq!(designator(&sc, "/ch[1]/RS"), "R40");
    let fwd = forward(&sc);
    assert!(fwd
        .changes
        .iter()
        .all(|c| c.kind != ChangeKind::ChangeDesignator || c.item != "R40"));

    // The gate swap rewrites the channel sheet.
    let gates = gates.clone();
    let mut board = sc.board.clone();
    eco::apply_back(
        &mut sc.design,
        &mut sc.lock,
        &mut board,
        &plan,
        &[gates.id.clone()].into(),
    )
    .unwrap();
    let ch = &sc.design.sheets["channel"];
    assert!(ch
        .connections
        .iter()
        .any(|c| c.ends.contains(&Endpoint::net("TIE_B"))
            && c.ends.contains(&Endpoint::pin("U", "C.OUT"))));
}

#[test]
fn back_annotated_footprint_picks_a_part_on_that_package() {
    let sc = eco_scenario();
    let mut board = sc.board.clone();
    let pkg = sc
        .pool
        .packages
        .values()
        .find(|p| p.name.starts_with("C_0805"))
        .unwrap()
        .clone();
    let f = board
        .footprints
        .values_mut()
        .find(|f| f.reference == "C900")
        .unwrap();
    // The schematic is on 0805 already in the edited design; use the synced one (0402).
    f.package = pkg;
    let s = Schematic {
        design: &sc.synced,
        netlist: &sc.synced_netlist,
        ..schematic(&sc)
    };
    board.sync.components.get_mut("/C900").unwrap().package = "C_0402_1005Metric_IPCN".into();
    let plan = eco::back_annotate(&s, &board, &ComparatorOptions::default());
    let fp = find(&plan, ChangeKind::ChangeFootprint, "C900");
    match &fp.op {
        Op::SchFootprint { part, package, .. } => {
            assert_eq!(package, "C_0805_2012Metric_IPCN");
            assert_eq!(part.as_deref(), Some("C0805-10U"));
        }
        other => panic!("{other:?}"),
    }
    assert!(fp
        .issues
        .iter()
        .any(|(s, r)| *s == Status::Warning && r.contains("C0805-10U")));
}

#[test]
fn agent_change_sets_are_reviewed_the_same_way() {
    let sc = eco_scenario();
    let mut decl = ComponentDecl::new("C950", PartRef::Part("C0402-100N".into()));
    decl.params.insert("note".into(), Param::text("agent"));
    let edits = vec![ModelEdit::AddComponent {
        sheet: "main".into(),
        decl,
        connect: vec![
            ("1".into(), Endpoint::net("+5V")),
            ("2".into(), Endpoint::net("GND")),
        ],
    }];
    let (_, plan) = eco::agent_change_set(
        &sc.design,
        &edits,
        "Add bulk decoupling",
        &sc.pool,
        &sc.lock,
        &sc.board,
        &ElabOptions {
            variant: Some("lite".into()),
        },
        &ComparatorOptions::default(),
    )
    .unwrap();
    // The agent's part is auto-annotated (`C950?fix0`); its line is found by identity.
    let add = plan
        .changes
        .iter()
        .find(|c| c.kind == ChangeKind::AddComponent && c.id.contains("C950"))
        .unwrap();
    assert_eq!(
        add.origin,
        Origin::Agent {
            title: "Add bulk decoupling".into()
        }
    );
    assert!(plan
        .changes
        .iter()
        .all(|c| matches!(c.origin, Origin::Agent { .. })));
    let idx = eco_index(&plan, &sc.board);
    let row = idx
        .groups
        .iter()
        .flat_map(|g| &g.rows)
        .find(|r| r.id == add.id)
        .unwrap();
    assert_eq!(
        (row.cause.origin.as_str(), row.cause.agent.as_deref()),
        ("agent", Some("Add bulk decoupling"))
    );
}

#[test]
fn deterministic_and_grouped_projection() {
    let a = eco_scenario();
    let b = eco_scenario();
    let (pa, pb) = (forward(&a), forward(&b));
    assert_eq!(pa, pb);
    let ia = eco_index(&pa, &a.board);
    let ib = eco_index(&pb, &b.board);
    let wrap = |eco| CircuitIndex {
        format_version: runmat_circuit::index::CIRCUIT_INDEX_FORMAT_VERSION,
        design: "io".into(),
        variant: None,
        variants: vec![],
        hierarchy: vec![],
        components: vec![],
        nets: vec![],
        net_classes: vec![],
        bom: vec![],
        eco,
        back_annotation: Default::default(),
        last_eco: None,
        erc: vec![],
        review: vec![],
        board: None,
    };
    assert_eq!(to_json(&wrap(ia.clone())), to_json(&wrap(ib)));
    assert_eq!(ia.total as usize, pa.changes.len());
    assert_eq!(ia.groups.iter().map(|g| g.count).sum::<u32>(), ia.total);
    let names: Vec<&str> = ia.groups.iter().map(|g| g.group.as_str()).collect();
    assert_eq!(
        names,
        [
            "components",
            "footprints",
            "parameters",
            "nets",
            "connections",
            "classes",
            "rooms",
            "rules"
        ]
    );
    assert!(ia.errors >= 1, "the conflict is an error");
    let row = ia
        .groups
        .iter()
        .flat_map(|g| &g.rows)
        .find(|r| r.item == "J1.9")
        .unwrap();
    assert!(row.impact.regions.contains(&"pad:J100.9".to_owned()));
    // The report lists every line by group.
    let checks = eco::validate(&pa, &a.board, &pa.ids());
    let text = eco::report(&pa, &checks, "io");
    assert!(text.contains("Pin connections (") && text.contains("[error] Change designator R60"));
}
