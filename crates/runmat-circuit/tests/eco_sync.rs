//! The ECO follow-ups on the multi-channel scenario: comparator ignore settings per difference
//! type (ignoring rooms leaves rooms alone), the board's sync record kept in the
//! `.circuit.layout` `sync { }` section, and Copy Room Formats.

use runmat_circuit::eco::{
    self, ApplyOptions, ChangeKind, ComparatorOptions, Difference, EcoPlan, Schematic,
};
use runmat_circuit::fixture::{eco_scenario, EcoScenario};
use runmat_circuit::geom::{mm, Point};
use runmat_circuit::index::eco_index;
use runmat_circuit::layout::{Board, TrackGeom};
use runmat_circuit::room_formats::{copy_room_formats, sibling_rooms, RoomFormatOptions};
use runmat_circuit::syntax::layout::{parse_layout, print_layout, write_sync, SyncSection};
use std::collections::{BTreeMap, BTreeSet};

fn schematic(sc: &EcoScenario) -> Schematic<'_> {
    Schematic {
        design: &sc.design,
        netlist: &sc.netlist,
        pool: &sc.pool,
        lock: &sc.lock,
    }
}

fn ignoring(ds: &[Difference]) -> ComparatorOptions {
    let mut o = ComparatorOptions::default();
    for d in ds {
        o.set_ignore(*d, true);
    }
    o
}

/// The plan's changes with every requirement that is still in the plan.
fn lines(plan: &EcoPlan) -> BTreeMap<String, Vec<String>> {
    plan.changes
        .iter()
        .map(|c| (c.id.clone(), c.requires.clone()))
        .collect()
}

/// What the review shows for a setting: the full compare minus the rows the setting hides,
/// with requirements on hidden rows dropped (the Desktop's filter).
fn filtered(full: &EcoPlan, o: &ComparatorOptions) -> BTreeMap<String, Vec<String>> {
    let kept: BTreeSet<&str> = full
        .changes
        .iter()
        .filter(|c| !o.hides(c.kind))
        .map(|c| c.id.as_str())
        .collect();
    full.changes
        .iter()
        .filter(|c| kept.contains(c.id.as_str()))
        .map(|c| {
            let requires = c
                .requires
                .iter()
                .filter(|r| kept.contains(r.as_str()))
                .cloned()
                .collect();
            (c.id.clone(), requires)
        })
        .collect()
}

#[test]
fn each_ignored_difference_type_produces_no_lines() {
    let sc = eco_scenario();
    let s = schematic(&sc);
    let full = eco::compare(&s, &sc.board, &ComparatorOptions::default());
    let present: BTreeSet<ChangeKind> = full.changes.iter().map(|c| c.kind).collect();
    for d in Difference::ALL {
        let o = ignoring(&[d]);
        let plan = eco::compare(&s, &sc.board, &o);
        let hidden: Vec<ChangeKind> = present
            .iter()
            .copied()
            .filter(|k| k.ignored_by().contains(&d))
            .collect();
        for c in &plan.changes {
            assert!(
                !c.kind.ignored_by().contains(&d),
                "{d:?} ignored, yet {} is listed",
                c.id
            );
        }
        // Each type the scenario exercises really did have lines to hide.
        if matches!(
            d,
            Difference::Rooms
                | Difference::ComponentClasses
                | Difference::Values
                | Difference::Parameters
                | Difference::Footprints
        ) {
            assert!(!hidden.is_empty(), "the scenario has {d:?} lines");
        }
        // The review's filter of the full list is exactly this compare.
        assert_eq!(lines(&plan), filtered(&full, &o), "{d:?}");
        let back = eco::back_annotate(&s, &sc.board, &o);
        assert!(back.changes.iter().all(|c| !o.hides(c.kind)));
    }
    let all = ignoring(&Difference::ALL);
    assert_eq!(
        lines(&eco::compare(&s, &sc.board, &all)),
        filtered(&full, &all)
    );
    // Rules are net-class parameters: ignoring net classes ignores them too.
    assert!(ignoring(&[Difference::NetClasses]).hides(ChangeKind::ChangeRule));
    assert!(!ignoring(&[Difference::Rules]).hides(ChangeKind::AddNetClass));
}

#[test]
fn ignoring_rooms_does_not_touch_rooms() {
    let sc = eco_scenario();
    let mut board = sc.board.clone();
    // A part that sits in a room and that the schematic no longer has.
    let stray = "R900";
    assert!(board.footprint_by_ref(stray).is_some());
    board.rooms[0].members.push(stray.into());
    let rooms_before = board.rooms.clone();
    assert!(
        rooms_before.len() >= 2,
        "the first sync made a room per channel"
    );

    let o = ignoring(&[Difference::Rooms]);
    let plan = eco::compare(&schematic(&sc), &board, &o);
    assert!(plan.changes.iter().all(|c| !matches!(
        c.kind,
        ChangeKind::AddRoom | ChangeKind::ChangeRoom | ChangeKind::RemoveRoom
    )));
    let full = eco::compare(&schematic(&sc), &board, &ComparatorOptions::default());
    assert!(
        full.changes.iter().any(|c| c.kind == ChangeKind::AddRoom),
        "without the setting the new channel gets a room"
    );

    // Everything that validates, the stray's removal and the new channel's parts included.
    let checks = eco::validate(&plan, &board, &plan.ids());
    let ok: BTreeSet<String> = plan
        .changes
        .iter()
        .filter(|c| checks[&c.id].status != eco::Status::Error)
        .map(|c| c.id.clone())
        .collect();
    let mut selected = BTreeSet::new();
    for id in &ok {
        let closed = plan.with_requirements(&[id.clone()].into());
        if closed.is_subset(&ok) {
            selected.extend(closed);
        }
    }
    let removes_stray = plan.changes.iter().any(|c| {
        selected.contains(&c.id) && c.kind == ChangeKind::RemoveComponent && c.item == stray
    });
    assert!(removes_stray, "the stray's removal is applied");
    let adds = plan
        .changes
        .iter()
        .filter(|c| selected.contains(&c.id) && c.kind == ChangeKind::AddComponent)
        .count();
    assert!(adds > 0, "new parts are applied");
    let renames: BTreeMap<String, String> = plan
        .changes
        .iter()
        .filter(|c| selected.contains(&c.id) && c.kind == ChangeKind::ChangeDesignator)
        .filter_map(|c| Some((c.from.clone()?, c.to.clone()?)))
        .collect();
    eco::apply(
        &mut board,
        &plan,
        &selected,
        &sc.pool,
        &ApplyOptions::default(),
    )
    .unwrap();

    assert!(board.footprint_by_ref(stray).is_none(), "the stray is gone");
    assert_eq!(
        board.rooms.len(),
        rooms_before.len(),
        "no room added or removed"
    );
    for (after, before) in board.rooms.iter().zip(&rooms_before) {
        assert_eq!(after.name, before.name);
        assert_eq!(
            after.outline, before.outline,
            "room {} keeps its outline",
            before.name
        );
        // Members are as they were; a member whose part was renumbered keeps its place under
        // the new designator.
        let expected: Vec<String> = before
            .members
            .iter()
            .map(|m| renames.get(m).cloned().unwrap_or_else(|| m.clone()))
            .collect();
        assert_eq!(
            after.members, expected,
            "room {} keeps its members",
            before.name
        );
    }
    assert!(board.rooms[0].members.iter().any(|m| m == stray));
    // The setting is saved with the board, and the next compare reads it.
    assert!(board.sync.options.ignore_rooms);
    let next = eco::compare_saved(&schematic(&sc), &board);
    assert!(next
        .changes
        .iter()
        .all(|c| c.kind.group() != eco::ChangeGroup::Rooms));
    let projection = eco_index(&next, &board);
    assert_eq!(projection.ignored, ["rooms"]);
}

#[test]
fn the_projection_tags_rows_with_what_hides_them() {
    let sc = eco_scenario();
    let full = eco::compare(&schematic(&sc), &sc.board, &ComparatorOptions::default());
    let index = eco_index(&full, &sc.board);
    let rows: Vec<_> = index.groups.iter().flat_map(|g| &g.rows).collect();
    for row in &rows {
        let kind = full.changes.iter().find(|c| c.id == row.id).unwrap().kind;
        let words: Vec<&str> = kind.ignored_by().iter().map(|d| d.as_str()).collect();
        assert_eq!(row.ignored_by, words, "{}", row.id);
    }
    assert!(rows.iter().any(|r| r.ignored_by == ["rooms"]));
    assert!(rows.iter().any(|r| r.ignored_by.is_empty()));
}

#[test]
fn update_board_offers_an_ignore_setting_per_difference_type() {
    let catalog = runmat_circuit::commands::catalog();
    let command = catalog.command("schematic.update-board").unwrap();
    for d in Difference::ALL {
        let field = command
            .inputs
            .iter()
            .find(|f| f.id == d.input_id())
            .unwrap_or_else(|| panic!("no input for {d:?}"));
        assert_eq!(field.label, d.label());
        assert!(field.help.is_some());
    }
    let inputs: BTreeMap<String, bool> = [("ignore_rooms".to_owned(), true)].into();
    let o = ComparatorOptions::default().with_inputs(&inputs);
    assert_eq!(o.ignored(), [Difference::Rooms]);
    assert!(catalog.lint().is_empty(), "{:#?}", catalog.lint());
}

fn unlinked(board: &Board) -> Board {
    let mut b = board.clone();
    b.sync = Default::default();
    for f in b.footprints.values_mut() {
        f.component = None;
    }
    b
}

#[test]
fn the_sync_record_round_trips_through_the_layout_file() {
    let sc = eco_scenario();
    let mut board = sc.board.clone();
    board.sync.options = ignoring(&[Difference::Values, Difference::Footprints]);
    board.sync.options.ignored_parameters = ["datasheet".to_owned()].into();
    let section = SyncSection::of(&board);
    assert!(!section.links.is_empty() && !section.sync.components.is_empty());
    let text = print_layout("io", &section);
    let parsed = parse_layout("boards/io.circuit.layout", &text);
    assert!(parsed.diagnostics.is_empty(), "{:#?}", parsed.diagnostics);
    assert_eq!(parsed.sync.as_ref(), Some(&section));

    // A board read back has the same base and links, so the three-way compare is the same.
    let mut read = unlinked(&board);
    assert!(parsed.sync.unwrap().install(&mut read).is_empty());
    assert_eq!(read, board);
    let s = schematic(&sc);
    assert_eq!(
        eco::compare_saved(&s, &read),
        eco::compare_saved(&s, &board)
    );
    assert_eq!(
        eco::back_annotate_saved(&s, &read),
        eco::back_annotate_saved(&s, &board)
    );
    // Without the section, the record is empty and the compare differs (two-way).
    let bare = unlinked(&board);
    assert_ne!(
        eco::compare(&s, &bare, &board.sync.options).ids(),
        eco::compare_saved(&s, &board).ids()
    );
}

#[test]
fn applying_an_eco_updates_the_section() {
    let sc = eco_scenario();
    let mut board = sc.board.clone();
    let file = write_sync("io.circuit.layout", "", &SyncSection::of(&board)).unwrap();
    let s = schematic(&sc);
    let o = ignoring(&[Difference::Rooms]);
    let plan = eco::compare(&s, &board, &o);
    let pick: BTreeSet<String> = plan
        .changes
        .iter()
        .filter(|c| c.kind == ChangeKind::AddComponent && c.cause.sheet == "ch[2]")
        .map(|c| c.id.clone())
        .collect();
    let selected = plan.with_requirements(&pick);
    eco::apply(
        &mut board,
        &plan,
        &selected,
        &sc.pool,
        &ApplyOptions::default(),
    )
    .unwrap();
    let updated = write_sync("io.circuit.layout", &file, &SyncSection::of(&board)).unwrap();
    let parsed = parse_layout("io.circuit.layout", &updated).sync.unwrap();
    assert_eq!(parsed, SyncSection::of(&board));
    assert!(parsed.sync.options.ignore_rooms);
    let added: Vec<_> = plan
        .changes
        .iter()
        .filter(|c| selected.contains(&c.id) && c.kind == ChangeKind::AddComponent)
        .collect();
    assert!(!added.is_empty());
    for c in added {
        let reference = &c.item;
        assert!(
            parsed.links.contains_key(reference),
            "{reference} is linked"
        );
        assert!(parsed
            .sync
            .components
            .values()
            .any(|r| &r.designator == reference));
    }
    // The applied lines are no longer pending against the board read back from the file.
    let mut read = unlinked(&board);
    parsed.install(&mut read);
    let left = eco::compare_saved(&s, &read).ids();
    for id in &selected {
        assert!(!left.contains(id), "{id} was applied");
    }
}

/// Routing in a room: a track between two pads of a net only the room's parts carry, and a
/// via. Returns the net and one of its pads (component path, pad name).
fn route_in_room(board: &mut Board, room: &str) -> (String, (String, String)) {
    let copper = board.copper_count();
    let r = board.rooms.iter().find(|r| r.name == room).unwrap().clone();
    let prefix = format!("/{room}/");
    let mut pads: BTreeMap<_, Vec<(Point, String, String, bool)>> = BTreeMap::new();
    for f in board.footprints.values() {
        let path = f.component.clone().unwrap_or_default();
        let ours = path.starts_with(&prefix);
        for p in f.pads(copper) {
            if let Some(n) = p.net {
                pads.entry(n)
                    .or_default()
                    .push((p.center, path.clone(), p.name.clone(), ours));
            }
        }
    }
    let inside = |p: Point| r.outline.contains(p) != Some(false);
    let (net, list) = pads
        .into_iter()
        .find(|(_, list)| {
            list.len() >= 2 && list.iter().all(|(at, _, _, ours)| *ours && inside(*at))
        })
        .expect("a net local to the room");
    board.add_track(Some(net), 0, mm(0.2), list[0].0, list[1].0);
    board.add_via(Some(net), list[0].0, mm(0.6), mm(0.3));
    (
        board.net_name(net).to_owned(),
        (list[0].1.clone(), list[0].2.clone()),
    )
}

#[test]
fn copy_room_formats_replicates_placement_and_routing() {
    let sc = eco_scenario();
    let mut board = sc.board.clone();
    assert_eq!(sibling_rooms(&board, "ch[0]"), ["ch[1]"]);
    let (net, (path, pad)) = route_in_room(&mut board, "ch[0]");
    let other = path.replacen("/ch[0]/", "/ch[1]/", 1);
    let target_net_id = board
        .footprints
        .values()
        .find(|f| f.component.as_deref() == Some(&other))
        .and_then(|f| f.pad_nets.get(&pad).copied())
        .expect("the counterpart pad has a net");
    assert_ne!(board.net(&net), Some(target_net_id), "a channel-local net");
    // Stale routing in the target room on that net is replaced.
    let dst = board
        .rooms
        .iter()
        .find(|r| r.name == "ch[1]")
        .unwrap()
        .clone();
    let c = dst.outline.bbox().center();
    let stale = board.add_track(
        Some(target_net_id),
        0,
        mm(0.2),
        c,
        c + Point::new(mm(0.5), 0),
    );
    let before = board.clone();

    let report =
        copy_room_formats(&mut board, "ch[0]", &[], &RoomFormatOptions::default()).unwrap();
    assert_eq!(report.copies.len(), 1);
    let copy = &report.copies[0];
    assert_eq!(copy.room, "ch[1]");
    assert!(!copy.placed.is_empty());
    // The scenario's own routing in the room comes over too, with the track and via added here.
    assert!(!copy.tracks_added.is_empty() && !copy.vias_added.is_empty());
    assert!(copy.tracks_removed.contains(&stale.0));

    // Counterparts sit at the same place relative to their room, by component path.
    let src_room = before.rooms.iter().find(|r| r.name == "ch[0]").unwrap();
    let dst_room = board.rooms.iter().find(|r| r.name == "ch[1]").unwrap();
    let d = dst_room.outline.bbox().min - src_room.outline.bbox().min;
    assert_eq!(
        dst_room.outline.bbox().max - dst_room.outline.bbox().min,
        src_room.outline.bbox().max - src_room.outline.bbox().min,
        "the target room takes the source room's shape"
    );
    let mut matched = 0;
    for f in before.footprints.values() {
        let Some(rest) = f
            .component
            .as_deref()
            .and_then(|p| p.strip_prefix("/ch[0]/"))
        else {
            continue;
        };
        let other = format!("/ch[1]/{rest}");
        let Some(t) = board
            .footprints
            .values()
            .find(|g| g.component.as_deref() == Some(&other))
        else {
            continue;
        };
        if t.locked {
            continue;
        }
        assert_eq!(t.at, f.at + d, "{} follows {}", t.reference, f.reference);
        assert_eq!((t.rotation, t.side), (f.rotation, f.side));
        matched += 1;
    }
    assert!(matched > 3);
    // The track added here is copied onto the other channel's net, moved by the same offset.
    let ours = before
        .tracks
        .values()
        .filter(|t| t.net == before.net(&net))
        .max_by_key(|t| t.id)
        .unwrap();
    let TrackGeom::Segment { a, b } = ours.geom else {
        panic!("a segment")
    };
    let copied = copy
        .tracks_added
        .iter()
        .map(|id| &board.tracks[&runmat_circuit::layout::ItemId(*id)])
        .find(|t| t.geom == TrackGeom::Segment { a: a + d, b: b + d })
        .expect("the track is copied with the room's offset");
    assert_eq!(copied.net, Some(target_net_id));
    assert_eq!((copied.layer, copied.width), (ours.layer, ours.width));
    assert!(copy
        .vias_added
        .iter()
        .map(|id| &board.vias[&runmat_circuit::layout::ItemId(*id)])
        .any(|v| v.at == a + d && v.net == Some(target_net_id)));
    // Only the target room's copper is replaced; the rest, the source room included, is as it was.
    let removed: BTreeSet<u64> = copy.tracks_removed.iter().copied().collect();
    for (id, t) in &before.tracks {
        if removed.contains(&id.0) {
            let (p, q) = t.endpoints();
            assert!(
                dst_room.outline.contains(p) != Some(false)
                    && dst_room.outline.contains(q) != Some(false)
            );
        } else {
            assert_eq!(board.tracks.get(id), Some(t));
        }
    }
    assert_eq!(*report.undo, before);
}

#[test]
fn copy_room_formats_respects_locks_and_refuses_bad_rooms() {
    let sc = eco_scenario();
    let mut board = sc.board.clone();
    let locked = board
        .footprints
        .values()
        .find(|f| {
            f.component
                .as_deref()
                .is_some_and(|p| p.starts_with("/ch[1]/"))
        })
        .unwrap()
        .id;
    board.footprints.get_mut(&locked).unwrap().locked = true;
    let at = board.footprints[&locked].at;
    let o = RoomFormatOptions {
        routing: false,
        room_shape: false,
        ..RoomFormatOptions::default()
    };
    let report = copy_room_formats(&mut board, "ch[0]", &["ch[1]".to_owned()], &o).unwrap();
    assert_eq!(board.footprints[&locked].at, at, "a locked part stays");
    let reference = board.footprints[&locked].reference.clone();
    assert!(report.copies[0]
        .skipped
        .iter()
        .any(|s| s == &format!("{reference}: locked")));
    assert!(report.copies[0].tracks_added.is_empty());

    let before = board.clone();
    for (source, targets) in [
        ("ch[9]", vec![]),
        ("ch[0]", vec!["ch[0]".to_owned()]),
        ("ch[0]", vec!["nowhere".to_owned()]),
    ] {
        assert!(copy_room_formats(&mut board, source, &targets, &o).is_err());
        assert_eq!(board, before, "a refused copy changes nothing");
    }
}

#[test]
fn copy_room_formats_is_a_board_command_on_rooms() {
    let catalog = runmat_circuit::commands::catalog();
    let command = catalog.command("board.copy-room-formats").unwrap();
    let ids: Vec<&str> = command.inputs.iter().map(|f| f.id.as_str()).collect();
    assert_eq!(
        ids,
        [
            "source",
            "targets",
            "placement",
            "routing",
            "pours",
            "room_shape"
        ]
    );
    let menu = catalog
        .context_menu(
            runmat_design_commands::SelectKind::Room,
            runmat_design_commands::EditorKind::CircuitLayout,
        )
        .expect("rooms have a context menu");
    assert_eq!(menu.quick[0], "board.copy-room-formats");
}

/// A rectangle `inset` inside the room's bounding box.
fn inside_room(board: &Board, room: &str, inset: f64) -> runmat_circuit::geom::poly::Polygon {
    let b = board
        .rooms
        .iter()
        .find(|r| r.name == room)
        .unwrap()
        .outline
        .bbox();
    let (x0, y0, x1, y1) = (
        b.min.x + mm(inset),
        b.min.y + mm(inset),
        b.max.x - mm(inset),
        b.max.y - mm(inset),
    );
    runmat_circuit::geom::poly::Polygon {
        outer: vec![
            Point::new(x0, y0),
            Point::new(x1, y0),
            Point::new(x1, y1),
            Point::new(x0, y1),
        ],
        holes: Vec::new(),
    }
}

#[test]
fn copy_room_formats_copies_pours_on_the_mapped_nets() {
    use runmat_circuit::layout::Zone;
    let sc = eco_scenario();
    let mut board = sc.board.clone();
    let (net, (path, pad)) = route_in_room(&mut board, "ch[0]");
    let other = path.replacen("/ch[0]/", "/ch[1]/", 1);
    let target_net = board
        .footprints
        .values()
        .find(|f| f.component.as_deref() == Some(&other))
        .and_then(|f| f.pad_nets.get(&pad).copied())
        .unwrap();
    let source_net = board.net(&net);
    // A pour on the channel's own net and a keep-clear pour with no net in ch[0]; in ch[1] a
    // stale unlocked pour on the counterpart net and a locked one.
    let add = |board: &mut Board, room: &str, net, inset: f64, locked: bool| {
        let id = board.alloc();
        let mut z = Zone::new(id, "", net, vec![0], inside_room(board, room, inset));
        z.locked = locked;
        board.zones.insert(id, z);
        id
    };
    add(&mut board, "ch[0]", source_net, 0.5, false);
    add(&mut board, "ch[0]", None, 1.0, false);
    let stale = add(&mut board, "ch[1]", Some(target_net), 0.7, false);
    let kept = add(&mut board, "ch[1]", Some(target_net), 0.9, true);
    let o = RoomFormatOptions {
        placement: false,
        routing: false,
        room_shape: false,
        pours: true,
    };
    let report = copy_room_formats(&mut board, "ch[0]", &["ch[1]".to_owned()], &o).unwrap();
    let copy = &report.copies[0];
    assert_eq!(copy.pours_added.len(), 2, "{copy:?}");
    assert!(copy.pours_removed.contains(&stale.0));
    assert!(!board.zones.contains_key(&stale));
    assert!(board.zones.contains_key(&kept), "a locked pour stays");
    let src = board
        .rooms
        .iter()
        .find(|r| r.name == "ch[0]")
        .unwrap()
        .outline
        .bbox();
    let dst = board
        .rooms
        .iter()
        .find(|r| r.name == "ch[1]")
        .unwrap()
        .outline
        .bbox();
    let d = dst.min - src.min;
    let added: Vec<&Zone> = copy
        .pours_added
        .iter()
        .map(|id| &board.zones[&runmat_circuit::layout::ItemId(*id)])
        .collect();
    assert!(added.iter().any(|z| z.net == Some(target_net)
        && z.outline.outer[0] == inside_room(&board, "ch[0]", 0.5).outer[0] + d));
    assert!(added.iter().any(|z| z.net.is_none()));

    // Without the option the pours stay.
    let mut again = sc.board.clone();
    add(&mut again, "ch[0]", None, 1.0, false);
    let zones = again.zones.len();
    let o = RoomFormatOptions {
        pours: false,
        ..RoomFormatOptions::default()
    };
    copy_room_formats(&mut again, "ch[0]", &["ch[1]".to_owned()], &o).unwrap();
    assert_eq!(again.zones.len(), zones);
}
