//! Exploded views: ordered translate/rotate/radial steps over stable occurrence paths, base
//! views, explode lines with jogs, validation diagnostics, the projection, and drawing views
//! that show an exploded state.

use runmat_cad::assembly::explode::*;
use runmat_cad::assembly::resolve::*;
use runmat_cad::assembly::source::*;
use runmat_cad::assembly::*;
use runmat_cad::diagnostics::{DiagCode, Repair, Severity};
use runmat_cad::ids::{FilePath, PublishName};
use runmat_cad::model::datum::Frame;
use runmat_cad::refs::lock::{LockSection, SeededMinter};
use runmat_cad::refs::InstanceKey;
use runmat_cad::units::StubUnits;
use runmat_cad::value::{NoDynamicValues, ValueExpr};

fn mm(v: f64) -> ValueExpr {
    ValueExpr::mm(v)
}

fn at(x: f64, y: f64, z: f64) -> Placement {
    Placement::Offset {
        offset: OffsetExpr::translation([mm(x), mm(y), mm(z)]),
    }
}

fn close(a: [f64; 3], b: [f64; 3]) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() < 1e-9)
}

fn add_part(
    src: &mut MemorySource,
    file: &str,
    bbox: Aabb,
    publishes: Vec<(&str, PublishedGeometry)>,
) {
    let mut s = DefinitionSummary::new(DefinitionRef::part(file, None));
    s.part_number = Some(file.to_uppercase());
    src.add_part(
        s,
        ConfiguredSummary {
            bbox: Some(bbox),
            publishes: publishes
                .into_iter()
                .map(|(k, v)| (PublishName::new(k), v))
                .collect(),
            ..Default::default()
        },
    );
}

fn holes() -> PublishedGeometry {
    PublishedGeometry::Pattern {
        frames: [("h1", 0.02, 0.02), ("h2", 0.18, 0.02), ("h3", 0.18, 0.18)]
            .iter()
            .map(|(k, x, y)| {
                (
                    InstanceKey::Row { key: k.to_string() },
                    Frame {
                        origin: [*x, *y, 0.11],
                        ..Frame::WORLD
                    },
                )
            })
            .collect(),
    }
}

fn bolt(key: &str) -> RelPath {
    if key == "h1" {
        RelPath::new(&["bolt"])
    } else {
        RelPath(vec![PathSegment::instance(
            "bolt",
            "bolts",
            InstanceKey::Row { key: key.into() },
        )])
    }
}

fn translate(id: &str, occ: Vec<RelPath>, v: [f64; 3], d: f64) -> ExplodeStep {
    ExplodeStep {
        id: id.into(),
        occurrences: occ,
        motion: ExplodeMotion::Translate {
            direction: ExplodeDirection::Vector { vector: v },
            distance: mm(d),
            spacing: None,
        },
    }
}

/// A gearbox: housing, cover, three cover bolts (a seed and two hole-pattern instances), a
/// spare (suppressed) and a shaft sub-assembly (gear, bearing).
fn gearbox() -> (MemorySource, AssemblyDocument) {
    let mut src = MemorySource::default();
    add_part(
        &mut src,
        "housing.part",
        Aabb::new([0.0; 3], [0.2, 0.2, 0.1]),
        vec![("holes", holes())],
    );
    add_part(
        &mut src,
        "cover.part",
        Aabb::new([0.0; 3], [0.2, 0.2, 0.01]),
        vec![],
    );
    add_part(
        &mut src,
        "bolt.part",
        Aabb::new([-0.003, -0.003, 0.0], [0.003, 0.003, 0.02]),
        vec![],
    );
    add_part(
        &mut src,
        "gear.part",
        Aabb::new([-0.02, -0.02, 0.0], [0.02, 0.02, 0.01]),
        vec![],
    );
    add_part(
        &mut src,
        "bearing.part",
        Aabb::new([-0.01, -0.01, 0.0], [0.01, 0.01, 0.005]),
        vec![],
    );
    let mut shaft = AssemblyDocument::new("shaft.assembly", "shaft");
    let mut gear = Occurrence::new("gear", DefinitionRef::part("gear.part", None));
    gear.placement = at(0.0, 0.0, 20.0);
    shaft.occurrences.push(gear);
    shaft.occurrences.push(Occurrence::new(
        "bearing",
        DefinitionRef::part("bearing.part", None),
    ));
    src.add_assembly(shaft);

    let mut top = AssemblyDocument::new("gearbox.assembly", "gearbox");
    top.occurrences.push(Occurrence::new(
        "housing",
        DefinitionRef::part("housing.part", None),
    ));
    let mut cover = Occurrence::new("cover", DefinitionRef::part("cover.part", None));
    cover.placement = at(0.0, 0.0, 100.0);
    top.occurrences.push(cover);
    let mut b = Occurrence::new("bolt", DefinitionRef::part("bolt.part", None));
    b.placement = at(20.0, 20.0, 110.0);
    top.occurrences.push(b);
    let mut spare = Occurrence::new("spare", DefinitionRef::part("bolt.part", None));
    spare.suppressed = true;
    top.occurrences.push(spare);
    let mut s = Occurrence::new("shaft", DefinitionRef::assembly("shaft.assembly"));
    s.placement = at(300.0, 0.0, 0.0);
    top.occurrences.push(s);
    top.patterns.push(OccurrencePattern {
        name: "bolts".into(),
        seeds: vec!["bolt".into()],
        layout: OccurrenceLayout::FollowPattern {
            occurrence: RelPath::new(&["housing"]),
            publish: "holes".into(),
        },
        skip: vec![],
        when: None,
    });

    let mut service = ExplodedView::new("service");
    service.steps.push(translate(
        "cover_up",
        vec![RelPath::new(&["cover"])],
        [0.0, 0.0, 2.0],
        100.0,
    ));
    service.steps.push(ExplodeStep {
        id: "bolts_out".into(),
        occurrences: vec![bolt("h1"), bolt("h2"), bolt("h3")],
        motion: ExplodeMotion::Translate {
            direction: ExplodeDirection::OccurrenceAxis {
                occurrence: bolt("h1"),
                axis: LocalAxis::Z,
                reversed: false,
            },
            distance: mm(150.0),
            spacing: Some(mm(10.0)),
        },
    });
    service.steps.push(ExplodeStep {
        id: "shaft_out".into(),
        occurrences: vec![RelPath::new(&["shaft"])],
        motion: ExplodeMotion::Radial {
            center: RadialCenter::Axis {
                point: [mm(0.0), mm(0.0), mm(0.0)],
                direction: [0.0, 0.0, 1.0],
            },
            distance: mm(50.0),
        },
    });
    service.steps.push(ExplodeStep {
        id: "gear_turn".into(),
        occurrences: vec![RelPath::new(&["shaft", "gear"])],
        motion: ExplodeMotion::Rotate {
            axis: ExplodeAxis::OccurrenceAxis {
                occurrence: RelPath::new(&["shaft", "gear"]),
                axis: LocalAxis::Z,
                offset: None,
            },
            angle: ValueExpr::deg(90.0),
        },
    });
    service.lines.push(ExplodeLine {
        id: "cover_trail".into(),
        occurrence: RelPath::new(&["cover"]),
        anchor: None,
        jogs: vec![Jog {
            segment: 0,
            at: 0.5,
            offset: [mm(10.0), mm(0.0), mm(0.0)],
        }],
    });
    top.exploded.push(service);
    let mut instructions = ExplodedView::new("instructions");
    instructions.base = Some("service".into());
    instructions.steps.push(translate(
        "bearing_out",
        vec![RelPath::new(&["shaft", "bearing"])],
        [0.0, -1.0, 0.0],
        30.0,
    ));
    instructions.lines.push(ExplodeLine {
        id: "bearing_trail".into(),
        occurrence: RelPath::new(&["shaft", "bearing"]),
        anchor: Some([mm(0.0), mm(0.0), mm(0.0)]),
        jogs: vec![],
    });
    top.exploded.push(instructions);
    src.add_assembly(top.clone());
    (src, top)
}

fn resolve_with(src: &MemorySource, lock: Option<&LockSection>) -> ResolvedAssembly {
    let ctx = ResolveContext {
        source: src,
        units: &StubUnits,
        dynamic: &NoDynamicValues,
        lock,
        catalog: None,
        tables: None,
        gauge: None,
    };
    resolve(
        &FilePath::new("gearbox.assembly"),
        &ResolveRequest::default(),
        &ctx,
    )
}

fn explode(
    src: &MemorySource,
    doc: &AssemblyDocument,
    tree: &ResolvedAssembly,
    view: &str,
    lock: Option<&LockSection>,
) -> ExplodedState {
    evaluate_exploded(
        doc,
        tree,
        view,
        &ExplodeContext {
            units: &StubUnits,
            dynamic: &NoDynamicValues,
            lock,
            source: Some(src),
        },
    )
}

#[test]
fn e01_ordered_steps_move_occurrences_and_their_subtrees() {
    let (src, doc) = gearbox();
    let tree = resolve_with(&src, None);
    assert!(
        tree.diagnostics.errors().next().is_none(),
        "{:?}",
        tree.diagnostics
    );
    assert!(validate_exploded(&doc).diagnostics.is_empty());
    let st = explode(&src, &doc, &tree, "service", None);
    assert!(
        st.diagnostics.diagnostics.is_empty(),
        "{:#?}",
        st.diagnostics
    );
    let w = |p: &str| st.world[tree.find_display(p).unwrap() as usize];
    let a = |p: &str| tree.node(tree.find_display(p).unwrap()).world;
    // Translate (the direction vector is normalised).
    assert!(close(w("cover").translation, [0.0, 0.0, 0.2]));
    // Along the seed bolt's own Z, spaced by listing order.
    for (p, d) in [
        ("bolt", 0.15),
        ("bolt@bolts[\"h2\"]", 0.16),
        ("bolt@bolts[\"h3\"]", 0.17),
    ] {
        let (x, y) = (a(p).translation, w(p).translation);
        assert!(close(y, [x[0], x[1], x[2] + d]), "{}: {:?} → {:?}", p, x, y);
    }
    // Radially away from the Z axis, moving the sub-assembly with its members.
    assert!(close(w("shaft").translation, [0.35, 0.0, 0.0]));
    assert!(close(w("shaft/bearing").translation, [0.35, 0.0, 0.0]));
    // Rotated a quarter turn about its own axis, after the radial move.
    let g = w("shaft/gear");
    assert!(close(g.translation, [0.35, 0.0, 0.02]));
    assert!(close(g.apply_vector([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]));
    // Untouched: housing.
    assert!(close(w("housing").translation, a("housing").translation));
    let moved: Vec<String> = st.moved.iter().map(|n| tree.path(*n)).collect();
    assert_eq!(
        moved.len(),
        7,
        "cover, three bolts, shaft and its two members: {:?}",
        moved
    );
    assert!(!moved.contains(&"housing".to_string()));
    // The offset is the motion, not the pose.
    let cover = tree.find_display("cover").unwrap();
    assert!(close(st.offset(&tree, cover).translation, [0.0, 0.0, 0.1]));
}

#[test]
fn e02_explode_lines_trail_each_step_and_jog() {
    let (src, doc) = gearbox();
    let tree = resolve_with(&src, None);
    let st = explode(&src, &doc, &tree, "service", None);
    let line = st.lines.iter().find(|l| l.id == "cover_trail").unwrap();
    assert_eq!(line.occurrence, "cover");
    // The cover's centre (bounding box) from its assembled to its exploded position, jogged
    // 10 mm in X half way.
    let expected = [
        [0.1, 0.1, 0.105],
        [0.1, 0.1, 0.155],
        [0.11, 0.1, 0.155],
        [0.11, 0.1, 0.205],
        [0.1, 0.1, 0.205],
    ];
    assert_eq!(line.points.len(), expected.len(), "{:?}", line.points);
    for (p, e) in line.points.iter().zip(expected) {
        assert!(close(*p, e), "{:?} vs {:?}", p, e);
    }
    // A view built on another runs the base's steps first, then its own; its lines follow
    // every step that moved their occurrence.
    let st = explode(&src, &doc, &tree, "instructions", None);
    assert!(
        st.diagnostics.diagnostics.is_empty(),
        "{:#?}",
        st.diagnostics
    );
    let b = st.world[tree.find_display("shaft/bearing").unwrap() as usize];
    assert!(close(b.translation, [0.35, -0.03, 0.0]));
    let trail = st.lines.iter().find(|l| l.id == "bearing_trail").unwrap();
    assert_eq!(trail.points.len(), 3, "radial step, then its own step");
    assert!(close(trail.points[0], [0.3, 0.0, 0.0]));
    assert!(close(trail.points[1], [0.35, 0.0, 0.0]));
    assert!(close(trail.points[2], [0.35, -0.03, 0.0]));
    assert_eq!(
        st.lines.len(),
        2,
        "the base's lines are part of the derived view"
    );
}

#[test]
fn e03_static_validation_has_exact_diagnostics() {
    let (_, mut doc) = gearbox();
    let mut dup = ExplodedView::new("dup");
    dup.steps.push(translate(
        "a",
        vec![RelPath::new(&["cover"])],
        [0.0, 0.0, 1.0],
        1.0,
    ));
    dup.steps.push(translate(
        "a",
        vec![RelPath::new(&["cover"]), RelPath::new(&["cover"])],
        [0.0, 0.0, 0.0],
        1.0,
    ));
    doc.exploded.push(dup);
    let mut orphan = ExplodedView::new("orphan");
    orphan.base = Some("nope".into());
    doc.exploded.push(orphan);
    let mut l1 = ExplodedView::new("loop1");
    l1.base = Some("loop2".into());
    let mut l2 = ExplodedView::new("loop2");
    l2.base = Some("loop1".into());
    doc.exploded.push(l1);
    doc.exploded.push(l2);
    let mut again = ExplodedView::new("service");
    again
        .steps
        .push(translate("x", vec![], [1.0, 0.0, 0.0], 1.0));
    doc.exploded.push(again);
    let rep = validate_exploded(&doc);
    let msgs: Vec<(DiagCode, String, String)> = rep
        .diagnostics
        .iter()
        .map(|d| (d.code, d.site.key(), d.message.clone()))
        .collect();
    let has = |c: DiagCode, key: &str, m: &str| {
        assert!(
            msgs.iter()
                .any(|(cc, k, mm)| *cc == c && k == key && mm == m),
            "missing {:?} {} {:?} in {:#?}",
            c,
            key,
            m,
            msgs
        );
    };
    has(
        DiagCode::ExplodeDuplicate,
        "exploded.service#name",
        "exploded view `service` is declared twice",
    );
    has(
        DiagCode::ExplodeDuplicate,
        "exploded.dup#steps.a",
        "step `a` appears twice in `dup`",
    );
    has(
        DiagCode::ExplodeDuplicate,
        "exploded.dup#steps.a.occurrences",
        "step `a` names `cover` twice",
    );
    has(
        DiagCode::ExplodeZeroDirection,
        "exploded.dup#steps.a.direction",
        "step `a` has a zero direction",
    );
    has(
        DiagCode::ExplodeUnknownView,
        "exploded.orphan#base",
        "`gearbox.assembly` has no exploded view `nope`",
    );
    has(
        DiagCode::ExplodeCycle,
        "exploded.loop1#base",
        "exploded views build on each other: loop1 → loop2 → loop1",
    );
    has(
        DiagCode::ExplodeCycle,
        "exploded.loop2#base",
        "exploded views build on each other: loop2 → loop1 → loop2",
    );
    has(
        DiagCode::ExplodeUnknownOccurrence,
        "exploded.service#steps.x.occurrences",
        "step `x` moves no occurrence",
    );
    assert!(rep
        .diagnostics
        .iter()
        .all(|d| d.severity == Severity::Error));
    // Evaluating an unknown view is specific.
    let (src, doc) = gearbox();
    let tree = resolve_with(&src, None);
    let st = explode(&src, &doc, &tree, "missing", None);
    assert_eq!(
        st.diagnostics.diagnostics[0].message,
        "`gearbox.assembly` has no exploded view `missing`"
    );
    assert!(st.moved.is_empty());
}

#[test]
fn e04_occurrence_problems_in_the_resolved_configuration() {
    let (src, mut doc) = gearbox();
    let mut v = ExplodedView::new("bad");
    v.steps.push(translate(
        "s1",
        vec![
            RelPath::new(&["nothing"]),
            RelPath::new(&["spare"]),
            RelPath::new(&["shaft", "nut"]),
            RelPath::new(&["shaft"]),
            RelPath::new(&["shaft", "gear"]),
        ],
        [0.0, 0.0, 1.0],
        10.0,
    ));
    v.steps.push(ExplodeStep {
        id: "s2".into(),
        occurrences: vec![RelPath::new(&["housing"])],
        motion: ExplodeMotion::Radial {
            center: RadialCenter::Point {
                point: [mm(100.0), mm(100.0), mm(50.0)],
            },
            distance: mm(10.0),
        },
    });
    v.steps.push(ExplodeStep {
        id: "s3".into(),
        occurrences: vec![RelPath::new(&["cover"])],
        motion: ExplodeMotion::Translate {
            direction: ExplodeDirection::Vector {
                vector: [1.0, 0.0, 0.0],
            },
            distance: ValueExpr::deg(5.0),
            spacing: None,
        },
    });
    v.lines.push(ExplodeLine {
        id: "l".into(),
        occurrence: RelPath::new(&["shaft"]),
        anchor: None,
        jogs: vec![
            Jog {
                segment: 0,
                at: 0.5,
                offset: [mm(0.0), mm(0.0), mm(5.0)],
            },
            Jog {
                segment: 4,
                at: 0.5,
                offset: [mm(1.0), mm(0.0), mm(0.0)],
            },
        ],
    });
    doc.exploded.push(v);
    let tree = resolve_with(&src, None);
    let st = explode(&src, &doc, &tree, "bad", None);
    let d: Vec<(DiagCode, Severity, String)> = st
        .diagnostics
        .diagnostics
        .iter()
        .map(|d| (d.code, d.severity, d.message.clone()))
        .collect();
    let want = [
        (
            DiagCode::ExplodeUnknownOccurrence,
            Severity::Error,
            "exploded view `bad` names `nothing`, which `gearbox.assembly` does not contain",
        ),
        (
            DiagCode::ExplodeAbsent,
            Severity::Warning,
            "`spare` is not in this configuration; the exploded view skips it",
        ),
        (
            DiagCode::ExplodeUnknownOccurrence,
            Severity::Error,
            "exploded view `bad` names `shaft/nut`, which `shaft` does not contain",
        ),
        (
            DiagCode::ExplodeNested,
            Severity::Info,
            "step `s1` moves `shaft/gear` and its assembly `shaft`; `shaft/gear` moves twice",
        ),
        (
            DiagCode::ExplodeZeroDirection,
            Severity::Warning,
            "`housing` sits on the centre of radial step `s2`; it has no direction to move in and stays",
        ),
        (
            DiagCode::ValueFailed,
            Severity::Error,
            "`steps.s3.distance`: unit mismatch: expected L, found angle^1",
        ),
        (
            DiagCode::ExplodeBadJog,
            Severity::Error,
            "line `l` has 1 segment(s); jog 1 is on segment 4",
        ),
        (
            DiagCode::ExplodeBadJog,
            Severity::Error,
            "jog 0 of line `l` must be offset perpendicular to its segment",
        ),
    ];
    for w in want {
        assert!(
            d.iter().any(|x| x.0 == w.0 && x.1 == w.1 && x.2 == w.2),
            "missing {:?} in {:#?}",
            w,
            d
        );
    }
    assert_eq!(d.len(), want.len(), "{:#?}", d);
    // The gear moved with the shaft and again on its own.
    let g = st.world[tree.find_display("shaft/gear").unwrap() as usize];
    assert!(close(g.translation, [0.3, 0.0, 0.04]));
}

#[test]
fn e05_renamed_occurrences_resolve_through_the_lock() {
    let (mut src, mut doc) = gearbox();
    let mut lock = LockSection::default();
    let mut m = SeededMinter { seed: "e".into() };
    lock_assembly(&doc, &mut lock, &mut m);
    let file = FilePath::new("gearbox.assembly");
    lock.rename(
        &item_path(&file, "occurrence", "cover"),
        &item_path(&file, "occurrence", "lid"),
    )
    .unwrap();
    doc.occurrences
        .iter_mut()
        .find(|o| o.name == "cover")
        .unwrap()
        .name = "lid".into();
    src.add_assembly(doc.clone());
    let tree = resolve_with(&src, Some(&lock));
    let st = explode(&src, &doc, &tree, "service", Some(&lock));
    let lid = tree.find_display("lid").unwrap();
    assert!(close(st.world[lid as usize].translation, [0.0, 0.0, 0.2]));
    let alias = st.diagnostics.with_code(DiagCode::AsmViaAlias);
    assert_eq!(
        alias.len(),
        2,
        "the step and the line: {:#?}",
        st.diagnostics
    );
    assert_eq!(
        alias[0].message,
        "`cover` was renamed to `lid`; update the exploded view"
    );
    assert!(matches!(
        &alias[0].repairs[..],
        [Repair::RewritePath { from, to, .. }] if from == "cover" && to == "lid"
    ));
    // Without the lock the old name is unknown.
    let st = explode(&src, &doc, &tree, "service", None);
    assert!(!st
        .diagnostics
        .with_code(DiagCode::ExplodeUnknownOccurrence)
        .is_empty());
}

#[test]
fn e06_serde_and_projection_are_deterministic() {
    let (src, doc) = gearbox();
    let json = serde_json::to_string_pretty(&doc).unwrap();
    let back: AssemblyDocument = serde_json::from_str(&json).unwrap();
    assert_eq!(back, doc);
    assert!(json.contains("\"motion\": \"radial\""));
    // Documents without exploded views do not carry the key.
    let plain = AssemblyDocument::new("x.assembly", "x");
    assert!(!serde_json::to_string(&plain).unwrap().contains("exploded"));
    let tree = resolve_with(&src, None);
    let a = exploded_index_json(&exploded_index(
        &tree,
        &explode(&src, &doc, &tree, "instructions", None),
    ));
    let b = exploded_index_json(&exploded_index(
        &tree,
        &explode(&src, &doc, &tree, "instructions", None),
    ));
    assert_eq!(a, b);
    let idx: ExplodedIndex = serde_json::from_str(&a).unwrap();
    assert_eq!(idx.format_version, EXPLODED_INDEX_FORMAT_VERSION);
    assert_eq!(idx.view, "instructions");
    let paths: Vec<&str> = idx.occurrences.iter().map(|o| o.path.as_str()).collect();
    let mut sorted = paths.clone();
    sorted.sort();
    assert_eq!(paths, sorted);
    assert_eq!(idx.lines[0].id, "bearing_trail");
    // Exploded views get stable lock ids like every other authored item.
    let mut lock = LockSection::default();
    let mut m = SeededMinter { seed: "e".into() };
    lock_assembly(&doc, &mut lock, &mut m);
    assert!(lock
        .lookup(&item_path(&doc.path, "exploded", "service"))
        .is_some());
}
