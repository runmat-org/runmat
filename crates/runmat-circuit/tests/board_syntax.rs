//! The `.circuit.layout` board front end: round trips, diagnostics, span maps, minimal-diff
//! structured edits (the catalog, ECO apply and Copy Room Formats through them) and scale.

use runmat_circuit::board_rules::{RuleSpec, RuleValue};
use runmat_circuit::eco::{self, ApplyOptions, Schematic};
use runmat_circuit::fixture::{eco_scenario, large, LargeParams};
use runmat_circuit::geom::poly::Polygon;
use runmat_circuit::geom::{mm, Point};
use runmat_circuit::layout::*;
use runmat_circuit::library::{Body, Drill, Graphic, Model3d, PadDef, PadKind, PadShape};
use runmat_circuit::room_formats::RoomFormatOptions;
use runmat_circuit::syntax::board::{canonical, parse_board, print_board, same_board};
use runmat_circuit::syntax::board_edit::{
    apply_board_edit, apply_eco, board_command, copy_room_formats, edit_board, BoardEdit,
};
use std::time::Instant;

const PATH: &str = "boards/ctl.circuit.layout";

fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<Point> {
    vec![
        Point::mm(x0, y0),
        Point::mm(x1, y0),
        Point::mm(x1, y1),
        Point::mm(x0, y1),
    ]
}

/// The ECO scenario's synced board with every other statement kind added.
fn rich_board() -> Board {
    let mut b = eco_scenario().board;
    b.name = "ctl".into();
    b.stackup.finish = "ENIG".into();
    let mut flex = Stackup::flex(2, 0.5, mm(0.05));
    flex.name = "flex 2".into();
    flex.layers[0].material.name = "Custom PI".into();
    flex.layers[0].material.dk = 3.37;
    b.stackups.insert(flex.name.clone(), flex);
    b.regions.push(StackRegion {
        name: "tail".into(),
        outline: Polygon {
            outer: rect(0.0, -20.0, 10.0, 0.0),
            holes: vec![rect(4.0, -12.0, 6.0, -10.0)],
        },
        stackup: "flex 2".into(),
        bends: vec![Bend {
            a: Point::mm(0.0, -10.0),
            b: Point::mm(10.0, -10.0),
            radius: mm(1.0),
            angle: 90.0,
            keepout: mm(0.5),
        }],
    });
    b.outline.holes.push(rect(40.0, 30.0, 41.25, 31.0000005));
    b.limits.min_clearance = mm(0.1);
    b.net_class_rules.insert(
        "power".into(),
        vec![
            ("track_width".into(), RuleValue::Len(mm(0.5))),
            ("clearance".into(), RuleValue::Len(mm(0.2))),
        ],
    );
    b.rules.push(
        RuleSpec::single("hv", ":net-class(hv)")
            .set("clearance", RuleValue::Len(mm(0.8)))
            .set("spoke_min", RuleValue::Int(3))
            .set("impedance", RuleValue::Num(50.0))
            .set("zone_connect", RuleValue::Text("solid".into()))
            .set(
                "via_types",
                RuleValue::ViaKinds(vec![ViaKind::Through, ViaKind::Micro]),
            )
            .priority(2),
    );
    b.rules.push(
        RuleSpec::pair("hv_lv", ":net-class(hv)", "net.GND")
            .set("clearance", RuleValue::Len(mm(1.5)))
            .set("hole_min", RuleValue::Bool(true)),
    );
    for name in ["+3V3", "true", "in", "a b"] {
        b.add_net(name, Some("hv class"));
    }
    let n33 = b.net("+3V3");
    let id = b.alloc();
    b.fiducials.insert(
        id,
        Fiducial {
            id,
            name: "FID1".into(),
            at: Point::mm(3.0, 3.0),
            side: Side::Bottom,
            diameter: mm(1.0),
            opening: mm(2.0),
            locked: true,
        },
    );
    let id = b.alloc();
    b.tracks.insert(
        id,
        Track {
            id,
            net: n33,
            layer: 1,
            width: mm(0.2),
            geom: TrackGeom::Arc {
                start: Point::mm(0.0, 0.0),
                mid: Point::mm(1.0, 1.0),
                end: Point::mm(2.0, 0.0),
            },
            locked: true,
        },
    );
    b.add_track(
        None,
        3,
        mm(0.15),
        Point::mm(-1.5, 2.0),
        Point::mm(1.0, -2.0),
    );
    let v = b.add_via(n33, Point::mm(5.0, 5.0), mm(0.45), mm(0.2));
    {
        let v = b.vias.get_mut(&v).unwrap();
        v.from = 0;
        v.to = 1;
        v.micro = true;
        v.tented = false;
        v.locked = true;
    }
    let id = b.alloc();
    let mut z = Zone::new(
        id,
        "",
        None,
        vec![0, 3],
        Polygon {
            outer: rect(0.0, 0.0, 5.0, 5.0),
            holes: vec![rect(1.0, 1.0, 2.0, 2.0)],
        },
    );
    z.priority = -1;
    z.clearance = Some(mm(0.4));
    z.connect = PadConnect::None;
    z.islands = IslandPolicy::RemoveBelow(0.5e12);
    z.locked = true;
    b.zones.insert(id, z);
    let id = b.alloc();
    let mut z = Zone::new(
        id,
        "odd area",
        n33,
        vec![1],
        Polygon {
            outer: rect(0.0, 0.0, 1.0, 1.0),
            holes: vec![],
        },
    );
    z.islands = IslandPolicy::RemoveBelow(1234.5);
    z.connect = PadConnect::ThermalThtOnly;
    b.zones.insert(id, z);
    for (name, layers) in [("antenna", vec![0u8]), ("", vec![])] {
        let id = b.alloc();
        b.keepouts.insert(
            id,
            Keepout {
                id,
                name: name.into(),
                outline: Polygon {
                    outer: rect(80.0, 0.0, 90.0, 10.0),
                    holes: vec![],
                },
                layers,
                rules: KeepoutRules {
                    tracks: true,
                    vias: true,
                    pads: false,
                    pours: true,
                    footprints: name.is_empty(),
                    max_height: Some(mm(3.0)),
                },
            },
        );
    }
    let id = b.alloc();
    b.texts.insert(
        id,
        BoardText {
            id,
            layer: Layer::Silk(Side::Top),
            at: Point::mm(90.0, 5.0),
            text: "REV \"A\"\nline 2".into(),
            size: mm(1.0),
            thickness: mm(0.15),
            angle: 45.5,
            mirrored: true,
        },
    );
    let graphics = vec![
        (
            Layer::Silk(Side::Bottom),
            Graphic::Line {
                points: rect(0.0, 0.0, 1.0, 1.0),
                width: mm(0.12),
            },
        ),
        (
            Layer::Fab(Side::Top),
            Graphic::Rect {
                a: Point::mm(0.0, 0.0),
                b: Point::mm(1.0, 1.0),
                width: mm(0.1),
                fill: true,
            },
        ),
        (
            Layer::Courtyard(Side::Bottom),
            Graphic::Circle {
                center: Point::mm(2.0, 2.0),
                radius: mm(0.5),
                width: 0,
                fill: false,
            },
        ),
        (
            Layer::Edge,
            Graphic::Arc {
                start: Point::mm(0.0, 0.0),
                mid: Point::mm(1.0, 1.0),
                end: Point::mm(2.0, 0.0),
                width: mm(0.05),
            },
        ),
        (
            Layer::User(3),
            Graphic::Polygon {
                points: rect(0.0, 0.0, 3.0, 1.0),
                width: mm(0.1),
                fill: true,
            },
        ),
        (
            Layer::Mask(Side::Top),
            Graphic::Text {
                at: Point::mm(1.0, 2.0),
                text: "logo".into(),
                size: mm(2.0),
                angle: 90.0,
            },
        ),
        (
            Layer::Paste(Side::Bottom),
            Graphic::Line {
                points: vec![Point::mm(0.0, 0.0), Point::mm(0.0, 1.0)],
                width: mm(0.2),
            },
        ),
        (
            Layer::Copper(2),
            Graphic::Line {
                points: vec![Point::mm(0.0, 0.0), Point::mm(0.0, 1.0)],
                width: mm(0.2),
            },
        ),
    ];
    for (layer, graphic) in graphics {
        let id = b.alloc();
        b.graphics.insert(id, BoardGraphic { id, layer, graphic });
    }
    let id = b.alloc();
    b.dimensions.insert(
        id,
        Dimension {
            id,
            layer: Layer::User(1),
            a: Point::mm(0.0, 0.0),
            b: Point::mm(100.0, 0.0),
            offset: mm(5.0),
            precision: 3,
        },
    );
    // A footprint with every pad shape and package field.
    let first = b.footprints.values().next().unwrap().package.clone();
    let mut p = first;
    p.name = "Odd pkg".into();
    p.description = "every pad shape".into();
    p.smd = false;
    let shapes = [
        PadShape::Circle { d: mm(1.0) },
        PadShape::Rect {
            w: mm(1.0),
            h: mm(2.0),
        },
        PadShape::RoundRect {
            w: mm(1.0),
            h: mm(2.0),
            ratio: 0.25,
        },
        PadShape::Oval {
            w: mm(1.0),
            h: mm(2.0),
        },
        PadShape::Trapezoid {
            w: mm(1.0),
            h: mm(2.0),
            delta: mm(0.1),
        },
        PadShape::Chamfered {
            w: mm(1.0),
            h: mm(2.0),
            chamfer: mm(0.2),
            corners: 5,
        },
        PadShape::Custom {
            points: rect(0.0, 0.0, 1.0, 1.0),
            holes: vec![rect(0.2, 0.2, 0.4, 0.4)],
        },
    ];
    p.pads = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| PadDef::smd(&format!("{}", i + 1), Point::mm(i as f64, 0.0), s.clone()))
        .collect();
    let mut tht = PadDef::tht(
        "A1",
        Point::mm(0.0, 5.0),
        PadShape::Oval {
            w: mm(2.0),
            h: mm(1.5),
        },
        mm(0.8),
    );
    tht.drill = Some(Drill {
        diameter: mm(0.8),
        slot: Some(mm(1.2)),
        plated: true,
    });
    tht.rotation = 30.0;
    tht.thermal_pad = true;
    tht.mask_margin = Some(mm(0.05));
    p.pads.push(tht);
    p.pads
        .push(PadDef::npth("MH", Point::mm(3.0, 5.0), mm(3.2)));
    let mut edge = PadDef::smd(
        "E1",
        Point::mm(5.0, 5.0),
        PadShape::Rect {
            w: mm(1.0),
            h: mm(3.0),
        },
    );
    edge.kind = PadKind::Connector;
    edge.paste = false;
    edge.copper = CopperSpanAlias::BACK;
    edge.paste_margin = Some(mm(-0.02));
    edge.mask = false;
    p.pads.push(edge);
    let mut no_drill = PadDef::tht("X", Point::ORIGIN, PadShape::Circle { d: mm(1.0) }, mm(0.5));
    no_drill.drill = None;
    p.pads.push(no_drill);
    p.silk = vec![Graphic::Text {
        at: Point::ORIGIN,
        text: "%R".into(),
        size: mm(1.0),
        angle: 0.0,
    }];
    p.fab = vec![Graphic::Circle {
        center: Point::ORIGIN,
        radius: mm(1.0),
        width: mm(0.1),
        fill: false,
    }];
    p.courtyard = vec![rect(-1.0, -1.0, 8.0, 6.0), rect(9.0, 0.0, 10.0, 1.0)];
    p.model = Some(Model3d {
        path: "models/odd.step".into(),
        offset: (0.1, 0.0, -0.25),
        rotation: (0.0, 0.0, 90.0),
        scale: (1.0, 1.0, 1.0),
        origin: "center".into(),
    });
    p.body = Some(Body {
        width: mm(8.0),
        length: mm(6.0),
        height: mm(1.2),
    });
    let gnd = b.net("GND").unwrap_or(NetId(0));
    let id = b.place("U99", p.clone(), Point::mm(50.0, 50.0), 270.0, Side::Bottom);
    {
        let f = b.footprints.get_mut(&id).unwrap();
        f.value = "10k 1%".into();
        f.locked = true;
        f.fitted = false;
        f.pad_nets.insert("1".into(), gnd);
        f.properties.insert("Stock code".into(), "A-1".into());
        f.properties.insert("mpn".into(), "true".into());
    }
    // Same package name, different content: the second gets its own key.
    let mut q = p;
    q.description = "another".into();
    b.place("U100", q, Point::mm(60.0, 50.0), 0.0, Side::Top);
    b.place(
        "U99",
        b.footprints[&id].package.clone(),
        Point::mm(70.0, 50.0),
        0.0,
        Side::Top,
    );
    b
}

/// `CopperSpan` without importing its module path twice.
struct CopperSpanAlias;
impl CopperSpanAlias {
    const BACK: runmat_circuit::library::CopperSpan = runmat_circuit::library::CopperSpan::Back;
}

fn round_trip(board: &Board) -> String {
    let text = print_board(board, &[]);
    let parsed = parse_board(PATH, &text);
    assert!(
        parsed.diagnostics.is_empty(),
        "{:#?}\n{}",
        &parsed.diagnostics[..parsed.diagnostics.len().min(5)],
        text.lines().take(60).collect::<Vec<_>>().join("\n")
    );
    let back = parsed.board.unwrap();
    let (a, b) = (canonical(board), canonical(&back));
    if a != b {
        // Narrow the failure down to the first different collection.
        assert_eq!(a.stackup, b.stackup);
        assert_eq!(a.stackups, b.stackups);
        assert_eq!(a.regions, b.regions);
        assert_eq!(a.outline, b.outline);
        assert_eq!(a.limits, b.limits);
        assert_eq!(a.rules, b.rules);
        assert_eq!(a.net_class_rules, b.net_class_rules);
        assert_eq!(a.nets, b.nets);
        assert_eq!(a.rooms, b.rooms);
        for (x, y) in a.footprints.values().zip(b.footprints.values()) {
            assert_eq!(x.package, y.package, "{}", x.reference);
            assert_eq!(x, y);
        }
        assert_eq!(a.footprints, b.footprints);
        assert_eq!(a.fiducials, b.fiducials);
        assert_eq!(a.tracks, b.tracks);
        assert_eq!(a.vias, b.vias);
        assert_eq!(a.zones, b.zones);
        assert_eq!(a.keepouts, b.keepouts);
        assert_eq!(a.texts, b.texts);
        assert_eq!(a.graphics, b.graphics);
        assert_eq!(a.dimensions, b.dimensions);
        assert_eq!(a.sync, b.sync);
        assert_eq!(a, b);
    }
    // Printing is a fixed point.
    assert_eq!(print_board(&back, &[]), text);
    text
}

#[test]
fn every_statement_kind_round_trips() {
    let board = rich_board();
    let text = round_trip(&board);
    for word in [
        "stackup rigid",
        "stackup \"flex 2\"",
        "region tail",
        "outline =",
        "cutout =",
        "limits =",
        "net_class power",
        "rule hv",
        "rule hv_lv",
        "net \"+3V3\" : \"hv class\"",
        "net \"true\"",
        "room ",
        "footprint U99",
        "fiducial FID1",
        "track ",
        "arc ",
        "via ",
        "pour ",
        "keepout antenna",
        "keepout =",
        "text \"REV",
        "graphic line",
        "graphic text",
        "dimension =",
        "package \"Odd pkg\"",
        "package \"Odd pkg#2\"",
        "sync {",
    ] {
        assert!(text.contains(word), "no `{word}` in\n{text}");
    }
    // Footprint statements never carry the schematic link; the sync section does.
    let fp_lines: Vec<&str> = text
        .lines()
        .filter(|l| l.starts_with("    footprint "))
        .collect();
    assert!(!fp_lines.is_empty());
    let links: Vec<String> = board
        .footprints
        .values()
        .filter_map(|f| f.component.as_ref().map(|c| format!("= \"{c}\"")))
        .collect();
    assert!(!links.is_empty());
    assert!(
        fp_lines
            .iter()
            .all(|l| !l.contains("component") && links.iter().all(|k| !l.contains(k.as_str()))),
        "{fp_lines:#?}"
    );
    assert!(
        text.contains("        footprint "),
        "the sync section holds the links"
    );
    // One statement per line outside the stackup and package blocks.
    let tracks = text
        .lines()
        .filter(|l| l.trim_start().starts_with("track "))
        .count();
    assert_eq!(
        tracks,
        board
            .tracks
            .values()
            .filter(|t| matches!(t.geom, TrackGeom::Segment { .. }))
            .count()
    );
}

#[test]
fn fixture_and_imported_boards_round_trip() {
    let (_, _, _, mut b) = large(&LargeParams {
        channels: 6,
        copper_layers: 6,
        route_every: 1,
        pours: true,
    });
    b.name = "ctl".into();
    round_trip(&b);
    let k = runmat_circuit::kicad::pcb::read(include_str!("kicad/board.kicad_pcb")).unwrap();
    let mut b = k.board;
    b.name = "ctl".into();
    round_trip(&b);
}

#[test]
fn sites_and_spans() {
    let board = rich_board();
    let text = print_board(&board, &[]);
    let parsed = parse_board(PATH, &text);
    let b = parsed.board.as_ref().unwrap();
    for (id, site) in &parsed.sites {
        let span = parsed.spans[site];
        let stmt = &text[span.start..span.end];
        if let Some(f) = b.footprints.get(id) {
            assert!(site.starts_with("footprint."));
            assert!(stmt.starts_with("footprint "), "{stmt}");
            assert!(stmt.contains(&format!("{:?}", f.reference).trim_matches('"').to_string()));
        }
        if b.tracks.contains_key(id) {
            assert!(
                site.starts_with("track[") || site.starts_with("arc["),
                "{site}"
            );
        }
    }
    // Duplicate designators get an ordinal; statements are in source order.
    assert!(parsed.spans.contains_key("footprint.U99~2"));
    assert!(parsed
        .statements
        .windows(2)
        .all(|w| w[0].span.start < w[1].span.start));
    assert!(parsed.spans.contains_key("package.Odd pkg#2"));
    assert!(parsed.spans.contains_key("sync"));
}

#[test]
fn written_forms_and_diagnostics() {
    let file = |body: &str| format!("layout ctl {{\n{body}\n}}\n");
    // Units, comments and hand layout read as the same board.
    let a = parse_board(
        PATH,
        &file("    net GND\n    // a note\n    track GND = { layer = F.Cu, width = 10 mil, from = [0, 0], to = [1 in, 2.5 mm] }"),
    );
    assert!(a.diagnostics.is_empty(), "{:#?}", a.diagnostics);
    let t = a.board.unwrap().tracks.into_values().next().unwrap();
    assert_eq!(t.width, 254_000);
    assert_eq!(
        t.geom,
        TrackGeom::Segment {
            a: Point::ORIGIN,
            b: Point::new(25_400_000, 2_500_000)
        }
    );
    let cases = [
        ("    track VCC = { layer = F.Cu, width = 1, from = [0, 0], to = [1, 1] }", "not a net"),
        ("    net A\n    track A = { layer = In9.Cu, width = 1, from = [0, 0], to = [1, 1] }", "not a copper layer"),
        ("    net A\n    track A = { layer = F.Cu, width = 1, from = [0, 0] }", "needs `to`"),
        ("    net A\n    track A = { layer = F.Cu, width = 1, from = [0, 0], to = [1, 1], colour = red }", "has no `colour`"),
        ("    net A\n    track A = { layer = F.Cu, width = 3 V, from = [0, 0], to = [1, 1] }", "not a length unit"),
        ("    footprint R1 = { package = nope, at = [0, 0] }", "not defined"),
        ("    net A\n    net A", "given twice"),
        ("    wire A = 1", "does not start a board statement"),
        ("    sync {\n        version = 1\n        footprint R9 = \"/R9\"\n    }", "does not have"),
        ("    sync {\n        version = 4\n    }", "newer RunMat"),
        ("    pour = { layers = [F.Cu], outline = [[0, 0]], islands = maybe }", "islands are"),
        ("    rule r = { clearance = 1 mm }", "select"),
    ];
    for (body, message) in cases {
        let p = parse_board(PATH, &file(body));
        assert!(p.board.is_none(), "{body}");
        assert!(
            p.diagnostics.iter().any(|d| d.message.contains(message)),
            "{body}: {:#?}",
            p.diagnostics
        );
    }
    let named = parse_board("other.circuit.layout", &file(""));
    assert!(named.diagnostics[0].message.contains("its file is `other`"));
    // `use` lines are kept and printed.
    let used = parse_board(
        PATH,
        &format!(
            "runmat 0.7\nuse circuit ctl from \"./ctl.circuit\"\n{}",
            file("")
        ),
    );
    assert!(used.diagnostics.is_empty(), "{:#?}", used.diagnostics);
    assert_eq!(used.uses[0].symbol, Some(("circuit".into(), "ctl".into())));
    let printed = print_board(used.board.as_ref().unwrap(), &used.uses);
    assert!(printed.contains("use circuit ctl from \"./ctl.circuit\"\n"));
}

fn changed_lines(before: &str, after: &str) -> (usize, usize) {
    // Lines only in one side, counted with multiplicity.
    let mut count: std::collections::HashMap<&str, i64> = std::collections::HashMap::new();
    for l in before.lines() {
        *count.entry(l).or_default() += 1;
    }
    for l in after.lines() {
        *count.entry(l).or_default() -= 1;
    }
    let removed = count
        .values()
        .filter(|v| **v > 0)
        .map(|v| *v as usize)
        .sum();
    let added = count
        .values()
        .filter(|v| **v < 0)
        .map(|v| (-*v) as usize)
        .sum();
    (removed, added)
}

#[test]
fn edits_are_minimal_and_keep_comments() {
    let board = rich_board();
    let mut text = print_board(&board, &[]);
    // Hand edits the printer would not make: comments, blank lines and a reflowed record.
    text = text.replacen("    footprint ", "    // Placement.\n\n    footprint ", 1);
    let parsed = parse_board(PATH, &text);
    let site = parsed
        .sites
        .values()
        .find(|s| s.starts_with("footprint.U99"))
        .unwrap()
        .clone();
    // Move: one line changes.
    let (change, ()) = edit_board(PATH, &text, |b| {
        let id = *parsed.sites.iter().find(|(_, s)| **s == site).unwrap().0;
        let f = b.footprints.get_mut(&id).unwrap();
        f.at = Point::mm(51.0, 50.0);
        Ok(())
    })
    .unwrap();
    assert!(!change.reprinted);
    assert_eq!(change.edits.len(), 1, "{:#?}", change.edits);
    assert_eq!(changed_lines(&text, &change.source), (1, 1));
    assert!(change.source.contains("// Placement.\n\n"));
    // Add a track: one line in, after the last track.
    let (change, ()) = edit_board(PATH, &text, |b| {
        b.add_track(None, 0, mm(0.3), Point::mm(1.0, 1.0), Point::mm(2.0, 1.0));
        Ok(())
    })
    .unwrap();
    assert!(!change.reprinted);
    assert_eq!(changed_lines(&text, &change.source), (0, 1));
    // Delete a footprint: its line (and the comment above, when it is the first) goes.
    let first_fp = parsed
        .statements
        .iter()
        .find(|s| s.site.starts_with("footprint."))
        .unwrap()
        .site
        .clone();
    let change = board_command(
        PATH,
        &text,
        &BoardEdit::Delete {
            items: vec![first_fp],
        },
    )
    .unwrap();
    assert!(!change.reprinted);
    // The comment is set off by a blank line, so it stays.
    assert!(change.source.contains("// Placement.\n\n"));
    let reread = parse_board(PATH, &change.source).board.unwrap();
    assert_eq!(reread.footprints.len(), board.footprints.len() - 1);
    // A no-op edit changes nothing.
    let (change, ()) = edit_board(PATH, &text, |_| Ok(())).unwrap();
    assert!(change.edits.is_empty());
    assert_eq!(change.source, text);
}

#[test]
fn catalog_edits_run_through_the_minimal_diff() {
    let board = rich_board();
    let text = print_board(&board, &[]);
    let gnd = board
        .nets
        .iter()
        .find(|n| n.name == "GND")
        .map(|n| n.name.clone());
    let edits = vec![
        BoardEdit::Move {
            items: vec!["track[0]".into(), "via[0]".into()],
            dx: mm(1.0),
            dy: 0,
            rotation: 0.0,
        },
        BoardEdit::Flip {
            items: vec!["U100".into()],
        },
        BoardEdit::Lock {
            items: vec!["track[1]".into()],
            locked: true,
        },
        BoardEdit::Room {
            name: "new".into(),
            outline: Polygon {
                outer: rect(0.0, 0.0, 1.0, 1.0),
                holes: vec![],
            },
            members: vec!["U100".into()],
        },
        BoardEdit::Route {
            net: gnd.clone(),
            layer: "F.Cu".into(),
            width: mm(0.2),
            points: vec![
                Point::mm(0.0, 0.0),
                Point::mm(1.0, 0.0),
                Point::mm(1.0, 1.0),
            ],
        },
        BoardEdit::Via {
            net: gnd.clone(),
            at: Point::mm(1.0, 1.0),
            diameter: mm(0.6),
            drill: mm(0.3),
            span: Some(("F.Cu".into(), "In1.Cu".into())),
            micro: false,
        },
        BoardEdit::Pour {
            net: gnd.clone(),
            layers: vec!["B.Cu".into()],
            outline: Polygon {
                outer: rect(0.0, 0.0, 9.0, 9.0),
                holes: vec![],
            },
            clearance: Some(mm(0.3)),
            connect: PadConnect::Solid,
        },
        BoardEdit::Keepout {
            name: "mount".into(),
            outline: Polygon {
                outer: rect(0.0, 0.0, 2.0, 2.0),
                holes: vec![],
            },
            layers: vec![],
            rules: KeepoutRules {
                vias: true,
                ..Default::default()
            },
        },
        BoardEdit::Dimension {
            layer: Layer::User(2),
            a: Point::ORIGIN,
            b: Point::mm(10.0, 0.0),
            offset: mm(2.0),
        },
        BoardEdit::Rule(
            RuleSpec::single("hv", ":net-class(hv)").set("clearance", RuleValue::Len(mm(1.0))),
        ),
        BoardEdit::Outline {
            outline: Polygon {
                outer: rect(0.0, 0.0, 120.0, 90.0),
                holes: vec![],
            },
        },
        BoardEdit::TestPoint {
            net: "GND".into(),
            at: Point::mm(4.0, 4.0),
            side: Side::Bottom,
            diameter: mm(1.0),
        },
    ];
    let catalog = runmat_circuit::commands::catalog();
    for edit in &edits {
        assert!(
            catalog.command(edit.command()).is_some(),
            "{}",
            edit.command()
        );
        let change = board_command(PATH, &text, edit).unwrap_or_else(|e| panic!("{edit:?}: {e}"));
        assert!(!change.reprinted, "{edit:?}");
        let (removed, added) = changed_lines(&text, &change.source);
        assert!(removed <= 3 && added <= 12, "{edit:?}: -{removed} +{added}");
        // The source reads back as the edited board.
        let mut expect = parse_board(PATH, &text).board.unwrap();
        let sites = parse_board(PATH, &text).sites;
        apply_board_edit(&mut expect, &sites, edit).unwrap();
        assert!(
            same_board(&parse_board(PATH, &change.source).board.unwrap(), &expect),
            "{edit:?}"
        );
    }
    // The stackup edit renumbers copper and reprints only what changes.
    let change = board_command(
        PATH,
        &text,
        &BoardEdit::LayerStack {
            layers: 6,
            thickness: mm(1.6),
        },
    )
    .unwrap();
    assert!(!change.reprinted);
    assert!(
        parse_board(PATH, &change.source)
            .board
            .unwrap()
            .copper_count()
            == 6
    );
    assert!(board_command(
        PATH,
        &text,
        &BoardEdit::Delete {
            items: vec!["nothing".into()]
        }
    )
    .is_err());
}

#[test]
fn eco_apply_and_copy_room_formats_are_minimal_text_changes() {
    let sc = eco_scenario();
    let mut board = sc.board.clone();
    board.name = "ctl".into();
    let text = print_board(&board, &[]);
    let s = Schematic {
        design: &sc.design,
        netlist: &sc.netlist,
        pool: &sc.pool,
        lock: &sc.lock,
    };
    let source_board = parse_board(PATH, &text).board.unwrap();
    let plan = eco::compare_saved(&s, &source_board);
    let checks = eco::validate(&plan, &source_board, &plan.ids());
    let ok: std::collections::BTreeSet<String> = plan
        .changes
        .iter()
        .filter(|c| checks[&c.id].status != eco::Status::Error)
        .map(|c| c.id.clone())
        .collect();
    let mut selected = std::collections::BTreeSet::new();
    for id in &ok {
        let closed = plan.with_requirements(&[id.clone()].into());
        if closed.is_subset(&ok) {
            selected.extend(closed);
        }
    }
    assert!(selected.len() > 10);
    let (change, _) = apply_eco(
        PATH,
        &text,
        &plan,
        &selected,
        &sc.pool,
        &ApplyOptions::default(),
    )
    .unwrap();
    assert!(!change.reprinted);
    let mut expect = source_board.clone();
    eco::apply(
        &mut expect,
        &plan,
        &selected,
        &sc.pool,
        &ApplyOptions::default(),
    )
    .unwrap();
    // A rename and a new part on the renamed net leave one net of that name.
    let names: std::collections::BTreeSet<&str> =
        expect.nets.iter().map(|n| n.name.as_str()).collect();
    assert_eq!(names.len(), expect.nets.len());
    let got = parse_board(PATH, &change.source).board.unwrap();
    assert!(same_board(&got, &expect));
    // Exactly the lines whose printed form changed, and none else.
    assert_eq!(
        changed_lines(&text, &change.source),
        changed_lines(&text, &print_board(&expect, &[]))
    );
    assert!(change.edits.len() < parse_board(PATH, &text).statements.len() / 4);
    // Copy Room Formats.
    let (change, report) = copy_room_formats(
        PATH,
        &change.source,
        "ch[0]",
        &[],
        &RoomFormatOptions::default(),
    )
    .unwrap();
    assert!(!change.reprinted);
    assert!(!report.copies.is_empty());
    let mut expect = got;
    runmat_circuit::room_formats::copy_room_formats(
        &mut expect,
        "ch[0]",
        &[],
        &RoomFormatOptions::default(),
    )
    .unwrap();
    assert!(same_board(
        &parse_board(PATH, &change.source).board.unwrap(),
        &expect
    ));
}

#[test]
fn boards_at_scale() {
    // 50k+ primitives: 40k tracks, 8k vias, 2k footprints with pads, plus pours.
    let (_, _, _, seed) = large(&LargeParams {
        channels: 2,
        copper_layers: 8,
        route_every: 1,
        pours: true,
    });
    let mut b = seed;
    b.name = "ctl".into();
    let package = b.footprints.values().next().unwrap().package.clone();
    let nets: Vec<NetId> = (0..400)
        .map(|i| b.add_net(&format!("N{i}"), None))
        .collect();
    for i in 0..2_000i64 {
        let id = b.place(
            &format!("R{}", 1000 + i),
            package.clone(),
            Point::new(i * 500_000, 3_000_000),
            90.0,
            Side::Top,
        );
        let f = b.footprints.get_mut(&id).unwrap();
        for (k, pad) in package.pads.iter().enumerate() {
            f.pad_nets
                .insert(pad.name.clone(), nets[(i as usize + k) % nets.len()]);
        }
    }
    for i in 0..40_000i64 {
        let y = (i % 200) * 250_000;
        b.add_track(
            Some(nets[i as usize % 400]),
            (i % 8) as u8,
            150_000,
            Point::new(i * 1_250, y),
            Point::new(i * 1_250 + 1_000_000, y + 333),
        );
    }
    for i in 0..8_000i64 {
        b.add_via(
            Some(nets[i as usize % 400]),
            Point::new(i * 3_001, 7_777),
            450_000,
            200_000,
        );
    }
    let primitives = b.footprints.len() + b.tracks.len() + b.vias.len() + b.zones.len();
    assert!(primitives >= 50_000, "{primitives}");
    let t = Instant::now();
    let text = print_board(&b, &[]);
    let print = t.elapsed();
    let t = Instant::now();
    let tree =
        runmat_design_syntax::parse_source(runmat_design_syntax::Dialect::CircuitLayout, &text);
    let grammar = t.elapsed();
    let t = Instant::now();
    let statements = tree.file().items().len();
    let walk = t.elapsed();
    let t = Instant::now();
    let parsed = parse_board(PATH, &text);
    let parse = t.elapsed();
    eprintln!("scale: shared grammar {grammar:?}, item walk {walk:?} ({statements} items)");
    assert!(parsed.diagnostics.is_empty());
    assert!(same_board(parsed.board.as_ref().unwrap(), &b));
    let t = Instant::now();
    let (change, ()) = edit_board(PATH, &text, |b| {
        b.add_track(None, 0, mm(0.3), Point::mm(1.0, 1.0), Point::mm(2.0, 1.0));
        let id = *b.tracks.keys().nth(17).unwrap();
        b.tracks.get_mut(&id).unwrap().width = mm(0.5);
        Ok(())
    })
    .unwrap();
    let edit = t.elapsed();
    assert!(!change.reprinted);
    assert_eq!(change.edits.len(), 2);
    eprintln!(
        "scale: {primitives} primitives, {} statements, {} bytes; print {print:?}, parse {parse:?}, edit (parse, diff, apply, re-read) {edit:?}",
        parsed.statements.len(),
        text.len()
    );
}

#[test]
fn the_sync_record_survives_a_kicad_round_trip() {
    use runmat_circuit::kicad::pcb;
    use runmat_circuit::syntax::layout::SyncSection;
    let board = eco_scenario().board;
    let before = SyncSection::of(&board);
    assert!(!before.links.is_empty() && !before.sync.components.is_empty());
    // Export and import: the baseline, the settings and every footprint link come back.
    let text = pcb::write_board(&board);
    assert!(text.contains(pcb::SYNC_PROPERTY));
    let k = pcb::read(&text).unwrap();
    assert!(k.warnings.is_empty(), "{:?}", k.warnings);
    assert_eq!(SyncSection::of(&k.board), before);
    // An unedited write keeps the file as it was.
    assert_eq!(k.write(), pcb::read(&k.write()).unwrap().write());
    // An edit to the settings is written back; the rest of the file stays.
    let mut k = k;
    k.board.sync.options.relink_by_designator = false;
    let edited = k.write();
    let back = pcb::read(&edited).unwrap();
    assert!(!back.board.sync.options.relink_by_designator);
    assert_eq!(back.board.footprints.len(), board.footprints.len());
    // The imported board prints as a layout and reads back the same record.
    let mut imported = back.board;
    imported.name = "ctl".into();
    let layout = print_board(&imported, &[]);
    let parsed = parse_board(PATH, &layout);
    assert_eq!(
        SyncSection::of(parsed.board.as_ref().unwrap()),
        SyncSection::of(&imported)
    );
    // A record from a newer release is reported, and the board imports without it.
    let newer = text.replacen("version = 1", "version = 9", 1);
    let k = pcb::read(&newer).unwrap();
    assert!(
        k.warnings.iter().any(|w| w.contains("newer RunMat")),
        "{:?}",
        k.warnings
    );
}
