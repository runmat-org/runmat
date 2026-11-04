//! The projection carries the geometry the Desktop's board and schematic canvases draw: every
//! pad, track, via and pour at its board position, the outline, courtyards and ratsnest; and a
//! schematic page per sheet instance with placed symbols, pins, wires and labels.

use runmat_circuit::board_rules::{BoardGraph, BoardRules};
use runmat_circuit::drc::Drc;
use runmat_circuit::erc::{self, ErcOptions};
use runmat_circuit::fixture;
use runmat_circuit::index::{add_board, build_index, to_json, CircuitIndex};
use runmat_circuit::layout::TrackGeom;
use runmat_circuit::review::{self, ReviewOptions};
use runmat_circuit::zone::fill_zones;
use std::collections::BTreeSet;

fn scenario_index() -> (CircuitIndex, runmat_circuit::layout::Board) {
    let sc = fixture::eco_scenario();
    let erc = erc::run(&sc.netlist, &ErcOptions::default());
    let findings = review::run_all(&sc.netlist, &sc.pool, &ReviewOptions::default());
    let mut index = build_index(&sc.design, &sc.netlist, &sc.pool, &erc, &findings);
    let mut board = sc.board.clone();
    let (mut rules, _) = BoardRules::new(&board.limits, &[]);
    rules.rebuild(&BoardGraph::new(&board, &[]));
    let fills = fill_zones(&mut board, &rules, &[]);
    let drc = Drc::run(&board, &rules, &[], &fills);
    add_board(&mut index, &board, &rules, &drc);
    (index, board)
}

fn inside(b: &[[f64; 2]], p: [f64; 2]) -> bool {
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for q in b {
        for k in 0..2 {
            lo[k] = lo[k].min(q[k]);
            hi[k] = hi[k].max(q[k]);
        }
    }
    (0..2).all(|k| p[k] >= lo[k] - 1e-6 && p[k] <= hi[k] + 1e-6)
}

#[test]
fn board_items_carry_their_positions() {
    let (index, board) = scenario_index();
    let b = index.board.as_ref().unwrap();
    // Outline: the outer ring first, at the stack-up's thickness.
    assert!(!b.outline.polygons.is_empty());
    assert!(b.outline.polygons[0].len() >= 3);
    assert!((b.outline.thickness_mm - board.stackup.thickness() as f64 / 1e6).abs() < 1e-6);
    let outer = &b.outline.polygons[0];
    // Pads: absolute, inside the board, sized, THT pads drilled.
    assert_eq!(b.pads.len(), board.all_pads().len());
    for (row, placed) in b.pads.iter().zip(board.all_pads()) {
        assert!(
            (row.x_mm - placed.center.x as f64 / 1e6).abs() < 1e-6,
            "{}",
            row.id
        );
        assert!(
            (row.y_mm - placed.center.y as f64 / 1e6).abs() < 1e-6,
            "{}",
            row.id
        );
        assert!(row.size_x_mm > 0.0 && row.size_y_mm > 0.0, "{}", row.id);
        assert_eq!(row.drill_mm.is_some(), placed.drill.is_some(), "{}", row.id);
        assert!(
            inside(outer, [row.x_mm, row.y_mm]),
            "{} off the board",
            row.id
        );
    }
    // A footprint's rotation turns its pads.
    for f in board.footprints.values().filter(|f| f.rotation != 0.0) {
        let row = b.pads.iter().find(|p| p.component == f.reference).unwrap();
        let def = &f.package.pads[0];
        let want = f.transform().apply_angle(def.rotation);
        assert!((row.rotation - want).abs() < 1e-9, "{}", row.id);
    }
    // Tracks: a centreline from end to end.
    assert_eq!(b.tracks.len(), board.tracks.len());
    for (row, t) in b.tracks.iter().zip(board.tracks.values()) {
        assert!(row.points.len() >= 2);
        let (a, z) = t.endpoints();
        assert_eq!(row.points[0], [a.x as f64 / 1e6, a.y as f64 / 1e6]);
        assert_eq!(
            *row.points.last().unwrap(),
            [z.x as f64 / 1e6, z.y as f64 / 1e6]
        );
        assert_eq!(row.arc.is_some(), matches!(t.geom, TrackGeom::Arc { .. }));
    }
    // Vias at their centres.
    for (row, v) in b.vias.iter().zip(board.vias.values()) {
        assert_eq!(row.x_mm, Some(v.at.x as f64 / 1e6));
        assert_eq!(row.y_mm, Some(v.at.y as f64 / 1e6));
    }
    // Pours: one entry per layer, with the poured copper.
    assert!(!b.zones.is_empty());
    for (row, z) in b.zones.iter().zip(board.zones.values()) {
        assert_eq!(row.polygons.len(), z.layers.len());
        for layer in &row.polygons {
            assert!(layer.outline.len() >= 4);
            assert_eq!(layer.outline.first(), layer.outline.last(), "closed");
        }
    }
    assert!(
        b.zones
            .iter()
            .flat_map(|z| &z.polygons)
            .any(|l| !l.filled.is_empty()),
        "some pour has copper after clearances"
    );
    // Footprints draw their courtyard.
    assert!(
        b.footprints
            .iter()
            .filter(|f| !f.courtyard.is_empty())
            .count()
            * 2
            > b.footprints.len()
    );
    for f in &b.footprints {
        for ring in &f.courtyard {
            assert_eq!(ring.first(), ring.last());
        }
    }
    // Rooms keep their outline.
    for r in &b.rooms {
        assert!(r.outline.len() >= 3);
    }
    // Every airwire joins two points of its net's copper.
    let nets: BTreeSet<&str> = b.pads.iter().filter_map(|p| p.net.as_deref()).collect();
    for a in &b.unrouted {
        assert!(nets.contains(a.net.as_str()), "{}", a.net);
    }
}

#[test]
fn schematic_has_a_page_per_sheet_with_every_pin_connected() {
    let (index, _) = scenario_index();
    let sch = &index.schematic;
    let pages: Vec<&str> = sch.sheets.iter().map(|s| s.path.as_str()).collect();
    let hierarchy: Vec<&str> = index.hierarchy.iter().map(|s| s.path.as_str()).collect();
    assert_eq!(pages, hierarchy);
    // Every component appears on its own sheet, once per gate.
    for c in &index.components {
        let sheet = c
            .path
            .rsplit_once('/')
            .map(|(p, _)| if p.is_empty() { "/" } else { p })
            .unwrap();
        let page = sch.sheets.iter().find(|s| s.path == sheet).unwrap();
        assert!(
            page.symbols.iter().any(|s| s.designator == c.designator),
            "{}",
            c.designator
        );
    }
    let mut direct = 0;
    for page in &sch.sheets {
        let (w, h) = (page.width_mm, page.height_mm);
        for s in &page.symbols {
            assert!(
                s.x_mm > 0.0 && s.x_mm < w && s.y_mm > 0.0 && s.y_mm < h,
                "{} on the page",
                s.designator
            );
            assert_eq!(s.body.len(), 1 + s.pins.len(), "a box and a line per pin");
            for (pin, line) in s.pins.iter().zip(&s.body[1..]) {
                assert_eq!(
                    line[1],
                    [pin.x_mm, pin.y_mm],
                    "the pin line ends at the tip"
                );
                let Some(net) = &pin.net else {
                    assert!(page.no_connects.contains(&[pin.x_mm, pin.y_mm]));
                    continue;
                };
                // A connected pin starts a wire of its net.
                let wire = page
                    .wires
                    .iter()
                    .find(|w| {
                        &w.net == net
                            && (w.points[0] == [pin.x_mm, pin.y_mm]
                                || *w.points.last().unwrap() == [pin.x_mm, pin.y_mm])
                    })
                    .unwrap_or_else(|| panic!("{}.{} has no wire", s.designator, pin.name));
                let end = if wire.points[0] == [pin.x_mm, pin.y_mm] {
                    *wire.points.last().unwrap()
                } else {
                    wire.points[0]
                };
                let labelled = page
                    .labels
                    .iter()
                    .any(|l| &l.net == net && [l.x_mm, l.y_mm] == end);
                let to_pin = page
                    .symbols
                    .iter()
                    .flat_map(|o| &o.pins)
                    .any(|p| p.net.as_ref() == Some(net) && [p.x_mm, p.y_mm] == end);
                assert!(
                    labelled || to_pin,
                    "{}.{} ends at a label or a pin",
                    s.designator,
                    pin.name
                );
                if to_pin {
                    direct += 1;
                }
            }
        }
        for l in &page.labels {
            assert!(["local", "global", "hier", "power"].contains(&l.kind.as_str()));
        }
    }
    assert!(direct > 0, "neighbouring two-pin nets are drawn as wires");
    // Child sheets appear as sheet symbols on their parent.
    let root = sch.sheets.iter().find(|s| s.path == "/").unwrap();
    let children = index
        .hierarchy
        .iter()
        .filter(|s| s.path.matches('/').count() == 1 && s.path != "/")
        .count();
    assert_eq!(root.sheet_symbols.len(), children);
    // Rails are power ports; the rest of the global nets global labels.
    assert!(sch
        .sheets
        .iter()
        .flat_map(|s| &s.labels)
        .any(|l| l.kind == "power"));
}

#[test]
fn geometry_is_deterministic() {
    let (a, _) = scenario_index();
    let (b, _) = scenario_index();
    assert_eq!(to_json(&a), to_json(&b));
}
