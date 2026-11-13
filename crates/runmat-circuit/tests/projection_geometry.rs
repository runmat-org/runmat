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
    // One first page per sheet instance; a second page follows its first (unused units).
    let pages: Vec<&str> = sch
        .sheets
        .iter()
        .filter(|s| s.page == 1)
        .map(|s| s.path.as_str())
        .collect();
    let hierarchy: Vec<&str> = index.hierarchy.iter().map(|s| s.path.as_str()).collect();
    assert_eq!(pages, hierarchy);
    for (k, s) in sch.sheets.iter().enumerate().filter(|(_, s)| s.page > 1) {
        assert_eq!(
            sch.sheets[k - 1].path,
            s.path,
            "page {} of {} follows its first",
            s.page,
            s.path
        );
    }
    // Every component appears on its own sheet instance, once per gate.
    for c in &index.components {
        let sheet = c
            .path
            .rsplit_once('/')
            .map(|(p, _)| if p.is_empty() { "/" } else { p })
            .unwrap();
        assert!(
            sch.sheets
                .iter()
                .filter(|s| s.path == sheet)
                .any(|page| page.symbols.iter().any(|s| s.designator == c.designator)),
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
                    // A cross only where the design marks the pin no-connect; any other pin on
                    // no net is drawn open.
                    let crossed = page.no_connects.contains(&[pin.x_mm, pin.y_mm]);
                    assert_eq!(
                        crossed, pin.no_connect,
                        "{}.{} cross",
                        s.designator, pin.name
                    );
                    continue;
                };
                // A connected pin ends a wire of its net, or carries its net's port or label.
                let tip = [pin.x_mm, pin.y_mm];
                let wired = page.wires.iter().any(|w| {
                    &w.net == net && (w.points[0] == tip || *w.points.last().unwrap() == tip)
                });
                let ported = page
                    .labels
                    .iter()
                    .any(|l| &l.net == net && [l.x_mm, l.y_mm] == tip);
                assert!(
                    wired || ported,
                    "{}.{} is connected",
                    s.designator,
                    pin.name
                );
            }
        }
        // Every wire end lands on its net: a pin, a label or port, a sheet entry, or another
        // wire of the net (a T, drawn with a junction dot).
        let on_segment = |p: [f64; 2], w: &runmat_circuit::schematic::SchematicWire| {
            w.points.windows(2).any(|seg| {
                let (a, b) = (seg[0], seg[1]);
                let within = |v: f64, x: f64, y: f64| v >= x.min(y) - 1e-6 && v <= x.max(y) + 1e-6;
                ((a[0] - b[0]).abs() < 1e-6
                    && (p[0] - a[0]).abs() < 1e-6
                    && within(p[1], a[1], b[1]))
                    || ((a[1] - b[1]).abs() < 1e-6
                        && (p[1] - a[1]).abs() < 1e-6
                        && within(p[0], a[0], b[0]))
            })
        };
        for (k, w) in page.wires.iter().enumerate() {
            for end in [w.points[0], *w.points.last().unwrap()] {
                let pin = page
                    .symbols
                    .iter()
                    .flat_map(|o| &o.pins)
                    .any(|p| p.net.as_ref() == Some(&w.net) && [p.x_mm, p.y_mm] == end);
                let entry = page
                    .sheet_symbols
                    .iter()
                    .flat_map(|o| &o.pins)
                    .any(|p| p.net == w.net && [p.x_mm, p.y_mm] == end);
                let label = page
                    .labels
                    .iter()
                    .any(|l| l.net == w.net && [l.x_mm, l.y_mm] == end);
                let joined = page
                    .wires
                    .iter()
                    .enumerate()
                    .any(|(j, o)| j != k && o.net == w.net && on_segment(end, o));
                assert!(
                    pin || entry || label || joined,
                    "{} wire end {end:?} dangles",
                    w.net
                );
            }
            // A wire with no label at either end: the net is drawn as a wire, not as labels.
            let labelled = |p: [f64; 2]| {
                page.labels
                    .iter()
                    .any(|l| l.net == w.net && [l.x_mm, l.y_mm] == p)
            };
            if !labelled(w.points[0]) && !labelled(*w.points.last().unwrap()) {
                direct += 1;
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

#[test]
fn the_large_board_projection_reaches_the_scale_scene_size() {
    let params = runmat_circuit::fixture::LargeParams {
        channels: 400,
        ..Default::default()
    };
    let index = runmat_circuit::fixture::large_index(&params);
    let board = index.board.as_ref().expect("a board");
    let primitives =
        board.pads.len() + board.tracks.len() + board.vias.len() + board.footprints.len();
    eprintln!(
        "large board: {} pads, {} tracks, {} vias, {} footprints = {primitives}",
        board.pads.len(),
        board.tracks.len(),
        board.vias.len(),
        board.footprints.len()
    );
    assert!(primitives > 10_000, "{primitives}");
}

/// Every symbol and unit of io.circuit (the FPGA's bank and power units included) draws its pin
/// names, pin numbers, designator and value without any two texts overlapping: pins sit on the
/// 2.54 mm pitch with 1.27 mm text, names inside the body, numbers above the pin lines outside
/// it, and the body as wide as the longest names on both sides.
#[test]
fn schematic_texts_never_overlap() {
    use runmat_circuit::schematic::{text_box, PIN_TEXT};
    let (index, _) = scenario_index();
    let nm = |v: f64| (v * 1.0e6).round() as i64;
    let meets =
        |a: &[i64; 4], b: &[i64; 4]| a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3];
    let mut symbols = 0;
    let mut fpga_units = 0;
    for page in &index.schematic.sheets {
        for symbol in &page.symbols {
            symbols += 1;
            if symbol.designator == "U9000" {
                fpga_units += 1;
            }
            assert_eq!(nm(symbol.pin_text_mm), PIN_TEXT);
            let xs: Vec<f64> = symbol.body[0].iter().map(|p| p[0]).collect();
            let (left, right) = (
                nm(xs.iter().cloned().fold(f64::MAX, f64::min)),
                nm(xs.iter().cloned().fold(f64::MIN, f64::max)),
            );
            let ys: Vec<f64> = symbol.body[0].iter().map(|p| p[1]).collect();
            let (bottom, top) = (
                nm(ys.iter().cloned().fold(f64::MAX, f64::min)),
                nm(ys.iter().cloned().fold(f64::MIN, f64::max)),
            );
            let half = runmat_circuit::schematic::NO_CONNECT_HALF;
            for pin in &symbol.pins {
                // Two-pin parts (R, C, L, D) hide pin names and numbers.
                if pin.hidden {
                    continue;
                }
                let name = text_box(
                    &pin.name,
                    PIN_TEXT,
                    nm(pin.name_at[0]),
                    nm(pin.name_at[1]),
                    &pin.name_anchor,
                );
                assert!(
                    name[0] >= left && name[2] <= right && name[1] >= bottom && name[3] <= top,
                    "{} {} name leaves the body",
                    symbol.designator,
                    pin.name
                );
                if pin.number != pin.name {
                    let number = text_box(
                        &pin.number_text,
                        PIN_TEXT,
                        nm(pin.number_at[0]),
                        nm(pin.number_at[1]),
                        "middle",
                    );
                    let (tx, ty) = (nm(pin.x_mm), nm(pin.y_mm));
                    match pin.side.as_str() {
                        // Above the pin line, between the body and the tip.
                        "left" | "right" => {
                            assert!(
                                number[1] > ty,
                                "{} {} number on its line",
                                symbol.designator,
                                pin.number
                            );
                            let (lo, hi) = if pin.side == "left" {
                                (tx, left)
                            } else {
                                (right, tx)
                            };
                            assert!(
                                number[0] >= lo && number[2] <= hi,
                                "{} {} number leaves its pin",
                                symbol.designator,
                                pin.number_text
                            );
                        }
                        // Beside the pin line, between the body and the tip.
                        _ => {
                            assert!(
                                number[0] > tx,
                                "{} {} number on its line",
                                symbol.designator,
                                pin.number
                            );
                            let (lo, hi) = if pin.side == "bottom" {
                                (ty, bottom)
                            } else {
                                (top, ty)
                            };
                            assert!(
                                number[1] >= lo && number[3] <= hi,
                                "{} {} number leaves its pin",
                                symbol.designator,
                                pin.number_text
                            );
                        }
                    }
                    // Clear of a no-connect cross at the tip.
                    let cross = [tx - half, ty - half, tx + half, ty + half];
                    assert!(
                        !meets(&number, &cross),
                        "{} {} number under the tip's cross",
                        symbol.designator,
                        pin.number_text
                    );
                }
            }
            // Every text the symbol draws, as the canvas draws it: no two overlap.
            let boxes: Vec<(String, [i64; 4])> = runmat_circuit::schematic::symbol_texts(symbol)
                .into_iter()
                .enumerate()
                .map(|(k, b)| {
                    (
                        format!("text {k}"),
                        [nm(b[0]), nm(b[1]), nm(b[2]), nm(b[3])],
                    )
                })
                .collect();
            for (i, (a, ba)) in boxes.iter().enumerate() {
                for (b, bb) in &boxes[i + 1..] {
                    assert!(
                        !meets(ba, bb),
                        "{}{}: {a} overlaps {b}",
                        symbol.designator,
                        symbol.unit.as_deref().unwrap_or("")
                    );
                }
            }
        }
    }
    assert!(
        symbols > 10 && fpga_units >= 5,
        "{symbols} symbols, {fpga_units} FPGA units"
    );
}
