//! One page drawn from its placed cells: symbols with their texts, sheet symbols, the routed
//! connections, the frame that fits and the title block.

use super::route::{self, Term, Walls};
use super::symbol::{Art, Place, PinSide};
use super::*;

/// Gap between a body and the designator or value beside it.
pub(super) const TEXT_GAP: Coord = 500_000;

fn xy(p: Point) -> Xy {
    [mm(p.x), mm(p.y)]
}

fn grow(b: [Coord; 4], p: [Coord; 4]) -> [Coord; 4] {
    [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[2]), b[3].max(p[3])]
}

/// The designator line as the canvas draws it: `U1A · bank 34`.
pub fn designator_text(s: &SchematicSymbol) -> String {
    let name = match &s.unit {
        Some(u) => format!("{}{u}", s.designator),
        None => s.designator.clone(),
    };
    match &s.unit_label {
        Some(l) => format!("{name} · {l}"),
        None => name,
    }
}

/// Every text box a symbol draws, `[x0, y0, x1, y1]` in mm: the designator, the value, and
/// each shown pin name and number.
pub fn symbol_texts(s: &SchematicSymbol) -> Vec<[f64; 4]> {
    let n = mm_to_nm;
    let bx = |t: &str, size: f64, at: Xy, anchor: &str| {
        let b = text_box(t, n(size), n(at[0]), n(at[1]), anchor);
        [mm(b[0]), mm(b[1]), mm(b[2]), mm(b[3])]
    };
    let mut out = vec![bx(
        &designator_text(s),
        s.designator_text_mm,
        s.designator_at,
        &s.text_anchor,
    )];
    if let Some(v) = &s.value {
        out.push(bx(v, s.value_text_mm, s.value_at, &s.text_anchor));
    }
    for p in &s.pins {
        if p.hidden {
            continue;
        }
        out.push(bx(&p.name, s.pin_text_mm, p.name_at, &p.name_anchor));
        if p.number != p.name {
            out.push(bx(&p.number_text, s.pin_text_mm, p.number_at, "middle"));
        }
    }
    out
}

pub(super) fn page(
    nl: &Netlist,
    path: &NamePath,
    sheet_name: &str,
    sheets: &[(NamePath, String)],
    cells: &[Cell],
    places: &[Place],
    infos: &[NetInfo],
    centre: bool,
) -> SchematicSheet {
    let mut page = SchematicSheet {
        path: path.to_string(),
        sheet: sheet_name.to_owned(),
        page: 1,
        ..Default::default()
    };
    let mut terms: Vec<Term> = Vec::new();
    // Each term's pin as the lock's `wire` records name it (units only).
    let mut refs: Vec<Option<String>> = Vec::new();
    let mut walls = Walls {
        bodies: Vec::new(),
        pin_lines: Vec::new(),
        texts: Vec::new(),
    };
    let mut bbox = [Coord::MAX, Coord::MAX, Coord::MIN, Coord::MIN];
    let text_nm = |b: [f64; 4]| [mm_to_nm(b[0]), mm_to_nm(b[1]), mm_to_nm(b[2]), mm_to_nm(b[3])];
    for (ci, cell) in cells.iter().enumerate() {
        let pl = &places[ci];
        let lay = &cell.lay;
        let body = pl.rect([-lay.half_w, -lay.half_h, lay.half_w, lay.half_h]);
        walls.bodies.push(body);
        bbox = grow(bbox, pl.rect(lay.extent()));
        // Pins: tips, sides and the lines from the body.
        let mut side_terms: BTreeMap<PinSide, Vec<usize>> = BTreeMap::new();
        for (k, lp) in lay.pins.iter().enumerate() {
            let tip = pl.apply(lay.tip(lp.side, lp.at));
            let root = pl.apply(lay.root(lp.side, lp.at));
            let side = pl.side(lp.side);
            if lay.pin_len > GRID {
                let (dx, dy) = side.out();
                walls
                    .pin_lines
                    .push((root, Point::new(tip.x - dx * GRID, tip.y - dy * GRID)));
            }
            side_terms.entry(side).or_default().push(terms.len());
            refs.push(match cell.kind {
                CellKind::Unit { comp, .. } => cell.pins.get(k).map(|&p| super::hand::pin_ref(&nl.components[comp], p)),
                CellKind::Sheet { .. } => None,
            });
            terms.push(Term {
                net: cell.nets[k],
                tip,
                side,
                first_on_side: false,
                last_on_side: false,
                above: None,
                below: None,
            });
        }
        for (side, mut list) in side_terms {
            if side.horizontal() {
                list.sort_by_key(|&t| std::cmp::Reverse(terms[t].tip.y));
                for w in list.windows(2) {
                    if terms[w[0]].tip.y - terms[w[1]].tip.y == PIN_PITCH {
                        terms[w[0]].below = Some(w[1]);
                        terms[w[1]].above = Some(w[0]);
                    }
                }
            }
            if let (Some(&f), Some(&l)) = (list.first(), list.last()) {
                terms[f].first_on_side = true;
                terms[l].last_on_side = true;
            }
        }
        match cell.kind {
            CellKind::Unit { comp, gate } => {
                let s = symbol(nl, cell, comp, gate, pl, path, &terms);
                for t in symbol_texts(&s) {
                    walls.texts.push(text_nm(t));
                    bbox = grow(bbox, text_nm(t));
                }
                page.symbols.push(s);
            }
            CellKind::Sheet { child } => {
                let sym = sheet_symbol(nl, cell, child, sheets, pl, body);
                // The path above the box and the sheet name below it, as the canvas draws them.
                let (x0, y0, y1) = (body[0], body[1], body[3]);
                walls.texts.push(text_box(&sym.path, 1_600_000, x0, y1 + 800_000, "start"));
                walls.texts.push(text_box(&sym.sheet, 1_300_000, x0, y0 - 2_000_000, "start"));
                page.sheet_symbols.push(sym);
            }
        }
    }
    if bbox[0] > bbox[2] {
        bbox = [0, -GRID, GRID, 0];
    }
    // Notes: at their recorded places (the lock's frame), else stacked above the drawing.
    let mut stack = bbox[3] + 2 * GRID;
    for (name, text) in nl.sheet_texts.get(sheet_name).map(Vec::as_slice).unwrap_or(&[]) {
        let at = match nl.sch_places.get(&text_key(sheet_name, name)).filter(|p| p.page == 1 && !centre) {
            Some(s) => Point::new(s.x - MARGIN, MARGIN - s.y),
            None => {
                let p = Point::new(bbox[0], stack);
                stack += 2 * GRID;
                p
            }
        };
        let b = text_box(text, NOTE_TEXT, at.x, at.y, "start");
        walls.texts.push(b);
        bbox = grow(bbox, b);
        page.texts.push(SheetText { name: name.clone(), text: text.clone(), x_mm: mm(at.x), y_mm: mm(at.y), size_mm: mm(NOTE_TEXT) });
    }
    // Hand-drawn wires apply on a page whose symbols the lock places (its frame is the lock's).
    let hands: Vec<super::hand::Hand> = if centre {
        Vec::new()
    } else {
        let find = |r: &str| refs.iter().position(|x| x.as_deref() == Some(r));
        nl.sch_wires
            .iter()
            .filter(|(k, _)| k.sheet == sheet_name)
            .filter_map(|(k, w)| {
                Some(super::hand::Hand {
                    a: find(&k.a)?,
                    b: find(&k.b)?,
                    mode: w.mode,
                    points: w.points.iter().map(|&(x, y)| Point::new(x - MARGIN, MARGIN - y)).collect(),
                })
            })
            .collect()
    };
    let drawn = route::connect(&terms, infos, &walls, bbox, &hands);
    for (n, pts) in &drawn.wires {
        page.wires.push(SchematicWire {
            net: nl.nets[*n].name.clone(),
            points: pts.iter().map(|p| xy(*p)).collect(),
        });
    }
    page.junctions = drawn.junctions.iter().map(|p| xy(*p)).collect();
    page.no_connects = drawn.no_connects.iter().map(|p| xy(*p)).collect();
    for l in &drawn.labels {
        page.labels.push(NetLabel {
            net: nl.nets[l.net].name.clone(),
            x_mm: mm(l.at.x),
            y_mm: mm(l.at.y),
            kind: l.kind.to_owned(),
            text: infos[l.net].short.clone(),
            side: match l.side {
                PinSide::Left => "left",
                PinSide::Right => "right",
                PinSide::Top => "up",
                PinSide::Bottom => "down",
            }
            .to_owned(),
        });
    }
    frame(&mut page, nl, centre);
    page
}

fn symbol(
    nl: &Netlist,
    cell: &Cell,
    comp: usize,
    gate: usize,
    pl: &Place,
    path: &NamePath,
    _terms: &[Term],
) -> SchematicSymbol {
    let c = &nl.components[comp];
    let us = units(c);
    let unit = &us[gate];
    let lay = &cell.lay;
    let (hw, hh) = (lay.half_w, lay.half_h);
    let corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh), (-hw, -hh)];
    let mut body = vec![corners
        .iter()
        .map(|(x, y)| xy(pl.apply(Point::new(*x, *y))))
        .collect::<Vec<_>>()];
    let small = lay.art.small();
    let mut pins = Vec::new();
    for (k, lp) in lay.pins.iter().enumerate() {
        let tip = pl.apply(lay.tip(lp.side, lp.at));
        body.push(vec![xy(pl.apply(lay.root(lp.side, lp.at))), xy(tip)]);
        let (name_at, anchor) = lay.name_at(lp.side, lp.at);
        let anchor = match (pl.mirror, anchor) {
            (true, "start") => "end",
            (true, "end") => "start",
            (_, a) => a,
        };
        let shown = number_text(&lp.pads);
        let number_at = lay.number_at(lp.side, lp.at, &shown);
        let net = c.pins[cell.pins[k]].net;
        pins.push(SchematicPin {
            name: lp.name.clone(),
            number: lp.pads.join(","),
            x_mm: mm(tip.x),
            y_mm: mm(tip.y),
            side: pl.side(lp.side).name().to_owned(),
            name_at: xy(pl.apply(name_at)),
            name_anchor: anchor.to_owned(),
            number_text: shown,
            number_at: xy(pl.apply(number_at)),
            hidden: small,
            kind: pin_kind(lp.kind).to_owned(),
            net: net.map(|n| nl.nets[n].name.clone()),
        });
    }
    let r = pl.rect([-hw, -hh, hw, hh]);
    let o = pl.origin;
    let (ds, vs) = (DESIGNATOR_TEXT, VALUE_TEXT);
    let (designator_at, value_at, anchor) = if lay.art == Art::Box {
        // Top-left and bottom-left corners, clear of the edge pins (right-aligned); a
        // mirrored box keeps them on the right.
        let (x, a) = if pl.mirror { (r[2], "end") } else { (r[0], "start") };
        (
            Point::new(x, r[3] + TEXT_GAP + ds / 5),
            Point::new(x, r[1] - TEXT_GAP - vs * 4 / 5),
            a,
        )
    } else if pl.upright() || lay.art == Art::Hole {
        let x = r[2] + 2 * TEXT_GAP;
        (
            Point::new(x, o.y + 2 * TEXT_GAP / 5),
            Point::new(x, o.y - 2 * TEXT_GAP / 5 - vs),
            "start",
        )
    } else {
        (
            Point::new(o.x, r[3] + TEXT_GAP + ds / 5),
            Point::new(o.x, r[1] - TEXT_GAP - vs * 4 / 5),
            "middle",
        )
    };
    SchematicSymbol {
        designator: c.designator.clone(),
        unit: (us.len() > 1).then(|| unit.name.clone()),
        unit_label: unit.label.clone(),
        path: c.path.to_string(),
        sheet_path: path.to_string(),
        key: cell.key.clone(),
        x_mm: mm(o.x),
        y_mm: mm(o.y),
        rotation: pl.rotation as f64,
        mirror: pl.mirror,
        art: lay.art.name().to_owned(),
        body,
        strokes: lay
            .strokes()
            .iter()
            .map(|s| s.iter().map(|p| xy(pl.apply(*p))).collect())
            .collect(),
        pins,
        value: crate::index::component_value(c),
        designator_at: xy(designator_at),
        value_at: xy(value_at),
        text_anchor: anchor.to_owned(),
        pin_text_mm: mm(PIN_TEXT),
        designator_text_mm: mm(DESIGNATOR_TEXT),
        value_text_mm: mm(VALUE_TEXT),
    }
}

fn sheet_symbol(
    nl: &Netlist,
    cell: &Cell,
    child: usize,
    sheets: &[(NamePath, String)],
    pl: &Place,
    body: [Coord; 4],
) -> SheetSymbol {
    SheetSymbol {
        path: sheets[child].0.to_string(),
        sheet: sheets[child].1.clone(),
        key: cell.key.clone(),
        x_mm: mm(body[0]),
        y_mm: mm(body[1]),
        w_mm: mm(body[2] - body[0]),
        h_mm: mm(body[3] - body[1]),
        pins: cell
            .lay
            .pins
            .iter()
            .enumerate()
            .map(|(k, lp)| {
                let tip = pl.apply(cell.lay.tip(lp.side, lp.at));
                SheetPin {
                    net: nl.nets[cell.nets[k].unwrap()].name.clone(),
                    x_mm: mm(tip.x),
                    y_mm: mm(tip.y),
                    side: pl.side(lp.side).name().to_owned(),
                    name: lp.name.clone(),
                    direction: (if cell.sheet_out[k] { "out" } else { "in" }).to_owned(),
                }
            })
            .collect(),
    }
}

/// Every box the page draws, in mm: symbols with their texts, sheet symbols, labels, wires.
pub fn page_boxes(page: &SchematicSheet) -> Vec<[f64; 4]> {
    let mut out = Vec::new();
    let span = |pts: &[Xy]| {
        pts.iter().fold([f64::MAX, f64::MAX, f64::MIN, f64::MIN], |b, p| {
            [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])]
        })
    };
    for s in &page.symbols {
        for line in &s.body {
            out.push(span(line));
        }
        out.extend(symbol_texts(s));
    }
    for t in &page.texts {
        let n = mm_to_nm;
        let b = text_box(&t.text, n(t.size_mm), n(t.x_mm), n(t.y_mm), "start");
        out.push([mm(b[0]), mm(b[1]), mm(b[2]), mm(b[3])]);
    }
    for s in &page.sheet_symbols {
        out.push([s.x_mm, s.y_mm - 3.5, s.x_mm + s.w_mm, s.y_mm + s.h_mm + 2.5]);
    }
    for l in &page.labels {
        out.push(l.extent());
    }
    for w in &page.wires {
        out.push(span(&w.points));
    }
    out
}

/// Picks the smallest frame the page fits on, clear of the title block, and moves the content
/// from its top-left corner (at the origin, y up) onto the paper.
/// With `centre` (no recorded places on the page), the content is centred on the paper.
fn frame(page: &mut SchematicSheet, nl: &Netlist, centre: bool) {
    let boxes = page_boxes(page);
    let content = boxes
        .iter()
        .fold([0.0f64, 0.0, 0.0, 0.0], |b, p| [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[2]), b[3].max(p[3])]);
    let margin = mm(MARGIN);
    let rows = 5.0 + if nl.variant.is_some() { 1.0 } else { 0.0 };
    let title_h = rows * TITLE_ROW_MM;
    let fits = |w: f64, h: f64, dx: f64, dy: f64| {
        if content[2] + dx > w - margin
            || content[1] + dy < margin
            || content[3] + dy > h - margin + 1e-6
        {
            return false;
        }
        let tb = [w - FRAME_MM - TITLE_W_MM - 2.0, 0.0, w, FRAME_MM + title_h + 2.0];
        !boxes.iter().any(|b| {
            let b = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy];
            b[0] < tb[2] && tb[0] < b[2] && b[1] < tb[3] && tb[1] < b[3]
        })
    };
    let (mut name, mut w, mut h) = FRAMES[FRAMES.len() - 1];
    let mut found = false;
    for f in FRAMES {
        if fits(f.1, f.2, margin - content[0].min(0.0), f.2 - margin) {
            (name, w, h) = f;
            found = true;
            break;
        }
    }
    if !found {
        w = w.max(content[2] + 2.0 * margin + TITLE_W_MM);
        h = h.max(-content[1] + 2.0 * margin + title_h);
    }
    page.frame = name.to_owned();
    page.width_mm = w;
    page.height_mm = mm(mm_to_nm(h));
    let (mut dx, mut dy) = (margin, page.height_mm - margin);
    if centre && found {
        // Whole grid steps, so pins stay on KiCad's grid (y counted down from the top).
        let grid = mm(GRID);
        let steps = |v: f64| (v / grid).floor().max(0.0) * grid;
        let cx = dx + steps((w - 2.0 * margin - (content[2] - content[0])) / 2.0 - content[0]);
        let cy = dy - steps((h - 2.0 * margin - (content[3] - content[1])) / 2.0 + content[3]);
        if fits(w, h, cx, cy) {
            (dx, dy) = (cx, cy);
        }
    }
    shift(page, dx, dy);
    page.title_block = vec![
        ("Title".into(), nl.design.clone()),
        ("Sheet".into(), page.sheet.clone()),
        (
            "Path".into(),
            if page.path == "/" {
                "/ (root)".into()
            } else {
                page.path.clone()
            },
        ),
        ("Size".into(), name.to_owned()),
    ];
    if let Some(v) = &nl.variant {
        page.title_block.push(("Variant".into(), v.clone()));
    }
}

pub(super) fn shift(page: &mut SchematicSheet, dx: f64, dy: f64) {
    let r = |v: f64| (v * 1.0e6).round() / 1.0e6;
    let m = |p: &mut Xy| {
        p[0] = r(p[0] + dx);
        p[1] = r(p[1] + dy);
    };
    for s in &mut page.symbols {
        s.x_mm = r(s.x_mm + dx);
        s.y_mm = r(s.y_mm + dy);
        s.body.iter_mut().flatten().for_each(m);
        s.strokes.iter_mut().flatten().for_each(m);
        m(&mut s.designator_at);
        m(&mut s.value_at);
        for p in &mut s.pins {
            p.x_mm = r(p.x_mm + dx);
            p.y_mm = r(p.y_mm + dy);
            m(&mut p.name_at);
            m(&mut p.number_at);
        }
    }
    for s in &mut page.sheet_symbols {
        s.x_mm = r(s.x_mm + dx);
        s.y_mm = r(s.y_mm + dy);
        for p in &mut s.pins {
            p.x_mm = r(p.x_mm + dx);
            p.y_mm = r(p.y_mm + dy);
        }
    }
    for w in &mut page.wires {
        w.points.iter_mut().for_each(m);
    }
    page.junctions.iter_mut().for_each(m);
    page.no_connects.iter_mut().for_each(m);
    for t in &mut page.texts {
        t.x_mm = r(t.x_mm + dx);
        t.y_mm = r(t.y_mm + dy);
    }
    for l in &mut page.labels {
        l.x_mm = r(l.x_mm + dx);
        l.y_mm = r(l.y_mm + dy);
    }
}
