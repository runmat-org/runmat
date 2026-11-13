//! The drill map and the fabrication drawing as vector drawings in board coordinates.
//!
//! A [`Drawing`] holds stroked polylines and left-aligned text. The Gerber writer
//! ([`Drawing::gerber`]) draws the text with the stroke font; a sheet writer (PDF, DXF) can keep
//! it as text. Both drawings put the board where it lies, so the drill map overlays the copper
//! layers (`%TF.SameCoordinates%`), and place their tables to the right of the board.
//!
//! - [`drill_map`]: the outline, one symbol per tool at every hole, and the drill table
//!   (symbol, diameter, count, plating, span).
//! - [`fabrication_drawing`]: the drill map plus the overall size, the layer and stackup table
//!   (order, layer, type, material, thickness, copper weight, total thickness, finish) and the
//!   fabrication notes from the board's limits.

use super::drill::{drill_table, holes, DrillRow};
use crate::geom::{mm, to_mm, Coord, Point, Rect};
use crate::layout::{copper_name, Board, StackKind};
use std::fmt::Write as _;

/// One mark of a drawing.
#[derive(Clone, Debug, PartialEq)]
pub enum Mark {
    /// A stroked polyline (closed when the last point equals the first).
    Line { points: Vec<Point>, width: Coord },
    /// Left-aligned text: `at` is the left end of the baseline, `size` the cap height.
    Text {
        at: Point,
        size: Coord,
        text: String,
    },
}

/// A vector drawing in board coordinates.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Drawing {
    pub title: String,
    pub marks: Vec<Mark>,
}

const PEN: Coord = 100_000; // 0.1 mm
const THIN: Coord = 50_000;

/// Advance width of `text` at cap height `size`, as the stroke font sets it.
pub fn text_width(text: &str, size: Coord) -> Coord {
    let n = text.chars().count() as f64;
    if n == 0.0 {
        return 0;
    }
    ((5.4 * n - 1.4) * size as f64 / 6.0).round() as Coord
}

impl Drawing {
    pub(crate) fn line(&mut self, points: Vec<Point>, width: Coord) {
        self.marks.push(Mark::Line { points, width });
    }
    pub(crate) fn rect(&mut self, r: Rect, width: Coord) {
        let (a, b) = (r.min, r.max);
        self.line(
            vec![a, Point::new(b.x, a.y), b, Point::new(a.x, b.y), a],
            width,
        );
    }
    pub(crate) fn text(&mut self, at: Point, size: Coord, text: impl Into<String>) {
        self.marks.push(Mark::Text {
            at,
            size,
            text: text.into(),
        });
    }

    /// The extent of every mark.
    pub fn extent(&self) -> Rect {
        let mut r: Option<Rect> = None;
        let mut add = |p: Point| r = Some(r.map_or(Rect::new(p, p), |r| r.include(p)));
        for m in &self.marks {
            match m {
                Mark::Line { points, .. } => points.iter().for_each(|p| add(*p)),
                Mark::Text { at, size, text } => {
                    add(*at);
                    add(Point::new(at.x + text_width(text, *size), at.y + size));
                }
            }
        }
        r.unwrap_or(Rect::new(Point::ORIGIN, Point::ORIGIN))
    }

    /// The drawing as one Gerber X2 file of the given file function (`Drillmap`,
    /// `FabricationDrawing`).
    pub fn gerber(&self, board: &Board, function: &str) -> String {
        self.gerber_in(board, function, super::GerberFormat::default(), None)
    }

    /// [`Self::gerber`] in `format`.
    pub fn gerber_in(
        &self,
        board: &Board,
        function: &str,
        format: super::GerberFormat,
        revision: Option<&str>,
    ) -> String {
        let mut w = super::Writer::new(format);
        self.emit(&mut w);
        w.finish(&super::header(board, function, "Positive", revision))
    }

    /// Draws every mark into `sink`: lines as strokes, text in the stroke font.
    pub(crate) fn emit(&self, sink: &mut impl super::Sink) {
        for m in &self.marks {
            match m {
                Mark::Line { points, width } => {
                    let ap = sink.aperture(super::Aperture::Circle(*width), "Other,Drawing");
                    sink.draw(ap, points);
                }
                Mark::Text { at, size, text } => {
                    let ap = sink.aperture(
                        super::Aperture::Circle(crate::font::default_thickness(*size)),
                        "Other,Drawing",
                    );
                    let c = Point::new(at.x + text_width(text, *size) / 2, at.y + size / 2);
                    for pl in crate::font::strokes(text, c, *size, 0.0, false) {
                        if pl.len() == 1 {
                            sink.flash(ap, pl[0]);
                        } else {
                            sink.draw(ap, &pl);
                        }
                    }
                }
            }
        }
    }
}

/// The symbol of drill tool `i` (0-based) centred at `c`, `s` across: a cross, an X, a square,
/// a triangle, a diamond, a circle, a hexagon and an hourglass, then the same eight inside a
/// circle, and so on with a second ring.
pub fn symbol(i: usize, c: Point, s: Coord) -> Vec<Vec<Point>> {
    let h = s / 2;
    let p =
        |x: f64, y: f64| Point::new(c.x + (x * h as f64) as Coord, c.y + (y * h as f64) as Coord);
    let poly = |n: usize, phase: f64| -> Vec<Point> {
        (0..=n)
            .map(|k| {
                let a = phase + std::f64::consts::TAU * k as f64 / n as f64;
                p(a.cos(), a.sin())
            })
            .collect()
    };
    let mut out = match i % 8 {
        0 => vec![
            vec![p(-1.0, 0.0), p(1.0, 0.0)],
            vec![p(0.0, -1.0), p(0.0, 1.0)],
        ],
        1 => vec![
            vec![p(-0.7, -0.7), p(0.7, 0.7)],
            vec![p(-0.7, 0.7), p(0.7, -0.7)],
        ],
        2 => vec![vec![
            p(-0.7, -0.7),
            p(0.7, -0.7),
            p(0.7, 0.7),
            p(-0.7, 0.7),
            p(-0.7, -0.7),
        ]],
        3 => vec![poly(3, std::f64::consts::FRAC_PI_2)],
        4 => vec![poly(4, 0.0)],
        5 => vec![poly(32, 0.0)],
        6 => vec![poly(6, 0.0)],
        _ => vec![vec![
            p(-0.7, -0.7),
            p(0.7, 0.7),
            p(-0.7, 0.7),
            p(0.7, -0.7),
            p(-0.7, -0.7),
        ]],
    };
    for ring in 0..i / 8 {
        let r = 1.25 + 0.3 * ring as f64;
        out.push(
            (0..=32)
                .map(|k| {
                    let a = std::f64::consts::TAU * k as f64 / 32.0;
                    p(r * a.cos(), r * a.sin())
                })
                .collect(),
        );
    }
    out
}

const SYMBOL: Coord = 1_000_000; // 1 mm across
const TEXT: Coord = 1_200_000; // 1.2 mm cap height
const ROW: Coord = 2_400_000;

fn outline(d: &mut Drawing, board: &Board) {
    for r in board.outline.rings() {
        let mut pts = r.clone();
        pts.push(r[0]);
        d.line(pts, PEN);
    }
}

fn symbols(d: &mut Drawing, board: &Board, rows: &[DrillRow]) {
    let spans = holes(board);
    for (i, row) in rows.iter().enumerate() {
        for h in spans
            .get(&row.span)
            .into_iter()
            .flatten()
            .filter(|h| h.dia == row.dia)
        {
            let at = match h.slot {
                Some((a, b)) => {
                    d.line(vec![a, b], THIN);
                    Point::new((a.x + b.x) / 2, (a.y + b.y) / 2)
                }
                None => h.at,
            };
            for pl in symbol(i, at, SYMBOL) {
                d.line(pl, THIN);
            }
        }
    }
}

/// A table: a title, a header row and rows, with ruled columns, its top-left corner at `at`.
/// Returns the table's bottom edge.
fn table(
    d: &mut Drawing,
    at: Point,
    title: &str,
    head: &[&str],
    rows: &[Vec<String>],
    first_col: Coord,
) -> Coord {
    let pad = mm(0.8);
    let mut widths: Vec<Coord> = head.iter().map(|h| text_width(h, TEXT)).collect();
    for r in rows {
        for (k, c) in r.iter().enumerate() {
            widths[k] = widths[k].max(text_width(c, TEXT));
        }
    }
    widths[0] = widths[0].max(first_col);
    let widths: Vec<Coord> = widths.iter().map(|w| w + 2 * pad).collect();
    let total: Coord = widths.iter().sum();
    d.text(Point::new(at.x, at.y + mm(0.8)), TEXT, title);
    let n = rows.len() as Coord + 1;
    let bottom = at.y - n * ROW;
    d.rect(
        Rect::new(Point::new(at.x, bottom), Point::new(at.x + total, at.y)),
        PEN,
    );
    for k in 1..=n {
        let y = at.y - k * ROW;
        if k < n {
            d.line(
                vec![Point::new(at.x, y), Point::new(at.x + total, y)],
                if k == 1 { PEN } else { THIN },
            );
        }
    }
    let mut x = at.x;
    for (k, w) in widths.iter().enumerate() {
        if k > 0 {
            d.line(vec![Point::new(x, at.y), Point::new(x, bottom)], THIN);
        }
        d.text(
            Point::new(x + pad, at.y - ROW + (ROW - TEXT) / 2),
            TEXT,
            head[k],
        );
        for (ri, r) in rows.iter().enumerate() {
            let y = at.y - (ri as Coord + 2) * ROW + (ROW - TEXT) / 2;
            d.text(Point::new(x + pad, y), TEXT, r[k].clone());
        }
        x += w;
    }
    bottom
}

fn span_name(board: &Board, row: &DrillRow) -> String {
    let n = board.copper_count();
    format!(
        "{} {}-{}",
        row.span.kind(n),
        copper_name(row.span.from, n),
        copper_name(row.span.to, n)
    )
}

/// Draws the drill table at `at`, with each tool's symbol in its first column; returns the
/// bottom edge.
fn drill_rows(d: &mut Drawing, board: &Board, rows: &[DrillRow], at: Point) -> Coord {
    let body: Vec<Vec<String>> = rows
        .iter()
        .map(|r| {
            vec![
                String::new(),
                format!("{:.3}", to_mm(r.dia)),
                if r.slots > 0 {
                    format!("{} ({} slots)", r.count, r.slots)
                } else {
                    r.count.to_string()
                },
                if r.span.plated {
                    "plated".into()
                } else {
                    "non-plated".into()
                },
                span_name(board, r),
            ]
        })
        .collect();
    let bottom = table(
        d,
        at,
        "DRILL TABLE (mm)",
        &["Symbol", "Diameter", "Count", "Plating", "Span"],
        &body,
        SYMBOL * 2,
    );
    let col = text_width("Symbol", TEXT).max(SYMBOL * 2) + 2 * mm(0.8);
    for (i, _) in rows.iter().enumerate() {
        let c = Point::new(at.x + col / 2, at.y - (i as Coord + 1) * ROW - ROW / 2);
        for pl in symbol(i, c, SYMBOL) {
            d.line(pl, THIN);
        }
    }
    let total: usize = rows.iter().map(|r| r.count).sum();
    d.text(
        Point::new(at.x, bottom - mm(2.0)),
        TEXT,
        format!("{total} holes in {} tools", rows.len()),
    );
    bottom - mm(2.0)
}

fn margin_x(board: &Board) -> Coord {
    board.outline.bbox().max.x + mm(10.0)
}

/// The drill map: the board outline, a symbol per tool at every hole, and the drill table.
pub fn drill_map(board: &Board) -> Drawing {
    let mut d = Drawing {
        title: format!("{} drill map", board.name),
        marks: Vec::new(),
    };
    let rows = drill_table(board);
    outline(&mut d, board);
    symbols(&mut d, board, &rows);
    let top = board.outline.bbox().max.y;
    drill_rows(&mut d, board, &rows, Point::new(margin_x(board), top));
    d
}

/// The layer and stackup table rows: order, layer, type, material, thickness (mm), copper
/// weight.
pub fn stackup_rows(board: &Board) -> Vec<Vec<String>> {
    let n = board.copper_count();
    let mut cu: crate::layout::CopperIndex = 0;
    board
        .stackup
        .layers
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let (name, kind, weight) = match &l.kind {
                StackKind::Copper { weight_oz } => {
                    let name = copper_name(cu, n);
                    cu += 1;
                    let label = if l.name == name || l.name.is_empty() {
                        name
                    } else {
                        format!("{name} ({})", l.name)
                    };
                    (label, "copper".to_string(), format!("{weight_oz} oz"))
                }
                StackKind::Dielectric { core } => (
                    l.name.clone(),
                    if *core { "core" } else { "prepreg" }.to_string(),
                    String::new(),
                ),
                StackKind::SolderMask => (l.name.clone(), "solder mask".into(), String::new()),
                StackKind::Silkscreen => (l.name.clone(), "legend".into(), String::new()),
                StackKind::Paste => (l.name.clone(), "paste".into(), String::new()),
            };
            vec![
                (i + 1).to_string(),
                name,
                kind,
                l.material.name.clone(),
                format!("{:.4}", to_mm(l.thickness)),
                weight,
            ]
        })
        .collect()
}

/// One note per back-drill face and must-not-cut layer: how many holes, the drill, the depth
/// from that face and the residual stub the fabricator must hold.
pub fn backdrill_notes(board: &Board) -> Vec<String> {
    use crate::high_speed::stubs::board_backdrills;
    let names = board.stackup.copper_names();
    let mut groups: std::collections::BTreeMap<_, Vec<_>> = std::collections::BTreeMap::new();
    for (_, b) in board_backdrills(board) {
        groups.entry((b.face, b.must_not_cut)).or_default().push(b);
    }
    groups
        .into_iter()
        .map(|((face, keep), bs)| {
            let dias: std::collections::BTreeSet<_> = bs.iter().map(|b| b.diameter).collect();
            let depth = bs.iter().map(|b| b.depth).max().unwrap_or(0);
            let residual = bs.iter().map(|b| b.residual).max().unwrap_or(0);
            format!(
                "Back-drill {} hole{} from the {} face with a {} mm drill, {:.3} mm deep; must not cut layer {} ({}); stub left at most {:.3} mm",
                bs.len(),
                if bs.len() == 1 { "" } else { "s" },
                face.word(),
                dias.iter().map(|d| format!("{:.3}", to_mm(*d))).collect::<Vec<_>>().join(" / "),
                to_mm(depth),
                keep + 1,
                names.get(keep as usize).cloned().unwrap_or_default(),
                to_mm(residual)
            )
        })
        .collect()
}

/// An arrowhead at `tip` pointing along (`dx`, `dy`) (a unit axis direction): two strokes.
fn arrowhead(d: &mut Drawing, tip: Point, dx: Coord, dy: Coord) {
    let (len, half) = (mm(1.0), mm(0.35));
    let back = Point::new(tip.x - dx * len, tip.y - dy * len);
    let side = Point::new(-dy * half, dx * half);
    d.line(
        vec![
            Point::new(back.x + side.x, back.y + side.y),
            tip,
            Point::new(back.x - side.x, back.y - side.y),
        ],
        THIN,
    );
}

/// The fabrication drawing: the drill map, the overall size, the layer and stackup table and
/// the fabrication notes. The title names the board's `revision` when it has one.
pub fn fabrication_drawing(board: &Board, revision: Option<&str>) -> Drawing {
    let rev = revision.map(str::trim).filter(|r| !r.is_empty());
    let mut d = Drawing {
        title: match rev {
            Some(r) => format!("{} fabrication drawing, revision {r}", board.name),
            None => format!("{} fabrication drawing", board.name),
        },
        marks: Vec::new(),
    };
    let rows = drill_table(board);
    outline(&mut d, board);
    symbols(&mut d, board, &rows);
    let bb = board.outline.bbox();
    // Overall size: a dimension line under the board and one to its left.
    let y = bb.min.y - mm(5.0);
    d.line(vec![Point::new(bb.min.x, y), Point::new(bb.max.x, y)], THIN);
    arrowhead(&mut d, Point::new(bb.min.x, y), -1, 0);
    arrowhead(&mut d, Point::new(bb.max.x, y), 1, 0);
    for x in [bb.min.x, bb.max.x] {
        d.line(
            vec![
                Point::new(x, bb.min.y - mm(1.0)),
                Point::new(x, y - mm(1.0)),
            ],
            THIN,
        );
    }
    let wtxt = format!("{:.2}", to_mm(bb.width()));
    d.text(
        Point::new(bb.center().x - text_width(&wtxt, TEXT) / 2, y - mm(2.8)),
        TEXT,
        wtxt,
    );
    let x = bb.min.x - mm(5.0);
    d.line(vec![Point::new(x, bb.min.y), Point::new(x, bb.max.y)], THIN);
    arrowhead(&mut d, Point::new(x, bb.min.y), 0, -1);
    arrowhead(&mut d, Point::new(x, bb.max.y), 0, 1);
    for yy in [bb.min.y, bb.max.y] {
        d.line(
            vec![
                Point::new(bb.min.x - mm(1.0), yy),
                Point::new(x - mm(1.0), yy),
            ],
            THIN,
        );
    }
    let htxt = format!("{:.2}", to_mm(bb.height()));
    d.text(
        Point::new(
            x - text_width(&htxt, TEXT) - mm(1.5),
            bb.center().y - TEXT / 2,
        ),
        TEXT,
        htxt,
    );

    let left = margin_x(board);
    let mut top = bb.max.y;
    d.text(
        Point::new(left, top + mm(6.0)),
        mm(2.0),
        match rev {
            Some(r) => format!(
                "{} FABRICATION DRAWING  REV {}",
                board.name.to_uppercase(),
                r.to_uppercase()
            ),
            None => format!("{} FABRICATION DRAWING", board.name.to_uppercase()),
        },
    );
    top = drill_rows(&mut d, board, &rows, Point::new(left, top)) - mm(6.0);
    let stack = stackup_rows(board);
    let total: Coord = board.stackup.layers.iter().map(|l| l.thickness).sum();
    top = table(
        &mut d,
        Point::new(left, top),
        "LAYERS AND STACKUP (top to bottom)",
        &["#", "Layer", "Type", "Material", "Thickness mm", "Copper"],
        &stack,
        0,
    );
    let mut notes = vec![format!(
        "Finished thickness {:.3} mm over {} copper layers",
        to_mm(total),
        board.copper_count()
    )];
    if !board.stackup.finish.is_empty() {
        notes.push(format!("Surface finish {}", board.stackup.finish));
    }
    let l = &board.limits;
    notes.push(format!(
        "Minimum track {:.3} mm, minimum clearance {:.3} mm",
        to_mm(l.min_track),
        to_mm(l.min_clearance)
    ));
    notes.push(format!(
        "Minimum drill {:.3} mm, minimum annular ring {:.3} mm, hole to hole {:.3} mm",
        to_mm(l.min_drill),
        to_mm(l.min_annular),
        to_mm(l.min_hole_to_hole)
    ));
    notes.push(format!(
        "Copper to board edge at least {:.3} mm",
        to_mm(l.min_copper_to_edge)
    ));
    notes.extend(backdrill_notes(board));
    notes.push("Dimensions in millimeters; drill diameters are finished hole sizes".into());
    let mut y = top - mm(4.0);
    d.text(Point::new(left, y), TEXT, "NOTES");
    for (i, n) in notes.iter().enumerate() {
        y -= ROW;
        let mut s = String::new();
        let _ = write!(s, "{}. {n}", i + 1);
        d.text(Point::new(left, y), TEXT, s);
    }
    d
}
