//! Silkscreen reference designators for footprints whose package carries none on the silkscreen.
//!
//! A designator is placed as Altium's autoposition places it: beside the part's body (above,
//! below, right, then left), clear of every pad and body and of the designators already placed.
//! Large parts are placed first, so they take the obvious spots and small parts fit around them.
//! The text height follows the part: 1 mm for an IC down to 0.6 mm for an 0402.
//!
//! The board display and the KiCad export both read this placement, so the canvas and the
//! exported board show the same silkscreen.

use crate::geom::{mm, to_mm, Coord, Point, Rect};
use crate::layout::{Board, Footprint, ItemId};
use crate::library::Graphic;
use std::collections::BTreeMap;

/// A placed designator: the text's centre (board nm) and its height (nm).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Designator {
    pub at: Point,
    pub size: Coord,
}

/// Width of a designator glyph and the box height, as fractions of the text height.
const GLYPH_WIDTH: f64 = 0.62;
const BOX_HEIGHT: f64 = 0.78;
/// Clearance between the designator and the part's body (mm).
const GAP_MM: f64 = 0.2;

/// Whether the package names its part on the silkscreen already (`${REFERENCE}`, `%R` or the
/// reference itself).
pub fn names_itself(f: &Footprint) -> bool {
    f.package.silk.iter().any(|g| {
        matches!(g, Graphic::Text { text, .. }
            if text.contains("${REFERENCE}") || text.contains("%R") || *text == f.reference)
    })
}

#[derive(Clone, Copy)]
struct MmBox {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
}

impl MmBox {
    fn of(r: Rect) -> Self {
        MmBox {
            x0: to_mm(r.min.x),
            y0: to_mm(r.min.y),
            x1: to_mm(r.max.x),
            y1: to_mm(r.max.y),
        }
    }
    fn overlap(&self, o: &MmBox) -> f64 {
        let w = self.x1.min(o.x1) - self.x0.max(o.x0);
        let h = self.y1.min(o.y1) - self.y0.max(o.y0);
        if w > 0.0 && h > 0.0 {
            w * h
        } else {
            0.0
        }
    }
    fn area(&self) -> f64 {
        (self.x1 - self.x0).max(0.0) * (self.y1 - self.y0).max(0.0)
    }
}

/// Boxes bucketed on a 2 mm grid, for overlap queries.
struct Occupancy {
    cells: BTreeMap<(i64, i64), Vec<usize>>,
    boxes: Vec<MmBox>,
}

impl Occupancy {
    const CELL: f64 = 2.0;
    fn keys(b: &MmBox) -> impl Iterator<Item = (i64, i64)> {
        let (x0, x1) = (
            (b.x0 / Self::CELL).floor() as i64,
            (b.x1 / Self::CELL).floor() as i64,
        );
        let (y0, y1) = (
            (b.y0 / Self::CELL).floor() as i64,
            (b.y1 / Self::CELL).floor() as i64,
        );
        (x0..=x1).flat_map(move |x| (y0..=y1).map(move |y| (x, y)))
    }
    fn add(&mut self, b: MmBox) {
        let i = self.boxes.len();
        self.boxes.push(b);
        for k in Self::keys(&b) {
            self.cells.entry(k).or_default().push(i);
        }
    }
    fn overlap(&self, b: &MmBox) -> f64 {
        let mut seen = std::collections::BTreeSet::new();
        let mut area = 0.0;
        for k in Self::keys(b) {
            for &i in self.cells.get(&k).into_iter().flatten() {
                if seen.insert(i) {
                    area += b.overlap(&self.boxes[i]);
                }
            }
        }
        area
    }
}

/// Designators for every footprint whose package doesn't name it, by footprint id.
pub fn designators(board: &Board) -> BTreeMap<ItemId, Designator> {
    let copper = board.copper_count();
    let mut occupied = Occupancy {
        cells: BTreeMap::new(),
        boxes: Vec::new(),
    };
    for f in board.footprints.values() {
        for p in f.pads(copper) {
            occupied.add(MmBox::of(p.prim.bbox()));
        }
    }
    let mut parts: Vec<(&Footprint, MmBox)> = Vec::new();
    for f in board.footprints.values() {
        let body = f.bbox();
        let body = if body.min.x > body.max.x {
            let (x, y) = (to_mm(f.at.x), to_mm(f.at.y));
            MmBox {
                x0: x - 0.5,
                y0: y - 0.5,
                x1: x + 0.5,
                y1: y + 0.5,
            }
        } else {
            MmBox::of(body)
        };
        occupied.add(body);
        if !names_itself(f) {
            parts.push((f, body));
        }
    }
    // Large parts first; equal sizes keep board order.
    parts.sort_by(|a, b| b.1.area().total_cmp(&a.1.area()));
    let mut out = BTreeMap::new();
    for (f, e) in parts {
        let size = (0.45 * (e.x1 - e.x0).min(e.y1 - e.y0) + 0.25).clamp(0.6, 1.0);
        let (w, h) = (
            GLYPH_WIDTH * size * f.reference.chars().count() as f64,
            BOX_HEIGHT * size,
        );
        let (cx, cy) = ((e.x0 + e.x1) / 2.0, (e.y0 + e.y1) / 2.0);
        let spots = [
            (cx, e.y1 + GAP_MM + h / 2.0),
            (cx, e.y0 - GAP_MM - h / 2.0),
            (e.x1 + GAP_MM + w / 2.0, cy),
            (e.x0 - GAP_MM - w / 2.0, cy),
        ];
        let mut best: Option<(MmBox, (f64, f64), f64)> = None;
        for at in spots {
            let b = MmBox {
                x0: at.0 - w / 2.0,
                y0: at.1 - h / 2.0,
                x1: at.0 + w / 2.0,
                y1: at.1 + h / 2.0,
            };
            let area = occupied.overlap(&b);
            if best.is_none_or(|(_, _, a)| area < a - 1e-9) {
                best = Some((b, at, area));
            }
            if area <= 1e-9 {
                break;
            }
        }
        let Some((b, at, _)) = best else { continue };
        occupied.add(b);
        out.insert(
            f.id,
            Designator {
                at: Point::mm(at.0, at.1),
                size: mm(size),
            },
        );
    }
    out
}

/// `p` (board nm) in the package's own frame (nm), undoing the footprint's placement: the
/// offset, then its rotation, then the bottom-side mirror.
pub fn to_local(f: &Footprint, p: Point) -> Point {
    let (dx, dy) = ((p.x - f.at.x) as f64, (p.y - f.at.y) as f64);
    let r = (-f.rotation).to_radians();
    let (x, y) = (dx * r.cos() - dy * r.sin(), dx * r.sin() + dy * r.cos());
    let x = if f.side == crate::layout::Side::Bottom {
        -x
    } else {
        x
    };
    Point::new(x.round() as Coord, y.round() as Coord)
}
