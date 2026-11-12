//! Recorded places on a page, and where the parts an edit adds go among them.
//!
//! - **Recorded places stick.** A symbol whose place the lock records stays exactly there.
//! - **New parts keep their company.** The automatic layout draws a new part where it belongs:
//!   a regulator in the power chain with its input and output caps in a row under it, a pull-up
//!   above its net. A part drawn with a recorded part (a cap added to an existing IC) keeps its
//!   automatic offset from that part's recorded place. New parts drawn together (a new IC and its
//!   new caps) move as one block, so the caps stay under the pins they serve.
//! - **The block lands by its neighbours.** It moves by as much as the recorded part nearest to
//!   it in the automatic layout, among those on its nets, was moved by hand; then to the nearest
//!   free spot, in whole grid steps, clear of every placed symbol's ports and texts, inside the
//!   page's frame and off its title block.

use super::place::{reach, Placed};
use super::{Cell, CellKind, NetInfo, Place, FRAMES, FRAME_MM, GRID, MARGIN, TITLE_W_MM};
use crate::geom::{Coord, Point};
use crate::ids::SchPlace;
use std::collections::BTreeMap;

type Rect = [Coord; 4];

/// Clearance kept round each symbol's reach when a new part looks for room.
const CLEAR: Coord = GRID;
/// How far the search for room goes: rings two grids apart.
const RINGS: Coord = 400;

fn nm(v: f64) -> Coord {
    (v * 1.0e6).round() as Coord
}

fn hits(a: &Rect, b: &Rect) -> bool {
    a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]
}

fn snap(v: Coord) -> Coord {
    (v as f64 / GRID as f64).round() as Coord * GRID
}

/// Recorded places override the automatic ones (stored in paper mm, y down from the top; the
/// content's top-left corner sits [`MARGIN`] in); unrecorded symbols on a page with recorded
/// ones go where the module notes say. `title_rows` is the title block's row count.
pub(super) fn stick(
    cells: &[Cell],
    nets: &[NetInfo],
    placed: &mut Placed,
    stored: &BTreeMap<String, SchPlace>,
    title_rows: f64,
) {
    let auto = placed.places.clone();
    let places = &mut placed.places;
    let mut fixed: Vec<usize> = Vec::new();
    for (i, c) in cells.iter().enumerate() {
        let Some(s) = stored.get(&c.key) else {
            continue;
        };
        let o = Point::new(s.x - MARGIN, MARGIN - s.y);
        places[i] = match c.kind {
            // A sheet symbol is recorded by its top-left corner.
            CellKind::Sheet { .. } => Place::at(Point::new(o.x + c.lay.half_w, o.y - c.lay.half_h)),
            _ => Place {
                origin: o,
                rotation: s.rotation,
                mirror: s.mirror,
            },
        };
        fixed.push(i);
    }
    if fixed.is_empty() {
        return;
    }
    let is_fixed: Vec<bool> = (0..cells.len()).map(|i| fixed.contains(&i)).collect();
    let rect_with = |i: usize, p: &Place, clear: Coord| {
        let e = reach(cells, nets, i, p);
        let o = p.origin;
        [
            o.x + e[0] - clear,
            o.y + e[1] - clear,
            o.x + e[2] + clear,
            o.y + e[3] + clear,
        ]
    };
    let rect = |i: usize, p: &Place| rect_with(i, p, CLEAR);
    let mut taken: Vec<Rect> = fixed.iter().map(|&i| rect(i, &places[i])).collect();
    // The frame is the one the recorded symbols' bodies fit, so it never grows past the one the
    // page is drawn on.
    let bodies: Vec<Rect> = fixed
        .iter()
        .map(|&i| places[i].rect(cells[i].lay.extent()))
        .collect();
    let room = Room::of(&bodies, title_rows);
    // Each new part's block: the part it's drawn with when that is new too, followed up the
    // chain; a part drawn with a recorded one is a block of its own, anchored to it.
    let root = |mut i: usize| {
        let mut seen = 0;
        while let Some(h) = placed.attach[i].filter(|&h| !is_fixed[h] && h != i) {
            i = h;
            seen += 1;
            if seen > cells.len() {
                break;
            }
        }
        i
    };
    let mut blocks: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in (0..cells.len()).filter(|&i| !is_fixed[i]) {
        blocks.entry(root(i)).or_default().push(i);
    }
    for (r, members) in blocks {
        let shift = match placed.attach[r].filter(|&h| is_fixed[h]) {
            // Drawn with a recorded part: its automatic offset from that part.
            Some(h) => Point::new(
                places[h].origin.x - auto[h].origin.x,
                places[h].origin.y - auto[h].origin.y,
            ),
            None => moved_by(cells, &auto, places, &fixed, &members),
        };
        let start = Point::new(snap(shift.x), snap(shift.y));
        let at = |i: usize, d: Point| Place {
            origin: Point::new(auto[i].origin.x + d.x, auto[i].origin.y + d.y),
            ..auto[i]
        };
        // The block's spot nearest its start, in whole two-grid steps over every point of each
        // ring; first with the clearance round every symbol, then touching reach to reach.
        let mut best = None;
        'passes: for clear in [CLEAR, 0] {
            let fits = |d: Point| -> bool {
                members.iter().all(|&i| {
                    let r = rect_with(i, &at(i, d), clear);
                    room.holds(&r) && !taken.iter().any(|t| hits(&r, t))
                })
            };
            for ring in 0..RINGS {
                let mut spots: Vec<(Coord, Coord)> = Vec::new();
                for k in -ring..=ring {
                    spots.extend([(k, -ring), (k, ring), (-ring, k), (ring, k)]);
                }
                spots.sort_by_key(|&(x, y)| (x * x + y * y, -x, y));
                spots.dedup();
                for (x, y) in spots {
                    let d = Point::new(start.x + x * 2 * GRID, start.y + y * 2 * GRID);
                    if fits(d) {
                        best = Some(d);
                        break 'passes;
                    }
                }
                if room
                    .area
                    .is_some_and(|a| ring * 2 * GRID > (a[2] - a[0]) + (a[3] - a[1]))
                {
                    break;
                }
            }
        }
        // No room anywhere on the frame: the block keeps its first spot and the page grows.
        let d = best.unwrap_or(start);
        let rs: Vec<Rect> = members.iter().map(|&i| rect(i, &at(i, d))).collect();
        for &i in &members {
            places[i].origin = Point::new(auto[i].origin.x + d.x, auto[i].origin.y + d.y);
        }
        taken.extend(rs);
    }
}

/// How far the recorded part nearest a new block in the automatic layout, among those sharing
/// a net with it, was moved from its automatic place; nothing when none shares
/// one.
fn moved_by(
    cells: &[Cell],
    auto: &[Place],
    places: &[Place],
    fixed: &[usize],
    members: &[usize],
) -> Point {
    let centre = {
        let n = members.len() as Coord;
        let (x, y) = members.iter().fold((0, 0), |(x, y), &i| {
            (x + auto[i].origin.x, y + auto[i].origin.y)
        });
        Point::new(x / n, y / n)
    };
    let shares = |f: usize| {
        members.iter().any(|&m| {
            cells[m]
                .nets
                .iter()
                .flatten()
                .any(|k| cells[f].nets.contains(&Some(*k)))
        })
    };
    let dist = |f: usize| {
        let (dx, dy) = (auto[f].origin.x - centre.x, auto[f].origin.y - centre.y);
        (dx as i128) * (dx as i128) + (dy as i128) * (dy as i128)
    };
    fixed
        .iter()
        .copied()
        .filter(|&f| shares(f))
        .min_by_key(|&f| (dist(f), f))
        .map(|f| {
            Point::new(
                places[f].origin.x - auto[f].origin.x,
                places[f].origin.y - auto[f].origin.y,
            )
        })
        .unwrap_or(Point::ORIGIN)
}

/// Where new parts may go: inside the smallest frame the recorded symbols' bodies fit, off its title
/// block, in the page's content coordinates (top-left content corner at the origin, y up).
struct Room {
    area: Option<Rect>,
    title: Rect,
}

impl Room {
    fn of(bodies: &[Rect], title_rows: f64) -> Room {
        let title_h = super::title_height_mm(title_rows as usize);
        let frame = |w: f64, h: f64| {
            let area = [0, -(nm(h) - 2 * MARGIN), nm(w) - 2 * MARGIN, 0];
            let title = [
                nm(w - FRAME_MM - TITLE_W_MM - 2.0) - MARGIN,
                Coord::MIN / 4,
                Coord::MAX / 4,
                nm(FRAME_MM + title_h + 2.0) - nm(h) + MARGIN,
            ];
            Room {
                area: Some(area),
                title,
            }
        };
        for (_, w, h) in FRAMES {
            let r = frame(w, h);
            if bodies.iter().all(|b| r.holds(b)) {
                return r;
            }
        }
        // The recorded parts fit no frame: keep only off the largest frame's title block.
        let (_, w, h) = FRAMES[FRAMES.len() - 1];
        Room {
            area: None,
            ..frame(w, h)
        }
    }

    fn holds(&self, r: &Rect) -> bool {
        let inside = self
            .area
            .is_none_or(|a| r[0] >= a[0] && r[1] >= a[1] && r[2] <= a[2] && r[3] <= a[3]);
        inside && !hits(r, &self.title)
    }
}
