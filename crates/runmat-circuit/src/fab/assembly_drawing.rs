//! The assembly drawing: one sheet per side with the board outline, each fitted part's outline
//! (its courtyard, else its pads' extent) and its designator in the middle, and a title. The
//! bottom side is drawn as seen from below (mirrored left to right about the board's centre),
//! the way an assembler holds the board.

use super::drawing::{text_width, Drawing};
use crate::geom::{mm, Coord, Point, Rect};
use crate::layout::{Board, Side};

const PEN: Coord = 100_000;
const THIN: Coord = 50_000;

/// The drawing of each side that has parts: `("top", …)` and `("bottom", …)`.
pub fn assembly_drawings(board: &Board) -> Vec<(&'static str, Drawing)> {
    [(Side::Top, "top"), (Side::Bottom, "bottom")]
        .into_iter()
        .filter(|(side, _)| board.footprints.values().any(|f| f.side == *side && f.fitted))
        .map(|(side, name)| (name, assembly_drawing(board, side)))
        .collect()
}

/// One side's assembly drawing.
pub fn assembly_drawing(board: &Board, side: Side) -> Drawing {
    let bb = board.outline.bbox();
    let cx = bb.min.x + bb.max.x;
    let flip = |p: Point| if side == Side::Bottom { Point::new(cx - p.x, p.y) } else { p };
    let mut d = Drawing { title: format!("{} assembly drawing, {} side", board.name, if side == Side::Top { "top" } else { "bottom" }), marks: Vec::new() };
    for r in board.outline.rings() {
        let mut pts: Vec<Point> = r.iter().map(|p| flip(*p)).collect();
        pts.push(pts[0]);
        d.line(pts, PEN);
    }
    let copper = board.copper_count();
    for f in board.footprints.values().filter(|f| f.side == side && f.fitted) {
        let rings = f.courtyard();
        let extent = if rings.is_empty() {
            let pads = f.pads(copper);
            let r = pads.iter().fold(Rect::new(f.at, f.at), |r, p| r.union(p.prim.bbox()));
            d.rect(Rect::new(flip(r.min), flip(r.max)), THIN);
            r
        } else {
            let mut r = Rect::new(rings[0][0], rings[0][0]);
            for ring in &rings {
                let mut pts: Vec<Point> = ring.iter().map(|p| flip(*p)).collect();
                pts.push(pts[0]);
                for p in ring {
                    r = r.include(*p);
                }
                d.line(pts, THIN);
            }
            r
        };
        // The designator fits inside the outline: up to 1 mm, down to 0.4 mm.
        let span = extent.width().min(extent.height() * 3);
        let mut size = mm(1.0);
        while size > mm(0.4) && text_width(&f.reference, size) > span {
            size -= mm(0.1);
        }
        let c = flip(f.at);
        d.text(Point::new(c.x - text_width(&f.reference, size) / 2, c.y - size / 2), size, f.reference.clone());
    }
    let size = mm(1.5);
    d.text(Point::new(bb.min.x, bb.min.y - mm(4.0)), size, d.title.clone());
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_side_with_parts_draws_its_designators() {
        let outline = crate::geom::poly::Polygon::from_ring(vec![Point::ORIGIN, Point::new(mm(20.0), 0), Point::new(mm(20.0), mm(10.0)), Point::new(0, mm(10.0))]);
        let mut board = Board::new("t", outline, crate::layout::Stackup::standard(2, mm(1.6), 1.0, 0.5));
        let r0402 = crate::library::footprint::from_string("R0402").unwrap();
        board.place("R1", r0402.clone(), Point::new(mm(5.0), mm(5.0)), 0.0, Side::Top);
        board.place("C7", r0402, Point::new(mm(5.0), mm(5.0)), 90.0, Side::Bottom);
        let drawings = assembly_drawings(&board);
        assert_eq!(drawings.iter().map(|(s, _)| *s).collect::<Vec<_>>(), ["top", "bottom"]);
        let texts = |d: &Drawing| -> Vec<String> {
            d.marks.iter().filter_map(|m| match m {
                super::super::drawing::Mark::Text { text, .. } => Some(text.clone()),
                _ => None,
            }).collect()
        };
        assert!(texts(&drawings[0].1).contains(&"R1".to_string()));
        assert!(texts(&drawings[1].1).contains(&"C7".to_string()));
        // Seen from below, a part 5 mm from the left edge is 15 mm from it.
        let c7 = drawings[1].1.marks.iter().find_map(|m| match m {
            super::super::drawing::Mark::Text { text, at, .. } if text == "C7" => Some(*at),
            _ => None,
        }).unwrap();
        assert!(c7.x > mm(13.0), "{c7:?}");
        assert!(drawings[0].1.gerber(&board, "AssemblyDrawing,Top").contains("%TF.FileFunction,AssemblyDrawing,Top*%"));
    }
}
