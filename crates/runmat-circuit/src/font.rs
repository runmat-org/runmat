//! Legend and copper text in fabrication outputs, set in the shared single-line font
//! ([`runmat_types::font`]).
//!
//! [`strokes`] lays out a string: `size` is the cap height, the advance is `size × 0.9`, lines
//! advance by `size × 1.6`, and the block is centred on the anchor (the usual default
//! justification for board text), then mirrored (bottom-side text) and rotated counter-clockwise.

use crate::geom::{Coord, Point};
use runmat_types::font::{layout, Anchor, CAP};

/// The text as polylines in board coordinates (see the module docs for the layout).
pub fn strokes(text: &str, at: Point, size: Coord, angle: f64, mirrored: bool) -> Vec<Vec<Point>> {
    let u = size as f64 / CAP;
    let (s, c) = angle.to_radians().sin_cos();
    layout(text, Anchor::Center)
        .into_iter()
        .map(|pl| {
            pl.iter()
                .map(|&(gx, gy)| {
                    let mut x = gx * u;
                    let y = gy * u;
                    if mirrored {
                        x = -x;
                    }
                    Point::from_f(at.x as f64 + x * c - y * s, at.y as f64 + x * s + y * c)
                })
                .collect()
        })
        .collect()
}

/// The default stroke width for text of cap height `size`: 0.15 of the height.
pub fn default_thickness(size: Coord) -> Coord {
    (size * 3 / 20).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::mm;

    #[test]
    fn layout_centres_rotates_and_mirrors() {
        // Cells are centred on the anchor (full-width glyphs fill their cells).
        let s = strokes("HM", Point::ORIGIN, mm(1.0), 0.0, false);
        let all: Vec<Point> = s.iter().flatten().copied().collect();
        let r = crate::geom::Rect::from_points(all.iter().copied());
        assert!(
            (r.center().x).abs() <= 2 && (r.center().y).abs() <= 2,
            "{r:?}"
        );
        assert!((r.height() - mm(1.0)).abs() <= 2);
        let m = strokes("HM", Point::ORIGIN, mm(1.0), 0.0, true);
        assert_eq!(m[0][0].x, -s[0][0].x);
        let rot = strokes("HM", Point::ORIGIN, mm(1.0), 90.0, false);
        let rr = crate::geom::Rect::from_points(rot.iter().flatten().copied());
        assert!((rr.width() - r.height()).abs() <= 2 && (rr.height() - r.width()).abs() <= 2);
        // Small capitals for lowercase.
        let lo = strokes("r", Point::ORIGIN, mm(1.0), 0.0, false);
        let lr = crate::geom::Rect::from_points(lo.iter().flatten().copied());
        assert!(lr.height() < mm(0.7));
    }
}
