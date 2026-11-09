//! Where a legend box sits inside its axes, from MATLAB's `Location` names.
//!
//! The eight compass locations put the box against that side or corner of the plot area.
//! `best` puts it where it covers the least drawn data, trying the corners, the side
//! midpoints and a grid of positions between them. The data is an occupancy grid of the
//! plot area: a line marks every cell along its segments and a bar or patch every cell it
//! fills, so the box avoids a curve between its samples and a bar's face, not only their
//! vertices. Ties go to the earlier candidate, starting with `northeast`, so a plot with
//! free space at its upper right keeps MATLAB's default place. Unknown names and the
//! `*outside` locations fall back to `northeast` inside the axes.

use egui::{Pos2, Rect, Vec2};

/// The occupancy cell's side in screen points.
const CELL: f32 = 3.0;

/// The drawn data over a plot area, as filled cells of a grid.
pub struct Obstacles {
    origin: Pos2,
    cols: usize,
    rows: usize,
    filled: Vec<bool>,
}

impl Obstacles {
    pub fn new(plot: Rect) -> Self {
        let cols = (plot.width() / CELL).ceil().max(1.0) as usize;
        let rows = (plot.height() / CELL).ceil().max(1.0) as usize;
        Self {
            origin: plot.min,
            cols,
            rows,
            filled: vec![false; cols * rows],
        }
    }

    #[cfg(test)]
    pub fn from_points(plot: Rect, points: &[Pos2]) -> Self {
        let mut o = Self::new(plot);
        for p in points {
            o.point(*p);
        }
        o
    }

    fn cell(&self, p: Pos2) -> Option<(usize, usize)> {
        let (cx, cy) = ((p.x - self.origin.x) / CELL, (p.y - self.origin.y) / CELL);
        (cx >= 0.0 && cy >= 0.0 && (cx as usize) < self.cols && (cy as usize) < self.rows)
            .then(|| (cx as usize, cy as usize))
    }

    pub fn point(&mut self, p: Pos2) {
        if let Some((x, y)) = self.cell(p) {
            self.filled[y * self.cols + x] = true;
        }
    }

    fn bounds(&self) -> Rect {
        Rect::from_min_size(
            self.origin,
            Vec2::new(self.cols as f32 * CELL, self.rows as f32 * CELL),
        )
    }

    /// Marks every cell the segment from `a` to `b` crosses inside the plot area.
    pub fn segment(&mut self, a: Pos2, b: Pos2) {
        let Some((a, b)) = clip(a, b, self.bounds()) else {
            return;
        };
        let steps = ((b - a).length() / (0.5 * CELL)).ceil().max(1.0) as usize;
        for k in 0..=steps {
            self.point(a + (b - a) * (k as f32 / steps as f32));
        }
    }

    /// Marks the triangle's edges and every cell whose center lies inside it.
    pub fn triangle(&mut self, a: Pos2, b: Pos2, c: Pos2) {
        self.segment(a, b);
        self.segment(b, c);
        self.segment(c, a);
        let area = (b - a).x * (c - a).y - (b - a).y * (c - a).x;
        if area.abs() < 1e-3 {
            return;
        }
        let lo = |v: f32, o: f32, n: usize| (((v - o) / CELL).floor().max(0.0) as usize).min(n);
        let hi = |v: f32, o: f32, n: usize| (((v - o) / CELL).ceil().max(0.0) as usize).min(n);
        let (x0, x1) = (
            lo(a.x.min(b.x).min(c.x), self.origin.x, self.cols),
            hi(a.x.max(b.x).max(c.x), self.origin.x, self.cols),
        );
        let (y0, y1) = (
            lo(a.y.min(b.y).min(c.y), self.origin.y, self.rows),
            hi(a.y.max(b.y).max(c.y), self.origin.y, self.rows),
        );
        let edge = |p: Pos2, q: Pos2, r: Pos2| (q - p).x * (r - p).y - (q - p).y * (r - p).x;
        for y in y0..y1 {
            for x in x0..x1 {
                let p = Pos2::new(
                    self.origin.x + (x as f32 + 0.5) * CELL,
                    self.origin.y + (y as f32 + 0.5) * CELL,
                );
                let (w0, w1, w2) = (edge(b, c, p), edge(c, a, p), edge(a, b, p));
                if (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) || (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0) {
                    self.filled[y * self.cols + x] = true;
                }
            }
        }
    }

    /// Summed-area table of the filled cells, `(cols + 1) x (rows + 1)`.
    fn summed(&self) -> Vec<u32> {
        let w = self.cols + 1;
        let mut s = vec![0u32; w * (self.rows + 1)];
        for y in 0..self.rows {
            for x in 0..self.cols {
                s[(y + 1) * w + x + 1] = u32::from(self.filled[y * self.cols + x])
                    + s[y * w + x + 1]
                    + s[(y + 1) * w + x]
                    - s[y * w + x];
            }
        }
        s
    }
}

/// The part of the segment `a`–`b` inside `r` (Liang–Barsky), if any.
fn clip(a: Pos2, b: Pos2, r: Rect) -> Option<(Pos2, Pos2)> {
    let d = b - a;
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for (p, q) in [
        (-d.x, a.x - r.min.x),
        (d.x, r.max.x - a.x),
        (-d.y, a.y - r.min.y),
        (d.y, r.max.y - a.y),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let t = q / p;
            if p < 0.0 {
                t0 = t0.max(t);
            } else {
                t1 = t1.min(t);
            }
            if t0 > t1 {
                return None;
            }
        }
    }
    Some((a + d * t0, a + d * t1))
}

/// The box's top-left corner for a legend of `size` in `plot`, `margin` in from the sides.
/// Only `best` reads `obstacles`.
pub fn legend_origin(
    plot: Rect,
    size: Vec2,
    margin: f32,
    location: Option<&str>,
    obstacles: &Obstacles,
) -> Pos2 {
    let location = location.map(|s| s.trim().to_ascii_lowercase());
    let at = |fx: f32, fy: f32| -> Pos2 {
        let free_x = (plot.width() - size.x - 2.0 * margin).max(0.0);
        let free_y = (plot.height() - size.y - 2.0 * margin).max(0.0);
        Pos2::new(
            plot.min.x + margin + fx * free_x,
            plot.min.y + margin + fy * free_y,
        )
    };
    match location.as_deref() {
        Some("northwest") | Some("northwestoutside") => at(0.0, 0.0),
        Some("north") | Some("northoutside") => at(0.5, 0.0),
        Some("southeast") | Some("southeastoutside") => at(1.0, 1.0),
        Some("southwest") | Some("southwestoutside") => at(0.0, 1.0),
        Some("south") | Some("southoutside") => at(0.5, 1.0),
        Some("east") | Some("eastoutside") => at(1.0, 0.5),
        Some("west") | Some("westoutside") => at(0.0, 0.5),
        Some("best") | Some("bestoutside") => best(obstacles, size, &at),
        _ => at(1.0, 0.0),
    }
}

/// Candidates in preference order: the four corners (MATLAB's default first), the side
/// midpoints, then a 9 x 9 grid between them.
fn best(obstacles: &Obstacles, size: Vec2, at: &dyn Fn(f32, f32) -> Pos2) -> Pos2 {
    let mut candidates = vec![
        (1.0, 0.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (0.5, 0.0),
        (0.5, 1.0),
        (1.0, 0.5),
        (0.0, 0.5),
    ];
    for i in 0..9 {
        for j in 0..9 {
            candidates.push((i as f32 / 8.0, j as f32 / 8.0));
        }
    }
    let summed = obstacles.summed();
    let w = obstacles.cols + 1;
    let covered = |origin: Pos2| -> u32 {
        let rect = Rect::from_min_size(origin, size).expand(2.0);
        let col = |x: f32| {
            (((x - obstacles.origin.x) / CELL).round().max(0.0) as usize).min(obstacles.cols)
        };
        let row = |y: f32| {
            (((y - obstacles.origin.y) / CELL).round().max(0.0) as usize).min(obstacles.rows)
        };
        let (x0, x1, y0, y1) = (
            col(rect.min.x),
            col(rect.max.x),
            row(rect.min.y),
            row(rect.max.y),
        );
        summed[y1 * w + x1] + summed[y0 * w + x0] - summed[y0 * w + x1] - summed[y1 * w + x0]
    };
    let mut chosen = at(1.0, 0.0);
    let mut fewest = u32::MAX;
    for (fx, fy) in candidates {
        let origin = at(fx, fy);
        let n = covered(origin);
        if n < fewest {
            fewest = n;
            chosen = origin;
            if n == 0 {
                break;
            }
        }
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plot() -> Rect {
        Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(400.0, 300.0))
    }

    #[test]
    fn compass_locations_sit_against_their_sides() {
        let size = Vec2::new(100.0, 40.0);
        let none = Obstacles::new(plot());
        assert_eq!(
            legend_origin(plot(), size, 8.0, None, &none),
            Pos2::new(292.0, 8.0)
        );
        assert_eq!(
            legend_origin(plot(), size, 8.0, Some("southwest"), &none),
            Pos2::new(8.0, 252.0)
        );
        assert_eq!(
            legend_origin(plot(), size, 8.0, Some("North"), &none),
            Pos2::new(150.0, 8.0)
        );
    }

    #[test]
    fn best_keeps_northeast_when_it_is_free_and_avoids_the_data_otherwise() {
        let size = Vec2::new(100.0, 40.0);
        // A falling line along the bottom: the upper right is free.
        let low: Vec<Pos2> = (0..100)
            .map(|k| Pos2::new(k as f32 * 4.0, 250.0 + k as f32 * 0.4))
            .collect();
        assert_eq!(
            legend_origin(
                plot(),
                size,
                8.0,
                Some("best"),
                &Obstacles::from_points(plot(), &low)
            ),
            Pos2::new(292.0, 8.0)
        );
        // A response flat across the top, as a magnitude plot's passband: the box drops below it.
        let high: Vec<Pos2> = (0..100)
            .map(|k| {
                Pos2::new(
                    k as f32 * 4.0,
                    20.0 + if k > 70 { (k - 70) as f32 * 8.0 } else { 0.0 },
                )
            })
            .collect();
        let origin = legend_origin(
            plot(),
            size,
            8.0,
            Some("best"),
            &Obstacles::from_points(plot(), &high),
        );
        let rect = Rect::from_min_size(origin, size);
        assert!(
            high.iter().all(|p| !rect.contains(*p)),
            "the legend at {origin:?} covers data"
        );
    }

    /// Two-point lines (spec limits drawn as `plot([x x], [y0 y1])`) have no vertex inside the
    /// box, but their segments cross it: the box must avoid the segments.
    #[test]
    fn best_avoids_a_line_between_its_vertices() {
        let size = Vec2::new(100.0, 40.0);
        let mut o = Obstacles::new(plot());
        // Vertical lines from top to bottom at x = 60, 200 and 340: every corner box is crossed.
        for x in [60.0, 200.0, 340.0] {
            o.segment(Pos2::new(x, 0.0), Pos2::new(x, 300.0));
        }
        let origin = legend_origin(plot(), size, 8.0, Some("best"), &o);
        let rect = Rect::from_min_size(origin, size).expand(2.0);
        for x in [60.0, 200.0, 340.0] {
            assert!(
                !(rect.min.x..=rect.max.x).contains(&x),
                "the legend at {origin:?} crosses the line at x = {x}"
            );
        }
    }

    /// A histogram's bars are triangles: the box goes over the short bars, not across a tall
    /// bar's face, though no bar has a vertex inside any corner box.
    #[test]
    fn best_avoids_the_faces_of_filled_bars() {
        let size = Vec2::new(120.0, 60.0);
        let mut o = Obstacles::new(plot());
        // Ten bars from the bottom: the outer two on each side reach y = 10, the middle six y = 250.
        for k in 0..10 {
            let (x0, x1) = (k as f32 * 40.0, k as f32 * 40.0 + 36.0);
            let top = if (2..8).contains(&k) { 250.0 } else { 10.0 };
            let (a, b, c, d) = (
                Pos2::new(x0, 300.0),
                Pos2::new(x1, 300.0),
                Pos2::new(x1, top),
                Pos2::new(x0, top),
            );
            o.triangle(a, b, c);
            o.triangle(a, c, d);
        }
        let origin = legend_origin(plot(), size, 8.0, Some("best"), &o);
        let rect = Rect::from_min_size(origin, size);
        assert!(
            rect.min.x >= 76.0 && rect.max.x <= 320.0 && rect.max.y <= 250.0,
            "the legend at {origin:?} covers a bar"
        );
    }

    #[test]
    fn clip_keeps_the_part_inside() {
        let (a, b) = clip(Pos2::new(-100.0, 150.0), Pos2::new(500.0, 150.0), plot()).unwrap();
        assert_eq!((a, b), (Pos2::new(0.0, 150.0), Pos2::new(400.0, 150.0)));
        assert!(clip(Pos2::new(-10.0, -10.0), Pos2::new(-5.0, 500.0), plot()).is_none());
    }
}
