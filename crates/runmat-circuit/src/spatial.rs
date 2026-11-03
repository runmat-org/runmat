//! A uniform-grid spatial index over bounding boxes.
//!
//! Each cell keeps its entries with their boxes inline, so a query never looks a key up in a
//! second map. An entry covering several cells is reported once per query: only from the cell
//! holding the lower-left corner of the intersection of its cell range with the query's
//! (the reference-cell rule), so queries need no de-duplication set.
//!
//! Cells are keyed in an ordered map and [`Grid::query`] returns keys sorted, so results are
//! deterministic. Insert, remove and update are O(cells covered); a query is O(cells covered +
//! hits). The cell size should be near the typical item size plus the largest clearance
//! (1–2 mm on most boards). Items much larger than a cell (zone fills, outlines) should be
//! indexed by their edges instead.

use crate::geom::poly::Ring;
use crate::geom::shape::Prim;
use crate::geom::{Coord, Point, Rect};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct Grid<K: Ord + Copy> {
    cell: Coord,
    cells: BTreeMap<(i64, i64), Vec<(K, Rect)>>,
    boxes: BTreeMap<K, Rect>,
}

impl<K: Ord + Copy> Grid<K> {
    pub fn new(cell: Coord) -> Self {
        Grid {
            cell: cell.max(1),
            cells: BTreeMap::new(),
            boxes: BTreeMap::new(),
        }
    }
    fn range(&self, r: &Rect) -> (i64, i64, i64, i64) {
        (
            r.min.x.div_euclid(self.cell),
            r.min.y.div_euclid(self.cell),
            r.max.x.div_euclid(self.cell),
            r.max.y.div_euclid(self.cell),
        )
    }
    pub fn len(&self) -> usize {
        self.boxes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.boxes.is_empty()
    }
    pub fn clear(&mut self) {
        self.cells.clear();
        self.boxes.clear();
    }
    pub fn bbox(&self, k: K) -> Option<Rect> {
        self.boxes.get(&k).copied()
    }
    pub fn insert(&mut self, k: K, r: Rect) {
        if r.is_empty() {
            return;
        }
        self.remove(k);
        let (x0, y0, x1, y1) = self.range(&r);
        for x in x0..=x1 {
            for y in y0..=y1 {
                self.cells.entry((x, y)).or_default().push((k, r));
            }
        }
        self.boxes.insert(k, r);
    }
    pub fn remove(&mut self, k: K) -> bool {
        let Some(r) = self.boxes.remove(&k) else {
            return false;
        };
        let (x0, y0, x1, y1) = self.range(&r);
        for x in x0..=x1 {
            for y in y0..=y1 {
                if let Some(v) = self.cells.get_mut(&(x, y)) {
                    v.retain(|e| e.0 != k);
                    if v.is_empty() {
                        self.cells.remove(&(x, y));
                    }
                }
            }
        }
        true
    }
    /// Calls `f` once for every entry whose box intersects `r`, in an order fixed by the
    /// grid's contents (not sorted).
    pub fn visit(&self, r: &Rect, mut f: impl FnMut(K, &Rect)) {
        if r.is_empty() {
            return;
        }
        let (x0, y0, x1, y1) = self.range(r);
        for x in x0..=x1 {
            for (&(cx, cy), v) in self.cells.range((x, y0)..=(x, y1)) {
                for (k, b) in v {
                    if !b.intersects(r) {
                        continue;
                    }
                    // Reference cell: the first cell both ranges share.
                    let (bx, by) = (b.min.x.div_euclid(self.cell), b.min.y.div_euclid(self.cell));
                    if cx == bx.max(x0) && cy == by.max(y0) {
                        f(*k, b);
                    }
                }
            }
        }
    }
    /// Keys whose boxes intersect `r`, sorted.
    pub fn query(&self, r: &Rect) -> Vec<K> {
        let mut out = Vec::new();
        self.visit(r, |k, _| out.push(k));
        out.sort_unstable();
        out
    }
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.boxes.keys()
    }
}

/// The edges of a large filled ring set (a zone fill, the board outline) indexed for the two
/// queries the DRC and connectivity need: edges near a box, and point containment.
///
/// Edges sit in a [`Grid`] for proximity. For containment each edge is also listed in every
/// horizontal band of height `cell` its `y` range overlaps; the non-zero winding number of a
/// point only needs the edges crossing its horizontal line, which all lie in the point's band.
/// So a containment query costs the edges of one band, not of the whole set.
#[derive(Clone, Debug)]
pub struct EdgeIndex {
    edges: Vec<(Point, Point)>,
    grid: Grid<u32>,
    cell: Coord,
    bands: BTreeMap<i64, Vec<u32>>,
    bbox: Rect,
}

impl EdgeIndex {
    /// Indexes the rings (outers counter-clockwise, holes clockwise, filled by non-zero).
    pub fn new<'a>(rings: impl IntoIterator<Item = &'a Ring>, cell: Coord) -> EdgeIndex {
        let cell = cell.max(1);
        let mut idx = EdgeIndex {
            edges: Vec::new(),
            grid: Grid::new(cell),
            cell,
            bands: BTreeMap::new(),
            bbox: Rect::EMPTY,
        };
        for r in rings {
            let m = r.len();
            if m < 3 {
                continue;
            }
            for i in 0..m {
                let (a, b) = (r[i], r[(i + 1) % m]);
                if a == b {
                    continue;
                }
                let k = idx.edges.len() as u32;
                let bb = Rect::new(a, b);
                idx.grid.insert(k, bb);
                for band in bb.min.y.div_euclid(cell)..=bb.max.y.div_euclid(cell) {
                    idx.bands.entry(band).or_default().push(k);
                }
                idx.bbox = idx.bbox.union(bb);
                idx.edges.push((a, b));
            }
        }
        idx
    }
    pub fn bbox(&self) -> Rect {
        self.bbox
    }
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
    /// Calls `f` for every edge whose box meets `r`.
    pub fn visit_near(&self, r: &Rect, mut f: impl FnMut(Point, Point)) {
        self.grid.visit(r, |k, _| {
            let (a, b) = self.edges[k as usize];
            f(a, b)
        });
    }
    /// Whether any edge's box meets `r`.
    pub fn any_near(&self, r: &Rect) -> bool {
        let mut hit = false;
        self.grid.visit(r, |_, _| hit = true);
        hit
    }
    /// `Some(true)` inside, `Some(false)` outside, `None` on an edge.
    pub fn contains(&self, p: Point) -> Option<bool> {
        if !self.bbox.contains(p) {
            return Some(false);
        }
        let Some(band) = self.bands.get(&p.y.div_euclid(self.cell)) else {
            return Some(false);
        };
        let mut w = 0i64;
        for &k in band {
            let (a, b) = self.edges[k as usize];
            if crate::geom::shape::point_on_segment(p, a, b) {
                return None;
            }
            if a.y <= p.y {
                if b.y > p.y && crate::geom::orient(a, b, p) > 0 {
                    w += 1;
                }
            } else if b.y <= p.y && crate::geom::orient(a, b, p) < 0 {
                w -= 1;
            }
        }
        Some(w != 0)
    }
    /// Smallest gap from `prim` to the set's boundary, searching only within `radius` of it;
    /// `f64::INFINITY` when no edge is that close.
    pub fn boundary_gap(&self, prim: &Prim, radius: Coord) -> f64 {
        self.probe(prim, radius).gap
    }
    /// Whether `prim` overlaps or touches the filled set.
    pub fn touches(&self, prim: &Prim) -> bool {
        let p = self.probe(prim, 1);
        p.gap <= 0.0 || p.inside == Some(true)
    }
    /// Gap from `prim` to the boundary (within `radius`) and, when the primitive does not
    /// cross the boundary, whether it lies inside the set.
    ///
    /// Inside/outside comes from the nearest edge when that is decisive: if the point's
    /// closest boundary point is interior to an edge, the point is inside exactly when it is
    /// left of that edge (interior is on the left of every ring edge). Otherwise, or when the
    /// nearest edge might lie outside the searched box, the band winding test decides.
    pub fn probe(&self, prim: &Prim, radius: Coord) -> Probe {
        let pt = match &prim.core {
            crate::geom::shape::Core::Point(p) => *p,
            crate::geom::shape::Core::Segment(a, _) => *a,
            crate::geom::shape::Core::Poly(rs) => match rs.first().and_then(|r| r.first()) {
                Some(p) => *p,
                None => {
                    return Probe {
                        gap: f64::INFINITY,
                        inside: Some(false),
                    }
                }
            },
        };
        let bx = prim.bbox().inflate(radius.max(0));
        let mut gap = f64::INFINITY;
        let mut near: Option<(f64, Point, Point)> = None;
        if self.bbox.intersects(&bx) {
            self.visit_near(&bx, |a, b| {
                gap = gap.min(prim.gap(&Prim::segment(a, b, 0)));
                let d = crate::geom::shape::point_segment_distance(pt, a, b);
                if near.is_none_or(|n| d < n.0) {
                    near = Some((d, a, b));
                }
            });
        }
        if gap <= 0.0 {
            return Probe { gap, inside: None };
        }
        let margin = (pt.x - bx.min.x)
            .min(bx.max.x - pt.x)
            .min(pt.y - bx.min.y)
            .min(bx.max.y - pt.y) as f64;
        if let Some((d, a, b)) = near {
            let interior = (pt - a).dot(b - a) > 0 && (pt - b).dot(a - b) > 0;
            let o = crate::geom::orient(a, b, pt);
            if d <= margin && interior && o != 0 {
                return Probe {
                    gap,
                    inside: Some(o > 0),
                };
            }
        }
        Probe {
            gap,
            inside: self.contains(pt),
        }
    }
}

/// Result of [`EdgeIndex::probe`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Probe {
    /// Gap to the boundary, `f64::INFINITY` when no edge is within the search radius.
    pub gap: f64,
    /// Inside the set (`None` when the primitive crosses or touches the boundary).
    pub inside: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_query_remove() {
        let mut g: Grid<u32> = Grid::new(1000);
        for i in 0..100u32 {
            let x = (i % 10) as i64 * 1500;
            let y = (i / 10) as i64 * 1500;
            g.insert(i, Rect::new(Point::new(x, y), Point::new(x + 500, y + 500)));
        }
        let hits = g.query(&Rect::new(Point::new(0, 0), Point::new(1600, 1600)));
        assert_eq!(hits, vec![0, 1, 10, 11]);
        g.remove(11);
        let hits = g.query(&Rect::new(Point::new(0, 0), Point::new(1600, 1600)));
        assert_eq!(hits, vec![0, 1, 10]);
        g.insert(
            0,
            Rect::new(Point::new(-5000, -5000), Point::new(-4000, -4000)),
        );
        let hits = g.query(&Rect::new(Point::new(0, 0), Point::new(1600, 1600)));
        assert_eq!(hits, vec![1, 10]);
        assert_eq!(g.len(), 99);
    }

    #[test]
    fn edge_index_contains_and_near() {
        use crate::geom::poly::{boolean, rect_ring, BoolOp};
        let outer = rect_ring(Rect::new(Point::new(0, 0), Point::new(10_000, 10_000)));
        let hole = rect_ring(Rect::new(Point::new(3000, 3000), Point::new(6000, 6000)));
        let set = boolean(&[outer], &[hole], BoolOp::Difference);
        let rings = set.rings();
        let idx = EdgeIndex::new(rings.iter(), 1000);
        for (p, want) in [
            (Point::new(1000, 1000), Some(true)),
            (Point::new(4500, 4500), Some(false)),
            (Point::new(20_000, 5000), Some(false)),
            (Point::new(0, 5000), None),
            (Point::new(3000, 4000), None),
            (Point::new(9999, 9999), Some(true)),
        ] {
            assert_eq!(idx.contains(p), want, "{p:?}");
            assert_eq!(idx.contains(p), set.contains(p), "{p:?}");
        }
        assert!(idx.touches(&Prim::circle(Point::new(4500, 4500), 1600)));
        assert!(!idx.touches(&Prim::circle(Point::new(4500, 4500), 1400)));
        assert!(idx.touches(&Prim::circle(Point::new(1000, 1000), 10)));
        assert_eq!(
            idx.boundary_gap(&Prim::circle(Point::new(4500, 4500), 1000), 1000),
            500.0
        );
    }
}
