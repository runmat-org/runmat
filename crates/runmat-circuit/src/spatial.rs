//! A uniform-grid spatial index over bounding boxes.
//!
//! Cells are keyed in an ordered map, so queries return keys in a deterministic order (sorted
//! and de-duplicated). Insert, remove and update are O(cells covered); a query is O(cells
//! covered + hits). The cell size should be near the typical item size plus the largest
//! clearance (1–2 mm on most boards). Items much larger than a cell (zone fills, outlines)
//! should be indexed by their edges instead.

use crate::geom::{Coord, Rect};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug)]
pub struct Grid<K: Ord + Copy> {
    cell: Coord,
    cells: BTreeMap<(i64, i64), Vec<K>>,
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
                self.cells.entry((x, y)).or_default().push(k);
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
                    v.retain(|e| *e != k);
                    if v.is_empty() {
                        self.cells.remove(&(x, y));
                    }
                }
            }
        }
        true
    }
    /// Keys whose boxes intersect `r`, sorted.
    pub fn query(&self, r: &Rect) -> Vec<K> {
        if r.is_empty() {
            return Vec::new();
        }
        let (x0, y0, x1, y1) = self.range(r);
        let mut out = BTreeSet::new();
        for x in x0..=x1 {
            for (_, v) in self.cells.range((x, y0)..=(x, y1)) {
                for k in v {
                    if self.boxes[k].intersects(r) {
                        out.insert(*k);
                    }
                }
            }
        }
        out.into_iter().collect()
    }
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.boxes.keys()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Point;

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
}
