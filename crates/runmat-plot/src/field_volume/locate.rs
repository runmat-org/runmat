//! Point location in a tetrahedral mesh: a uniform bucket grid over the elements' bounding
//! boxes, the previous hit tried first (consecutive points along a streamline stay in or near
//! it), and barycentric coordinates from each straight-sided tetrahedron's inverse edge matrix.

use super::TetMesh;

const BARYCENTRIC_TOLERANCE: f64 = 1.0e-9;

/// A point inside an element: the element and the point's barycentric coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Located {
    pub element: usize,
    pub weights: [f64; 4],
}

pub struct TetrahedronLocator<'a> {
    mesh: &'a TetMesh,
    /// Row-major inverse of [p1-p0, p2-p0, p3-p0] per element; `None` when degenerate.
    inverses: Vec<Option<[f64; 9]>>,
    origin: [f64; 3],
    cell: [f64; 3],
    dims: [usize; 3],
    buckets: Vec<Vec<u32>>,
    last: std::cell::Cell<Option<usize>>,
}

impl<'a> TetrahedronLocator<'a> {
    pub fn new(mesh: &'a TetMesh) -> Self {
        let (min, max) = mesh.bounds();
        let count = mesh.element_count().max(1);
        let per_axis = ((count as f64).cbrt().ceil() as usize).clamp(1, 128);
        let dims = [per_axis; 3];
        let cell = [0, 1, 2].map(|a| (max[a] - min[a]).max(1.0e-12) / per_axis as f64);
        let mut buckets = vec![Vec::new(); per_axis * per_axis * per_axis];
        let mut inverses = Vec::with_capacity(mesh.element_count());
        for element in 0..mesh.element_count() {
            let p = mesh.corners(element);
            inverses.push(invert_edges(p));
            let lo = [0, 1, 2].map(|a| p.iter().map(|q| q[a]).fold(f64::INFINITY, f64::min));
            let hi = [0, 1, 2].map(|a| p.iter().map(|q| q[a]).fold(f64::NEG_INFINITY, f64::max));
            let c0 = cell_of(lo, min, cell, dims);
            let c1 = cell_of(hi, min, cell, dims);
            for i in c0[0]..=c1[0] {
                for j in c0[1]..=c1[1] {
                    for k in c0[2]..=c1[2] {
                        buckets[(i * dims[1] + j) * dims[2] + k].push(element as u32);
                    }
                }
            }
        }
        Self { mesh, inverses, origin: min, cell, dims, buckets, last: std::cell::Cell::new(None) }
    }

    pub fn mesh(&self) -> &TetMesh {
        self.mesh
    }

    /// The element containing `point`, or `None` outside the mesh.
    pub fn locate(&self, point: [f64; 3]) -> Option<Located> {
        if let Some(previous) = self.last.get() {
            if let Some(weights) = self.weights(previous, point) {
                return Some(Located { element: previous, weights });
            }
        }
        let inside = (0..3).all(|a| {
            let slack = self.cell[a] * 1.0e-9;
            point[a] >= self.origin[a] - slack && point[a] <= self.origin[a] + self.cell[a] * self.dims[a] as f64 + slack
        });
        if !inside {
            return None;
        }
        let c = cell_of(point, self.origin, self.cell, self.dims);
        for &element in &self.buckets[(c[0] * self.dims[1] + c[1]) * self.dims[2] + c[2]] {
            if let Some(weights) = self.weights(element as usize, point) {
                self.last.set(Some(element as usize));
                return Some(Located { element: element as usize, weights });
            }
        }
        None
    }

    fn weights(&self, element: usize, q: [f64; 3]) -> Option<[f64; 4]> {
        let inverse = self.inverses.get(element)?.as_ref()?;
        let p0 = self.mesh.corners(element)[0];
        let d = [q[0] - p0[0], q[1] - p0[1], q[2] - p0[2]];
        let l1 = inverse[0] * d[0] + inverse[1] * d[1] + inverse[2] * d[2];
        let l2 = inverse[3] * d[0] + inverse[4] * d[1] + inverse[5] * d[2];
        let l3 = inverse[6] * d[0] + inverse[7] * d[1] + inverse[8] * d[2];
        let all = [1.0 - l1 - l2 - l3, l1, l2, l3];
        if !all.iter().all(|&l| l >= -BARYCENTRIC_TOLERANCE) {
            return None;
        }
        let clamped = all.map(|l| l.max(0.0));
        let total: f64 = clamped.iter().sum();
        Some(clamped.map(|l| l / total))
    }
}

fn cell_of(q: [f64; 3], origin: [f64; 3], cell: [f64; 3], dims: [usize; 3]) -> [usize; 3] {
    [0, 1, 2].map(|a| (((q[a] - origin[a]) / cell[a]).floor().max(0.0) as usize).min(dims[a] - 1))
}

/// Inverse of the 3x3 matrix whose columns are the edges from corner 0.
fn invert_edges(p: [[f64; 3]; 4]) -> Option<[f64; 9]> {
    let e = [0, 1, 2].map(|k| [p[k + 1][0] - p[0][0], p[k + 1][1] - p[0][1], p[k + 1][2] - p[0][2]]);
    let m = [e[0][0], e[1][0], e[2][0], e[0][1], e[1][1], e[2][1], e[0][2], e[1][2], e[2][2]];
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    let scale = e.iter().map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()).product::<f64>();
    if !det.is_finite() || det.abs() <= 1.0e-12 * scale.max(f64::MIN_POSITIVE) {
        return None;
    }
    let inv = 1.0 / det;
    Some([
        (m[4] * m[8] - m[5] * m[7]) * inv,
        (m[2] * m[7] - m[1] * m[8]) * inv,
        (m[1] * m[5] - m[2] * m[4]) * inv,
        (m[5] * m[6] - m[3] * m[8]) * inv,
        (m[0] * m[8] - m[2] * m[6]) * inv,
        (m[2] * m[3] - m[0] * m[5]) * inv,
        (m[3] * m[7] - m[4] * m[6]) * inv,
        (m[1] * m[6] - m[0] * m[7]) * inv,
        (m[0] * m[4] - m[1] * m[3]) * inv,
    ])
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::super::TetOrder;
    use super::*;

    #[test]
    fn locates_interior_points_with_weights_that_rebuild_them() {
        let mesh = unit_cube(TetOrder::Linear);
        let locator = TetrahedronLocator::new(&mesh);
        for point in [[0.2, 0.3, 0.4], [0.9, 0.1, 0.5], [0.5, 0.5, 0.5], [0.01, 0.99, 0.02]] {
            let located = locator.locate(point).expect("inside");
            let rebuilt = mesh.point(located.element, located.weights);
            for axis in 0..3 {
                assert!((rebuilt[axis] - point[axis]).abs() < 1e-12);
            }
        }
        assert!(locator.locate([1.5, 0.5, 0.5]).is_none());
        assert!(locator.locate([0.5, -0.1, 0.5]).is_none());
    }
}
