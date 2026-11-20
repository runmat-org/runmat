//! Nearest ray hits on a preview surface through a uniform grid of triangle buckets.
use super::{cross, dot, sub, Triangle};

/// Grid cells at most, whatever the triangle count.
const MAXIMUM_CELLS: usize = 2_000_000;
const MAXIMUM_CELLS_PER_AXIS: usize = 256;

pub(super) struct RayGrid {
    corners: Vec<[[f64; 3]; 3]>,
    /// Outward unit normal of each triangle.
    normals: Vec<[f64; 3]>,
    origin: [f64; 3],
    cell: [f64; 3],
    dims: [usize; 3],
    offsets: Vec<u32>,
    items: Vec<u32>,
    /// Hits closer than this are the ray's own surface.
    epsilon: f64,
}

impl RayGrid {
    pub(super) fn build(positions: &[[f64; 3]], triangles: &[Triangle], outward: f64) -> Self {
        let corners: Vec<[[f64; 3]; 3]> = triangles
            .iter()
            .map(|t| t.corners.map(|corner| positions[corner]))
            .collect();
        let normals = triangles
            .iter()
            .map(|t| t.unit_normal.map(|x| x * outward))
            .collect();
        let mut lower = [f64::INFINITY; 3];
        let mut upper = [f64::NEG_INFINITY; 3];
        for point in corners.iter().flatten() {
            for axis in 0..3 {
                lower[axis] = lower[axis].min(point[axis]);
                upper[axis] = upper[axis].max(point[axis]);
            }
        }
        let extent = sub(upper, lower);
        let diagonal = dot(extent, extent).sqrt().max(f64::MIN_POSITIVE);
        let floor = diagonal * 1.0e-6;
        let padded = extent.map(|x| x.max(floor));
        let target = triangles.len().clamp(1, MAXIMUM_CELLS) as f64;
        let side = (padded[0] * padded[1] * padded[2] / target).cbrt();
        let dims = padded.map(|x| ((x / side).ceil() as usize).clamp(1, MAXIMUM_CELLS_PER_AXIS));
        let cell = std::array::from_fn(|axis| padded[axis] / dims[axis] as f64);
        let mut grid = Self {
            corners,
            normals,
            origin: lower,
            cell,
            dims,
            offsets: Vec::new(),
            items: Vec::new(),
            epsilon: diagonal * 1.0e-7,
        };
        grid.bucket();
        grid
    }

    fn cell_of(&self, point: [f64; 3]) -> [usize; 3] {
        std::array::from_fn(|axis| {
            let index = ((point[axis] - self.origin[axis]) / self.cell[axis]).floor();
            (index.max(0.0) as usize).min(self.dims[axis] - 1)
        })
    }

    fn linear(&self, cell: [usize; 3]) -> usize {
        (cell[2] * self.dims[1] + cell[1]) * self.dims[0] + cell[0]
    }

    fn bucket(&mut self) {
        let count = self.dims.iter().product::<usize>();
        let spans: Vec<([usize; 3], [usize; 3])> = self
            .corners
            .iter()
            .map(|t| {
                let low = std::array::from_fn(|a| t[0][a].min(t[1][a]).min(t[2][a]));
                let high = std::array::from_fn(|a| t[0][a].max(t[1][a]).max(t[2][a]));
                (self.cell_of(low), self.cell_of(high))
            })
            .collect();
        let mut counts = vec![0_u32; count + 1];
        for span in &spans {
            self.visit(span, &mut |cell| counts[cell + 1] += 1);
        }
        for index in 0..count {
            counts[index + 1] += counts[index];
        }
        let mut cursor = counts.clone();
        let mut items = vec![0_u32; counts[count] as usize];
        for (triangle, span) in spans.iter().enumerate() {
            self.visit(span, &mut |cell| {
                items[cursor[cell] as usize] = triangle as u32;
                cursor[cell] += 1;
            });
        }
        self.offsets = counts;
        self.items = items;
    }

    fn visit(&self, span: &([usize; 3], [usize; 3]), f: &mut dyn FnMut(usize)) {
        for z in span.0[2]..=span.1[2] {
            for y in span.0[1]..=span.1[1] {
                for x in span.0[0]..=span.1[0] {
                    f(self.linear([x, y, z]));
                }
            }
        }
    }

    /// Distance along `direction` (unit) from `origin` to the nearest other triangle the ray
    /// meets from the expected side: a ray inside the material meets the opposite wall from
    /// behind (`from_inside`), a ray in empty space meets a facing surface from the front.
    pub(super) fn first_hit(
        &self,
        source: usize,
        origin: [f64; 3],
        direction: [f64; 3],
        from_inside: bool,
    ) -> Option<f64> {
        let mut cell = self.cell_of(origin);
        let mut step = [0_isize; 3];
        let mut next = [f64::INFINITY; 3];
        let mut delta = [f64::INFINITY; 3];
        for axis in 0..3 {
            if direction[axis] > 0.0 {
                step[axis] = 1;
                let boundary = self.origin[axis] + (cell[axis] + 1) as f64 * self.cell[axis];
                next[axis] = (boundary - origin[axis]) / direction[axis];
                delta[axis] = self.cell[axis] / direction[axis];
            } else if direction[axis] < 0.0 {
                step[axis] = -1;
                let boundary = self.origin[axis] + cell[axis] as f64 * self.cell[axis];
                next[axis] = (boundary - origin[axis]) / direction[axis];
                delta[axis] = -self.cell[axis] / direction[axis];
            }
        }
        let mut best = f64::INFINITY;
        loop {
            let linear = self.linear(cell);
            let bucket = self.offsets[linear] as usize..self.offsets[linear + 1] as usize;
            for &triangle in &self.items[bucket] {
                let triangle = triangle as usize;
                if triangle == source {
                    continue;
                }
                let facing = dot(self.normals[triangle], direction);
                if (from_inside && facing <= 0.0) || (!from_inside && facing >= 0.0) {
                    continue;
                }
                if let Some(t) = intersect(origin, direction, &self.corners[triangle]) {
                    if t > self.epsilon && t < best {
                        best = t;
                    }
                }
            }
            let axis = if next[0] <= next[1] && next[0] <= next[2] {
                0
            } else if next[1] <= next[2] {
                1
            } else {
                2
            };
            if best <= next[axis] || !next[axis].is_finite() {
                break;
            }
            let moved = cell[axis] as isize + step[axis];
            if moved < 0 || moved >= self.dims[axis] as isize {
                break;
            }
            cell[axis] = moved as usize;
            next[axis] += delta[axis];
        }
        best.is_finite().then_some(best)
    }
}

/// Ray parameter of the intersection with a triangle (Moller-Trumbore), either side.
fn intersect(origin: [f64; 3], direction: [f64; 3], corners: &[[f64; 3]; 3]) -> Option<f64> {
    let edge1 = sub(corners[1], corners[0]);
    let edge2 = sub(corners[2], corners[0]);
    let p = cross(direction, edge2);
    let determinant = dot(edge1, p);
    if determinant.abs() <= f64::EPSILON * dot(edge1, edge1).max(dot(edge2, edge2)) {
        return None;
    }
    let inverse = 1.0 / determinant;
    let s = sub(origin, corners[0]);
    let u = dot(s, p) * inverse;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross(s, edge1);
    let v = dot(direction, q) * inverse;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    Some(dot(edge2, q) * inverse)
}
