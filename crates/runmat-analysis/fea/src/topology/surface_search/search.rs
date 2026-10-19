//! Quadratic patches differ from their chord by a bounded Bernstein control offset.
//! Subdivision shrinks that offset; exhaustion returns no accepted geometry result.
use super::*;
#[derive(Clone, Copy)]
struct Patch {
    vertices: [[f64; 3]; 3],
    depth: u8,
}
impl BoundarySearchPolicy {
    /// Requested heap for the depth-first frontier, excluding input/output and allocator overhead.
    pub fn resident_bytes(self) -> Result<u64, BoundarySearchError> {
        if !self.distance_tolerance_m.is_finite()
            || self.distance_tolerance_m <= 0.0
            || self.maximum_patches == 0
            || self.maximum_depth > 32
        {
            return Err(BoundarySearchError::InvalidInput);
        }
        super::super::storage::slots::<Patch>(1 + 3 * usize::from(self.maximum_depth))
            .map_err(|_| BoundarySearchError::Capacity)
    }
}
impl BoundarySurface {
    pub fn closest_point(
        &self,
        query: [f64; 3],
        policy: BoundarySearchPolicy,
    ) -> Result<ClosestBoundaryPoint, BoundarySearchError> {
        policy.resident_bytes()?;
        if query.iter().any(|x| !x.is_finite()) {
            return Err(BoundarySearchError::InvalidInput);
        }
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled);
        }
        let guard = 256.0
            * f64::EPSILON
            * self.nodes[..self.count]
                .iter()
                .flatten()
                .chain(query.iter())
                .map(|x| x.abs())
                .fold(f64::MIN_POSITIVE, f64::max);
        if policy.distance_tolerance_m <= guard {
            return Err(BoundarySearchError::Resolution);
        }
        let tolerance = policy.distance_tolerance_m - guard;
        let capacity = 1 + 3 * usize::from(policy.maximum_depth);
        let mut frontier = Vec::new();
        frontier
            .try_reserve_exact(capacity)
            .map_err(|_| BoundarySearchError::Capacity)?;
        frontier.push(Patch {
            vertices: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            depth: 0,
        });
        let mut best = None::<ClosestBoundaryPoint>;
        let mut visited = 0_u64;
        while let Some(patch) = frontier.pop() {
            if crate::is_fea_cancelled() {
                return Err(BoundarySearchError::Cancelled);
            }
            if visited == policy.maximum_patches {
                return Err(BoundarySearchError::Exhausted);
            }
            visited += 1;
            let vertices = patch
                .vertices
                .map(|b| self.evaluate(b).map(|p| p.position_m));
            let [a, b, c] = vertices;
            let physical = [a?, b?, c?];
            let (local, chord_distance) = chord::closest(physical, query)?;
            let barycentric =
                std::array::from_fn(|i| (0..3).map(|j| local[j] * patch.vertices[j][i]).sum());
            let point = self.evaluate(barycentric)?;
            let distance = norm(subtract(point.position_m, query));
            if !distance.is_finite() {
                return Err(BoundarySearchError::Resolution);
            }
            if best
                .as_ref()
                .is_none_or(|previous| distance < previous.distance_m)
            {
                best = Some(ClosestBoundaryPoint {
                    barycentric,
                    point,
                    distance_m: distance,
                    signed_normal_offset_m: dot(
                        subtract(query, point.position_m),
                        point.unit_normal,
                    ),
                    distance_error_m: policy.distance_tolerance_m,
                    visited_patches: visited,
                });
            }
            let midpoints = [(0, 1), (1, 2), (2, 0)].map(|(i, j)| {
                std::array::from_fn(|axis| {
                    0.5 * (patch.vertices[i][axis] + patch.vertices[j][axis])
                })
            });
            let mut deviation = 0.0_f64;
            if self.order == BoundaryTriangleOrder::Tri6 {
                for (midpoint, (i, j)) in midpoints.iter().zip([(0, 1), (1, 2), (2, 0)]) {
                    let position = self.evaluate(*midpoint)?.position_m;
                    let delta = std::array::from_fn(|axis| {
                        2.0 * position[axis] - physical[i][axis] - physical[j][axis]
                    });
                    deviation = deviation.max(norm(delta));
                }
            }
            if !deviation.is_finite() {
                return Err(BoundarySearchError::Resolution);
            }
            let lower = (chord_distance - deviation - guard).max(0.0);
            if best.as_ref().unwrap().distance_m - lower <= tolerance {
                continue;
            }
            if patch.depth == policy.maximum_depth {
                return Err(BoundarySearchError::Exhausted);
            }
            let [a, b, c] = patch.vertices;
            let [ab, bc, ca] = midpoints;
            for vertices in [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]] {
                frontier.push(Patch {
                    vertices,
                    depth: patch.depth + 1,
                });
            }
            debug_assert!(frontier.len() <= capacity);
        }
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled);
        }
        let mut best = best.ok_or(BoundarySearchError::InvalidGeometry)?;
        best.visited_patches = visited;
        Ok(best)
    }
}
