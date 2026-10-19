//! Numerical Bernstein control hulls of original boundary geometry; not interval proofs.
use super::*;
#[derive(Clone, Copy, Debug)]
pub struct BoundarySurfaceBounds {
    pub minimum_m: [f64; 3],
    pub maximum_m: [f64; 3],
    pub roundoff_guard_m: f64,
}
impl BoundarySurfaceBounds {
    pub fn distance_lower_bound_m(&self, query: [f64; 3]) -> Result<f64, BoundarySearchError> {
        if query.iter().any(|x| !x.is_finite()) {
            return Err(BoundarySearchError::InvalidInput);
        }
        let delta: [f64; 3] = std::array::from_fn(|i| {
            (self.minimum_m[i] - query[i])
                .max(query[i] - self.maximum_m[i])
                .max(0.0)
        });
        let distance = delta[0].hypot(delta[1]).hypot(delta[2]);
        let query_guard =
            256.0 * f64::EPSILON * query.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        if !distance.is_finite() {
            return Err(BoundarySearchError::Resolution);
        }
        Ok((distance - self.roundoff_guard_m - query_guard).max(0.0))
    }
}
impl BoundarySurface {
    pub fn bounds(&self) -> Result<BoundarySurfaceBounds, BoundarySearchError> {
        let mut controls = self.nodes;
        if self.order == BoundaryTriangleOrder::Tri6 {
            for (mid, (a, b)) in [(0, 1), (1, 2), (2, 0)].into_iter().enumerate() {
                controls[3 + mid] = std::array::from_fn(|i| {
                    2.0 * self.nodes[3 + mid][i] - 0.5 * self.nodes[a][i] - 0.5 * self.nodes[b][i]
                });
            }
        }
        if controls[..self.count]
            .iter()
            .flatten()
            .any(|x| !x.is_finite())
        {
            return Err(BoundarySearchError::Resolution);
        }
        let minimum_m = std::array::from_fn(|i| {
            controls[..self.count]
                .iter()
                .map(|p| p[i])
                .fold(f64::INFINITY, f64::min)
        });
        let maximum_m = std::array::from_fn(|i| {
            controls[..self.count]
                .iter()
                .map(|p| p[i])
                .fold(f64::NEG_INFINITY, f64::max)
        });
        let scale = controls[..self.count]
            .iter()
            .flatten()
            .chain(self.nodes[..self.count].iter().flatten())
            .map(|x| x.abs())
            .fold(f64::MIN_POSITIVE, f64::max);
        Ok(BoundarySurfaceBounds {
            minimum_m,
            maximum_m,
            roundoff_guard_m: 256.0 * f64::EPSILON * scale,
        })
    }
}
