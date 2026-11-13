//! Implicit differential at a resolved stationary INTERIOR local minimum, not a uniqueness proof.
use super::*;
pub(super) mod geometry;
mod mixed;
pub use mixed::{BoundaryProjectionDirection, BoundaryProjectionMixedVariation};
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundaryProjectionDifferentialPolicy {
    pub stationarity_tolerance_m: f64,
    pub minimum_barycentric: f64,
    /// Minimum eigenvalue/maximum eigenvalue of the local squared-distance metric.
    pub minimum_metric_ratio: f64,
}
#[derive(Clone, Copy, Debug)]
pub struct BoundaryProjectionVariation {
    pub parameter_increment: [f64; 2],
    pub surface_tangent_increment_m: [[f64; 3]; 2],
    pub signed_gap_increment_m: f64,
    pub unit_normal_increment: [f64; 3],
    pub point_position_increment_m: [f64; 3],
    pub gap_gradient_query: [f64; 3],
    pub gap_gradient_query_increment: [f64; 3],
    shape_increment: [f64; 6],
    gap_gradient_nodes: [[f64; 3]; 6],
    gap_gradient_nodes_increment: [[f64; 3]; 6],
    count: usize,
}
impl BoundaryProjectionVariation {
    pub fn shape_increment(&self) -> &[f64] {
        &self.shape_increment[..self.count]
    }
    pub fn gap_gradient_nodes(&self) -> &[[f64; 3]] {
        &self.gap_gradient_nodes[..self.count]
    }
    pub fn gap_gradient_nodes_increment(&self) -> &[[f64; 3]] {
        &self.gap_gradient_nodes_increment[..self.count]
    }
    pub(super) fn orient(mut self, side: f64) -> Self {
        self.signed_gap_increment_m *= side;
        self.unit_normal_increment = self.unit_normal_increment.map(|x| side * x);
        self.gap_gradient_query = self.gap_gradient_query.map(|x| side * x);
        self.gap_gradient_query_increment = self.gap_gradient_query_increment.map(|x| side * x);
        self.gap_gradient_nodes = self.gap_gradient_nodes.map(|n| n.map(|x| side * x));
        self.gap_gradient_nodes_increment = self
            .gap_gradient_nodes_increment
            .map(|n| n.map(|x| side * x));
        self
    }
}
impl BoundarySurface {
    pub fn stationary_projection_variation(
        &self,
        query: [f64; 3],
        barycentric: [f64; 3],
        query_increment_m: [f64; 3],
        node_increments_m: &[[f64; 3]],
        policy: BoundaryProjectionDifferentialPolicy,
    ) -> Result<BoundaryProjectionVariation, BoundarySearchError> {
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled);
        }
        if node_increments_m.len() != self.count
            || query_increment_m
                .iter()
                .chain(node_increments_m.iter().flatten())
                .any(|x| !x.is_finite())
        {
            return Err(BoundarySearchError::InvalidInput);
        }
        let geometry = geometry::ProjectionGeometry::derive(self, query, barycentric, policy)?;
        let geometry::ProjectionGeometry {
            point,
            d,
            t,
            weights,
            first,
            curvature,
            ..
        } = geometry;
        let fixed_position =
            quadrature::weighted_position(node_increments_m, &weights[..self.count]);
        let relative = subtract(query_increment_m, fixed_position);
        let fixed_tangent = [0, 1]
            .map(|i| quadrature::weighted_derivative(node_increments_m, &first[..self.count], i));
        let rhs = [0, 1].map(|i| dot(t[i], relative) + dot(d, fixed_tangent[i]));
        let parameter_increment = geometry.solve(rhs);
        let tangent_increment: [[f64; 3]; 2] = std::array::from_fn(|i| {
            std::array::from_fn(|k| {
                fixed_tangent[i][k]
                    + (0..2)
                        .map(|j| curvature[i][j][k] * parameter_increment[j])
                        .sum::<f64>()
            })
        });
        let raw = cross(t[0], t[1]);
        let raw_increment = {
            let a = cross(tangent_increment[0], t[1]);
            let b = cross(t[0], tangent_increment[1]);
            std::array::from_fn(|i| a[i] + b[i])
        };
        let n = point.unit_normal;
        let normal_increment =
            std::array::from_fn(|i| (raw_increment[i] - n[i] * dot(n, raw_increment)) / norm(raw));
        let shape_increment = std::array::from_fn(|i| {
            first[i][0] * parameter_increment[0] + first[i][1] * parameter_increment[1]
        });
        let result = BoundaryProjectionVariation {
            parameter_increment,
            surface_tangent_increment_m: tangent_increment,
            signed_gap_increment_m: dot(n, relative),
            unit_normal_increment: normal_increment,
            point_position_increment_m: std::array::from_fn(|k| {
                fixed_position[k]
                    + t[0][k] * parameter_increment[0]
                    + t[1][k] * parameter_increment[1]
            }),
            gap_gradient_query: n,
            gap_gradient_query_increment: normal_increment,
            gap_gradient_nodes: std::array::from_fn(|i| n.map(|x| -weights[i] * x)),
            gap_gradient_nodes_increment: std::array::from_fn(|i| {
                std::array::from_fn(|k| {
                    -shape_increment[i] * n[k] - weights[i] * normal_increment[k]
                })
            }),
            shape_increment,
            count: self.count,
        };
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled);
        }
        if result
            .parameter_increment
            .iter()
            .chain(result.unit_normal_increment.iter())
            .chain(result.point_position_increment_m.iter())
            .chain(result.gap_gradient_nodes_increment.iter().flatten())
            .any(|x| !x.is_finite())
            || !result.signed_gap_increment_m.is_finite()
        {
            return Err(BoundarySearchError::Resolution);
        }
        Ok(result)
    }
}
#[cfg(test)]
mod tests;
