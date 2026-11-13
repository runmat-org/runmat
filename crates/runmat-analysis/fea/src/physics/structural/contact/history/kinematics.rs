//! Objective coordinates and their analytic first/mixed derivatives for a held material pair.
use super::*;
use crate::algebra::NormalizedVectorDifferential;
use crate::topology::{BoundaryProjectionDirection, BoundarySurfacePoint};
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn sum(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] + b[i])
}
pub(super) struct Kinematics {
    pub surface: OrientedBoundarySurface,
    pub point: BoundarySurfacePoint,
    pub query: [f64; 3],
    pub barycentric: [f64; 3],
    pub relative_old_material: [f64; 3],
    pub frame: ContactPointFrame,
    pub policy: BoundaryProjectionDifferentialPolicy,
    pub old_shape: [f64; 6],
    pub count: usize,
}
impl Kinematics {
    fn relative_direction(&self, d: BoundaryProjectionDirection<'_>) -> [f64; 3] {
        std::array::from_fn(|k| {
            d.query_m[k]
                - d.nodes_m
                    .iter()
                    .zip(self.old_shape)
                    .map(|(p, n)| p[k] * n)
                    .sum::<f64>()
        })
    }
    pub fn first(
        &self,
        d: BoundaryProjectionDirection<'_>,
    ) -> Result<[f64; 3], ContactHistoryError> {
        let variation = self.surface.stationary_projection_variation(
            self.query,
            self.barycentric,
            d.query_m,
            d.nodes_m,
            self.policy,
        )?;
        let tangent = NormalizedVectorDifferential::derive(
            self.point.tangents_m[0],
            variation.surface_tangent_increment_m[0],
            [0.0; 3],
            [0.0; 3],
        )
        .map_err(|_| ContactHistoryError::InvalidHistory)?;
        let axes = self.frame.axes();
        let de0 = tangent.first[0];
        let de1 = sum(
            cross(variation.unit_normal_increment, axes[0]),
            cross(axes[2], de0),
        );
        let dr = self.relative_direction(d);
        Ok([
            dot(de0, self.relative_old_material) + dot(axes[0], dr),
            dot(de1, self.relative_old_material) + dot(axes[1], dr),
            variation.signed_gap_increment_m,
        ])
    }
    pub fn mixed(
        &self,
        directions: [BoundaryProjectionDirection<'_>; 2],
    ) -> Result<[f64; 3], ContactHistoryError> {
        let [a, b] = directions;
        let va = self.surface.stationary_projection_variation(
            self.query,
            self.barycentric,
            a.query_m,
            a.nodes_m,
            self.policy,
        )?;
        let vb = self.surface.stationary_projection_variation(
            self.query,
            self.barycentric,
            b.query_m,
            b.nodes_m,
            self.policy,
        )?;
        let mixed = self.surface.stationary_projection_mixed_variation(
            self.query,
            self.barycentric,
            directions,
            self.policy,
        )?;
        let tangent = NormalizedVectorDifferential::derive(
            self.point.tangents_m[0],
            va.surface_tangent_increment_m[0],
            vb.surface_tangent_increment_m[0],
            mixed.surface_tangent_mixed_m[0],
        )
        .map_err(|_| ContactHistoryError::InvalidHistory)?;
        let axes = self.frame.axes();
        let e0a = tangent.first[0];
        let e0b = tangent.first[1];
        let e0ab = tangent.mixed;
        let e1a = sum(
            cross(va.unit_normal_increment, axes[0]),
            cross(axes[2], e0a),
        );
        let e1b = sum(
            cross(vb.unit_normal_increment, axes[0]),
            cross(axes[2], e0b),
        );
        let e1ab = sum(
            sum(
                cross(mixed.unit_normal_mixed, axes[0]),
                cross(va.unit_normal_increment, e0b),
            ),
            sum(cross(vb.unit_normal_increment, e0a), cross(axes[2], e0ab)),
        );
        let ra = self.relative_direction(a);
        let rb = self.relative_direction(b);
        Ok([
            dot(e0ab, self.relative_old_material) + dot(e0a, rb) + dot(e0b, ra),
            dot(e1ab, self.relative_old_material) + dot(e1a, rb) + dot(e1b, ra),
            mixed.signed_gap_mixed_m,
        ])
    }
}
pub(super) fn basis(
    column: usize,
    nodes: &mut [[f64; 3]; 6],
    count: usize,
) -> BoundaryProjectionDirection<'_> {
    let mut query = [0.0; 3];
    if column < 3 {
        query[column] = 1.0;
    } else {
        nodes[(column - 3) / 3][(column - 3) % 3] = 1.0;
    }
    BoundaryProjectionDirection {
        query_m: query,
        nodes_m: &nodes[..count],
    }
}
