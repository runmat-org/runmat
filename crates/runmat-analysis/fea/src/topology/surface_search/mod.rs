//! Closest-point search on original Tri3/Tri6 geometry, for contact and other consumers.
//! A geometry result is not contact enforcement, a unique projection, or a physics verdict.
use super::{
    element::{cross, dot, norm, subtract},
    quadrature,
};
use runmat_meshing_core::BoundaryTriangleOrder;
use thiserror::Error;
mod stationary;
pub use stationary::{BoundaryStationaryPolicy, StationaryBoundaryProjection};
mod differential;
pub use differential::{
    BoundaryProjectionDifferentialPolicy, BoundaryProjectionDirection,
    BoundaryProjectionMixedVariation, BoundaryProjectionVariation,
};
mod bounds;
mod index;
pub use bounds::BoundarySurfaceBounds;
pub use index::{
    BoundaryPairingError, BoundaryPairingPolicy, BoundarySurfaceIndex, BoundarySurfaceMatch,
};
mod chord;
mod search;
mod snapshot;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundarySearchPolicy {
    pub distance_tolerance_m: f64,
    pub maximum_patches: u64,
    pub maximum_depth: u8,
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum BoundarySearchError {
    #[error("invalid boundary surface or folded quadratic map")]
    InvalidGeometry,
    #[error("invalid boundary search controls or nonfinite point")]
    InvalidInput,
    #[error("boundary search cancelled")]
    Cancelled,
    #[error("boundary search exhausted its work/depth envelope")]
    Exhausted,
    #[error("boundary search tolerance is below floating-point resolution")]
    Resolution,
    #[error("boundary search storage is not addressable")]
    Capacity,
    #[error("projection lies on a nonsmooth boundary feature")]
    NonSmoothProjection,
    #[error("supplied projection is not stationary within the declared tolerance")]
    NonStationaryProjection,
    #[error("projection distance Hessian is not a resolved positive definite local metric")]
    DegenerateProjection,
    #[error("stationary projection exhausted its refinement/backtrack envelope")]
    RefinementExhausted,
}

#[derive(Clone, Copy, Debug)]
pub struct BoundarySurface {
    order: BoundaryTriangleOrder,
    nodes: [[f64; 3]; 6],
    count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundarySurfacePoint {
    pub position_m: [f64; 3],
    pub tangents_m: [[f64; 3]; 2],
    /// Oriented by the supplied node winding. Caller retains the outward-side authority.
    pub unit_normal: [f64; 3],
    weights: [f64; 6],
    count: usize,
}
impl BoundarySurfacePoint {
    pub fn shape_values(&self) -> &[f64] {
        &self.weights[..self.count]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosestBoundaryPoint {
    pub barycentric: [f64; 3],
    pub point: BoundarySurfacePoint,
    pub distance_m: f64,
    /// Query displacement along the selected, winding-oriented surface normal.
    /// It is not a contact penetration/complementarity verdict.
    pub signed_normal_offset_m: f64,
    /// Numerical global distance error envelope, including a scale-based roundoff guard.
    /// This is not an interval-arithmetic proof or a uniqueness certificate.
    pub distance_error_m: f64,
    pub visited_patches: u64,
}

impl BoundarySurface {
    pub fn derive(
        order: BoundaryTriangleOrder,
        coordinates: &[[f64; 3]],
    ) -> Result<Self, BoundarySearchError> {
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled);
        }
        let count = match order {
            BoundaryTriangleOrder::Tri3 => 3,
            BoundaryTriangleOrder::Tri6 => 6,
        };
        if coordinates.len() != count || coordinates.iter().flatten().any(|x| !x.is_finite()) {
            return Err(BoundarySearchError::InvalidGeometry);
        }
        if !runmat_meshing_core::is_valid_boundary_face(order, coordinates) {
            return Err(BoundarySearchError::InvalidGeometry);
        }
        let mut nodes = [[0.0; 3]; 6];
        nodes[..count].copy_from_slice(coordinates);
        Ok(Self {
            order,
            nodes,
            count,
        })
    }

    pub fn evaluate(
        &self,
        barycentric: [f64; 3],
    ) -> Result<BoundarySurfacePoint, BoundarySearchError> {
        if barycentric
            .iter()
            .any(|b| !b.is_finite() || *b < 0.0 || *b > 1.0)
            || (barycentric.iter().sum::<f64>() - 1.0).abs() > 16.0 * f64::EPSILON
        {
            return Err(BoundarySearchError::InvalidInput);
        }
        let (weights, derivatives) = quadrature::boundary_shape(self.order, barycentric);
        let coordinates = &self.nodes[..self.count];
        let position_m = quadrature::weighted_position(coordinates, &weights[..self.count]);
        let tangents_m = std::array::from_fn(|i| {
            quadrature::weighted_derivative(coordinates, &derivatives[..self.count], i)
        });
        let normal = cross(tangents_m[0], tangents_m[1]);
        let length = norm(normal);
        if !length.is_finite() || length <= 0.0 || position_m.iter().any(|x| !x.is_finite()) {
            return Err(BoundarySearchError::InvalidGeometry);
        }
        Ok(BoundarySurfacePoint {
            position_m,
            tangents_m,
            unit_normal: normal.map(|x| x / length),
            weights,
            count: self.count,
        })
    }
}

#[cfg(test)]
mod tests;

pub use snapshot::OrientedBoundarySurface;
