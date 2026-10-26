//! Fixed-frame transformations. Geometry/frame derivatives are required separately in FE assembly.
use super::*;
use crate::topology::BoundarySurfacePoint;
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContactPointFrame {
    axes: [[f64; 3]; 3],
}
impl ContactPointFrame {
    pub fn from_surface(point: &BoundarySurfacePoint) -> Result<Self, ContactPointError> {
        let axes = crate::physics::structural::frame::material_surface_basis(
            point.unit_normal,
            point.tangents_m[0],
        )
        .map_err(|_| ContactPointError::InvalidInput)?;
        Ok(Self { axes })
    }
    pub fn axes(&self) -> [[f64; 3]; 3] {
        self.axes
    }
    /// Caller supplies objective relative material motion, not a difference of closest-point gaps.
    pub fn tangential_increment(
        &self,
        relative_material_increment: Quantity<kind::Length, [f64; 3]>,
    ) -> Result<TangentialMotion, ContactPointError> {
        let relative_material_increment_m = relative_material_increment.canonical_value();
        if relative_material_increment_m.iter().any(|x| !x.is_finite()) {
            return Err(ContactPointError::InvalidInput);
        }
        let local = std::array::from_fn(|i| {
            (0..3)
                .map(|j| self.axes[i][j] * relative_material_increment_m[j])
                .sum::<f64>()
        });
        if local.iter().any(|x| !x.is_finite()) {
            return Err(ContactPointError::InvalidInput);
        }
        Ok(Quantity::new(local))
    }
    /// Compare the SAME paired material locations in accepted/current frames.
    /// Closest-point separation at a newly projected primary location is not material slip.
    /// Frame/point lineage and their geometry derivatives remain assembly authority.
    pub fn objective_tangential_increment(
        &self,
        current_relative_material_vector: Quantity<kind::Length, [f64; 3]>,
        accepted_frame: &Self,
        accepted_relative_material_vector: Quantity<kind::Length, [f64; 3]>,
    ) -> Result<TangentialMotion, ContactPointError> {
        let current = self
            .tangential_increment(current_relative_material_vector)?
            .canonical_value();
        let accepted = accepted_frame
            .tangential_increment(accepted_relative_material_vector)?
            .canonical_value();
        let increment = std::array::from_fn(|i| current[i] - accepted[i]);
        if increment.iter().any(|x| !x.is_finite()) {
            return Err(ContactPointError::InvalidInput);
        }
        Ok(Quantity::new(increment))
    }
    pub fn internal_traction_world(
        &self,
        response: &ContactPointResponse,
    ) -> Result<Quantity<kind::Traction, [f64; 3]>, ContactPointError> {
        let local = response.internal_traction_local_pa.canonical_value();
        let world =
            std::array::from_fn(|i| (0..3).map(|j| self.axes[j][i] * local[j]).sum::<f64>());
        if world.iter().any(|x| !x.is_finite()) {
            return Err(ContactPointError::NonFiniteResponse);
        }
        Ok(Quantity::new(world))
    }
    /// F C Fᵀ for a held material frame. Excludes derivatives of F and of surface pairing.
    pub fn tangent_world(
        &self,
        response: &ContactPointResponse,
    ) -> Result<Quantity<kind::TractionDisplacementStiffness, [[f64; 3]; 3]>, ContactPointError>
    {
        let local = response.tangent_pa_per_m.canonical_value();
        let matrix: [[f64; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                (0..3)
                    .flat_map(|a| {
                        (0..3).map(move |b| self.axes[a][i] * local[a][b] * self.axes[b][j])
                    })
                    .sum::<f64>()
            })
        });
        if matrix.iter().flatten().any(|x| !x.is_finite()) {
            return Err(ContactPointError::NonFiniteResponse);
        }
        Ok(Quantity::new(matrix))
    }
}
