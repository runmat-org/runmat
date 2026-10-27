//! Normal penalty potential on a smooth interior pair, integrated over FIXED reference area.
//! Includes projection/normal derivatives. No friction, active-feature switch or global assembly.
use super::{ContactPointError, PenaltyContactLaw};
use crate::topology::{
    BoundaryProjectionDifferentialPolicy, BoundarySearchError, OrientedBoundarySurface,
};
use thiserror::Error;
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum NormalContactError {
    #[error(transparent)]
    Geometry(#[from] BoundarySearchError),
    #[error(transparent)]
    Law(#[from] ContactPointError),
    #[error("invalid reference-area weight or nonfinite contact contribution")]
    InvalidContribution,
}
pub struct NormalContactPoint {
    surface: OrientedBoundarySurface,
    query_m: [f64; 3],
    barycentric: [f64; 3],
    reference_area_m2: f64,
    policy: BoundaryProjectionDifferentialPolicy,
    normal_traction_pa: f64,
    normal_stiffness_pa_per_m: f64,
    response: NormalContactContribution,
}
#[derive(Clone, Copy, Debug)]
pub struct NormalContactContribution {
    pub signed_gap_m: f64,
    /// Nominal traction per reference area, not spatial pressure per deformed area.
    pub nominal_pressure_pa: runmat_types::physical::Pressure,
    pub elastic_energy_j: runmat_types::physical::Energy,
    pub query_internal_force_n: [f64; 3],
    primary_internal_force_n: [[f64; 3]; 6],
    count: usize,
}
impl NormalContactContribution {
    pub fn primary_internal_force_n(&self) -> &[[f64; 3]] {
        &self.primary_internal_force_n[..self.count]
    }
}
#[derive(Clone, Copy, Debug)]
pub struct NormalContactTangentAction {
    pub query_force_increment_n: [f64; 3],
    primary_force_increment_n: [[f64; 3]; 6],
    count: usize,
}
impl NormalContactTangentAction {
    pub fn primary_force_increment_n(&self) -> &[[f64; 3]] {
        &self.primary_force_increment_n[..self.count]
    }
}
impl NormalContactPoint {
    pub fn derive(
        surface: OrientedBoundarySurface,
        query_m: [f64; 3],
        barycentric: [f64; 3],
        reference_area_m2: f64,
        policy: BoundaryProjectionDifferentialPolicy,
        law: &PenaltyContactLaw,
    ) -> Result<Self, NormalContactError> {
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled.into());
        }
        if !reference_area_m2.is_finite() || reference_area_m2 <= 0.0 {
            return Err(NormalContactError::InvalidContribution);
        }
        let point = surface.evaluate(barycentric)?;
        let count = point.shape_values().len();
        let geometry = surface.stationary_projection_variation(
            query_m,
            barycentric,
            [0.0; 3],
            &[[0.0; 3]; 6][..count],
            policy,
        )?;
        let gap = (0..3)
            .map(|i| (query_m[i] - point.position_m[i]) * point.unit_normal[i])
            .sum::<f64>();
        // Normal potential is history independent. Tangential output from the point law is not used.
        let local = law.update(
            runmat_types::physical::Length::new(gap),
            runmat_types::physical::Quantity::new([0.0; 2]),
            &law.initial_state(),
        )?;
        let traction = local.internal_traction_local_pa.canonical_values()[2];
        let stiffness = local.tangent_pa_per_m.canonical_values()[2][2];
        let response = NormalContactContribution {
            signed_gap_m: gap,
            nominal_pressure_pa: local.normal_pressure_pa,
            elastic_energy_j: runmat_types::physical::Energy::new(
                reference_area_m2 * local.normal_elastic_energy_j_per_m2.canonical_value(),
            ),
            query_internal_force_n: geometry
                .gap_gradient_query
                .map(|n| reference_area_m2 * traction * n),
            primary_internal_force_n: std::array::from_fn(|i| {
                if i < count {
                    geometry.gap_gradient_nodes()[i].map(|n| reference_area_m2 * traction * n)
                } else {
                    [0.0; 3]
                }
            }),
            count,
        };
        if !response.elastic_energy_j.canonical_value().is_finite()
            || response
                .query_internal_force_n
                .iter()
                .chain(response.primary_internal_force_n.iter().flatten())
                .any(|x| !x.is_finite())
        {
            return Err(NormalContactError::InvalidContribution);
        }
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled.into());
        }
        Ok(Self {
            surface,
            query_m,
            barycentric,
            reference_area_m2,
            policy,
            normal_traction_pa: traction,
            normal_stiffness_pa_per_m: stiffness,
            response,
        })
    }
    pub fn contribution(&self) -> &NormalContactContribution {
        &self.response
    }
    /// Complete derivative of this normal potential's nodal residual at the held reference weight.
    /// Excludes friction, secondary interpolation changes, active features and changing area measures.
    pub fn tangent_action(
        &self,
        query_increment_m: [f64; 3],
        primary_node_increments_m: &[[f64; 3]],
    ) -> Result<NormalContactTangentAction, NormalContactError> {
        let geometry = self.surface.stationary_projection_variation(
            self.query_m,
            self.barycentric,
            query_increment_m,
            primary_node_increments_m,
            self.policy,
        )?;
        let change = self.normal_stiffness_pa_per_m * geometry.signed_gap_increment_m;
        let query_force_increment_n = std::array::from_fn(|k| {
            self.reference_area_m2
                * (change * geometry.gap_gradient_query[k]
                    + self.normal_traction_pa * geometry.gap_gradient_query_increment[k])
        });
        let primary_force_increment_n = std::array::from_fn(|i| {
            std::array::from_fn(|k| {
                if i < self.response.count {
                    self.reference_area_m2
                        * (change * geometry.gap_gradient_nodes()[i][k]
                            + self.normal_traction_pa
                                * geometry.gap_gradient_nodes_increment()[i][k])
                } else {
                    0.0
                }
            })
        });
        if query_force_increment_n
            .iter()
            .chain(primary_force_increment_n.iter().flatten())
            .any(|x| !x.is_finite())
        {
            return Err(NormalContactError::InvalidContribution);
        }
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled.into());
        }
        Ok(NormalContactTangentAction {
            query_force_increment_n,
            primary_force_increment_n,
            count: self.response.count,
        })
    }
}
#[cfg(test)]
mod tests;
