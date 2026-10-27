//! Point trial/residual/tangent. No global convergence, restart authorization or feature transfer.
use super::kinematics::{basis, Kinematics};
use super::*;
mod assembly;
use crate::topology::BoundaryProjectionDirection;
pub struct ContactPointTrial {
    history: ContactMaterialPointHistory,
    local: ContactPointResponse,
    kinematics: Kinematics,
    reference_area_m2: f64,
    query_force_n: [f64; 3],
    primary_force_n: [[f64; 3]; 6],
    secondary_shape: [f64; 6],
    secondary_count: usize,
}
impl ContactPointTrial {
    pub fn history(&self) -> &ContactMaterialPointHistory {
        &self.history
    }
    pub fn point_response(&self) -> &ContactPointResponse {
        &self.local
    }
    pub fn query_internal_force_n(&self) -> [f64; 3] {
        self.query_force_n
    }
    pub fn primary_internal_force_n(&self) -> &[[f64; 3]] {
        &self.primary_force_n[..self.kinematics.count]
    }
    pub fn secondary_internal_force_n(&self) -> [[f64; 3]; 6] {
        std::array::from_fn(|i| {
            if i < self.secondary_count {
                self.query_force_n.map(|f| self.secondary_shape[i] * f)
            } else {
                [0.0; 3]
            }
        })
    }
    /// Complete derivative of this point residual on a smooth held-face branch, at fixed reference area.
    pub fn tangent_action(
        &self,
        query_increment_m: [f64; 3],
        primary_node_increments_m: &[[f64; 3]],
    ) -> Result<ContactPointTangentAction, ContactHistoryError> {
        let d = BoundaryProjectionDirection {
            query_m: query_increment_m,
            nodes_m: primary_node_increments_m,
        };
        let b = self.kinematics.first(d)?;
        let traction_increment: [f64; 3] = std::array::from_fn(|i| {
            (0..3)
                .map(|j| self.local.tangent_pa_per_m.canonical_values()[i][j] * b[j])
                .sum()
        });
        let mut force = [0.0; 21];
        for (column, out) in force[..3 * (1 + self.kinematics.count)]
            .iter_mut()
            .enumerate()
        {
            let mut nodes = [[0.0; 3]; 6];
            let a = basis(column, &mut nodes, self.kinematics.count);
            let gradient = self.kinematics.first(a)?;
            let h = self.kinematics.mixed([a, d])?;
            *out = self.reference_area_m2
                * (0..3)
                    .map(|i| {
                        gradient[i] * traction_increment[i]
                            + self.local.internal_traction_local_pa.canonical_values()[i] * h[i]
                    })
                    .sum::<f64>();
        }
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled.into());
        }
        if force.iter().any(|f| !f.is_finite()) {
            return Err(ContactHistoryError::InvalidHistory);
        }
        Ok(ContactPointTangentAction {
            query_force_increment_n: [force[0], force[1], force[2]],
            primary_force_increment_n: std::array::from_fn(|i| {
                if i < self.kinematics.count {
                    [force[3 + 3 * i], force[4 + 3 * i], force[5 + 3 * i]]
                } else {
                    [0.0; 3]
                }
            }),
            count: self.kinematics.count,
        })
    }
}
#[derive(Clone, Copy, Debug)]
pub struct ContactPointTangentAction {
    pub query_force_increment_n: [f64; 3],
    primary_force_increment_n: [[f64; 3]; 6],
    count: usize,
}
impl ContactPointTangentAction {
    pub fn primary_force_increment_n(&self) -> &[[f64; 3]] {
        &self.primary_force_increment_n[..self.count]
    }
}
impl ContactPointBinding<'_> {
    pub fn trial(
        &self,
        accepted: &ContactMaterialPointHistory,
        expected_ordinal: u64,
        accepted_positions_m: &[[f64; 3]],
        current_positions_m: &[[f64; 3]],
        primary_barycentric: [f64; 3],
    ) -> Result<ContactPointTrial, ContactHistoryError> {
        self.validate_history(accepted, expected_ordinal, accepted_positions_m)?;
        let index = self.primary_index(accepted.primary_face)?;
        let configuration = self.configuration(index, primary_barycentric, current_positions_m)?;
        let point = configuration.surface.evaluate(primary_barycentric)?;
        let old_material = configuration
            .surface
            .evaluate(accepted.primary_barycentric)?;
        let relative = std::array::from_fn(|i| configuration.query[i] - old_material.position_m[i]);
        let increment = configuration.frame.objective_tangential_increment(
            runmat_types::physical::Quantity::new(relative),
            &accepted.frame,
            runmat_types::physical::Quantity::new(accepted.relative_material_vector_m),
        )?;
        let local = self.law.update(
            runmat_types::physical::Length::new(configuration.gap),
            increment,
            &accepted.point,
        )?;
        let history = ContactMaterialPointHistory {
            schema: 1,
            scope: self.scope,
            ordinal: expected_ordinal
                .checked_add(1)
                .ok_or(ContactHistoryError::OrdinalOverflow)?,
            primary_face: accepted.primary_face,
            primary_barycentric,
            primary_coordinates_m: configuration.primary,
            secondary_coordinates_m: configuration.secondary,
            frame: configuration.frame,
            relative_material_vector_m: configuration.relative,
            point: local.trial_state.clone(),
        };
        let mut old_shape = [0.0; 6];
        old_shape[..old_material.shape_values().len()].copy_from_slice(old_material.shape_values());
        let kinematics = Kinematics {
            surface: configuration.surface,
            point,
            query: configuration.query,
            barycentric: primary_barycentric,
            relative_old_material: relative,
            frame: configuration.frame,
            policy: self.policy,
            old_shape,
            count: point.shape_values().len(),
        };
        let mut force = [0.0; 21];
        for (column, out) in force[..3 * (1 + kinematics.count)].iter_mut().enumerate() {
            let mut nodes = [[0.0; 3]; 6];
            let d = basis(column, &mut nodes, kinematics.count);
            let gradient = kinematics.first(d)?;
            *out = self.reference.area_weight_m2
                * (0..3)
                    .map(|i| gradient[i] * local.internal_traction_local_pa.canonical_values()[i])
                    .sum::<f64>();
        }
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled.into());
        }
        if force.iter().any(|f| !f.is_finite()) {
            return Err(ContactHistoryError::InvalidHistory);
        }
        let mut secondary_shape = [0.0; 6];
        let secondary_count = self.reference.shape_values().len();
        secondary_shape[..secondary_count].copy_from_slice(self.reference.shape_values());
        Ok(ContactPointTrial {
            history,
            local,
            kinematics,
            reference_area_m2: self.reference.area_weight_m2,
            query_force_n: [force[0], force[1], force[2]],
            primary_force_n: std::array::from_fn(|i| {
                if i < point.shape_values().len() {
                    [force[3 + 3 * i], force[4 + 3 * i], force[5 + 3 * i]]
                } else {
                    [0.0; 3]
                }
            }),
            secondary_shape,
            secondary_count,
        })
    }
}
