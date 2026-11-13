use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{CanonicalConnectorEntry, CanonicalStructuralConnectorPlan};
use crate::assembly::dofs::{StructuralDofKind, StructuralDofLayout};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum CanonicalConnectorResponse {
    AxialSpring {
        connector_id: String,
        extension_meters: f64,
        axial_force_newtons: f64,
        resultant_force_newtons: [f64; 3],
        stored_energy_joules: f64,
    },
    AxialDashpot {
        connector_id: String,
        relative_axial_velocity_meters_per_second: f64,
        axial_force_newtons: f64,
        resultant_force_newtons: [f64; 3],
        stored_energy_joules: f64,
        dissipated_power_watts: f64,
    },
    IsotropicGroundSpring {
        connector_id: String,
        displacement_meters: [f64; 3],
        resultant_force_newtons: [f64; 3],
        stored_energy_joules: f64,
    },
    SurfaceNormalFoundation {
        connector_id: String,
        resultant_force_newtons: [f64; 3],
        stored_energy_joules: f64,
    },
}

impl CanonicalConnectorResponse {
    pub fn connector_id(&self) -> &str {
        match self {
            Self::AxialSpring { connector_id, .. }
            | Self::AxialDashpot { connector_id, .. }
            | Self::IsotropicGroundSpring { connector_id, .. }
            | Self::SurfaceNormalFoundation { connector_id, .. } => connector_id,
        }
    }

    pub const fn resultant_force_newtons(&self) -> [f64; 3] {
        match self {
            Self::AxialSpring {
                resultant_force_newtons,
                ..
            }
            | Self::AxialDashpot {
                resultant_force_newtons,
                ..
            }
            | Self::IsotropicGroundSpring {
                resultant_force_newtons,
                ..
            }
            | Self::SurfaceNormalFoundation {
                resultant_force_newtons,
                ..
            } => *resultant_force_newtons,
        }
    }

    pub const fn stored_energy_joules(&self) -> f64 {
        match self {
            Self::AxialSpring {
                stored_energy_joules,
                ..
            }
            | Self::AxialDashpot {
                stored_energy_joules,
                ..
            }
            | Self::IsotropicGroundSpring {
                stored_energy_joules,
                ..
            }
            | Self::SurfaceNormalFoundation {
                stored_energy_joules,
                ..
            } => *stored_energy_joules,
        }
    }

    pub const fn dissipated_power_watts(&self) -> f64 {
        match self {
            Self::AxialDashpot {
                dissipated_power_watts,
                ..
            } => *dissipated_power_watts,
            Self::AxialSpring { .. }
            | Self::IsotropicGroundSpring { .. }
            | Self::SurfaceNormalFoundation { .. } => 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalStructuralConnectorResult {
    pub schema_version: u16,
    pub plan: CanonicalStructuralConnectorPlan,
    pub responses: Vec<CanonicalConnectorResponse>,
    pub total_stored_energy_joules: f64,
    pub total_dissipated_power_watts: f64,
}

impl CanonicalStructuralConnectorResult {
    pub fn recover_static(
        plan: CanonicalStructuralConnectorPlan,
        layout: &StructuralDofLayout,
        solution: &[f64],
    ) -> Result<Self, ConnectorRecoveryError> {
        Self::recover(plan, layout, solution, None)
    }

    pub fn recover_dynamic(
        plan: CanonicalStructuralConnectorPlan,
        layout: &StructuralDofLayout,
        displacement: &[f64],
        velocity: &[f64],
    ) -> Result<Self, ConnectorRecoveryError> {
        Self::recover(plan, layout, displacement, Some(velocity))
    }

    fn recover(
        plan: CanonicalStructuralConnectorPlan,
        layout: &StructuralDofLayout,
        displacement: &[f64],
        velocity: Option<&[f64]>,
    ) -> Result<Self, ConnectorRecoveryError> {
        plan.validate()?;
        validate_state_length("displacement", layout, displacement)?;
        if let Some(velocity) = velocity {
            validate_state_length("velocity", layout, velocity)?;
        }
        let responses = plan
            .entries
            .iter()
            .map(|entry| recover_entry(entry, layout, displacement, velocity))
            .collect::<Result<Vec<_>, _>>()?;
        let result = Self {
            schema_version: 1,
            total_stored_energy_joules: responses
                .iter()
                .map(CanonicalConnectorResponse::stored_energy_joules)
                .sum(),
            total_dissipated_power_watts: responses
                .iter()
                .map(CanonicalConnectorResponse::dissipated_power_watts)
                .sum(),
            plan,
            responses,
        };
        result.validate()?;
        Ok(result)
    }

    pub fn validate(&self) -> Result<(), ConnectorRecoveryError> {
        self.plan.validate()?;
        if self.schema_version != 1
            || self.responses.len() != self.plan.entries.len()
            || !self
                .responses
                .iter()
                .zip(&self.plan.entries)
                .all(|(response, entry)| response_matches_entry(response, entry))
            || self.responses.iter().any(|response| {
                response
                    .resultant_force_newtons()
                    .iter()
                    .any(|value| !value.is_finite())
                    || !response.stored_energy_joules().is_finite()
                    || response.stored_energy_joules() < 0.0
                    || !response.dissipated_power_watts().is_finite()
                    || response.dissipated_power_watts() < 0.0
            })
        {
            return Err(ConnectorRecoveryError::Invalid(
                "connector result does not match its canonical plan".into(),
            ));
        }
        let total = self
            .responses
            .iter()
            .map(CanonicalConnectorResponse::stored_energy_joules)
            .sum::<f64>();
        if !self.total_stored_energy_joules.is_finite()
            || (self.total_stored_energy_joules - total).abs() > 1.0e-10 * total.abs().max(1.0)
        {
            return Err(ConnectorRecoveryError::Invalid(
                "connector result total energy is inconsistent".into(),
            ));
        }
        let total_power = self
            .responses
            .iter()
            .map(CanonicalConnectorResponse::dissipated_power_watts)
            .sum::<f64>();
        if !self.total_dissipated_power_watts.is_finite()
            || !approximately_equal(self.total_dissipated_power_watts, total_power)
        {
            return Err(ConnectorRecoveryError::Invalid(
                "connector result total dissipated power is inconsistent".into(),
            ));
        }
        Ok(())
    }
}

fn response_matches_entry(
    response: &CanonicalConnectorResponse,
    entry: &CanonicalConnectorEntry,
) -> bool {
    if response.connector_id() != entry.connector_id() {
        return false;
    }
    match (response, entry) {
        (
            CanonicalConnectorResponse::AxialSpring {
                extension_meters,
                axial_force_newtons,
                resultant_force_newtons,
                stored_energy_joules,
                ..
            },
            CanonicalConnectorEntry::AxialSpring {
                unit_axis,
                stiffness_newtons_per_meter,
                ..
            },
        ) => {
            approximately_equal(
                *axial_force_newtons,
                stiffness_newtons_per_meter * extension_meters,
            ) && resultant_force_newtons
                .iter()
                .zip(unit_axis)
                .all(|(force, axis)| approximately_equal(*force, axial_force_newtons * axis))
                && approximately_equal(
                    *stored_energy_joules,
                    0.5 * stiffness_newtons_per_meter * extension_meters * extension_meters,
                )
        }
        (
            CanonicalConnectorResponse::AxialDashpot {
                relative_axial_velocity_meters_per_second,
                axial_force_newtons,
                resultant_force_newtons,
                stored_energy_joules,
                dissipated_power_watts,
                ..
            },
            CanonicalConnectorEntry::AxialDashpot {
                unit_axis,
                damping_newton_seconds_per_meter,
                ..
            },
        ) => {
            approximately_equal(
                *axial_force_newtons,
                damping_newton_seconds_per_meter * relative_axial_velocity_meters_per_second,
            ) && resultant_force_newtons
                .iter()
                .zip(unit_axis)
                .all(|(force, axis)| approximately_equal(*force, axial_force_newtons * axis))
                && *stored_energy_joules == 0.0
                && approximately_equal(
                    *dissipated_power_watts,
                    damping_newton_seconds_per_meter
                        * relative_axial_velocity_meters_per_second
                        * relative_axial_velocity_meters_per_second,
                )
        }
        (
            CanonicalConnectorResponse::IsotropicGroundSpring {
                displacement_meters,
                resultant_force_newtons,
                stored_energy_joules,
                ..
            },
            CanonicalConnectorEntry::IsotropicGroundSpring {
                stiffness_newtons_per_meter,
                ..
            },
        ) => {
            resultant_force_newtons
                .iter()
                .zip(displacement_meters)
                .all(|(force, displacement)| {
                    approximately_equal(*force, -stiffness_newtons_per_meter * displacement)
                })
                && approximately_equal(
                    *stored_energy_joules,
                    0.5 * stiffness_newtons_per_meter
                        * dot(*displacement_meters, *displacement_meters),
                )
        }
        (
            CanonicalConnectorResponse::SurfaceNormalFoundation { .. },
            CanonicalConnectorEntry::SurfaceNormalFoundation { .. },
        ) => true,
        _ => false,
    }
}

fn approximately_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-10 * left.abs().max(right.abs()).max(1.0)
}

fn recover_entry(
    entry: &CanonicalConnectorEntry,
    layout: &StructuralDofLayout,
    solution: &[f64],
    velocity: Option<&[f64]>,
) -> Result<CanonicalConnectorResponse, ConnectorRecoveryError> {
    match entry {
        CanonicalConnectorEntry::AxialSpring {
            connector_id,
            primary,
            secondary,
            unit_axis,
            stiffness_newtons_per_meter,
            ..
        } => {
            let primary_displacement = displacement(primary.node_index, layout, solution)?;
            let secondary_displacement = displacement(secondary.node_index, layout, solution)?;
            let extension = dot(
                *unit_axis,
                std::array::from_fn(|index| {
                    secondary_displacement[index] - primary_displacement[index]
                }),
            );
            let axial_force = stiffness_newtons_per_meter * extension;
            Ok(CanonicalConnectorResponse::AxialSpring {
                connector_id: connector_id.clone(),
                extension_meters: extension,
                axial_force_newtons: axial_force,
                resultant_force_newtons: unit_axis.map(|component| axial_force * component),
                stored_energy_joules: 0.5 * stiffness_newtons_per_meter * extension * extension,
            })
        }
        CanonicalConnectorEntry::AxialDashpot {
            connector_id,
            primary,
            secondary,
            unit_axis,
            damping_newton_seconds_per_meter,
            ..
        } => {
            let relative_velocity = if let Some(velocity) = velocity {
                let primary_velocity = displacement(primary.node_index, layout, velocity)?;
                let secondary_velocity = displacement(secondary.node_index, layout, velocity)?;
                dot(
                    *unit_axis,
                    std::array::from_fn(|index| {
                        secondary_velocity[index] - primary_velocity[index]
                    }),
                )
            } else {
                0.0
            };
            let axial_force = damping_newton_seconds_per_meter * relative_velocity;
            Ok(CanonicalConnectorResponse::AxialDashpot {
                connector_id: connector_id.clone(),
                relative_axial_velocity_meters_per_second: relative_velocity,
                axial_force_newtons: axial_force,
                resultant_force_newtons: unit_axis.map(|component| axial_force * component),
                stored_energy_joules: 0.0,
                dissipated_power_watts: damping_newton_seconds_per_meter
                    * relative_velocity
                    * relative_velocity,
            })
        }
        CanonicalConnectorEntry::IsotropicGroundSpring {
            connector_id,
            target,
            stiffness_newtons_per_meter,
            ..
        } => {
            let displacement = displacement(target.node_index, layout, solution)?;
            let force = displacement.map(|value| -stiffness_newtons_per_meter * value);
            Ok(CanonicalConnectorResponse::IsotropicGroundSpring {
                connector_id: connector_id.clone(),
                displacement_meters: displacement,
                resultant_force_newtons: force,
                stored_energy_joules: 0.5
                    * stiffness_newtons_per_meter
                    * dot(displacement, displacement),
            })
        }
        CanonicalConnectorEntry::SurfaceNormalFoundation {
            connector_id,
            modulus_newtons_per_cubic_meter,
            quadrature,
            ..
        } => {
            let mut resultant = [0.0; 3];
            let mut energy = 0.0;
            for point in quadrature {
                let mut point_displacement = [0.0; 3];
                for (node, shape) in point.nodes.iter().zip(&point.shape_values) {
                    let nodal = displacement(node.node_index, layout, solution)?;
                    for component in 0..3 {
                        point_displacement[component] += shape * nodal[component];
                    }
                }
                let normal_displacement = dot(point.unit_normal, point_displacement);
                let force_scale =
                    -modulus_newtons_per_cubic_meter * normal_displacement * point.area_weight_m2;
                for (component, value) in resultant.iter_mut().enumerate() {
                    *value += force_scale * point.unit_normal[component];
                }
                energy += 0.5
                    * modulus_newtons_per_cubic_meter
                    * normal_displacement
                    * normal_displacement
                    * point.area_weight_m2;
            }
            Ok(CanonicalConnectorResponse::SurfaceNormalFoundation {
                connector_id: connector_id.clone(),
                resultant_force_newtons: resultant,
                stored_energy_joules: energy,
            })
        }
    }
}

fn validate_state_length(
    state_name: &str,
    layout: &StructuralDofLayout,
    state: &[f64],
) -> Result<(), ConnectorRecoveryError> {
    if state.len() == layout.total_dof_count() {
        Ok(())
    } else {
        Err(ConnectorRecoveryError::Invalid(format!(
            "connector recovery {state_name} does not match the structural DOF layout"
        )))
    }
}

fn displacement(
    node: usize,
    layout: &StructuralDofLayout,
    solution: &[f64],
) -> Result<[f64; 3], ConnectorRecoveryError> {
    [
        StructuralDofKind::Ux,
        StructuralDofKind::Uy,
        StructuralDofKind::Uz,
    ]
    .map(|dof| {
        layout
            .index(node, dof)
            .and_then(|index| solution.get(index).copied())
            .ok_or_else(|| {
                ConnectorRecoveryError::Invalid(
                    "connector node lacks a solved translational degree of freedom".into(),
                )
            })
    })
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?
    .try_into()
    .map_err(|_| ConnectorRecoveryError::Invalid("connector displacement is invalid".into()))
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter().zip(right).map(|(a, b)| a * b).sum()
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ConnectorRecoveryError {
    #[error(transparent)]
    Contract(#[from] super::ConnectorContractError),
    #[error("connector recovery is invalid: {0}")]
    Invalid(String),
}

#[cfg(test)]
mod tests;
