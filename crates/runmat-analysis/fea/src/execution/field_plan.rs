use runmat_analysis_core::{
    FieldLocation, ObservationQuantity, ObservationValueShape, ProcedureStep, RecoveryPolicy,
};
use serde::{Deserialize, Serialize};

use crate::results::field::{
    RecoveredFieldDefinition, RecoveredFieldEntityCounts, RecoveredFieldEntityDomain,
    RecoveredFieldSupport, StructuralRecoveredFieldTopology,
};

use super::{AnalysisExecutionError, AnalysisExecutionResult};

/// Topology-resolved work required to publish one requested physical field.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldRecoveryCost {
    pub source_entity_count: u64,
    pub output_entity_count: u64,
    pub topology_incidence_count: u64,
    pub arithmetic_operation_count: u64,
}

/// One exact field derivation requested from a family-owned recovery kernel.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldRecoveryRequirement {
    pub quantity: ObservationQuantity,
    pub source_location: FieldLocation,
    pub output_location: FieldLocation,
    pub recovery: RecoveryPolicy,
    pub value_shape: ObservationValueShape,
    pub entity_domain: RecoveredFieldEntityDomain,
    pub cost: FieldRecoveryCost,
    /// The entities of the domain the field covers (the fluid regions of a conjugate model).
    #[serde(default, skip_serializing_if = "RecoveredFieldSupport::is_complete")]
    pub support: RecoveredFieldSupport,
}

/// Closed, immutable publication plan derived from authored observations and admitted topology.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldRecoveryPlan {
    pub requirements: Vec<FieldRecoveryRequirement>,
}

impl FieldRecoveryPlan {
    pub fn resolve_structural(
        step: &ProcedureStep,
        topology: StructuralRecoveredFieldTopology,
        constraint_group_count: usize,
    ) -> AnalysisExecutionResult<Self> {
        let mut requirements = Vec::new();
        for observation in &step.observations {
            let Some(spec) = observation.spec() else {
                continue;
            };
            let (output_location, recovery) = match spec.recovery {
                RecoveryPolicy::Native => (spec.source_location, RecoveryPolicy::Native),
                RecoveryPolicy::ShapeFunctionInterpolation => {
                    (spec.source_location, RecoveryPolicy::Native)
                }
                RecoveryPolicy::VolumeWeightedAverage
                    if spec.source_location == FieldLocation::Element
                        && spec.output_location == FieldLocation::Node =>
                {
                    (FieldLocation::Node, RecoveryPolicy::VolumeWeightedAverage)
                }
                _ => {
                    return Err(AnalysisExecutionError::Invalid(format!(
                        "structural field recovery does not implement {:?} from {:?} to {:?}",
                        spec.recovery, spec.source_location, spec.output_location
                    )))
                }
            };
            let entity_domain = structural_entity_domain(spec.quantity, output_location)?;
            let source_domain = structural_entity_domain(spec.quantity, spec.source_location)?;
            let source_entity_count = entity_count(
                topology.entity_counts,
                source_domain,
                constraint_group_count,
            )?;
            let output_entity_count = entity_count(
                topology.entity_counts,
                entity_domain,
                constraint_group_count,
            )?;
            let topology_incidence_count = if recovery == RecoveryPolicy::VolumeWeightedAverage {
                u64::try_from(topology.solid_node_incidence_count).map_err(|_| {
                    AnalysisExecutionError::Invalid(
                        "solid incidence count exceeds portable recovery limits".into(),
                    )
                })?
            } else {
                0
            };
            let arithmetic_operation_count = if recovery == RecoveryPolicy::VolumeWeightedAverage {
                volume_weighted_operation_upper_bound(
                    source_entity_count,
                    output_entity_count,
                    topology_incidence_count,
                )?
            } else {
                source_entity_count
            };
            let requirement = FieldRecoveryRequirement {
                support: Default::default(),
                quantity: spec.quantity,
                source_location: spec.source_location,
                output_location,
                recovery,
                value_shape: spec.value_shape.clone(),
                entity_domain,
                cost: FieldRecoveryCost {
                    source_entity_count,
                    output_entity_count,
                    topology_incidence_count,
                    arithmetic_operation_count,
                },
            };
            if !requirements.contains(&requirement) {
                requirements.push(requirement);
            }
        }
        Ok(Self { requirements })
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        for requirement in &self.requirements {
            if requirement.cost.source_entity_count == 0
                || requirement.cost.output_entity_count == 0
                || requirement.value_shape.component_count().is_none()
                || (requirement.recovery == RecoveryPolicy::VolumeWeightedAverage
                    && (requirement.source_location != FieldLocation::Element
                        || requirement.output_location != FieldLocation::Node
                        || requirement.entity_domain != RecoveredFieldEntityDomain::Nodes
                        || requirement.cost.topology_incidence_count == 0))
                || (requirement.recovery == RecoveryPolicy::Native
                    && (requirement.source_location != requirement.output_location
                        || requirement.cost.topology_incidence_count != 0))
                || !matches!(
                    requirement.recovery,
                    RecoveryPolicy::Native | RecoveryPolicy::VolumeWeightedAverage
                )
            {
                return Err(AnalysisExecutionError::Invalid(
                    "field recovery plan contains an invalid or unsupported requirement".into(),
                ));
            }
        }
        Ok(())
    }

    pub fn validate_definitions(
        &self,
        definitions: impl IntoIterator<Item = RecoveredFieldDefinition>,
    ) -> AnalysisExecutionResult<()> {
        let definitions = definitions.into_iter().collect::<Vec<_>>();
        for requirement in &self.requirements {
            let matches = definitions
                .iter()
                .filter(|definition| {
                    definition.quantity() == requirement.quantity
                        && definition.value_shape() == &requirement.value_shape
                        && definition.entity_domain() == requirement.entity_domain
                        && definition.support() == &requirement.support
                        && definition.satisfies_recovery(
                            requirement.source_location,
                            requirement.output_location,
                            requirement.recovery,
                        )
                })
                .count();
            if matches != 1 {
                return Err(AnalysisExecutionError::Invalid(
                    "field recovery output does not uniquely satisfy its requested derivation"
                        .into(),
                ));
            }
        }
        Ok(())
    }
}

fn volume_weighted_operation_upper_bound(
    source_entity_count: u64,
    output_entity_count: u64,
    topology_incidence_count: u64,
) -> AnalysisExecutionResult<u64> {
    // Per incidence: weighted multiply, weighted accumulation, and volume accumulation. Per
    // source element: a conservative bound for exact signed-volume geometry. Per output node:
    // the final normalization division.
    topology_incidence_count
        .checked_mul(3)
        .and_then(|count| count.checked_add(source_entity_count.checked_mul(32)?))
        .and_then(|count| count.checked_add(output_entity_count))
        .ok_or_else(|| {
            AnalysisExecutionError::Invalid(
                "field recovery operation count exceeds portable limits".into(),
            )
        })
}

fn entity_count(
    counts: RecoveredFieldEntityCounts,
    domain: RecoveredFieldEntityDomain,
    constraint_group_count: usize,
) -> AnalysisExecutionResult<u64> {
    let count = if domain == RecoveredFieldEntityDomain::ConstraintGroups {
        constraint_group_count
    } else {
        counts.entity_count(domain)
    };
    u64::try_from(count).map_err(|_| {
        AnalysisExecutionError::Invalid("field entity count exceeds portable limits".into())
    })
}

fn structural_entity_domain(
    quantity: ObservationQuantity,
    location: FieldLocation,
) -> AnalysisExecutionResult<RecoveredFieldEntityDomain> {
    use ObservationQuantity as Quantity;
    let domain = match location {
        FieldLocation::Node => RecoveredFieldEntityDomain::Nodes,
        FieldLocation::Connector => RecoveredFieldEntityDomain::Connectors,
        FieldLocation::Constraint => RecoveredFieldEntityDomain::ConstraintGroups,
        FieldLocation::IntegrationPoint => RecoveredFieldEntityDomain::VolumeIntegrationPoints,
        FieldLocation::Element => match quantity {
            Quantity::BeamAxialForce
            | Quantity::BeamShearForce
            | Quantity::BeamTorsionMoment
            | Quantity::BeamBendingMoment
            | Quantity::BeamBendingStress
            | Quantity::BeamTorsionStress => RecoveredFieldEntityDomain::BeamElements,
            Quantity::ShellMembraneForce
            | Quantity::ShellBendingMoment
            | Quantity::ShellTransverseShear
            | Quantity::ShellVonMises => RecoveredFieldEntityDomain::ShellElements,
            _ => RecoveredFieldEntityDomain::VolumeElements,
        },
        _ => {
            return Err(AnalysisExecutionError::Invalid(format!(
                "structural field recovery cannot publish {:?} fields",
                location
            )))
        }
    };
    Ok(domain)
}

#[cfg(test)]
mod tests {
    use super::volume_weighted_operation_upper_bound;

    #[test]
    fn volume_weighted_cost_covers_geometry_incidence_and_normalization_work() {
        assert_eq!(volume_weighted_operation_upper_bound(2, 7, 8).unwrap(), 95);
        assert!(volume_weighted_operation_upper_bound(u64::MAX, 1, 1).is_err());
    }
}
