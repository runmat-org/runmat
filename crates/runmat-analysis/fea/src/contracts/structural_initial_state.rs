//! Complete initial-state realization evidence, including immutable nodal source lineage.
use runmat_analysis_core::{InitialConditionTarget, SignalId, SignalSampleApproach};
use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralInitialConditionQuantity {
    Displacement,
    Velocity,
    AngularVelocity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum StructuralInitialValueSourceEvidence {
    UniformVector3,
    FieldArtifact {
        reference: runmat_analysis_core::InitialFieldArtifactRef,
    },
    SignalSample {
        signal_id: SignalId,
        abscissa_bits: u64,
        approach: SignalSampleApproach,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppliedStructuralInitialConditionEvidence {
    pub initial_condition_id: String,
    pub quantity: StructuralInitialConditionQuantity,
    pub target: InitialConditionTarget,
    pub resolved_node_identities: Vec<StableDigest>,
    pub source: StructuralInitialValueSourceEvidence,
    /// Uniform/signal components; nodal field content is bound by its source reference.
    pub realized_component_bits: Option<[u64; 3]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum StructuralInitialStateEvidenceBasis {
    CanonicalTopology {
        discretization_identity: Digest,
        mesh_digest: Option<StableDigest>,
    },
    OperatorBoundZeroDefault {
        dof_count: usize,
        operator_identity: Digest,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "StructuralInitialStateEvidenceData")]
pub struct StructuralInitialStateEvidence {
    pub procedure_step_id: String,
    pub basis: StructuralInitialStateEvidenceBasis,
    pub total_dof_count: usize,
    pub authored_displacement_dof_count: usize,
    pub constraint_derived_displacement_dof_count: usize,
    pub zero_default_displacement_dof_count: usize,
    pub authored_velocity_dof_count: usize,
    pub constraint_derived_velocity_dof_count: usize,
    pub zero_default_velocity_dof_count: usize,
    pub direct_constrained_dof_count: usize,
    pub constraint_equation_count: usize,
    pub maximum_displacement_constraint_residual_bits: u64,
    pub maximum_velocity_constraint_residual_bits: u64,
    pub applied_conditions: Vec<AppliedStructuralInitialConditionEvidence>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralInitialStateEvidenceData {
    procedure_step_id: String,
    basis: StructuralInitialStateEvidenceBasis,
    total_dof_count: usize,
    authored_displacement_dof_count: usize,
    constraint_derived_displacement_dof_count: usize,
    zero_default_displacement_dof_count: usize,
    authored_velocity_dof_count: usize,
    constraint_derived_velocity_dof_count: usize,
    zero_default_velocity_dof_count: usize,
    direct_constrained_dof_count: usize,
    constraint_equation_count: usize,
    maximum_displacement_constraint_residual_bits: u64,
    maximum_velocity_constraint_residual_bits: u64,
    applied_conditions: Vec<AppliedStructuralInitialConditionEvidence>,
}

impl TryFrom<StructuralInitialStateEvidenceData> for StructuralInitialStateEvidence {
    type Error = String;

    fn try_from(data: StructuralInitialStateEvidenceData) -> Result<Self, Self::Error> {
        if data.procedure_step_id.is_empty() || data.total_dof_count == 0 {
            return Err("structural initial-state evidence has an empty identity domain".into());
        }
        let accounted_displacement_dofs = data
            .authored_displacement_dof_count
            .checked_add(data.constraint_derived_displacement_dof_count)
            .and_then(|count| count.checked_add(data.zero_default_displacement_dof_count));
        let accounted_velocity_dofs = data
            .authored_velocity_dof_count
            .checked_add(data.constraint_derived_velocity_dof_count)
            .and_then(|count| count.checked_add(data.zero_default_velocity_dof_count));
        if accounted_displacement_dofs != Some(data.total_dof_count)
            || accounted_velocity_dofs != Some(data.total_dof_count)
            || data.direct_constrained_dof_count > data.total_dof_count
        {
            return Err("structural initial-state evidence DOF accounting is inconsistent".into());
        }
        let displacement_residual =
            f64::from_bits(data.maximum_displacement_constraint_residual_bits);
        let velocity_residual = f64::from_bits(data.maximum_velocity_constraint_residual_bits);
        if !displacement_residual.is_finite()
            || displacement_residual < 0.0
            || displacement_residual.to_bits() == (-0.0_f64).to_bits()
            || !velocity_residual.is_finite()
            || velocity_residual < 0.0
            || velocity_residual.to_bits() == (-0.0_f64).to_bits()
            || data.applied_conditions.iter().any(|condition| {
                condition.initial_condition_id.is_empty()
                    || condition.resolved_node_identities.is_empty()
                    || condition.resolved_node_identities.iter().enumerate().any(
                        |(index, identity)| {
                            condition.resolved_node_identities[index + 1..].contains(identity)
                        },
                    )
                    || condition
                        .realized_component_bits
                        .iter().flatten()
                        .any(|bits| !f64::from_bits(*bits).is_finite())
                    || matches!((&condition.source, condition.realized_component_bits),
                        (StructuralInitialValueSourceEvidence::FieldArtifact { .. }, Some(_))
                        | (StructuralInitialValueSourceEvidence::UniformVector3 | StructuralInitialValueSourceEvidence::SignalSample { .. }, None))
                    || matches!(&condition.source,
                        StructuralInitialValueSourceEvidence::FieldArtifact { reference }
                        if reference.artifact_id.trim().is_empty() || reference.field_id.trim().is_empty()
                            || reference.content_digest.parse::<Digest>().is_err()
                            || reference.topology_digest.parse::<Digest>().is_err())
                    || matches!(
                        &condition.source,
                        StructuralInitialValueSourceEvidence::SignalSample {
                            signal_id,
                            abscissa_bits,
                            ..
                        } if signal_id.0.is_empty()
                            || !f64::from_bits(*abscissa_bits).is_finite()
                            || *abscissa_bits == (-0.0_f64).to_bits()
                    )
            })
            || data
                .applied_conditions
                .windows(2)
                .any(|pair| pair[0].initial_condition_id >= pair[1].initial_condition_id)
        {
            return Err(
                "structural initial-state evidence contains invalid realized values".into(),
            );
        }
        match data.basis {
            StructuralInitialStateEvidenceBasis::CanonicalTopology {
                discretization_identity,
                mesh_digest,
            } if discretization_identity.bytes() == &[0; 32]
                || discretization_identity == Digest::sha256([])
                || mesh_digest == Some(StableDigest::ZERO) =>
            {
                return Err(
                    "structural initial-state evidence has an invalid topology identity".into(),
                );
            }
            StructuralInitialStateEvidenceBasis::OperatorBoundZeroDefault {
                dof_count,
                operator_identity,
            } => {
                if dof_count != data.total_dof_count {
                    return Err(
                        "operator-bound structural initial-state evidence has inconsistent DOF identity"
                            .into(),
                    );
                }
                if operator_identity.bytes() == &[0; 32] || operator_identity == Digest::sha256([])
                {
                    return Err("operator-bound structural initial-state evidence has an invalid operator identity".into());
                }
                if data.authored_displacement_dof_count != 0
                    || data.authored_velocity_dof_count != 0
                    || !data.applied_conditions.is_empty()
                {
                    return Err(
                        "operator-bound structural initial-state evidence is not an implicit zero state"
                            .into(),
                    );
                }
            }
            _ => {}
        }
        Ok(Self {
            procedure_step_id: data.procedure_step_id,
            basis: data.basis,
            total_dof_count: data.total_dof_count,
            authored_displacement_dof_count: data.authored_displacement_dof_count,
            constraint_derived_displacement_dof_count: data
                .constraint_derived_displacement_dof_count,
            zero_default_displacement_dof_count: data.zero_default_displacement_dof_count,
            authored_velocity_dof_count: data.authored_velocity_dof_count,
            constraint_derived_velocity_dof_count: data.constraint_derived_velocity_dof_count,
            zero_default_velocity_dof_count: data.zero_default_velocity_dof_count,
            direct_constrained_dof_count: data.direct_constrained_dof_count,
            constraint_equation_count: data.constraint_equation_count,
            maximum_displacement_constraint_residual_bits: data
                .maximum_displacement_constraint_residual_bits,
            maximum_velocity_constraint_residual_bits: data
                .maximum_velocity_constraint_residual_bits,
            applied_conditions: data.applied_conditions,
        })
    }
}
