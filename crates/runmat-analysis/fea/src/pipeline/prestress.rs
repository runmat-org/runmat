use crate::{
    algebra::{
        AssembledRealSystem, CanonicalRealOperators, MatrixProperties, OperatorRole,
        RealSparseMatrix,
    },
    assembly::AssemblySummary,
    execution::{CanonicalSemanticState, CanonicalStateTransition},
    pipeline::structural_linearization::StructuralLinearizationArtifact,
};
use runmat_analysis_core::StateKind;
use runmat_execution::Digest;

pub(crate) fn structural_linearization_from_state<'a>(
    transition: &CanonicalStateTransition,
    state: &'a CanonicalSemanticState,
    topology_digest: Digest,
) -> Result<&'a StructuralLinearizationArtifact, String> {
    transition.validate().map_err(|error| error.to_string())?;
    state.validate().map_err(|error| error.to_string())?;
    if state.kind != StateKind::GeometricStiffness
        || state.identity != transition.source_state
        || state.topology_digest != topology_digest
        || transition.target_topology_digest != topology_digest
    {
        return Err(
            "prestress transition, geometric-stiffness state, and consumer topology differ".into(),
        );
    }
    state
        .payload
        .structural_linearization()
        .ok_or_else(|| "prestress state does not carry a structural-linearization payload".into())
}

/// Builds the small-perturbation structural operator about one verified preload state.
///
/// The current model remains authoritative for elastic stiffness, mass, damping, constraints, and
/// recovery. The transferred state contributes only its geometric stiffness after proving that it
/// was formed from the same elastic operator. Numerical coercivity of the resulting tangent is
/// deliberately established by the consuming solver after constraint reduction.
pub(crate) fn prestressed_structural_summary(
    summary: &AssemblySummary,
    linearization: &StructuralLinearizationArtifact,
) -> Result<AssemblySummary, String> {
    linearization
        .validate()
        .map_err(|error| error.to_string())?;
    let elastic = summary.operator.operators().stiffness();
    let elastic_digest = elastic
        .numerical_identity()
        .map_err(|error| error.to_string())?;
    if linearization.elastic_stiffness_digest() != elastic_digest
        || linearization.geometric_stiffness().rows() != summary.dof_count
        || linearization.geometric_stiffness().columns() != summary.dof_count
    {
        return Err(
            "prestress state differs from the current elastic operator or DOF space".into(),
        );
    }
    let tangent = RealSparseMatrix::scaled_sum(
        &[(1.0, elastic), (1.0, linearization.geometric_stiffness())],
        OperatorRole::EffectiveTangent,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )
    .map_err(|error| error.to_string())?;
    let operators = CanonicalRealOperators::try_new(
        tangent,
        summary.operator.operators().mass().clone(),
        summary.operator.operators().damping().clone(),
    )
    .map_err(|error| error.to_string())?;
    let operator = AssembledRealSystem::try_new(
        operators,
        summary.operator.constrained_dofs().to_vec(),
        summary.operator.load().to_vec(),
    )
    .map_err(|error| error.to_string())?;
    let mut prestressed = summary.clone();
    prestressed.operator = operator;
    Ok(prestressed)
}
