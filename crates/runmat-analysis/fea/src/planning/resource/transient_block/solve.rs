//! Reserve the selected algorithm's workspace, including sparse SPD proof construction.
use super::*;
use crate::{
    algebra::CoercivityWorkEnvelope, constraints::CanonicalStructuralConstraintPlan,
    planning::resource::structural_facts::derive_structural_resource_facts,
};

pub(super) fn proof_storage(
    model: &runmat_analysis_core::AnalysisModel,
    mesh: &SolverMeshArtifact,
) -> Result<u64, ResourcePlanningError> {
    let constraints = CanonicalStructuralConstraintPlan::compile_for_model(model, Some(mesh))
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let facts = derive_structural_resource_facts(model, Some(mesh), &constraints)?;
    let full = CoercivityWorkEnvelope::derive(facts.symbolic.stiffness_upper_bound())
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    // Initial acceleration and Newmark may use the full directly constrained system.
    // Affine support motion and normalized equilibrium corrections use Tᵀ A T.
    // Their proofs are sequential, so reserve the larger construction envelope.
    let reduction = facts
        .constraint_matrix
        .affine_reduction()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let reduced = if reduction.free_dof_indices.is_empty() {
        0
    } else {
        let pattern = reduction
            .reduce_sparse_pattern(facts.symbolic.stiffness_upper_bound())
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        CoercivityWorkEnvelope::derive(&pattern)
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?
            .resident_bytes_limit()
    };
    add(
        full.resident_bytes_limit().max(reduced),
        crate::algebra::LinearPreparationReuse::RESIDENT_BYTES,
        "transient retained exact solver certificate",
    )
}
