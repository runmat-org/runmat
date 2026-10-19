//! Remaining component budget inside a caller-selected ceiling or planned task request.
//! Source facts are independently derived by the caller, including the resource-plan input.
use super::{ByteCount, PreparedInputResourceFacts, ResourcePlanningError};

/// Reserve original encoded closures, decoded values and caller-owned live work before
/// compiler construction. Excludes allocator overhead/RSS; allocated execution resources
/// must still be reconciled by the shared execution owner.
pub fn constraint_compilation_available_bytes(
    maximum: u64,
    inputs: &PreparedInputResourceFacts,
    additional_retained: ByteCount,
) -> Result<u64, ResourcePlanningError> {
    let required = inputs
        .transport_closure_bytes
        .0
        .checked_add(inputs.decoded_resident_bytes_admission_budget.0)
        .and_then(|value| value.checked_add(additional_retained.0))
        .ok_or(ResourcePlanningError::Overflow(
            "constraint compilation retained inputs",
        ))?;
    maximum
        .checked_sub(required)
        .ok_or(ResourcePlanningError::AccuracyCeiling {
            resource: "constraint compilation retained inputs",
            required,
            maximum,
        })
}

/// Original construction scope and exact plan comparison, shared by structural planners.
pub(super) fn original_compilation_peak(
    compiled: &crate::constraints::ConstraintCompilation<'_>,
    model: &runmat_analysis_core::AnalysisModel,
    mesh: Option<&runmat_meshing_core::SolverMeshArtifact>,
    expected: &crate::constraints::CanonicalStructuralConstraintPlan,
) -> Result<ByteCount, ResourcePlanningError> {
    compiled
        .require_original_inputs(model, mesh)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    if compiled.plan() != expected {
        return Err(ResourcePlanningError::Invalid(
            "resource constraints differ from original compilation".into(),
        ));
    }
    Ok(ByteCount(compiled.workspace().peak_requested_bytes()))
}

/// Plan-only callers reconstruct original bounded construction; decoded counts are not evidence.
pub(super) fn compile_original_inventory<'a>(
    model: &'a runmat_analysis_core::AnalysisModel,
    mesh: Option<&'a runmat_meshing_core::SolverMeshArtifact>,
    expected: &crate::constraints::CanonicalStructuralConstraintPlan,
    maximum: u64,
) -> Result<crate::constraints::ConstraintCompilation<'a>, ResourcePlanningError> {
    let compiled = crate::constraints::CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(model, mesh, maximum)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    original_compilation_peak(&compiled, model, mesh, expected)?;
    Ok(compiled)
}
