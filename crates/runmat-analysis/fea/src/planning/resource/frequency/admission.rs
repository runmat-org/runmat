//! Original compiler admission and reusable frequency resource projections.
use super::*;

impl StructuralFrequencyEntryResourceProjection {
    pub fn new(
        mesh: &SolverMeshArtifact,
        constraints: &CanonicalStructuralConstraintPlan,
        workload: &StructuralFrequencyWorkload,
        states: StructuralFrequencyStateInputs<'_>,
        accelerators: &[AcceleratorRequest],
    ) -> Result<Self, ResourcePlanningError> {
        let compiled = super::super::constraint_compilation::compile_original_inventory(
            workload.model(),
            Some(mesh),
            constraints,
            workload
                .procedure_step()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?
                .accuracy
                .resources
                .resident_byte_budget(),
        )?;
        Self::new_from_compilation(mesh, &compiled, workload, states, accelerators)
    }
    pub fn new_from_compilation(
        mesh: &SolverMeshArtifact,
        compiled: &crate::constraints::ConstraintCompilation<'_>,
        workload: &StructuralFrequencyWorkload,
        states: StructuralFrequencyStateInputs<'_>,
        accelerators: &[AcceleratorRequest],
    ) -> Result<Self, ResourcePlanningError> {
        let peak = super::super::constraint_compilation::original_compilation_peak(
            compiled,
            workload.model(),
            Some(mesh),
            compiled.plan(),
        )?;
        Self::derive_entry(mesh, compiled.plan(), peak, workload, states, accelerators)
    }
    pub(super) fn for_recovery(
        mesh: &SolverMeshArtifact,
        constraints: &CanonicalStructuralConstraintPlan,
        workload: &StructuralFrequencyWorkload,
        states: StructuralFrequencyStateInputs<'_>,
        accelerators: &[AcceleratorRequest],
    ) -> Result<Self, ResourcePlanningError> {
        let compiled = super::super::constraint_compilation::compile_original_inventory(
            workload.model(),
            Some(mesh),
            constraints,
            workload
                .procedure_step()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?
                .accuracy
                .resources
                .resident_byte_budget(),
        )?;
        Self::for_recovery_from_compilation(mesh, &compiled, workload, states, accelerators)
    }
    pub fn for_recovery_from_compilation(
        mesh: &SolverMeshArtifact,
        compiled: &crate::constraints::ConstraintCompilation<'_>,
        workload: &StructuralFrequencyWorkload,
        states: StructuralFrequencyStateInputs<'_>,
        accelerators: &[AcceleratorRequest],
    ) -> Result<Self, ResourcePlanningError> {
        let peak = super::super::constraint_compilation::original_compilation_peak(
            compiled,
            workload.model(),
            Some(mesh),
            compiled.plan(),
        )?;
        Self::derive_recovery(mesh, compiled.plan(), peak, workload, states, accelerators)
    }
}

/// Builds the exact CPU resource envelope for one independently scheduled frequency entry.
pub fn build_structural_frequency_entry_resource_plan(
    mesh: &SolverMeshArtifact,
    constraints: &CanonicalStructuralConstraintPlan,
    workload: &StructuralFrequencyWorkload,
    request: &StructuralFrequencyEntryRequest,
    states: StructuralFrequencyStateInputs<'_>,
    accelerator_requests: &[AcceleratorRequest],
    prepared: PreparedPhaseResourceFacts,
    local_execution: LocalExecutionPlan,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    StructuralFrequencyEntryResourceProjection::new(
        mesh,
        constraints,
        workload,
        states,
        accelerator_requests,
    )?
    .plan_for(
        request,
        prepared.inputs,
        prepared.output,
        prepared.output_set,
        local_execution,
    )
}

/// Builds the exact CPU resource envelope for one independently scheduled frequency recovery.
pub fn build_structural_frequency_recovery_resource_plan(
    mesh: &SolverMeshArtifact,
    constraints: &CanonicalStructuralConstraintPlan,
    workload: &StructuralFrequencyWorkload,
    recovery: StructuralFrequencyRecoveryResourceInput<'_>,
    states: StructuralFrequencyStateInputs<'_>,
    accelerator_requests: &[AcceleratorRequest],
    prepared: PreparedPhaseResourceFacts,
    local_execution: LocalExecutionPlan,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    StructuralFrequencyEntryResourceProjection::for_recovery(
        mesh,
        constraints,
        workload,
        states,
        accelerator_requests,
    )?
    .recovery_plan_for(
        recovery.request,
        recovery.result,
        prepared.inputs,
        prepared.output,
        prepared.output_set,
        local_execution,
    )
}

/// Reuse original bounded compilation while deriving the canonical phase request.
pub fn build_structural_frequency_entry_resource_plan_from_compilation(
    mesh: &SolverMeshArtifact,
    compiled: &crate::constraints::ConstraintCompilation<'_>,
    workload: &StructuralFrequencyWorkload,
    request: &StructuralFrequencyEntryRequest,
    states: StructuralFrequencyStateInputs<'_>,
    accelerator_requests: &[AcceleratorRequest],
    prepared: PreparedPhaseResourceFacts,
    local_execution: LocalExecutionPlan,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    StructuralFrequencyEntryResourceProjection::new_from_compilation(
        mesh,
        compiled,
        workload,
        states,
        accelerator_requests,
    )?
    .plan_for(
        request,
        prepared.inputs,
        prepared.output,
        prepared.output_set,
        local_execution,
    )
}

/// Reuse original bounded compilation while deriving the canonical phase request.
pub fn build_structural_frequency_recovery_resource_plan_from_compilation(
    mesh: &SolverMeshArtifact,
    compiled: &crate::constraints::ConstraintCompilation<'_>,
    workload: &StructuralFrequencyWorkload,
    recovery: StructuralFrequencyRecoveryResourceInput<'_>,
    states: StructuralFrequencyStateInputs<'_>,
    accelerator_requests: &[AcceleratorRequest],
    prepared: PreparedPhaseResourceFacts,
    local_execution: LocalExecutionPlan,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    StructuralFrequencyEntryResourceProjection::for_recovery_from_compilation(
        mesh,
        compiled,
        workload,
        states,
        accelerator_requests,
    )?
    .recovery_plan_for(
        recovery.request,
        recovery.result,
        prepared.inputs,
        prepared.output,
        prepared.output_set,
        local_execution,
    )
}
