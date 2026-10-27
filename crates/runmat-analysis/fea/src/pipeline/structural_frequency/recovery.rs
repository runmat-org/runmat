//! Original frequency-entry field recovery and source-bound constraint admission.
use super::*;
use crate::{
    execution::{FieldRecoveryRequest, FieldRecoverySource},
    post::{
        complex_structural_field_layout, recover_complex_connector_force_field,
        recover_complex_constraint_group_observable_fields,
        recover_complex_structural_result_fields, ComplexStructuralFieldCardinality,
        ComplexStructuralFieldState,
    },
    results::field::{RecoveredFieldArtifact, StructuralRecoveredFieldTopology},
};

/// Recovers formulation-owned complex fields for one independently completed frequency entry.
///
/// Recovery remains colocatable with the entry solve: it reassembles only this exact frequency,
/// consumes solver-authoritative dynamic reactions, and retains authored-constraint ownership as
/// a separate complex field domain.
pub fn recover_structural_frequency_entry_fields(
    workload: &StructuralFrequencyWorkload,
    solver_mesh: &SolverMeshArtifact,
    request: &FieldRecoveryRequest,
    result: &StructuralFrequencyEntryResult,
) -> AnalysisExecutionResult<RecoveredFieldArtifact> {
    if workload.prestress_transition().is_some() {
        return Err(AnalysisExecutionError::Invalid(
            "prestressed structural frequency recovery requires its exact base state".into(),
        ));
    }
    recover_structural_frequency_entry_fields_with_state(
        workload,
        solver_mesh,
        request,
        result,
        None,
    )
}

pub fn recover_prestressed_structural_frequency_entry_fields(
    workload: &StructuralFrequencyWorkload,
    solver_mesh: &SolverMeshArtifact,
    request: &FieldRecoveryRequest,
    result: &StructuralFrequencyEntryResult,
    prestress_state: &CanonicalSemanticState,
) -> AnalysisExecutionResult<RecoveredFieldArtifact> {
    recover_structural_frequency_entry_fields_with_state(
        workload,
        solver_mesh,
        request,
        result,
        Some(prestress_state),
    )
}

fn recover_structural_frequency_entry_fields_with_state(
    workload: &StructuralFrequencyWorkload,
    solver_mesh: &SolverMeshArtifact,
    request: &FieldRecoveryRequest,
    result: &StructuralFrequencyEntryResult,
    prestress_state: Option<&CanonicalSemanticState>,
) -> AnalysisExecutionResult<RecoveredFieldArtifact> {
    request.validate()?;
    result.validate()?;
    let (prepared, assembly) =
        prepare_entry_execution(workload, solver_mesh, result.request(), prestress_state)?;
    let topology = StructuralRecoveredFieldTopology::derive(workload.model(), Some(solver_mesh))
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    if request.producer != prepared.step
        || request.source != FieldRecoverySource::StructuralFrequencyEntry(result.identity())
        || request.topology.require_structural()? != topology.identity
        || result.request().scope() != &prepared.scope
    {
        return Err(AnalysisExecutionError::Identity(
            "structural frequency recovery inputs differ from the completed frequency entry",
        ));
    }
    let constraint_plan = crate::constraints::CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
        workload.model(),
        Some(solver_mesh),
        workload.procedure_step()?.accuracy.resources.resident_byte_budget(),
    ).map(|compiled| compiled.into_plan())
    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    let expected_recovery_plan = crate::execution::FieldRecoveryPlan::resolve_structural(
        workload.procedure_step()?,
        topology,
        constraint_plan.groups.len(),
    )?;
    if request.plan != expected_recovery_plan {
        return Err(AnalysisExecutionError::Identity(
            "structural frequency recovery plan differs from its workload and topology",
        ));
    }
    let sample = result.sample();
    let frequency_hz = sample.entry().frequency_hz();
    let mut fields = recover_complex_structural_result_fields(
        &assembly.real,
        frequency_hz,
        ComplexStructuralFieldState {
            displacement_real: sample.displacement().real(),
            displacement_imaginary: sample.displacement().imaginary(),
            reaction_real: sample.reaction().real(),
            reaction_imaginary: sample.reaction().imaginary(),
        },
    )
    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    let real_constraints =
        crate::constraints::CanonicalStructuralConstraintResult::recover_from_solver_state(
            &constraint_plan,
            solver_mesh,
            sample.displacement().real(),
            sample.reaction().real(),
        )
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    let imaginary_constraints =
        crate::constraints::CanonicalStructuralConstraintResult::recover_from_solver_state(
            &constraint_plan,
            solver_mesh,
            sample.displacement().imaginary(),
            sample.reaction().imaginary(),
        )
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    fields.extend(
        recover_complex_constraint_group_observable_fields(
            &real_constraints,
            &imaginary_constraints,
            frequency_hz,
        )
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?,
    );
    let connector_plan = match assembly.real.structural_connector_plan.clone() {
        Some(plan) => plan,
        None if workload.model().connectors.is_empty() => {
            crate::connectors::CanonicalStructuralConnectorPlan::empty(
                constraint_plan.topology,
                assembly.real.structural_node_count,
            )
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?
        }
        None => {
            return Err(AnalysisExecutionError::Invalid(
                "structural frequency assembly did not publish a canonical connector plan".into(),
            ))
        }
    };
    connector_plan
        .validate_against_authored(&workload.model().connectors)
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    let connector_count = connector_plan.entries.len();
    if let Some(field) = recover_complex_connector_force_field(
        connector_plan,
        &assembly.real.structural_dof_layout,
        frequency_hz,
        sample.displacement().real(),
        sample.displacement().imaginary(),
    )
    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?
    {
        fields.push(field);
    }
    let layout = complex_structural_field_layout(ComplexStructuralFieldCardinality {
        nodes: assembly.real.structural_dof_layout.node_count(),
        volume_elements: assembly.real.structural_solid_recovery.len(),
        beam_elements: assembly.real.structural_beam_recovery.len(),
        shell_elements: assembly.real.structural_shell_recovery.len(),
        connectors: connector_count,
        constraint_groups: constraint_plan.groups.len(),
        has_rotational_dofs: assembly.real.structural_rotational_dof_count > 0,
    })
    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    let actual_storage_values = fields.iter().try_fold(0_usize, |total, field| {
        total.checked_add(
            field
                .storage()
                .as_host_f64()
                .map_or(0, |values| values.len()),
        )
    });
    if fields
        .iter()
        .map(|field| field.definition())
        .ne(layout.definitions.iter())
        || actual_storage_values != Some(layout.storage_value_count)
    {
        return Err(AnalysisExecutionError::Identity(
            "structural frequency recovery differs from its canonical field layout",
        ));
    }
    request
        .plan
        .validate_definitions(fields.iter().map(|field| field.definition().clone()))?;
    RecoveredFieldArtifact::new(
        request.identity(),
        request.source,
        topology.identity.discretization_digest(),
        layout.entity_counts,
        fields,
    )
    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))
}
