use std::f64::consts::TAU;

use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    assembly::{
        complex_structural::{
            assemble_complex_structural_system, ComplexStructuralAssemblySummary,
        },
        identity::structural_operator_digest,
        StructuralFormulationRequirement,
    },
    contracts::{SolverMethod, SolverPreconditioner},
    execution::{
        structural_identity::{
            derive_structural_prepared_identities, structural_contract_digest,
            StructuralPreparedIdentityInput,
        },
        AnalysisExecutionError, AnalysisExecutionResult, CanonicalSemanticState,
        ExecutionPlanIdentity, FieldRecoveryRequest, FieldRecoverySource, StepIdentity,
        StructuralFrequencyEntryRequest, StructuralFrequencyEntryResult,
        StructuralFrequencyEntrySolverEvidence, StructuralFrequencyPhysicalSample,
        StructuralFrequencyProblemIdentity, StructuralFrequencyScope,
        StructuralFrequencyScopeInput, StructuralFrequencySolutionSpace,
        StructuralFrequencySolverEvidenceInput, StructuralFrequencySolverPlan,
        StructuralFrequencyWorkload, StructuralPhasorConvention, StudyIdentity,
    },
    post::{
        complex_structural_field_layout, recover_complex_constraint_group_observable_fields,
        recover_complex_structural_result_fields, ComplexStructuralFieldCardinality,
        ComplexStructuralFieldState,
    },
    results::field::{RecoveredFieldArtifact, StructuralRecoveredFieldTopology},
    solve::{
        backend::kind::LinearAlgebraBackendKind,
        dynamics::{
            evaluate_harmonic_physical_state, solve_direct_harmonic, solve_modal_harmonic,
            ModalHarmonicLoad, ModalReducedSystem,
        },
    },
    AnalysisRunIdentity,
};

/// Re-derives the physical formulation and admits a published frequency study directly from its
/// typed multi-sample result.
pub fn admit_structural_frequency_publication(
    workload: &StructuralFrequencyWorkload,
    solver_mesh: &SolverMeshArtifact,
    result: &crate::execution::StructuralFrequencySolverResult,
    publication: &crate::execution::StructuralFrequencyPublicationManifest,
) -> AnalysisExecutionResult<AnalysisRunIdentity> {
    if workload.prestress_transition().is_some() {
        return Err(AnalysisExecutionError::Invalid(
            "prestressed structural frequency publication requires its exact base state".into(),
        ));
    }
    admit_structural_frequency_publication_with_state(
        workload,
        solver_mesh,
        result,
        publication,
        None,
        None,
    )
}

pub fn admit_prestressed_structural_frequency_publication(
    workload: &StructuralFrequencyWorkload,
    solver_mesh: &SolverMeshArtifact,
    prestress_state: &CanonicalSemanticState,
    result: &crate::execution::StructuralFrequencySolverResult,
    publication: &crate::execution::StructuralFrequencyPublicationManifest,
) -> AnalysisExecutionResult<AnalysisRunIdentity> {
    admit_structural_frequency_publication_with_state(
        workload,
        solver_mesh,
        result,
        publication,
        Some(prestress_state),
        None,
    )
}

pub fn admit_prestressed_modal_structural_frequency_publication(
    workload: &StructuralFrequencyWorkload,
    solver_mesh: &SolverMeshArtifact,
    prestress_state: &CanonicalSemanticState,
    modal_state: &CanonicalSemanticState,
    result: &crate::execution::StructuralFrequencySolverResult,
    publication: &crate::execution::StructuralFrequencyPublicationManifest,
) -> AnalysisExecutionResult<AnalysisRunIdentity> {
    admit_structural_frequency_publication_with_state(
        workload,
        solver_mesh,
        result,
        publication,
        Some(prestress_state),
        Some(modal_state),
    )
}

fn admit_structural_frequency_publication_with_state(
    workload: &StructuralFrequencyWorkload,
    solver_mesh: &SolverMeshArtifact,
    result: &crate::execution::StructuralFrequencySolverResult,
    publication: &crate::execution::StructuralFrequencyPublicationManifest,
    prestress_state: Option<&CanonicalSemanticState>,
    modal_state: Option<&CanonicalSemanticState>,
) -> AnalysisExecutionResult<AnalysisRunIdentity> {
    workload.validate()?;
    solver_mesh
        .validate_canonical()
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    result.validate()?;
    publication.validate_for_workload(workload)?;
    publication.receipt().validate_for_result(result)?;
    if publication.result() != result.manifest() {
        return Err(AnalysisExecutionError::Identity(
            "structural frequency publication differs from its physical result",
        ));
    }
    let assembly = assemble_entry(
        workload,
        solver_mesh,
        workload.plan().reference_frequency_hz(),
        prestress_state,
    )?;
    match (workload.solver(), modal_state) {
        (StructuralFrequencySolverPlan::Direct { .. }, None) => {}
        (StructuralFrequencySolverPlan::ModalReduced { basis_state }, Some(state)) => {
            state.validate()?;
            let topology_digest =
                runmat_execution::Digest::from_bytes(*solver_mesh.canonical_digest.bytes());
            let basis = state.payload.modal_basis().ok_or_else(|| {
                AnalysisExecutionError::Invalid(
                    "modal structural frequency publication requires a modal-basis state".into(),
                )
            })?;
            if state.identity != *basis_state || state.topology_digest != topology_digest {
                return Err(AnalysisExecutionError::Identity(
                    "modal structural frequency publication names another basis or topology",
                ));
            }
            ModalReducedSystem::project(&assembly.real, basis)
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        }
        _ => {
            return Err(AnalysisExecutionError::Invalid(
                "structural frequency solver and modal-state presence differ".into(),
            ))
        }
    }
    let identity = super::admission::admit_structural(
        &assembly.real,
        workload.procedure_step()?,
        workload.formulation_requirement(),
    )
    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    if identity.solution_space() != workload.solver().solution_space() {
        return Err(AnalysisExecutionError::Identity(
            "admitted structural frequency formulation has another solution space",
        ));
    }
    Ok(identity)
}

/// Canonical physical scope plus standard execution identities for frequency fan-out.
pub struct PreparedStructuralFrequency {
    pub study: StudyIdentity,
    pub step: StepIdentity,
    pub execution_plan: ExecutionPlanIdentity,
    pub scope: StructuralFrequencyScope,
}

pub fn prepare_structural_frequency(
    workload: &StructuralFrequencyWorkload,
    solver_mesh: &SolverMeshArtifact,
) -> AnalysisExecutionResult<PreparedStructuralFrequency> {
    if workload.prestress_transition().is_some() {
        return Err(AnalysisExecutionError::Invalid(
            "prestressed structural frequency preparation requires its exact base state".into(),
        ));
    }
    prepare_structural_frequency_with_state(workload, solver_mesh, None)
}

pub fn prepare_prestressed_structural_frequency(
    workload: &StructuralFrequencyWorkload,
    solver_mesh: &SolverMeshArtifact,
    prestress_state: &CanonicalSemanticState,
) -> AnalysisExecutionResult<PreparedStructuralFrequency> {
    prepare_structural_frequency_with_state(workload, solver_mesh, Some(prestress_state))
}

fn prepare_structural_frequency_with_state(
    workload: &StructuralFrequencyWorkload,
    solver_mesh: &SolverMeshArtifact,
    prestress_state: Option<&CanonicalSemanticState>,
) -> AnalysisExecutionResult<PreparedStructuralFrequency> {
    workload.validate()?;
    solver_mesh
        .validate_canonical()
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    let reference = assemble_entry(
        workload,
        solver_mesh,
        workload.plan().reference_frequency_hz(),
        prestress_state,
    )?;
    // The admitted structural formulation owns frequency-invariant K/M/C, constraints, DOF
    // layout, and recovery maps. Frequency changes only the sampled complex excitation and the
    // dynamic coefficients applied by each entry solve, so one reference assembly establishes the
    // shared physical scope without performing an O(entry-count) driver-side assembly scan.
    derive_prepared_frequency(workload, solver_mesh, &reference)
}

fn derive_prepared_frequency(
    workload: &StructuralFrequencyWorkload,
    solver_mesh: &SolverMeshArtifact,
    reference: &ComplexStructuralAssemblySummary,
) -> AnalysisExecutionResult<PreparedStructuralFrequency> {
    let step = workload.procedure_step()?;
    let operator_digest = structural_operator_digest(&reference.real)
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    let identities = derive_structural_prepared_identities(StructuralPreparedIdentityInput {
        model: workload.model(),
        step,
        solver_mesh,
        assembly: &reference.real,
        input_state: workload
            .prestress_transition()
            .map(|transition| transition.identity),
        excitation_contract: &(
            &workload.model().signals,
            &workload.model().excitations,
            &workload.model().excitation_cases,
            workload.plan(),
        ),
        solver_contract: &(workload.backend(), workload.solver()),
        execution_contract: &(
            "independent-frequency-entry-fanout",
            workload.plan().identity(),
        ),
        kinematics_contract: &(&workload.model().constraints, workload.plan()),
    })?;
    let physical_problem_digest = structural_contract_digest(
        "structural-frequency-physical-problem",
        &(
            &workload.model().geometry_id,
            workload.model().geometry_revision,
            workload.model().units,
            &workload.model().frame,
            &workload.model().materials,
            &workload.model().material_assignments,
            &workload.model().structural,
            &workload.model().constraints,
            &workload.model().boundaries,
            &workload.model().connectors,
            &workload.model().signals,
            &workload.model().excitations,
            &workload.model().excitation_cases,
            &step.activations,
            workload.plan(),
            workload.prestress_transition(),
        ),
    )?;
    let scope = StructuralFrequencyScope::new(StructuralFrequencyScopeInput {
        problem: StructuralFrequencyProblemIdentity::from_physical_digest(physical_problem_digest)?,
        plan: workload.plan().clone(),
        discretization: identities.discretization,
        constraint_plan: identities.constraint_plan,
        stiffness_operator: identities.stiffness_operator,
        mass_operator: identities.mass_operator,
        damping_operator: identities.damping_operator,
        operator_digest,
        excitations: identities.excitations,
        dof_layout: identities.dof_layout,
        formulation: identities.formulation,
        kinematics: identities.kinematics,
        physical_dof_count: reference.real.dof_count,
        phasor_convention: StructuralPhasorConvention::PeakAmplitude,
    })?;
    Ok(PreparedStructuralFrequency {
        study: identities.study,
        step: identities.step,
        execution_plan: identities.execution_plan,
        scope,
    })
}

pub fn execute_structural_frequency_entry(
    workload: &StructuralFrequencyWorkload,
    solver_mesh: &SolverMeshArtifact,
    request: &StructuralFrequencyEntryRequest,
    modal_state: Option<&CanonicalSemanticState>,
) -> AnalysisExecutionResult<StructuralFrequencyEntryResult> {
    if workload.prestress_transition().is_some() {
        return Err(AnalysisExecutionError::Invalid(
            "prestressed structural frequency execution requires its exact base state".into(),
        ));
    }
    execute_structural_frequency_entry_with_state(workload, solver_mesh, request, None, modal_state)
}

pub fn execute_prestressed_structural_frequency_entry(
    workload: &StructuralFrequencyWorkload,
    solver_mesh: &SolverMeshArtifact,
    request: &StructuralFrequencyEntryRequest,
    prestress_state: &CanonicalSemanticState,
    modal_state: Option<&CanonicalSemanticState>,
) -> AnalysisExecutionResult<StructuralFrequencyEntryResult> {
    execute_structural_frequency_entry_with_state(
        workload,
        solver_mesh,
        request,
        Some(prestress_state),
        modal_state,
    )
}

fn execute_structural_frequency_entry_with_state(
    workload: &StructuralFrequencyWorkload,
    solver_mesh: &SolverMeshArtifact,
    request: &StructuralFrequencyEntryRequest,
    prestress_state: Option<&CanonicalSemanticState>,
    modal_state: Option<&CanonicalSemanticState>,
) -> AnalysisExecutionResult<StructuralFrequencyEntryResult> {
    let (_, assembly) = prepare_entry_execution(workload, solver_mesh, request, prestress_state)?;
    if modal_state.is_some_and(|state| {
        state.topology_digest
            != runmat_execution::Digest::from_bytes(*solver_mesh.canonical_digest.bytes())
    }) {
        return Err(AnalysisExecutionError::Identity(
            "modal structural frequency state names another solver topology",
        ));
    }
    let omega = TAU * request.entry().frequency_hz();
    execute_prepared_entry(workload, request, modal_state, assembly, omega)
}

/// Re-derives one entry's canonical physical scope without evaluating unrelated frequencies.
pub fn validate_structural_frequency_entry(
    workload: &StructuralFrequencyWorkload,
    solver_mesh: &SolverMeshArtifact,
    request: &StructuralFrequencyEntryRequest,
) -> AnalysisExecutionResult<PreparedStructuralFrequency> {
    prepare_entry_execution(workload, solver_mesh, request, None).map(|(prepared, _)| prepared)
}

/// Re-derives one prestressed entry's canonical physical scope from its exact base state.
pub fn validate_prestressed_structural_frequency_entry(
    workload: &StructuralFrequencyWorkload,
    solver_mesh: &SolverMeshArtifact,
    request: &StructuralFrequencyEntryRequest,
    prestress_state: &CanonicalSemanticState,
) -> AnalysisExecutionResult<PreparedStructuralFrequency> {
    prepare_entry_execution(workload, solver_mesh, request, Some(prestress_state))
        .map(|(prepared, _)| prepared)
}

fn prepare_entry_execution(
    workload: &StructuralFrequencyWorkload,
    solver_mesh: &SolverMeshArtifact,
    request: &StructuralFrequencyEntryRequest,
    prestress_state: Option<&CanonicalSemanticState>,
) -> AnalysisExecutionResult<(
    PreparedStructuralFrequency,
    ComplexStructuralAssemblySummary,
)> {
    request.validate_for_workload(workload)?;
    solver_mesh
        .validate_canonical()
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    let assembly = assemble_entry(
        workload,
        solver_mesh,
        request.entry().frequency_hz(),
        prestress_state,
    )?;
    let prepared = derive_prepared_frequency(workload, solver_mesh, &assembly)?;
    if request.scope() != &prepared.scope
        || request.study() != prepared.study
        || request.step() != prepared.step
        || request.execution_plan() != prepared.execution_plan
    {
        return Err(AnalysisExecutionError::Identity(
            "structural frequency entry authority or scope differs from canonical preparation",
        ));
    }
    Ok((prepared, assembly))
}

fn execute_prepared_entry(
    workload: &StructuralFrequencyWorkload,
    request: &StructuralFrequencyEntryRequest,
    modal_state: Option<&CanonicalSemanticState>,
    assembly: ComplexStructuralAssemblySummary,
    omega: f64,
) -> AnalysisExecutionResult<StructuralFrequencyEntryResult> {
    let (real, imaginary, iterations, residual_norm, relative_residual, solution_space) =
        match workload.solver() {
            StructuralFrequencySolverPlan::Direct { policy } => {
                if modal_state.is_some() {
                    return Err(AnalysisExecutionError::Invalid(
                        "direct structural frequency entry received a modal state".into(),
                    ));
                }
                let response =
                    solve_direct_harmonic(&assembly.real, &assembly.imaginary_load, omega, *policy)
                        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
                (
                    response.displacement_real,
                    response.displacement_imaginary,
                    response.iterations,
                    response.residual_norm,
                    response.relative_residual,
                    StructuralFrequencySolutionSpace::Direct,
                )
            }
            StructuralFrequencySolverPlan::ModalReduced { basis_state } => {
                let state = modal_state.ok_or_else(|| {
                    AnalysisExecutionError::Invalid(
                        "modal structural frequency entry is missing its basis state".into(),
                    )
                })?;
                state.validate()?;
                if state.identity != *basis_state {
                    return Err(AnalysisExecutionError::Identity(
                        "modal structural frequency entry received another basis state",
                    ));
                }
                let basis = state.payload.modal_basis().ok_or_else(|| {
                    AnalysisExecutionError::Invalid(
                        "modal structural frequency entry state has no modal basis".into(),
                    )
                })?;
                let reduced = ModalReducedSystem::project(&assembly.real, basis)
                    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
                let load = ModalHarmonicLoad {
                    real: reduced
                        .project_vector(assembly.real.operator.load())
                        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?,
                    imaginary: reduced
                        .project_vector(&assembly.imaginary_load)
                        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?,
                };
                let response = solve_modal_harmonic(&reduced, omega, &load)
                    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
                let real = reduced
                    .expand_vector(&response.displacement_real)
                    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
                let imaginary = reduced
                    .expand_vector(&response.displacement_imaginary)
                    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
                (
                    real,
                    imaginary,
                    0,
                    response.residual_norm,
                    response.relative_residual,
                    StructuralFrequencySolutionSpace::ModalReduced {
                        basis_state: *basis_state,
                        projection: reduced.evidence().clone(),
                    },
                )
            }
        };
    let physical = evaluate_harmonic_physical_state(
        &assembly.real,
        omega,
        &real,
        &imaginary,
        assembly.real.operator.load(),
        &assembly.imaginary_load,
    )
    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    let sample = StructuralFrequencyPhysicalSample::new(
        request.scope(),
        request.entry(),
        physical.displacement,
        physical.reaction,
        physical.balance,
    )?;
    let evidence = StructuralFrequencyEntrySolverEvidence::new(
        &sample,
        StructuralFrequencySolverEvidenceInput {
            solver_method: match solution_space {
                StructuralFrequencySolutionSpace::Direct => {
                    SolverMethod::StructuralDirectHarmonicGmres
                }
                StructuralFrequencySolutionSpace::ModalReduced { .. } => {
                    SolverMethod::StructuralModalHarmonicDenseDirect
                }
            },
            solver_backend: LinearAlgebraBackendKind::CpuReference,
            preconditioner: SolverPreconditioner::None,
            iterations,
            solver_residual_norm: residual_norm,
            solver_relative_residual: relative_residual,
            accelerator_evidence: Vec::new(),
        },
    )?;
    StructuralFrequencyEntryResult::new(request.clone(), sample, solution_space, evidence)
}

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
        || request.topology != topology.identity
        || result.request().scope() != &prepared.scope
    {
        return Err(AnalysisExecutionError::Identity(
            "structural frequency recovery inputs differ from the completed frequency entry",
        ));
    }
    let constraint_plan = crate::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
        workload.model(),
        Some(solver_mesh),
    )
    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
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
    let layout = complex_structural_field_layout(ComplexStructuralFieldCardinality {
        structural_nodes: assembly.real.structural_dof_layout.node_count(),
        solid_elements: assembly.real.structural_solid_recovery.len(),
        beam_elements: assembly.real.structural_beam_recovery.len(),
        shell_elements: assembly.real.structural_shell_recovery.len(),
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
    RecoveredFieldArtifact::new(
        request.identity(),
        request.source,
        topology.identity.discretization_digest(),
        layout.entity_counts,
        fields,
    )
    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))
}

fn assemble_entry(
    workload: &StructuralFrequencyWorkload,
    solver_mesh: &SolverMeshArtifact,
    frequency_hz: f64,
    prestress_state: Option<&CanonicalSemanticState>,
) -> AnalysisExecutionResult<ComplexStructuralAssemblySummary> {
    let mut assembly = assemble_complex_structural_system(
        workload.model(),
        workload.procedure_step()?,
        frequency_hz,
        Some(solver_mesh.clone()),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    let topology_digest =
        runmat_execution::Digest::from_bytes(*solver_mesh.canonical_digest.bytes());
    let linearization = match (workload.prestress_transition(), prestress_state) {
        (None, None) => None,
        (Some(transition), Some(state)) => Some(
            super::prestress::structural_linearization_from_state(
                transition,
                state,
                topology_digest,
            )
            .map_err(AnalysisExecutionError::Invalid)?,
        ),
        _ => {
            return Err(AnalysisExecutionError::Invalid(
                "structural frequency workload and prestress-state presence differ".into(),
            ));
        }
    };
    if let Some(linearization) = linearization {
        assembly.real =
            super::prestress::prestressed_structural_summary(&assembly.real, linearization)
                .map_err(AnalysisExecutionError::Invalid)?;
    }
    Ok(assembly)
}
