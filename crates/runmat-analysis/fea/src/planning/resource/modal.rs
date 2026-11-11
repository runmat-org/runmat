use crate::{
    algebra::CoercivityWorkEnvelope,
    constraints::CanonicalStructuralConstraintPlan,
    execution::{CanonicalSemanticState, StructuralModalWorkload},
    results::observation::ResolvedObservationPlan,
    solve::modal::MODAL_MASS_ORTHOGONALIZATION_PASSES,
    ComputeBackend,
};
use runmat_execution::resource::{AcceleratorRequest, Capability};
use runmat_meshing_core::SolverMeshArtifact;

use super::{
    assembly_sizing::{
        ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT, FIXED_ASSEMBLY_SCRATCH_BYTES_PER_WORKER,
    },
    build_resource_plan,
    structural_facts::derive_structural_resource_facts,
    AcceleratorRequirement, AccuracyResourceDemand, AssemblyResourcePolicy, ByteCount,
    CheckpointArtifactPolicy, Count, FeaResourceDemand, FeaResourcePhase, FeaResourcePlan,
    FeaResourcePlanningInput, ModalSolveSelection, ObservationResourceFacts, PartitioningPolicy,
    PhaseResourcePolicy, PreparedStructuralModalResourceFacts, ResourcePlanningError,
    SolveResourcePolicy, StructuralModalResourceDemand, StructuralOperatorResourceDemand,
    SubspaceSolvePath, WorkerResourcePolicy, RESOURCE_ESTIMATION_MODEL_REVISION,
};

const ACCELERATOR_CG_RESIDENT_VECTOR_COUNT: u64 = 6;
const FIXED_WORKER_MEMORY_BYTES: u64 = 16 * 1024 * 1024;
const FIXED_WORKER_SCRATCH_BYTES: u64 = 8 * 1024 * 1024;

mod accelerator;
use accelerator::modal_accelerator_requirements;
mod arithmetic;
mod energy;
mod observations;
pub(super) mod output;
mod sizing;
mod subspace;
use arithmetic::{checked_add, checked_bytes, checked_mul, count};
use observations::resolve_modal_observations;
pub(super) use sizing::structural_modal_accelerator_minimum_bytes;

use sizing::{
    modal_eigensolver_working_values, modal_output_value_count, modal_phase_value_count,
    modal_result_payload_bytes, modal_workspace_bytes, prepared_output_covers_transport,
};
pub use sizing::{
    structural_modal_receipt_payload_bytes_upper_bound,
    structural_modal_result_output_resource_facts,
    structural_modal_result_payload_bytes_upper_bound,
};

/// Builds the monolithic resource envelope for one canonical modal-extraction workload.
pub fn build_structural_modal_resource_plan(
    mesh: Option<&SolverMeshArtifact>,
    constraints: &CanonicalStructuralConstraintPlan,
    observations: &ResolvedObservationPlan,
    workload: &StructuralModalWorkload,
    prestress_state: Option<&CanonicalSemanticState>,
    accelerator_requests: &[AcceleratorRequest],
    prepared: PreparedStructuralModalResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let compiled = super::constraint_compilation::compile_original_inventory(
        workload.model(),
        mesh,
        constraints,
        workload
            .procedure_step()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?
            .accuracy
            .resources
            .resident_byte_budget(),
    )?;
    build_structural_modal_resource_plan_from_compilation(
        mesh,
        &compiled,
        observations,
        workload,
        prestress_state,
        accelerator_requests,
        prepared,
    )
}

/// Reuse the exact fresh construction; detached or cold inventory is not accepted. Plans
/// within the authored ceiling alone (no machine memory is known).
pub fn build_structural_modal_resource_plan_from_compilation(
    mesh: Option<&SolverMeshArtifact>,
    compilation: &crate::constraints::ConstraintCompilation<'_>,
    observations: &ResolvedObservationPlan,
    workload: &StructuralModalWorkload,
    prestress_state: Option<&CanonicalSemanticState>,
    accelerator_requests: &[AcceleratorRequest],
    prepared: PreparedStructuralModalResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    build_structural_modal_resource_plan_selecting(
        mesh,
        compilation,
        observations,
        workload,
        prestress_state,
        accelerator_requests,
        prepared,
        ModalSolveSelection::WithinCeiling {
            machine_memory_bytes: None,
        },
    )
}

/// The modal solve path a plan recorded (`None` for other plans).
pub fn recorded_modal_solve_path(plan: &FeaResourcePlan) -> Option<SubspaceSolvePath> {
    match &plan.input.demand {
        FeaResourceDemand::StructuralModalExtraction(demand) => {
            demand.operator.solve.subspace_solve_path()
        }
        _ => None,
    }
}

/// Plans a modal workload on the path `selection` names.
///
/// Path-selection law: with the effective ceiling min(authored resident budget, machine
/// memory when known), the shift-invert factorization is chosen when its whole plan (memory
/// and scratch) fits the ceiling; otherwise preconditioned LOBPCG when its plan fits;
/// otherwise [`ResourcePlanningError::ModalSolveCeiling`] names both requirements. The plan
/// records the path, and the worker's re-derivation and the solver follow the record.
#[allow(clippy::too_many_arguments)]
pub fn build_structural_modal_resource_plan_selecting(
    mesh: Option<&SolverMeshArtifact>,
    compilation: &crate::constraints::ConstraintCompilation<'_>,
    observations: &ResolvedObservationPlan,
    workload: &StructuralModalWorkload,
    prestress_state: Option<&CanonicalSemanticState>,
    accelerator_requests: &[AcceleratorRequest],
    prepared: PreparedStructuralModalResourceFacts,
    selection: ModalSolveSelection,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let plan_path = |path: SubspaceSolvePath| {
        build_structural_modal_resource_plan_on_path(
            mesh,
            compilation,
            observations,
            workload,
            prestress_state,
            accelerator_requests,
            prepared.clone(),
            path,
        )
    };
    let machine_memory_bytes = match selection {
        ModalSolveSelection::Recorded(path) => return plan_path(path),
        ModalSolveSelection::WithinCeiling {
            machine_memory_bytes,
        } => machine_memory_bytes,
    };
    let authored = workload
        .procedure_step()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?
        .accuracy
        .resources
        .resident_byte_budget();
    let ceiling = machine_memory_bytes.map_or(authored, |machine| machine.min(authored));
    let fits = |plan: &FeaResourcePlan| {
        plan.request.memory_bytes <= ceiling && plan.request.scratch_bytes <= ceiling
    };
    let required = |planned: Result<FeaResourcePlan, ResourcePlanningError>| match planned {
        Ok(plan) if fits(&plan) => Ok(Ok(plan)),
        Ok(plan) => Ok(Err(plan
            .request
            .memory_bytes
            .max(plan.request.scratch_bytes))),
        Err(ResourcePlanningError::AccuracyCeiling {
            resource: super::RESIDENT_MEMORY_RESOURCE,
            required,
            ..
        }) => Ok(Err(required)),
        Err(error) => Err(error),
    };
    let factorization_resident_bytes =
        match required(plan_path(SubspaceSolvePath::ShiftInvertFactorization))? {
            Ok(plan) => return Ok(plan),
            Err(bytes) => bytes,
        };
    let lobpcg_resident_bytes = match required(plan_path(SubspaceSolvePath::PreconditionedLobpcg))?
    {
        Ok(plan) => return Ok(plan),
        Err(bytes) => bytes,
    };
    Err(ResourcePlanningError::ModalSolveCeiling {
        factorization_resident_bytes,
        lobpcg_resident_bytes,
        maximum: ceiling,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_structural_modal_resource_plan_on_path(
    mesh: Option<&SolverMeshArtifact>,
    compilation: &crate::constraints::ConstraintCompilation<'_>,
    observations: &ResolvedObservationPlan,
    workload: &StructuralModalWorkload,
    prestress_state: Option<&CanonicalSemanticState>,
    accelerator_requests: &[AcceleratorRequest],
    prepared: PreparedStructuralModalResourceFacts,
    path: SubspaceSolvePath,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let constraints = compilation.plan();
    let compiler_peak = super::constraint_compilation::original_compilation_peak(
        compilation,
        workload.model(),
        mesh,
        constraints,
    )?;
    workload
        .validate()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let topology_digest = mesh
        .map(|mesh| runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes()))
        .unwrap_or_else(|| constraints.topology.discretization_digest());
    let linearization = match (workload.prestress_transition(), prestress_state) {
        (None, None) => None,
        (Some(transition), Some(state)) => Some(
            crate::pipeline::prestress::structural_linearization_from_state(
                transition,
                state,
                topology_digest,
            )
            .map_err(ResourcePlanningError::Invalid)?,
        ),
        _ => {
            return Err(ResourcePlanningError::Invalid(
                "modal workload and prestress-state presence differ".into(),
            ))
        }
    };
    let facts = derive_structural_resource_facts(workload.model(), mesh, constraints)?;
    let expected_observations = resolve_modal_observations(mesh, constraints, workload)?;
    if observations != &expected_observations {
        return Err(ResourcePlanningError::Invalid(
            "modal observation plan does not rederive from its workload".into(),
        ));
    }
    if observations.topology_digest != facts.topology_digest {
        return Err(ResourcePlanningError::Invalid(
            "observation plan and modal constraints name different topology".into(),
        ));
    }
    if facts
        .constraint_matrix
        .right_hand_side
        .iter()
        .any(|value| *value != 0.0)
    {
        return Err(ResourcePlanningError::Invalid(
            "modal resource planning requires homogeneous constraints".into(),
        ));
    }
    let reduction = facts
        .constraint_matrix
        .affine_reduction()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let full_dimension = count(reduction.full_dof_count, "modal full dimension")?;
    let reduced_dimension = count(reduction.reduced_dof_count, "modal reduced dimension")?;
    if reduced_dimension == 0 {
        return Err(ResourcePlanningError::Invalid(
            "modal extraction requires at least one unconstrained degree of freedom".into(),
        ));
    }
    let subspace::ModalSubspaceBounds {
        requested_mode_count,
        maximum_subspace_dimension,
        initial_subspace_dimension,
        boundary_probe_width,
        expansion_count,
    } = subspace::derive(reduction.reduced_dof_count, workload)?;
    let residual_vector_count = count(
        workload.residual_vector_directions().len(),
        "modal residual-vector count",
    )?;

    let tangent_pattern = match linearization {
        Some(linearization) => facts
            .symbolic
            .stiffness_upper_bound()
            .union(linearization.geometric_stiffness().pattern())
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
        None => facts.symbolic.stiffness_upper_bound().clone(),
    };
    let reduced_tangent = reduction
        .reduce_sparse_pattern(&tangent_pattern)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let reduced_mass = reduction
        .reduce_sparse_pattern(facts.symbolic.mass_upper_bound())
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let admission = CoercivityWorkEnvelope::derive(&reduced_tangent)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let reduced_tangent_nonzeros = count(
        reduced_tangent.column_indices().len(),
        "reduced tangent nonzeros",
    )?;
    let geometric_stiffness_nonzeros = linearization
        .map(|linearization| {
            count(
                linearization.geometric_stiffness().values().len(),
                "geometric stiffness nonzeros",
            )
        })
        .transpose()?
        .unwrap_or(0);
    let reduced_mass_nonzeros =
        count(reduced_mass.column_indices().len(), "reduced mass nonzeros")?;

    // The solver carries K, M, and the canonical damping operator through assembly/reduction,
    // even though eigenspace iteration applies only K and M. All three therefore remain admitted.
    let elastic_contribution_nonzeros = facts
        .sparse
        .operator_nonzeros
        .iter()
        .try_fold(0_u64, |total, value| {
            checked_add(total, value.0, "modal contributions")
        })?;
    let contribution_nonzeros = checked_add(
        elastic_contribution_nonzeros,
        geometric_stiffness_nonzeros,
        "modal elastic and prestress contributions",
    )?;
    let contribution_bytes = checked_bytes(contribution_nonzeros, 24, "modal contributions")?;
    let dense_projected_values = checked_mul(
        maximum_subspace_dimension,
        maximum_subspace_dimension,
        "modal dense projected values",
    )?;
    let source_bytes = energy::original_energy_source_bytes(
        mesh,
        &reduction,
        linearization.is_some(),
        workload.model(),
        admission,
    )?;
    let shifted_pattern = reduced_tangent
        .union(&reduced_mass)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    // The completeness proof's factor-free inertia count on the pencil's union pattern, with
    // the coercivity envelope's 25% headroom between the planned support and the numerical
    // pattern the solver analyzes.
    let inertia_bytes = {
        let words = |values: &[u64]| -> Result<Vec<usize>, ResourcePlanningError> {
            values
                .iter()
                .map(|&value| {
                    usize::try_from(value)
                        .map_err(|_| ResourcePlanningError::Overflow("modal inertia pattern"))
                })
                .collect()
        };
        let estimate = runmat_sparse_solvers::eigen::pencil_inertia_estimate(
            usize::try_from(reduced_dimension)
                .map_err(|_| ResourcePlanningError::Overflow("modal reduced dimension"))?,
            &words(shifted_pattern.row_offsets())?,
            &words(shifted_pattern.column_indices())?,
            &crate::solve::modal::modal_eigen_options(1, workload.convergence()).ordering,
        )
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        checked_mul(estimate.resident_bytes, 5, "modal inertia headroom")? / 4
    };
    drop(shifted_pattern);
    // The local-energy proof holds the operator's entry intervals plus its block-derived
    // storage, which the source inventory bounds (capture or proof additions, the larger).
    let energy_proof_bytes = if source_bytes == 0 {
        0
    } else {
        checked_add(
            admission.energy_resident_limit(),
            source_bytes,
            "modal local-energy proof",
        )?
    };
    let (eigensolver_values, residency) = match path {
        SubspaceSolvePath::ShiftInvertFactorization => (
            modal_eigensolver_working_values(
                reduced_dimension,
                maximum_subspace_dimension,
                workload.convergence(),
            )?,
            sizing::ModalSearchResidency::ShiftInvertFactor {
                factor_bytes: admission.factor_resident_limit(),
                energy_proof_bytes,
            },
        ),
        SubspaceSolvePath::PreconditionedLobpcg => (
            sizing::modal_lobpcg_working_values(
                reduced_dimension,
                maximum_subspace_dimension,
                workload.convergence(),
            )?,
            sizing::ModalSearchResidency::Multigrid {
                preconditioner_bytes: runmat_sparse_solvers::estimate::amg_bytes(
                    usize::try_from(reduced_dimension)
                        .map_err(|_| ResourcePlanningError::Overflow("modal reduced dimension"))?,
                    usize::try_from(reduced_tangent_nonzeros)
                        .map_err(|_| ResourcePlanningError::Overflow("modal stiffness nonzeros"))?,
                    3,
                ),
            },
        ),
    };
    let workspace_bytes = modal_workspace_bytes(
        sizing::ModalWorkspaceDimensions {
            full: full_dimension,
            reduced: reduced_dimension,
            subspace: maximum_subspace_dimension,
            residual_vectors: residual_vector_count,
        },
        sizing::ModalSearchStorage {
            stiffness_nonzeros: reduced_tangent_nonzeros,
            mass_nonzeros: reduced_mass_nonzeros,
            eigensolver_values,
            inertia_bytes,
            residency,
        },
    )?;

    let modal_output_values = modal_output_value_count(
        full_dimension,
        maximum_subspace_dimension,
        residual_vector_count,
    )?;
    let modal_phase_values = modal_phase_value_count(
        modal_output_values,
        full_dimension,
        maximum_subspace_dimension,
        count(facts.recovery_node_count, "modal recovery nodes")?,
        count(facts.recovery_element_count, "modal recovery elements")?,
    )?;
    let observations =
        ObservationResourceFacts::from_plan(observations, workload.observation_execution())?;
    let expected_result_output = output::result_output(
        modal_output_values,
        maximum_subspace_dimension,
        residual_vector_count,
        full_dimension,
        facts.topology.nodes.0,
    )?;
    let result_payload_bytes = expected_result_output.encoded_payload_bytes_upper_bound.0;
    let receipt_payload_bytes = structural_modal_receipt_payload_bytes_upper_bound();
    let profiles_payload_bytes =
        crate::execution::structural_state_publication_profiles_encoding_profile(
            workload.model(),
            workload
                .procedure_step()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
            &[crate::execution::SemanticStatePublicationSource::ModalBasis],
            topology_digest,
        )
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?
        .encoded_bytes_upper_bound;
    if prepared.result_output != expected_result_output
        || prepared.receipt_output.encoded_payload_bytes_upper_bound.0 != receipt_payload_bytes
        || !prepared_output_covers_transport(&prepared.result_output)
        || !prepared_output_covers_transport(&prepared.receipt_output)
        || prepared
            .state_profiles_output
            .encoded_payload_bytes_upper_bound
            .0
            != profiles_payload_bytes
        || !prepared_output_covers_transport(&prepared.state_profiles_output)
        || prepared.output_set.encoded_bytes_upper_bound.0 == 0
    {
        return Err(ResourcePlanningError::Invalid(
            "modal prepared outputs differ from their canonical payload or transport envelopes"
                .into(),
        ));
    }
    let result_transport = prepared.result_output.transport_closure_bytes_upper_bound.0;
    let receipt_transport = prepared
        .receipt_output
        .transport_closure_bytes_upper_bound
        .0;
    let artifact_payload_bytes = checked_add(
        checked_add(
            result_payload_bytes,
            profiles_payload_bytes,
            "modal profile payload",
        )?,
        receipt_payload_bytes,
        "modal artifact payloads",
    )?;
    let artifact_transport_bytes = checked_add(
        checked_add(
            result_transport,
            prepared
                .state_profiles_output
                .transport_closure_bytes_upper_bound
                .0,
            "modal profile transport",
        )?,
        receipt_transport,
        "modal artifact transport",
    )?;
    let artifact_overhead_bytes = artifact_transport_bytes
        .checked_sub(artifact_payload_bytes)
        .ok_or(ResourcePlanningError::Invalid(
            "modal output transport is smaller than its payloads".into(),
        ))?;
    let output_construction_bytes = checked_add(
        checked_add(
            prepared
                .state_profiles_output
                .construction_resident_bytes_admission_budget
                .0,
            prepared
                .result_output
                .construction_resident_bytes_admission_budget
                .0,
            "modal profile construction",
        )?,
        prepared
            .receipt_output
            .construction_resident_bytes_admission_budget
            .0,
        "modal output construction",
    )?;
    let phase_resident_bytes = checked_add(
        output_construction_bytes.max(modal_result_payload_bytes(modal_phase_values, 0)?),
        source_bytes,
        "modal phase and original source",
    )?;
    let accelerators = modal_accelerator_requirements(
        workload.backend(),
        accelerator_requests,
        reduced_dimension,
        reduced_tangent_nonzeros,
    )?;
    let required_capabilities = accelerators
        .iter()
        .map(|accelerator| Capability::Accelerator(accelerator.class.clone()))
        .collect();
    let step = workload
        .procedure_step()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;

    let numerical_scope_digest = match linearization {
        Some(linearization) => runmat_execution::Digest::sha256(
            [
                b"runmat-structural-prestressed-modal-operator/v1\0".as_slice(),
                facts.symbolic.identity().bytes().as_slice(),
                linearization.identity().digest().bytes().as_slice(),
            ]
            .concat(),
        ),
        None => facts.symbolic.identity(),
    };

    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::StructuralModalExtraction(workload.identity()),
        demand: FeaResourceDemand::StructuralModalExtraction(StructuralModalResourceDemand {
            constraint_compilation_peak_bytes: compiler_peak,
            original_energy_source_bytes: ByteCount(source_bytes),
            workload: workload.identity(),
            prestress_linearization: linearization.map(|linearization| linearization.identity()),
            geometric_stiffness_nonzero_count: Count(geometric_stiffness_nonzeros),
            reduced_tangent_nonzero_count_upper_bound: Count(reduced_tangent_nonzeros),
            retained_mode_count_upper_bound: Count(maximum_subspace_dimension),
            residual_vector_count: Count(residual_vector_count),
            residual_vector_static_iteration_count_upper_bound: Count(
                if residual_vector_count == 0 {
                    0
                } else {
                    count(
                        crate::pipeline::homogeneous_static::homogeneous_static_iteration_limit(
                            reduction.reduced_dof_count,
                        ),
                        "modal residual-vector static iteration limit",
                    )?
                },
            ),
            residual_vector_mass_orthogonalization_passes: Count(if residual_vector_count == 0 {
                0
            } else {
                count(
                    crate::pipeline::modal_residual::RESIDUAL_MASS_ORTHOGONALIZATION_PASSES,
                    "modal residual-vector mass-orthogonalization passes",
                )?
            }),
            modal_phase_value_count_upper_bound: Count(modal_phase_values),
            modal_output_value_count_upper_bound: Count(modal_output_values),
            inputs: prepared.inputs.clone(),
            result_output: prepared.result_output.clone(),
            receipt_output: prepared.receipt_output.clone(),
            state_profiles_output: prepared.state_profiles_output.clone(),
            operator: StructuralOperatorResourceDemand {
                excitation_workspace: super::excitation_workspace::derive_excitation_workspace(
                    workload.model(),
                    workload
                        .procedure_step()
                        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
                    mesh,
                )?,
                topology_digest: facts.topology_digest,
                numerical_scope_digest,
                topology: facts.topology,
                sparse: facts.sparse,
                constraints: facts.constraints,
                assembly: AssemblyResourcePolicy {
                    contribution_nonzero_upper_bound: Count(contribution_nonzeros),
                    contribution_bytes_upper_bound: ByteCount(
                        super::connector_buffers::include_connector_buffers(
                            contribution_bytes,
                            facts.connector_assembly_buffer_bytes,
                        )?,
                    ),
                    scratch_bytes_per_element: ByteCount(ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT),
                    fixed_scratch_bytes_per_worker: ByteCount(
                        FIXED_ASSEMBLY_SCRATCH_BYTES_PER_WORKER,
                    ),
                },
                solve: SolveResourcePolicy::SymmetricSubspaceIteration {
                    full_dimension: Count(full_dimension),
                    reduced_dimension: Count(reduced_dimension),
                    requested_mode_count: Count(requested_mode_count),
                    initial_subspace_dimension: Count(initial_subspace_dimension),
                    maximum_subspace_dimension: Count(maximum_subspace_dimension),
                    boundary_probe_width: Count(boundary_probe_width),
                    subspace_expansion_count_upper_bound: Count(expansion_count),
                    maximum_iterations_per_expansion: Count(count(
                        workload.convergence().maximum_subspace_iterations(),
                        "modal iteration limit",
                    )?),
                    orthogonalization_passes_per_block: Count(count(
                        MODAL_MASS_ORTHOGONALIZATION_PASSES,
                        "modal orthogonalization pass count",
                    )?),
                    dense_projected_value_count_upper_bound: Count(dense_projected_values),
                    workspace_bytes_upper_bound: ByteCount(workspace_bytes),
                    path,
                },
                phase_resources: PhaseResourcePolicy {
                    resident_value_count: Count(modal_phase_values),
                    resident_bytes_upper_bound: ByteCount(phase_resident_bytes),
                    scratch_bytes: ByteCount(phase_resident_bytes),
                    retained_bytes: ByteCount(artifact_payload_bytes),
                },
                field_recovery: Default::default(),
                observations: observations.clone(),
                publication: CheckpointArtifactPolicy {
                    result_artifact_bytes: ByteCount(artifact_payload_bytes),
                    checkpoint_bytes: ByteCount(0),
                    artifact_overhead_bytes: ByteCount(artifact_overhead_bytes),
                    relay_bytes: ByteCount(0),
                    result_egress_bytes: ByteCount(artifact_transport_bytes),
                    checkpoint_egress_bytes: ByteCount(0),
                },
                partitioning: PartitioningPolicy::Monolithic,
                accuracy_demand: AccuracyResourceDemand {
                    refinement_cycles: Count(0),
                    solver_tightenings: Count(0),
                    samples: observations.maximum_samples_per_observation,
                },
            },
        }),
        output_set: Some(prepared.output_set),
        worker: WorkerResourcePolicy {
            cpu_millicores: prepared
                .local_execution
                .checked_cpu_millicores()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
            maximum_wall_milliseconds: step.accuracy.resources.maximum_wall_time_milliseconds,
            fixed_memory_bytes: ByteCount(FIXED_WORKER_MEMORY_BYTES),
            fixed_scratch_bytes: ByteCount(FIXED_WORKER_SCRATCH_BYTES),
            required_capabilities,
            accelerators,
        },
        accuracy_ceiling: step.accuracy.resources,
    })
}
