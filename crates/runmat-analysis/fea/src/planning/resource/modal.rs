use crate::{
    algebra::CoercivitySymbolicPlan,
    constraints::CanonicalStructuralConstraintPlan,
    execution::{CanonicalSemanticState, StructuralModalWorkload},
    post::fields::structural_observable_field_definitions,
    results::observation::{
        resolve_structural_observation_plan, ObservationLimits, ResolvedObservationPlan,
    },
    solve::modal::{MODAL_BOUNDARY_PROBE_WIDTH, MODAL_MASS_ORTHOGONALIZATION_PASSES},
    ComputeBackend,
};
use runmat_execution::resource::{AcceleratorRequest, Capability};
use runmat_meshing_core::SolverMeshArtifact;

use super::{
    build_resource_plan, structural_facts::derive_structural_resource_facts,
    AcceleratorRequirement, AccuracyResourceDemand, AssemblyResourcePolicy, ByteCount,
    CheckpointArtifactPolicy, Count, FeaResourceDemand, FeaResourcePhase, FeaResourcePlan,
    FeaResourcePlanningInput, ObservationResourceFacts, PartitioningPolicy, PhaseResourcePolicy,
    PreparedOutputResourceFacts, PreparedStructuralModalResourceFacts, ResourcePlanningError,
    SolveResourcePolicy, StructuralModalResourceDemand, StructuralOperatorResourceDemand,
    WorkerResourcePolicy, RESOURCE_ESTIMATION_MODEL_REVISION,
};

const ACCELERATOR_CG_RESIDENT_VECTOR_COUNT: u64 = 6;
const ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT: u64 = 8 * 1024;
const FIXED_ASSEMBLY_SCRATCH_BYTES: u64 = 64 * 1024;
const FIXED_WORKER_MEMORY_BYTES: u64 = 16 * 1024 * 1024;
const FIXED_WORKER_SCRATCH_BYTES: u64 = 8 * 1024 * 1024;
const ARTIFACT_OVERHEAD_BYTES: u64 = 64 * 1024;
const MODAL_RECEIPT_PAYLOAD_BYTES_UPPER_BOUND: u64 = 16 * 1024;

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
    let requested_mode_count = count(
        workload.truncation().requested_mode_count(),
        "requested modal count",
    )?
    .min(reduced_dimension);
    let boundary_probe_width = count(MODAL_BOUNDARY_PROBE_WIDTH, "modal boundary probe width")?;
    let initial_subspace_dimension = requested_mode_count
        .checked_add(boundary_probe_width)
        .ok_or(ResourcePlanningError::Overflow("modal initial subspace"))?
        .min(reduced_dimension);
    let remaining = reduced_dimension - initial_subspace_dimension;
    let expansion_count = remaining
        .checked_add(boundary_probe_width - 1)
        .ok_or(ResourcePlanningError::Overflow("modal subspace expansions"))?
        / boundary_probe_width
        + 1;

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
    let admission = CoercivitySymbolicPlan::derive(&reduced_tangent)
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
        reduced_dimension,
        reduced_dimension,
        "modal dense projected values",
    )?;
    let workspace_bytes = modal_workspace_bytes(
        full_dimension,
        reduced_dimension,
        reduced_tangent_nonzeros,
        reduced_mass_nonzeros,
        admission.resident_bytes_upper_bound(),
    )?;
    let modal_output_values = modal_output_value_count(full_dimension, reduced_dimension)?;
    let modal_phase_values = modal_phase_value_count(
        modal_output_values,
        full_dimension,
        reduced_dimension,
        count(facts.recovery_node_count, "modal recovery nodes")?,
        count(facts.recovery_element_count, "modal recovery elements")?,
    )?;
    let observations =
        ObservationResourceFacts::from_plan(observations, workload.observation_execution())?;
    let result_payload_bytes = modal_result_payload_bytes(modal_output_values, 0)?;
    let receipt_payload_bytes = structural_modal_receipt_payload_bytes_upper_bound();
    if prepared.result_output.encoded_payload_bytes_upper_bound.0 != result_payload_bytes
        || prepared.receipt_output.encoded_payload_bytes_upper_bound.0 != receipt_payload_bytes
        || !prepared_output_covers_transport(&prepared.result_output)
        || !prepared_output_covers_transport(&prepared.receipt_output)
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
        result_payload_bytes,
        receipt_payload_bytes,
        "modal artifact payloads",
    )?;
    let artifact_transport_bytes = checked_add(
        result_transport,
        receipt_transport,
        "modal artifact transport",
    )?;
    let artifact_overhead_bytes = artifact_transport_bytes
        .checked_sub(artifact_payload_bytes)
        .ok_or(ResourcePlanningError::Invalid(
            "modal output transport is smaller than its payloads".into(),
        ))?;
    let output_construction_bytes = checked_add(
        prepared
            .result_output
            .construction_resident_bytes_admission_budget
            .0,
        prepared
            .receipt_output
            .construction_resident_bytes_admission_budget
            .0,
        "modal output construction",
    )?;
    let phase_resident_bytes =
        output_construction_bytes.max(modal_result_payload_bytes(modal_phase_values, 0)?);
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
            workload: workload.identity(),
            prestress_linearization: linearization.map(|linearization| linearization.identity()),
            geometric_stiffness_nonzero_count: Count(geometric_stiffness_nonzeros),
            reduced_tangent_nonzero_count_upper_bound: Count(reduced_tangent_nonzeros),
            retained_mode_count_upper_bound: Count(reduced_dimension),
            modal_phase_value_count_upper_bound: Count(modal_phase_values),
            modal_output_value_count_upper_bound: Count(modal_output_values),
            inputs: prepared.inputs.clone(),
            result_output: prepared.result_output.clone(),
            receipt_output: prepared.receipt_output.clone(),
            operator: StructuralOperatorResourceDemand {
                topology_digest: facts.topology_digest,
                numerical_scope_digest,
                topology: facts.topology,
                sparse: facts.sparse,
                constraints: facts.constraints,
                assembly: AssemblyResourcePolicy {
                    contribution_nonzero_upper_bound: Count(contribution_nonzeros),
                    contribution_bytes_upper_bound: ByteCount(contribution_bytes),
                    scratch_bytes_per_element: ByteCount(ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT),
                    fixed_scratch_bytes_per_worker: ByteCount(FIXED_ASSEMBLY_SCRATCH_BYTES),
                },
                solve: SolveResourcePolicy::SymmetricSubspaceIteration {
                    full_dimension: Count(full_dimension),
                    reduced_dimension: Count(reduced_dimension),
                    requested_mode_count: Count(requested_mode_count),
                    initial_subspace_dimension: Count(initial_subspace_dimension),
                    maximum_subspace_dimension: Count(reduced_dimension),
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
                },
                phase_resources: PhaseResourcePolicy {
                    resident_value_count: Count(modal_phase_values),
                    resident_bytes_upper_bound: ByteCount(phase_resident_bytes),
                    scratch_bytes: ByteCount(phase_resident_bytes),
                    retained_bytes: ByteCount(artifact_payload_bytes),
                },
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
            cpu_millicores: 1_000,
            maximum_wall_milliseconds: step.accuracy.resources.maximum_wall_time_milliseconds,
            fixed_memory_bytes: ByteCount(FIXED_WORKER_MEMORY_BYTES),
            fixed_scratch_bytes: ByteCount(FIXED_WORKER_SCRATCH_BYTES),
            required_capabilities,
            accelerators,
        },
        accuracy_ceiling: step.accuracy.resources,
    })
}

fn modal_workspace_bytes(
    full_dimension: u64,
    reduced_dimension: u64,
    reduced_stiffness_nonzeros: u64,
    reduced_mass_nonzeros: u64,
    ilu_bytes: u64,
) -> Result<u64, ResourcePlanningError> {
    let reduced_row_offsets = checked_bytes(
        checked_mul(
            checked_add(reduced_dimension, 1, "reduced row offsets")?,
            3,
            "reduced row offsets",
        )?,
        8,
        "reduced row offsets",
    )?;
    // Reduced damping can be as dense as K; it is retained by the current canonical carrier.
    let reduced_entries = checked_add(
        checked_add(
            reduced_stiffness_nonzeros,
            reduced_mass_nonzeros,
            "reduced K+M nonzeros",
        )?,
        reduced_stiffness_nonzeros,
        "reduced K+M+C nonzeros",
    )?;
    let reduced_operator_bytes = checked_add(
        reduced_row_offsets,
        checked_bytes(reduced_entries, 16, "reduced operator entries")?,
        "reduced operators",
    )?;
    // Peak subspace residency: old/full candidate modes, reduced candidates, and three dense
    // Rayleigh-Ritz matrices can coexist during boundary expansion and eigendecomposition.
    let vector_values = checked_add(
        checked_mul(full_dimension, reduced_dimension, "expanded modal vectors")?,
        checked_mul(
            checked_mul(
                reduced_dimension,
                reduced_dimension,
                "reduced modal vectors",
            )?,
            3,
            "reduced modal vector blocks",
        )?,
        "modal vector values",
    )?;
    let dense_values = checked_mul(
        checked_mul(reduced_dimension, reduced_dimension, "dense modal matrices")?,
        3,
        "dense modal matrices",
    )?;
    let vector_and_dense = checked_bytes(
        checked_add(vector_values, dense_values, "modal numerical workspace")?,
        8,
        "modal numerical workspace",
    )?;
    checked_add(
        checked_add(
            reduced_operator_bytes,
            ilu_bytes,
            "reduced operators and ILU",
        )?,
        vector_and_dense,
        "modal workspace",
    )
}

fn modal_output_value_count(
    full_dimension: u64,
    retained_modes: u64,
) -> Result<u64, ResourcePlanningError> {
    let modal_vectors = checked_mul(full_dimension, retained_modes, "modal basis vectors")?;
    let square_metrics = checked_mul(retained_modes, retained_modes, "modal square metrics")?;
    [
        modal_vectors,
        checked_mul(square_metrics, 2, "modal orthogonality metrics")?,
        checked_mul(retained_modes, 48, "modal scalar metrics")?,
        32,
    ]
    .into_iter()
    .try_fold(0_u64, |total, value| {
        checked_add(total, value, "modal output values")
    })
}

fn modal_phase_value_count(
    output_values: u64,
    full_dimension: u64,
    retained_modes: u64,
    nodes: u64,
    elements: u64,
) -> Result<u64, ResourcePlanningError> {
    let recovery_per_mode = checked_add(
        checked_mul(nodes, 6, "modal phase nodal recovery")?,
        checked_mul(elements, 13, "modal phase element recovery")?,
        "modal phase recovery",
    )?;
    [
        output_values,
        checked_mul(
            recovery_per_mode,
            retained_modes,
            "all recovered modal fields",
        )?,
        checked_mul(
            checked_mul(nodes, 6, "projected translation and rotation values")?,
            retained_modes,
            "projected modal fields",
        )?,
        checked_mul(full_dimension, retained_modes, "solver modal vectors")?,
    ]
    .into_iter()
    .try_fold(0_u64, |total, value| {
        checked_add(total, value, "modal phase values")
    })
}

/// Deterministic canonical payload ceiling for the modal solver-result object.
pub fn structural_modal_result_payload_bytes_upper_bound(
    mesh: Option<&SolverMeshArtifact>,
    constraints: &CanonicalStructuralConstraintPlan,
    workload: &StructuralModalWorkload,
) -> Result<u64, ResourcePlanningError> {
    workload
        .validate()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let facts = derive_structural_resource_facts(workload.model(), mesh, constraints)?;
    let reduction = facts
        .constraint_matrix
        .affine_reduction()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    if reduction.offset.iter().any(|value| *value != 0.0) || reduction.reduced_dof_count == 0 {
        return Err(ResourcePlanningError::Invalid(
            "modal payload planning requires a nonempty homogeneous reduction".into(),
        ));
    }
    let values = modal_output_value_count(
        count(reduction.full_dof_count, "modal full dimension")?,
        count(reduction.reduced_dof_count, "modal retained modes")?,
    )?;
    modal_result_payload_bytes(values, 0)
}

/// Canonical payload ceiling for the compact fixed-shape numerical receipt.
pub const fn structural_modal_receipt_payload_bytes_upper_bound() -> u64 {
    MODAL_RECEIPT_PAYLOAD_BYTES_UPPER_BOUND
}

fn modal_result_payload_bytes(
    values: u64,
    embedded_observation_bytes: u64,
) -> Result<u64, ResourcePlanningError> {
    checked_add(
        checked_add(
            checked_bytes(values, 8, "modal result values")?,
            embedded_observation_bytes,
            "modal result observations",
        )?,
        ARTIFACT_OVERHEAD_BYTES,
        "modal result payload",
    )
}

fn prepared_output_covers_transport(output: &PreparedOutputResourceFacts) -> bool {
    output.transport_closure_bytes_upper_bound >= output.encoded_payload_bytes_upper_bound
        && output.construction_resident_bytes_admission_budget
            >= output.encoded_payload_bytes_upper_bound
}

fn modal_accelerator_requirements(
    backend: ComputeBackend,
    requests: &[AcceleratorRequest],
    reduced_dimension: u64,
    stiffness_nonzeros: u64,
) -> Result<Vec<AcceleratorRequirement>, ResourcePlanningError> {
    if backend != ComputeBackend::Gpu {
        if !requests.is_empty() {
            return Err(ResourcePlanningError::Invalid(
                "CPU modal execution cannot claim accelerator resources".into(),
            ));
        }
        return Ok(Vec::new());
    }
    let [request] = requests else {
        return Err(ResourcePlanningError::Invalid(
            "GPU modal execution requires exactly one accelerator request".into(),
        ));
    };
    let required =
        structural_modal_accelerator_minimum_bytes(reduced_dimension, stiffness_nonzeros)?;
    if request.count != 1 || request.memory_bytes_each < required {
        return Err(ResourcePlanningError::Invalid(format!(
            "the modal accelerator requires one device with at least {required} bytes"
        )));
    }
    Ok(vec![AcceleratorRequirement {
        class: request.class.clone(),
        count: request.count,
        memory_bytes_each: ByteCount(request.memory_bytes_each),
    }])
}

/// Known device-resident lower envelope for the native sparse CG adapter used by modal solves.
pub(super) fn structural_modal_accelerator_minimum_bytes(
    reduced_dimension: u64,
    stiffness_nonzeros: u64,
) -> Result<u64, ResourcePlanningError> {
    let row_offsets = checked_bytes(
        checked_add(reduced_dimension, 1, "accelerator row offsets")?,
        8,
        "accelerator row offsets",
    )?;
    let sparse_entries = checked_bytes(stiffness_nonzeros, 16, "accelerator sparse entries")?;
    let diagonal = checked_bytes(reduced_dimension, 8, "accelerator diagonal")?;
    let vectors = checked_bytes(
        checked_mul(
            reduced_dimension,
            ACCELERATOR_CG_RESIDENT_VECTOR_COUNT,
            "accelerator vectors",
        )?,
        8,
        "accelerator vectors",
    )?;
    [row_offsets, sparse_entries, diagonal, vectors]
        .into_iter()
        .try_fold(0_u64, |total, value| {
            checked_add(total, value, "modal accelerator")
        })
}

fn resolve_modal_observations(
    mesh: Option<&SolverMeshArtifact>,
    constraints: &CanonicalStructuralConstraintPlan,
    workload: &StructuralModalWorkload,
) -> Result<ResolvedObservationPlan, ResourcePlanningError> {
    let fields = structural_observable_field_definitions()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    resolve_structural_observation_plan(
        workload
            .procedure_step()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
        workload.model(),
        mesh,
        constraints,
        &fields,
        ObservationLimits::default(),
    )
    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))
}

fn count(value: usize, label: &'static str) -> Result<u64, ResourcePlanningError> {
    u64::try_from(value).map_err(|_| ResourcePlanningError::Overflow(label))
}

fn checked_bytes(
    count: u64,
    width: u64,
    label: &'static str,
) -> Result<u64, ResourcePlanningError> {
    checked_mul(count, width, label)
}

fn checked_mul(left: u64, right: u64, label: &'static str) -> Result<u64, ResourcePlanningError> {
    left.checked_mul(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}

fn checked_add(left: u64, right: u64, label: &'static str) -> Result<u64, ResourcePlanningError> {
    left.checked_add(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}
