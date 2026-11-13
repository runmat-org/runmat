//! Assembly-free resource admission for one physical transient block worker.

use crate::{
    execution::{
        CanonicalSemanticState, StructuralTransientBlockRequest, StructuralTransientCheckpoint,
        StructuralTransientCoordinateSpace, StructuralTransientSolverPlan,
        StructuralTransientWorkload,
    },
    pipeline::transient::prepare_transient_preassembly,
};
use runmat_meshing_core::SolverMeshArtifact;

use super::{
    build_resource_plan, build_structural_transient_scope_resource_plan, post_process,
    transient_scope::transient_source_workload_identity, AccuracyResourceDemand, ByteCount,
    CheckpointArtifactPolicy, Count, FeaResourceDemand, FeaResourcePhase, FeaResourcePlan,
    FeaResourcePlanningInput, PartitioningPolicy, PhaseResourcePolicy, PreparedInputResourceFacts,
    PreparedOutputResourceFacts, PreparedOutputSetResourceFacts, ResourcePlanningError,
    SolveResourcePolicy, StructuralTransientBlockResourceDemand,
    RESOURCE_ESTIMATION_MODEL_REVISION,
};

mod solve;

const FULL_ORDER_WORK_VECTORS: u64 = 128;
const REDUCED_DENSE_WORK_MATRICES: u64 = 64;
const BLOCK_OPERATOR_COPIES: u64 = 4;

/// Conservative canonical bound for a segment manifest, checkpoint, and physical samples.
/// The source-payload term covers copied schedule/observation metadata; each record also gets
/// 64 KiB of canonical metadata in addition to its nine physical vectors.
pub fn structural_transient_block_payload_bytes_upper_bound(
    inputs: &PreparedInputResourceFacts,
    time_point_count: u64,
    physical_dof_count: u64,
) -> Result<u64, ResourcePlanningError> {
    let records = add(time_point_count, 4, "transient block output records")?;
    let vectors = multiply(records, 9, "transient block physical vectors")?;
    let numeric = multiply(vectors, physical_dof_count, "transient block output values")?;
    post_process::derived_output_payload_bytes_upper_bound(inputs, records, numeric)
}

/// Derive the worker's block-solve demand from the same authoritative model, mesh, and
/// schedule used by checkpoint-scope preparation. No physical operator is assembled here.
pub fn build_structural_transient_block_resource_plan(
    workload: &StructuralTransientWorkload,
    mesh: &SolverMeshArtifact,
    modal_state: Option<&CanonicalSemanticState>,
    request: &StructuralTransientBlockRequest,
    predecessor: Option<&StructuralTransientCheckpoint>,
    prepared_inputs: Option<PreparedInputResourceFacts>,
    output_set: Option<PreparedOutputSetResourceFacts>,
    local_execution: super::LocalExecutionPlan,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    request
        .validate_against_predecessor(predecessor)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    // Block segments always execute on cpu_reference: a GPU request is admitted (fallback with
    // one warning, or a strict failure) by `run_transient_block`, so the plan is the CPU plan.
    let prepared = prepare_transient_preassembly(&workload.model, workload.into())
        .map_err(ResourcePlanningError::from_preparation_error)?;
    let scope_plan = build_structural_transient_scope_resource_plan(
        workload,
        mesh,
        modal_state,
        &prepared,
        None,
        None,
        local_execution,
    )?;
    let FeaResourceDemand::StructuralTransientScopePreparation(scope_demand) =
        scope_plan.input.demand
    else {
        unreachable!("scope planner returned a different demand")
    };
    let workload_bytes = workload
        .canonical_bytes()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let (study, step, source_workload) =
        transient_source_workload_identity(workload, mesh, request.scope.execution_plan)?;
    if request.scope.study != study
        || request.scope.step != step
        || request.scope.source_workload != source_workload
        || request.scope.schedule != prepared.schedule
        || count(request.scope.dof_count, "transient physical DOFs")?
            != scope_demand.operator.sparse.rows.0
    {
        return Err(ResourcePlanningError::Invalid(
            "transient block scope differs from the authoritative workload or mesh".into(),
        ));
    }
    let physical_dofs = scope_demand.operator.sparse.rows.0;
    let coordinate_dofs = match (&workload.solver, &request.scope.coordinate_space) {
        (StructuralTransientSolverPlan::Direct, StructuralTransientCoordinateSpace::Direct) => {
            physical_dofs
        }
        (
            StructuralTransientSolverPlan::ModalReduced { .. },
            StructuralTransientCoordinateSpace::ModalReduced {
                coordinate_count, ..
            },
        ) => count(*coordinate_count, "transient modal coordinate count")?,
        _ => {
            return Err(ResourcePlanningError::Invalid(
                "transient block coordinate space differs from its solver".into(),
            ));
        }
    };
    let time_point_count = request
        .terminal_schedule_ordinal
        .checked_sub(request.start_schedule_ordinal)
        .ok_or(ResourcePlanningError::Overflow(
            "transient block time points",
        ))?;
    let encoded_inputs = [
        count(workload_bytes.len(), "transient workload payload")?,
        mesh.canonical_encoded_len()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
        count(
            request
                .canonical_bytes()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?
                .len(),
            "transient request payload",
        )?,
        modal_state
            .map(|value| {
                value
                    .canonical_bytes()
                    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))
                    .and_then(|bytes| count(bytes.len(), "transient modal payload"))
            })
            .transpose()?
            .unwrap_or(0),
        predecessor
            .map(|value| {
                value
                    .canonical_bytes()
                    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))
                    .and_then(|bytes| count(bytes.len(), "transient predecessor payload"))
            })
            .transpose()?
            .unwrap_or(0),
    ]
    .into_iter()
    .try_fold(0_u64, |sum, bytes| {
        add(sum, bytes, "transient block input payload")
    })?;
    let inputs = match prepared_inputs {
        Some(facts)
            if facts.encoded_payload_bytes.0 == encoded_inputs
                && facts.transport_closure_bytes.0 >= encoded_inputs
                && facts.decoded_resident_bytes_admission_budget.0 >= encoded_inputs =>
        {
            facts
        }
        Some(_) => {
            return Err(ResourcePlanningError::Invalid(
                "prepared transient block inputs differ from canonical payloads".into(),
            ))
        }
        None => PreparedInputResourceFacts {
            encoded_payload_bytes: ByteCount(encoded_inputs),
            transport_closure_bytes: ByteCount(encoded_inputs),
            decoded_resident_bytes_admission_budget: ByteCount(
                runmat_canonical_codec::projected_canonical_typed_decode_resident_bytes_estimate(
                    encoded_inputs,
                    3,
                )
                .ok_or(ResourcePlanningError::Overflow("transient decoded inputs"))?,
            ),
        },
    };
    let segment_bytes = structural_transient_block_payload_bytes_upper_bound(
        &inputs,
        time_point_count,
        physical_dofs,
    )?;
    let segment_closure = add(
        segment_bytes,
        crate::execution::ANALYSIS_MANIFEST_MAX_ENCODED_BYTES,
        "transient segment closure",
    )?;
    // The segment is encoded from typed solver vectors, never decoded from a generic value
    // tree: its construction holds the nine f64 physical vectors per record, the encoded
    // payload and closure, and fixed per-contract codec state.
    let segment_records = add(time_point_count, 4, "transient segment records")?;
    let segment_typed_values = multiply(
        multiply(segment_records, 9, "transient segment vectors")?,
        physical_dofs,
        "transient segment values",
    )?;
    let segment_construction = [
        multiply(segment_typed_values, 8, "transient segment typed bytes")?,
        segment_bytes,
        segment_closure,
        multiply(
            runmat_canonical_codec::CANONICAL_MATERIALIZATION_FIXED_BYTES_PER_CONTRACT,
            2,
            "transient segment contract state",
        )?,
    ]
    .into_iter()
    .try_fold(0_u64, |sum, bytes| {
        add(sum, bytes, "transient segment construction")
    })?;
    let segment_output = PreparedOutputResourceFacts {
        encoded_payload_bytes_upper_bound: ByteCount(segment_bytes),
        transport_closure_bytes_upper_bound: ByteCount(segment_closure),
        construction_resident_bytes_admission_budget: ByteCount(segment_construction),
    };
    // The output set is the one root manifest listing this phase's output roots. Its bound is
    // the root schema bound, not the numerical payload bound carried by the prepared output.
    let output_set = output_set.unwrap_or(PreparedOutputSetResourceFacts {
        encoded_bytes_upper_bound: ByteCount(
            crate::execution::ANALYSIS_OUTPUT_SET_MAX_ENCODED_BYTES,
        ),
    });
    if output_set.encoded_bytes_upper_bound.0 == 0
        || output_set.encoded_bytes_upper_bound.0
            > crate::execution::ANALYSIS_OUTPUT_SET_MAX_ENCODED_BYTES
    {
        return Err(ResourcePlanningError::Invalid(
            "transient block output set is outside the output-set root schema bound".into(),
        ));
    }
    let operator = scope_demand.operator;
    let operator_nonzeros = operator
        .sparse
        .operator_nonzeros
        .iter()
        .try_fold(0_u64, |sum, value| {
            add(sum, value.0, "transient block operator nonzeros")
        })?;
    let operator_storage = add(
        multiply(operator_nonzeros, 16, "transient block CSR entries")?,
        multiply(
            add(physical_dofs, 1, "transient block CSR rows")?,
            multiply(
                count(
                    operator.sparse.operator_nonzeros.len(),
                    "transient block operators",
                )?,
                8,
                "transient block row offsets",
            )?,
            "transient block row storage",
        )?,
        "transient block operator storage",
    )?;
    let (modal_dense_values, proof_bytes) = match workload.solver {
        StructuralTransientSolverPlan::Direct => {
            (0, solve::proof_storage(&prepared.active_model, mesh)?)
        }
        StructuralTransientSolverPlan::ModalReduced { .. } => (
            multiply(
                multiply(
                    coordinate_dofs,
                    coordinate_dofs,
                    "transient modal matrix values",
                )?,
                REDUCED_DENSE_WORK_MATRICES,
                "transient modal dense workspace",
            )?,
            0,
        ),
    };
    let solve_values = add(
        multiply(
            physical_dofs,
            FULL_ORDER_WORK_VECTORS,
            "transient full-order workspace",
        )?,
        modal_dense_values,
        "transient block solve values",
    )?;
    let phase_values = add(
        solve_values,
        multiply(
            time_point_count,
            multiply(physical_dofs, 9, "transient sample vectors")?,
            "transient retained samples",
        )?,
        "transient phase values",
    )?;
    let phase_bytes = add(
        add(
            multiply(phase_values, 8, "transient phase scalar bytes")?,
            multiply(
                operator_storage,
                BLOCK_OPERATOR_COPIES,
                "transient simultaneous operator copies",
            )?,
            "transient phase numerical bytes",
        )?,
        segment_construction,
        "transient block phase residency",
    )?;
    let mut operator = operator;
    // The exact source workspace is inherited from authoritative scope preparation.
    // Replacing time-history phase resources below must not discard or multiply it.
    operator.solve = SolveResourcePolicy::TransientBlock {
        physical_dimension: Count(physical_dofs),
        coordinate_dimension: Count(coordinate_dofs),
        workspace_bytes_upper_bound: ByteCount(add(
            multiply(solve_values, 8, "transient solve workspace bytes")?,
            proof_bytes,
            "transient solve and coercivity construction",
        )?),
    };
    operator.phase_resources = PhaseResourcePolicy {
        resident_value_count: Count(phase_values),
        resident_bytes_upper_bound: ByteCount(phase_bytes),
        scratch_bytes: ByteCount(multiply(
            operator_storage,
            BLOCK_OPERATOR_COPIES,
            "transient operator construction scratch",
        )?),
        retained_bytes: ByteCount(segment_construction),
    };
    // The segment payload and its manifest; the output-set root is charged separately.
    operator.publication = CheckpointArtifactPolicy {
        result_artifact_bytes: ByteCount(segment_closure),
        checkpoint_bytes: ByteCount(0),
        artifact_overhead_bytes: ByteCount(0),
        relay_bytes: ByteCount(0),
        result_egress_bytes: ByteCount(segment_closure),
        checkpoint_egress_bytes: ByteCount(0),
    };
    operator.partitioning = PartitioningPolicy::Monolithic;
    operator.accuracy_demand = AccuracyResourceDemand {
        refinement_cycles: Count(0),
        solver_tightenings: Count(0),
        samples: Count(time_point_count),
    };
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::StructuralTransientBlock(request.identity),
        demand: FeaResourceDemand::StructuralTransientBlock(
            StructuralTransientBlockResourceDemand {
                request: request.identity,
                scope_digest: request
                    .scope
                    .identity()
                    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?
                    .digest(),
                time_point_count: Count(time_point_count),
                operator,
                inputs,
                segment_output,
            },
        ),
        output_set: Some(output_set),
        worker: scope_plan.input.worker,
        accuracy_ceiling: scope_plan.input.accuracy_ceiling,
    })
}

fn count(value: usize, label: &'static str) -> Result<u64, ResourcePlanningError> {
    u64::try_from(value).map_err(|_| ResourcePlanningError::Overflow(label))
}

fn add(left: u64, right: u64, label: &'static str) -> Result<u64, ResourcePlanningError> {
    left.checked_add(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}

fn multiply(left: u64, right: u64, label: &'static str) -> Result<u64, ResourcePlanningError> {
    left.checked_mul(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}
