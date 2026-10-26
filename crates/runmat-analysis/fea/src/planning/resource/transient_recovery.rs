//! Assembly-free host preflight for completed transient field-history recovery.

use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    execution::{
        CanonicalSemanticState, StructuralTransientCoordinateSpace,
        StructuralTransientNumericalReceipt, StructuralTransientSolverPlan,
        StructuralTransientSolverResult, StructuralTransientWorkload,
    },
    pipeline::transient::prepare_transient_preassembly,
};

use super::{
    build_resource_plan, build_structural_transient_scope_resource_plan,
    field_metadata::FIELD_METADATA_BYTES, transient_scope::transient_source_workload_identity,
    AccuracyResourceDemand, ByteCount, CheckpointArtifactPolicy, Count, FeaResourceDemand,
    FeaResourcePhase, FeaResourcePlan, FeaResourcePlanningInput, FieldRecoveryResourceFacts,
    ObservationResourceFacts, PartitioningPolicy, PhaseResourcePolicy, PreparedInputResourceFacts,
    ResourcePlanningError, SolveResourcePolicy, StructuralTransientFieldRecoveryResourceDemand,
    RESOURCE_ESTIMATION_MODEL_REVISION,
};

// The result is already decoded on entry. These deliberately conservative values cover its
// physical state, reaction and balance vectors, per-sample heap headers, typed field histories,
// observations, and the one reporting assembly that coexists with them. Measured peak evidence
// is still required before treating the estimate as a production memory guarantee.
const SAMPLE_VALUES_PER_PHYSICAL_DOF: u64 = 16;
const SAMPLE_METADATA_BYTES: u64 = 8 * 1024;
const FIELD_VALUES_PER_ENTITY: u64 = 128;
const FIELD_VALUES_PER_OBSERVATION_ENTITY: u64 = 16;
const FIELDS_PER_SAMPLE: u64 = 32;
const FIELDS_PER_OBSERVATION: u64 = 4;
const RECOVERY_SCRATCH_BYTES_PER_ELEMENT: u64 = 4 * 1024;
const REPORTING_OPERATOR_COPIES: u64 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HistoryProjection {
    sample_values: u64,
    sample_bytes: u64,
    field_values: u64,
    field_bytes: u64,
}

fn project_history(
    sample_count: u64,
    physical_dofs: u64,
    recovery_entities: u64,
    observations: u64,
) -> Result<HistoryProjection, ResourcePlanningError> {
    let sample_values = multiply(
        multiply(
            physical_dofs,
            SAMPLE_VALUES_PER_PHYSICAL_DOF,
            "transient physical sample values",
        )?,
        sample_count,
        "transient retained sample values",
    )?;
    let sample_bytes = multiply(
        sample_count,
        add(
            multiply(
                physical_dofs,
                SAMPLE_VALUES_PER_PHYSICAL_DOF * 8,
                "transient sample vector bytes",
            )?,
            SAMPLE_METADATA_BYTES,
            "transient sample bytes",
        )?,
        "transient decoded sample history",
    )?;
    let field_values_per_entity = add(
        FIELD_VALUES_PER_ENTITY,
        multiply(
            observations,
            FIELD_VALUES_PER_OBSERVATION_ENTITY,
            "transient observation values per entity",
        )?,
        "transient field values per entity",
    )?;
    let field_values = multiply(
        sample_count,
        multiply(
            recovery_entities,
            field_values_per_entity,
            "transient fields per sample",
        )?,
        "transient retained field values",
    )?;
    let field_metadata = multiply(
        sample_count,
        multiply(
            add(
                FIELDS_PER_SAMPLE,
                multiply(
                    observations,
                    FIELDS_PER_OBSERVATION,
                    "transient observation fields",
                )?,
                "transient fields per sample",
            )?,
            FIELD_METADATA_BYTES,
            "transient field metadata per sample",
        )?,
        "transient retained field metadata",
    )?;
    let field_bytes = add(
        multiply(field_values, 8, "transient field scalar bytes")?,
        field_metadata,
        "transient retained typed field history",
    )?;
    Ok(HistoryProjection {
        sample_values,
        sample_bytes,
        field_values,
        field_bytes,
    })
}

pub fn build_structural_transient_recovery_resource_plan(
    workload: &StructuralTransientWorkload,
    mesh: &SolverMeshArtifact,
    modal_state: Option<&CanonicalSemanticState>,
    result: &StructuralTransientSolverResult,
    receipt: &StructuralTransientNumericalReceipt,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    receipt
        .validate_for_result(result)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let prepared = prepare_transient_preassembly(&workload.model, workload.into())
        .map_err(ResourcePlanningError::from_preparation_error)?;
    let scope_plan = build_structural_transient_scope_resource_plan(
        workload,
        mesh,
        modal_state,
        &prepared,
        None,
        None,
        super::LocalExecutionPlan::serial(),
    )?;
    let FeaResourceDemand::StructuralTransientScopePreparation(scope_demand) =
        scope_plan.input.demand
    else {
        unreachable!("scope planner returned a different demand")
    };
    let scope = &result.manifest().scope;
    let (authored_study, authored_step, expected_source) =
        transient_source_workload_identity(workload, mesh, scope.execution_plan)?;
    let coordinate_space_matches = matches!(
        (&workload.solver, &scope.coordinate_space),
        (
            StructuralTransientSolverPlan::Direct,
            StructuralTransientCoordinateSpace::Direct
        ) | (
            StructuralTransientSolverPlan::ModalReduced { .. },
            StructuralTransientCoordinateSpace::ModalReduced { .. }
        )
    );
    if scope.study != authored_study
        || scope.step != authored_step
        || scope.source_workload != expected_source
        || scope.schedule != prepared.schedule
        || count(scope.dof_count, "transient physical DOFs")? != scope_demand.operator.sparse.rows.0
        || !coordinate_space_matches
    {
        return Err(ResourcePlanningError::Invalid(
            "completed transient scope differs from the authoritative workload or mesh".into(),
        ));
    }

    let sample_count = count(result.samples().len(), "completed transient sample count")?;
    let physical_dofs = scope_demand.operator.sparse.rows.0;
    let nodes = scope_demand.operator.topology.nodes.0;
    let elements = scope_demand.operator.topology.elements.0;
    let connectors = count(workload.model.connectors.len(), "transient connectors")?;
    let observations = count(
        prepared.execution_step.observations.len(),
        "transient observations",
    )?;
    let recovery_entities = add(
        add(nodes, elements, "transient recovery topology entities")?,
        add(
            connectors,
            scope_demand.operator.constraints.rows.0,
            "transient connector and constraint entities",
        )?,
        "transient recovery entities",
    )?;
    let HistoryProjection {
        sample_values,
        sample_bytes,
        field_values,
        field_bytes,
    } = project_history(sample_count, physical_dofs, recovery_entities, observations)?;
    let manifest_bytes = count(
        result
            .manifest()
            .canonical_bytes()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?
            .len(),
        "transient result manifest bytes",
    )?;
    let receipt_bytes = count(
        receipt
            .canonical_bytes()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?
            .len(),
        "transient numerical receipt bytes",
    )?;
    let encoded_inputs = add(
        scope_demand.inputs.encoded_payload_bytes.0,
        add(
            manifest_bytes,
            receipt_bytes,
            "transient result control inputs",
        )?,
        "transient recovery control inputs",
    )?;
    let decoded_inputs =
        runmat_canonical_codec::projected_canonical_typed_decode_resident_bytes_estimate(
            encoded_inputs,
            3,
        )
        .ok_or(ResourcePlanningError::Overflow(
            "transient recovery decoded control inputs",
        ))?;
    let inputs = PreparedInputResourceFacts {
        encoded_payload_bytes: ByteCount(encoded_inputs),
        transport_closure_bytes: ByteCount(encoded_inputs),
        decoded_resident_bytes_admission_budget: ByteCount(decoded_inputs),
    };
    let mut operator = scope_demand.operator;
    let nonzeros = operator
        .sparse
        .operator_nonzeros
        .iter()
        .try_fold(0_u64, |sum, count| {
            add(sum, count.0, "transient recovery operator nonzeros")
        })?;
    let operator_bytes = add(
        multiply(nonzeros, 16, "transient recovery CSR entries")?,
        multiply(
            add(physical_dofs, 1, "transient recovery CSR rows")?,
            multiply(
                count(
                    operator.sparse.operator_nonzeros.len(),
                    "transient operators",
                )?,
                8,
                "transient recovery row-offset bytes",
            )?,
            "transient recovery CSR row offsets",
        )?,
        "transient recovery operator storage",
    )?;
    let scratch = add(
        multiply(
            elements,
            RECOVERY_SCRATCH_BYTES_PER_ELEMENT,
            "transient recovery element scratch",
        )?,
        multiply(physical_dofs, 64 * 8, "transient recovery vector scratch")?,
        "transient recovery scratch",
    )?;
    let phase_values = add(
        sample_values,
        field_values,
        "transient recovery resident values",
    )?;
    let phase_bytes = [
        sample_bytes,
        field_bytes,
        scratch,
        // Reporting assembly can allocate contributions after earlier field snapshots are
        // retained. The generic operator estimator compares assembly and recovery peaks, so
        // this overlap must also appear explicitly in the recovery phase.
        operator.assembly.contribution_bytes_upper_bound.0,
        multiply(
            operator_bytes,
            REPORTING_OPERATOR_COPIES,
            "transient reporting operator copies",
        )?,
    ]
    .into_iter()
    .try_fold(0_u64, |sum, value| {
        add(sum, value, "transient recovery phase residency")
    })?;
    operator.solve = SolveResourcePolicy::NoSolve {
        workspace_bytes_upper_bound: ByteCount(scratch),
    };
    operator.phase_resources = PhaseResourcePolicy {
        resident_value_count: Count(phase_values),
        resident_bytes_upper_bound: ByteCount(phase_bytes),
        scratch_bytes: ByteCount(scratch),
        retained_bytes: ByteCount(field_bytes),
    };
    operator.field_recovery = FieldRecoveryResourceFacts {
        requirement_count: Count(add(1, observations, "transient recovery requirements")?),
        source_entity_count: Count(recovery_entities),
        output_entity_count: Count(recovery_entities),
        topology_incidence_count: operator.topology.element_node_references,
        arithmetic_operation_count: Count(field_values),
    };
    operator.observations = ObservationResourceFacts {
        maximum_samples_per_observation: Count(sample_count),
        produced_sample_count: Count(multiply(
            sample_count,
            observations,
            "transient produced observation samples",
        )?),
        retained_sample_count: Count(multiply(
            sample_count,
            observations,
            "transient retained observation samples",
        )?),
        operation_count: Count(multiply(
            field_values,
            observations,
            "transient observation operations",
        )?),
        peak_resident_bytes: ByteCount(0),
        stored_bytes: ByteCount(0),
        transfer_bytes: ByteCount(0),
        collective_count: Count(0),
    };
    operator.publication = CheckpointArtifactPolicy {
        result_artifact_bytes: ByteCount(0),
        checkpoint_bytes: ByteCount(0),
        artifact_overhead_bytes: ByteCount(0),
        relay_bytes: ByteCount(0),
        result_egress_bytes: ByteCount(0),
        checkpoint_egress_bytes: ByteCount(0),
    };
    operator.partitioning = PartitioningPolicy::Monolithic;
    operator.accuracy_demand = AccuracyResourceDemand {
        refinement_cycles: Count(0),
        solver_tightenings: Count(0),
        samples: Count(sample_count),
    };
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::StructuralTransientFieldRecovery(result.manifest().identity),
        demand: FeaResourceDemand::StructuralTransientFieldRecovery(
            StructuralTransientFieldRecoveryResourceDemand {
                result: result.manifest().identity,
                sample_count: Count(sample_count),
                physical_dof_count: Count(physical_dofs),
                decoded_sample_bytes_upper_bound: ByteCount(sample_bytes),
                retained_field_bytes_upper_bound: ByteCount(field_bytes),
                operator,
                inputs,
            },
        ),
        output_set: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_history_projection_grows_with_samples_and_observations() {
        let short = project_history(2, 24, 16, 0).unwrap();
        let long = project_history(20, 24, 16, 0).unwrap();
        let observed = project_history(20, 24, 16, 3).unwrap();
        assert!(long.sample_bytes > short.sample_bytes);
        assert!(long.field_bytes > short.field_bytes);
        assert!(observed.field_bytes > long.field_bytes);
        assert!(matches!(
            project_history(u64::MAX, 24, 16, 0),
            Err(ResourcePlanningError::Overflow(_))
        ));
    }
}
