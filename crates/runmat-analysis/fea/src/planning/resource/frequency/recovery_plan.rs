use crate::execution::{FieldRecoveryRequest, FieldRecoverySource, StructuralFrequencyEntryResult};

use super::{
    add, build_resource_plan, count, empty_observations, multiply,
    payload::recovery_payload_bytes_upper_bound, AccuracyResourceDemand, ByteCount,
    CheckpointArtifactPolicy, Count, FeaResourceDemand, FeaResourcePhase, FeaResourcePlan,
    FeaResourcePlanningInput, PartitioningPolicy, PhaseResourcePolicy, PreparedInputResourceFacts,
    PreparedOutputResourceFacts, PreparedOutputSetResourceFacts, ResourcePlanningError,
    SolveResourcePolicy, StructuralFrequencyEntryResourceProjection,
    StructuralFrequencyOperatorResourceDemand, StructuralOperatorResourceDemand,
    RESOURCE_ESTIMATION_MODEL_REVISION,
};

impl StructuralFrequencyEntryResourceProjection {
    pub fn recovery_plan_for(
        &self,
        request: &FieldRecoveryRequest,
        result: &StructuralFrequencyEntryResult,
        inputs: PreparedInputResourceFacts,
        output: PreparedOutputResourceFacts,
        output_set: PreparedOutputSetResourceFacts,
        local_execution: super::LocalExecutionPlan,
    ) -> Result<FeaResourcePlan, ResourcePlanningError> {
        request
            .validate()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        result
            .validate()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        result
            .request()
            .validate_for_workload(&self.workload)
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        if request.producer != result.request().step()
            || request.source != FieldRecoverySource::StructuralFrequencyEntry(result.identity())
            || request.topology
                != (crate::execution::FieldTopologyIdentity::Structural {
                    topology: self.structural_topology,
                })
        {
            return Err(ResourcePlanningError::Invalid(
                "frequency recovery request differs from its workload, entry, or topology".into(),
            ));
        }
        let full_dofs = self.sparse.rows.0;
        let output_values = count(
            self.recovery_layout.storage_value_count,
            "complex recovery output values",
        )?;
        let constraint_rows = count(
            self.constraint_group_count,
            "complex recovery constraint groups",
        )?;
        let recovery_values = add(
            multiply(
                output_values,
                2,
                "real, imaginary, and complex recovery values",
            )?,
            add(
                multiply(
                    full_dofs,
                    4,
                    "complex displacement and reaction work values",
                )?,
                multiply(constraint_rows, 12, "complex constraint recovery values")?,
                "complex recovery work values",
            )?,
            "complex recovery live values",
        )?;
        let recovery_bytes = multiply(recovery_values, 8, "complex recovery live bytes")?;
        let output_payload_bytes = recovery_payload_bytes_upper_bound(&self.recovery_layout)?;
        if output.encoded_payload_bytes_upper_bound.0 != output_payload_bytes
            || output.transport_closure_bytes_upper_bound.0 < output_payload_bytes
            || output.construction_resident_bytes_admission_budget.0 < output_payload_bytes
        {
            return Err(ResourcePlanningError::Invalid(
                "frequency recovery output facts differ from its canonical payload ceiling".into(),
            ));
        }
        let output_overhead = output
            .transport_closure_bytes_upper_bound
            .0
            .checked_sub(output_payload_bytes)
            .ok_or_else(|| {
                ResourcePlanningError::Invalid(
                    "frequency recovery closure is smaller than its payload".into(),
                )
            })?;
        let resident_bytes = add(
            recovery_bytes,
            output.construction_resident_bytes_admission_budget.0,
            "complex recovery output construction residency",
        )?;
        build_resource_plan(FeaResourcePlanningInput {
            estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
            phase: FeaResourcePhase::StructuralFrequencyFieldRecovery(request.identity()),
            demand: FeaResourceDemand::StructuralFrequencyFieldRecovery(
                StructuralFrequencyOperatorResourceDemand {
                    constraint_compilation_peak_bytes: self.constraint_compilation_peak_bytes,
                    inputs,
                    output: output.clone(),
                    operator: StructuralOperatorResourceDemand {
                        excitation_workspace: Default::default(),
                        topology_digest: self.topology_digest,
                        numerical_scope_digest: self.numerical_scope_digest,
                        topology: self.topology.clone(),
                        sparse: self.sparse.clone(),
                        constraints: self.constraints.clone(),
                        assembly: self.assembly.clone(),
                        solve: SolveResourcePolicy::NoSolve {
                            workspace_bytes_upper_bound: ByteCount(multiply(
                                full_dofs,
                                32,
                                "complex recovery operator workspace",
                            )?),
                        },
                        phase_resources: PhaseResourcePolicy {
                            resident_value_count: Count(recovery_values),
                            resident_bytes_upper_bound: ByteCount(resident_bytes),
                            scratch_bytes: ByteCount(0),
                            retained_bytes: ByteCount(multiply(
                                output_values,
                                8,
                                "complex recovery retained bytes",
                            )?),
                        },
                        field_recovery: super::super::FieldRecoveryResourceFacts::from_plan(
                            &request.plan,
                        )?,
                        observations: empty_observations(),
                        publication: CheckpointArtifactPolicy {
                            result_artifact_bytes: ByteCount(output_payload_bytes),
                            checkpoint_bytes: ByteCount(0),
                            artifact_overhead_bytes: ByteCount(output_overhead),
                            relay_bytes: ByteCount(0),
                            result_egress_bytes: output.transport_closure_bytes_upper_bound,
                            checkpoint_egress_bytes: ByteCount(0),
                        },
                        partitioning: PartitioningPolicy::Monolithic,
                        accuracy_demand: AccuracyResourceDemand {
                            refinement_cycles: Count(0),
                            solver_tightenings: Count(0),
                            samples: Count(1),
                        },
                    },
                },
            ),
            output_set: Some(output_set),
            worker: self.worker(local_execution)?,
            accuracy_ceiling: self.accuracy_ceiling,
        })
    }
}
