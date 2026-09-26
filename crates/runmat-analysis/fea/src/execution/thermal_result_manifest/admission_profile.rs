//! Pre-solve projection of the actual manifest wrapper and its typed scalar inventories.
use super::*;
use crate::{
    admission::AnalysisRunIdentity,
    contracts::{
        ComputeBackend, ExecutionBackend, SolverMethod, SolverPreconditioner, ThermalEnergyBalance,
    },
    solve::{backend::kind::LinearAlgebraBackendKind, preconditioner::SpdPreconditionerKind},
};
use runmat_canonical_codec::{profile_serialized, SerializedCanonicalProfile};
use serde::ser::SerializeSeq;

const PROFILE_LIMITS: CanonicalLimits =
    CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64);

impl ThermalSolverResultManifest {
    /// Profile the exact identity tuple, excluding the separately composed initial-state record.
    pub(crate) fn numerical_identity_profile(
        identity: &AnalysisRunIdentity,
        nodes: usize,
        elements: usize,
        samples: usize,
        solves: usize,
    ) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
        use crate::execution::thermal_result::identity::IdentitySerialization;
        Ok(profile_serialized(
            &IdentitySerialization(
                ThermalNumericalIntentIdentity::derive("profile", &0_u8)?,
                identity,
                Repeated {
                    value: 0.0_f64,
                    count: samples,
                },
                // RecoveredFieldIdentity is a transparent Digest; fixed rows mirror four histories.
                Repeated {
                    value: [Digest::from_bytes([0; 32]); 4],
                    count: samples,
                },
                Repeated {
                    value: 0.0_f64,
                    count: solves,
                },
                0.0,
                Repeated {
                    value: ThermalEnergyBalance::from_powers(0.0, 0.0, 0.0, 0.0),
                    count: solves,
                },
                0.0,
                Digest::from_bytes([0; 32]),
                StepIdentity::derive("profile", &0_u8)?,
                Repeated {
                    value: 0_u64,
                    count: nodes,
                },
                Repeated {
                    value: 0_u64,
                    count: elements,
                },
                (),
            ),
            PROFILE_LIMITS,
        )?)
    }

    /// This is a serialization-only skeleton, not a valid or publishable numerical result.
    /// Variable evidence/catalog collections are empty and must be composed by the caller.
    /// Opaque identities have fixed width; the registry-issued run identity is never fabricated.
    pub(crate) fn wrapper_profile(
        identity: &AnalysisRunIdentity,
    ) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
        let mut skeleton = skeleton(identity)?;
        let mut bound = SerializedCanonicalProfile {
            encoded_bytes_upper_bound: 0,
            materialization_resident_bytes_estimate: 0,
        };
        // Cover the execution evidence emitted by the shared thermal linear solver. The entire
        // actual serializer is used for each combination, including enum wire spellings.
        for requested in [ComputeBackend::Cpu, ComputeBackend::Gpu] {
            for execution in [ExecutionBackend::Cpu, ExecutionBackend::Accelerator] {
                for backend in [
                    LinearAlgebraBackendKind::CpuReference,
                    LinearAlgebraBackendKind::Accelerator,
                ] {
                    for method in [SolverMethod::AssembledPcg, SolverMethod::MatrixFreePcg] {
                        for preconditioner in
                            [SpdPreconditionerKind::Jacobi, SpdPreconditionerKind::Ilu0]
                        {
                            skeleton.run.requested_backend = requested;
                            skeleton.run.execution_backend = execution;
                            skeleton.run.solver_backend = backend;
                            skeleton.run.solver_method = method;
                            skeleton.run.preconditioner =
                                SolverPreconditioner::from_spd(preconditioner);
                            let profile = profile_serialized(&skeleton, PROFILE_LIMITS)?;
                            bound.encoded_bytes_upper_bound = bound
                                .encoded_bytes_upper_bound
                                .max(profile.encoded_bytes_upper_bound);
                            bound.materialization_resident_bytes_estimate = bound
                                .materialization_resident_bytes_estimate
                                .max(profile.materialization_resident_bytes_estimate);
                        }
                    }
                }
            }
        }
        Ok(bound)
    }

    pub(crate) fn scalar_inventory_profile(
        nodes: usize,
        elements: usize,
        samples: usize,
        solves: usize,
    ) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
        // Tuple/container overhead conservatively exceeds inserting these inventories into the
        // already-profiled empty vectors. Values use the real energy-record serializer.
        Ok(profile_serialized(
            &(
                Repeated {
                    value: 0_u64,
                    count: nodes,
                },
                Repeated {
                    value: 0_u64,
                    count: elements,
                },
                Repeated {
                    value: 0.0_f64,
                    count: samples,
                },
                Repeated {
                    value: 0.0_f64,
                    count: solves,
                },
                Repeated {
                    value: ThermalEnergyBalance::from_powers(0.0, 0.0, 0.0, 0.0),
                    count: solves,
                },
            ),
            PROFILE_LIMITS,
        )?)
    }
}

struct Repeated<T> {
    value: T,
    count: usize,
}
impl<T: Serialize> Serialize for Repeated<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.count))?;
        for _ in 0..self.count {
            sequence.serialize_element(&self.value)?;
        }
        sequence.end()
    }
}

fn skeleton(
    identity: &AnalysisRunIdentity,
) -> AnalysisExecutionResult<ThermalSolverResultManifest> {
    Ok(ThermalSolverResultManifest {
        schema_version: 1,
        source_workload: AnalysisWorkloadIdentity::derive("profile", &0_u8)?,
        numerical_intent: ThermalNumericalIntentIdentity::derive("profile", &0_u8)?,
        identity: SolverResultIdentity::derive("profile", &0_u8)?,
        run: FeaRunResult {
            identity: identity.clone(),
            requested_backend: ComputeBackend::Cpu,
            execution_backend: ExecutionBackend::Cpu,
            solver_backend: LinearAlgebraBackendKind::CpuReference,
            solver_device_apply_k_ratio: 0.0,
            solver_method: SolverMethod::AssembledPcg,
            preconditioner: SolverPreconditioner::from_spd(SpdPreconditionerKind::Jacobi),
            solver_host_sync_count: 0,
            accelerator_evidence: Vec::new(),
            diagnostics: Vec::new(),
            spatial_projection_evidence: Vec::new(),
            structural_static_balance: None,
            structural_constraint_result: None,
            structural_connector_result: None,
            observations: Vec::new(),
            observation_histories: Vec::new(),
            series_reductions: Vec::new(),
            fields: Vec::new(),
        },
        producer: StepIdentity::derive("profile", &0_u8)?,
        topology_digest: Digest::from_bytes([0; 32]),
        ordered_node_ids: Vec::new(),
        ordered_element_ids: Vec::new(),
        time_points_s: Vec::new(),
        residual_norms: Vec::new(),
        residual_target: 0.0,
        energy_balances: Vec::new(),
        reference_temperature_k: 0.0,
        initial_state: ThermalInitialStateEvidence {
            procedure_step_id: String::new(),
            reference_temperature_k: 0.0,
            applied_initial_condition_ids: Vec::new(),
            authored_node_count: 0,
            prescribed_node_count: 0,
            authored_and_prescribed_node_count: 0,
            reference_node_count: 0,
            minimum_temperature_k: 0.0,
            maximum_temperature_k: 0.0,
        },
        fields: Vec::new(),
    })
}
