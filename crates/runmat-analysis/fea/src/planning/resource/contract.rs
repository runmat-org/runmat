use std::collections::BTreeSet;

use runmat_analysis_core::AccuracyResourceCeiling;
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::{
    resource::{AcceleratorRequest, Capability, ResourceRequest},
    Digest,
};
use serde::{Deserialize, Serialize};

use crate::execution::{
    AnalysisNumericalObjectKind, FieldRecoveryRequestIdentity, ObservationAssemblyRequestIdentity,
    ObservationHistoryCheckpointRequestIdentity, ObservationPartitionEvaluationRequestIdentity,
    ObservationPartitionReductionRequestIdentity, StructuralBucklingWorkloadIdentity,
    StructuralFrequencyEntryRequestIdentity, StructuralFrequencyResultAssemblyRequestIdentity,
    StructuralModalResponseSpectrumAssemblyRequestIdentity,
    StructuralModalResponseSpectrumDirectionRequestIdentity, StructuralModalWorkloadIdentity,
};
use crate::pipeline::structural_linearization::StructuralLinearizationIdentity;

const RESOURCE_PREFIX: &[u8] = b"runmat-fea-resource-plan/v1\0";
pub const RESOURCE_PLAN_MAX_ENCODED_BYTES: u64 = 64 * 1024;
const RESOURCE_LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024, 8_192, 65_536, 64);
pub const RESOURCE_ESTIMATION_MODEL_REVISION: u16 = 2;

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ByteCount(pub u64);

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Count(pub u64);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopologyResourceFacts {
    pub nodes: Count,
    pub elements: Count,
    pub boundary_faces: Count,
    pub element_node_references: Count,
    pub boundary_node_references: Count,
    pub node_neighbor_references: Count,
    /// Deterministic estimate produced from the complete authoritative projection by the
    /// resource-estimation model named by `estimation_model_revision`.
    pub estimated_resident_bytes: ByteCount,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseResourceFacts {
    pub rows: Count,
    pub columns: Count,
    /// Nonzero upper bound for each independently stored canonical operator pattern.
    /// K, M, and C do not implicitly share row-offset or column-index storage.
    pub operator_nonzeros: Vec<Count>,
}

impl SparseResourceFacts {
    pub(crate) fn maximum_nonzeros(&self) -> u64 {
        self.operator_nonzeros
            .iter()
            .map(|count| count.0)
            .max()
            .unwrap_or(0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstraintResourceFacts {
    pub rows: Count,
    pub coefficients: Count,
    pub reference_points: Count,
    /// Deterministic estimate produced from the canonical constraint plan.
    pub estimated_resident_bytes: ByteCount,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationResourceFacts {
    /// Largest produced sample count for any one observation. Accuracy ceilings apply per
    /// observation rather than to the number of independently requested outputs.
    pub maximum_samples_per_observation: Count,
    /// Sum of samples that observation evaluators may produce across the resolved plan.
    pub produced_sample_count: Count,
    /// Sum of samples retained after applying each observation's storage policy.
    pub retained_sample_count: Count,
    pub operation_count: Count,
    pub peak_resident_bytes: ByteCount,
    pub stored_bytes: ByteCount,
    pub transfer_bytes: ByteCount,
    pub collective_count: Count,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccuracyResourceDemand {
    pub refinement_cycles: Count,
    pub solver_tightenings: Count,
    pub samples: Count,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssemblyResourcePolicy {
    pub contribution_nonzero_upper_bound: Count,
    /// Upper bound from the selected assembly backend's concrete contribution format.
    pub contribution_bytes_upper_bound: ByteCount,
    pub scratch_bytes_per_element: ByteCount,
    pub fixed_scratch_bytes_per_worker: ByteCount,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreconditionerStorage {
    None,
    Diagonal,
    SparseCsr {
        nonzero_upper_bound: Count,
        resident_bytes_upper_bound: ByteCount,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericalScalar {
    RealF64,
    ComplexF64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverAdmissionResourcePolicy {
    pub dimension: Count,
    pub factor_nonzero_upper_bound: Count,
    pub update_count_upper_bound: Count,
    pub resident_bytes_upper_bound: ByteCount,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolveResourcePolicy {
    NoSolve {
        workspace_bytes_upper_bound: ByteCount,
    },
    Iterative {
        vector_count: Count,
        scalar: NumericalScalar,
        preconditioner: PreconditionerStorage,
        admission: SolverAdmissionResourcePolicy,
    },
    DirectFactorization {
        vector_count: Count,
        scalar: NumericalScalar,
        factor_nonzero_upper_bound: Count,
        factor_bytes_upper_bound: ByteCount,
    },
    RestartedGmres {
        block_dimension: Count,
        restart: Count,
        maximum_iterations: Count,
        workspace_bytes_upper_bound: ByteCount,
    },
    ModalDenseHarmonic {
        full_dimension: Count,
        mode_count: Count,
        workspace_bytes_upper_bound: ByteCount,
    },
    SymmetricSubspaceIteration {
        full_dimension: Count,
        reduced_dimension: Count,
        requested_mode_count: Count,
        initial_subspace_dimension: Count,
        maximum_subspace_dimension: Count,
        boundary_probe_width: Count,
        subspace_expansion_count_upper_bound: Count,
        maximum_iterations_per_expansion: Count,
        orthogonalization_passes_per_block: Count,
        dense_projected_value_count_upper_bound: Count,
        workspace_bytes_upper_bound: ByteCount,
    },
}

impl SolveResourcePolicy {
    pub(crate) fn vector_count(&self) -> Option<u64> {
        match self {
            Self::Iterative { vector_count, .. }
            | Self::DirectFactorization { vector_count, .. } => Some(vector_count.0),
            Self::NoSolve { .. }
            | Self::RestartedGmres { .. }
            | Self::ModalDenseHarmonic { .. }
            | Self::SymmetricSubspaceIteration { .. } => None,
        }
    }

    pub(crate) fn scalar_bytes(&self) -> u64 {
        let scalar = match self {
            Self::Iterative { scalar, .. } | Self::DirectFactorization { scalar, .. } => scalar,
            Self::NoSolve { .. }
            | Self::RestartedGmres { .. }
            | Self::ModalDenseHarmonic { .. }
            | Self::SymmetricSubspaceIteration { .. } => {
                return 8;
            }
        };
        match scalar {
            NumericalScalar::RealF64 => 8,
            NumericalScalar::ComplexF64 => 16,
        }
    }

    pub(crate) const fn explicit_workspace_bytes(&self) -> Option<ByteCount> {
        match self {
            Self::NoSolve {
                workspace_bytes_upper_bound,
            }
            | Self::RestartedGmres {
                workspace_bytes_upper_bound,
                ..
            }
            | Self::ModalDenseHarmonic {
                workspace_bytes_upper_bound,
                ..
            }
            | Self::SymmetricSubspaceIteration {
                workspace_bytes_upper_bound,
                ..
            } => Some(*workspace_bytes_upper_bound),
            Self::Iterative { .. } | Self::DirectFactorization { .. } => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseResourcePolicy {
    pub resident_value_count: Count,
    pub resident_bytes_upper_bound: ByteCount,
    pub scratch_bytes: ByteCount,
    pub retained_bytes: ByteCount,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointArtifactPolicy {
    pub result_artifact_bytes: ByteCount,
    pub checkpoint_bytes: ByteCount,
    pub artifact_overhead_bytes: ByteCount,
    pub relay_bytes: ByteCount,
    pub result_egress_bytes: ByteCount,
    pub checkpoint_egress_bytes: ByteCount,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerResourcePolicy {
    pub cpu_millicores: u32,
    pub maximum_wall_milliseconds: u64,
    pub fixed_memory_bytes: ByteCount,
    pub fixed_scratch_bytes: ByteCount,
    pub required_capabilities: BTreeSet<Capability>,
    pub accelerators: Vec<AcceleratorRequirement>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceleratorRequirement {
    pub class: String,
    pub count: u16,
    pub memory_bytes_each: ByteCount,
}

impl From<&AcceleratorRequirement> for AcceleratorRequest {
    fn from(value: &AcceleratorRequirement) -> Self {
        Self {
            class: value.class.clone(),
            count: value.count,
            memory_bytes_each: value.memory_bytes_each.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartitioningPolicy {
    Monolithic,
    /// A recommendation for an operator/domain-partitioning implementation. This
    /// does not assert that the current solve is distributed.
    OperatorDomain {
        maximum_partitions: u32,
        replicated_memory_bytes_per_partition: ByteCount,
        halo_memory_bytes_per_partition: ByteCount,
        replicated_scratch_bytes_per_partition: ByteCount,
        halo_scratch_bytes_per_partition: ByteCount,
    },
}

/// Canonical numerical phase governed by one exact resource request.
///
/// Task request identities prevent a plan estimated for one frequency entry or observation task
/// from authorizing another. The linear-pipeline variant preserves the existing whole-step
/// envelope until that path is decomposed into independently scheduled phases.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "request")]
pub enum FeaResourcePhase {
    StructuralLinearPipeline,
    StructuralModalExtraction(StructuralModalWorkloadIdentity),
    StructuralBuckling(StructuralBucklingWorkloadIdentity),
    StructuralFrequencyEntry(StructuralFrequencyEntryRequestIdentity),
    StructuralFrequencyFieldRecovery(FieldRecoveryRequestIdentity),
    StructuralObservationPartition(ObservationPartitionEvaluationRequestIdentity),
    StructuralObservationReduction(ObservationPartitionReductionRequestIdentity),
    StructuralObservationHistoryCheckpoint(ObservationHistoryCheckpointRequestIdentity),
    StructuralObservationAssembly(ObservationAssemblyRequestIdentity),
    StructuralFrequencyResultAssembly(StructuralFrequencyResultAssemblyRequestIdentity),
    StructuralModalResponseSpectrumDirection(
        StructuralModalResponseSpectrumDirectionRequestIdentity,
    ),
    StructuralModalResponseSpectrumAssembly(StructuralModalResponseSpectrumAssemblyRequestIdentity),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralOperatorResourceDemand {
    pub topology_digest: Digest,
    /// Identity of the complete numerical structure governing this task. For an operator task this
    /// is its symbolic pattern; for post-processing it is the exact ordered input structure.
    pub numerical_scope_digest: Digest,
    pub topology: TopologyResourceFacts,
    pub sparse: SparseResourceFacts,
    pub constraints: ConstraintResourceFacts,
    pub assembly: AssemblyResourcePolicy,
    pub solve: SolveResourcePolicy,
    pub phase_resources: PhaseResourcePolicy,
    pub observations: ObservationResourceFacts,
    pub publication: CheckpointArtifactPolicy,
    pub partitioning: PartitioningPolicy,
    pub accuracy_demand: AccuracyResourceDemand,
}

/// Exact pre-execution resource facts for the current monolithic structural modal kernel.
///
/// The embedded operator demand accounts for K and M plus the damping operator retained by the
/// current canonical operator carrier. The additional fields bind adaptive truncation-boundary
/// probing and dense Rayleigh-Ritz work that a generic linear-solve policy cannot represent.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralModalResourceDemand {
    pub workload: StructuralModalWorkloadIdentity,
    pub operator: StructuralOperatorResourceDemand,
    /// Exact preload linearization carried by a prestressed-modal task.
    pub prestress_linearization: Option<StructuralLinearizationIdentity>,
    /// Full-space nonzeros imported from the exact geometric-stiffness state, or zero for an
    /// ordinary modal extraction.
    pub geometric_stiffness_nonzero_count: Count,
    /// Reduced nonzeros in the exact symbolic union K + Kg used by the modal solve.
    pub reduced_tangent_nonzero_count_upper_bound: Count,
    pub retained_mode_count_upper_bound: Count,
    pub residual_vector_count: Count,
    pub residual_vector_static_iteration_count_upper_bound: Count,
    pub residual_vector_mass_orthogonalization_passes: Count,
    pub modal_phase_value_count_upper_bound: Count,
    pub modal_output_value_count_upper_bound: Count,
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralBucklingResourceDemand {
    pub workload: StructuralBucklingWorkloadIdentity,
    pub operator: StructuralOperatorResourceDemand,
    pub requested_mode_count: Count,
    pub phase_value_count_upper_bound: Count,
    pub output_value_count_upper_bound: Count,
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
}

/// Prepared transport envelopes for the two modal task outputs and their shared output-set root.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedStructuralModalResourceFacts {
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
    pub output_set: PreparedOutputSetResourceFacts,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedInputResourceFacts {
    /// Canonical numerical payload bytes before object chunking.
    pub encoded_payload_bytes: ByteCount,
    /// Unique content-addressed object bytes imported by the worker.
    pub transport_closure_bytes: ByteCount,
    /// Conservative admission budget for decoded numerical objects that coexist at phase
    /// execution. Actual scheduler accounting remains authoritative.
    pub decoded_resident_bytes_admission_budget: ByteCount,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedOutputResourceFacts {
    /// Canonical numerical payload bytes emitted before object chunking.
    pub encoded_payload_bytes_upper_bound: ByteCount,
    /// Content-addressed chunks and manifest for this numerical output. The single shared
    /// output-set root is accounted separately at the workload level.
    pub transport_closure_bytes_upper_bound: ByteCount,
    /// Conservative admission budget for output values and encoding buffers that coexist during
    /// construction. Actual scheduler accounting remains authoritative.
    pub construction_resident_bytes_admission_budget: ByteCount,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedOutputSetResourceFacts {
    /// The one output-set root shared by every numerical output of a workload.
    pub encoded_bytes_upper_bound: ByteCount,
}

/// Prepared transport and materialization facts shared by one independently admitted phase.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedPhaseResourceFacts {
    pub inputs: PreparedInputResourceFacts,
    pub output: PreparedOutputResourceFacts,
    pub output_set: PreparedOutputSetResourceFacts,
}

/// Exact prepared-object inputs imported by an independently scheduled structural-frequency
/// operator phase. Keeping these facts beside, rather than inside, the reusable numerical
/// operator demand prevents the legacy whole-pipeline plan from pretending it has an object
/// transport boundary that it does not yet expose.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralFrequencyOperatorResourceDemand {
    pub operator: StructuralOperatorResourceDemand,
    pub inputs: PreparedInputResourceFacts,
    pub output: PreparedOutputResourceFacts,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ObservationPartitionPayloadLayout {
    SelectedValues {
        entity_count: Count,
        component_count: Count,
    },
    WeightedAccumulator {
        component_count: Count,
    },
    ExtremaCandidates {
        candidate_count_upper_bound: Count,
        component_count: Count,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationPartitionResourceDemand {
    pub topology_digest: Digest,
    pub partition_entity_count: Count,
    pub source_entity_count: Count,
    pub source_partition_count: Count,
    pub source_value_count: Count,
    pub output_payload: ObservationPartitionPayloadLayout,
    pub operation_count_upper_bound: Count,
    pub inputs: PreparedInputResourceFacts,
    pub evaluator_workspace_bytes_upper_bound: ByteCount,
    pub output: PreparedOutputResourceFacts,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationReductionResourceDemand {
    pub topology_digest: Digest,
    pub contribution_count: Count,
    pub contribution_encoded_bytes: ByteCount,
    pub contribution_value_count: Count,
    pub output_entity_count: Count,
    pub output_value_count: Count,
    pub accumulator_value_count: Count,
    pub inputs: PreparedInputResourceFacts,
    pub workspace_bytes_upper_bound: ByteCount,
    pub output: PreparedOutputResourceFacts,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationHistoryCheckpointResourceDemand {
    pub topology_digest: Digest,
    pub predecessor_present: bool,
    pub predecessor_sample_count: Count,
    pub retained_sample_count: Count,
    pub inputs: PreparedInputResourceFacts,
    pub transition_workspace_bytes_upper_bound: ByteCount,
    pub output: PreparedOutputResourceFacts,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationAssemblyResourceDemand {
    pub topology_digest: Digest,
    pub result_count: Count,
    pub result_encoded_bytes: ByteCount,
    pub evidence_count: Count,
    pub inputs: PreparedInputResourceFacts,
    pub ordering_workspace_bytes_upper_bound: ByteCount,
    pub output: PreparedOutputResourceFacts,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralFrequencyResultAssemblyResourceDemand {
    pub scope_digest: Digest,
    pub entry_count: Count,
    pub entry_encoded_bytes: ByteCount,
    pub physical_dof_count: Count,
    pub inputs: PreparedInputResourceFacts,
    pub ordering_workspace_bytes_upper_bound: ByteCount,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralModalResponseSpectrumDirectionResourceDemand {
    pub operator: StructuralOperatorResourceDemand,
    pub spectrum_point_count: Count,
    pub mode_count_upper_bound: Count,
    pub correlation_pair_count_upper_bound: Count,
    pub response_vector_count_upper_bound: Count,
    pub inputs: PreparedInputResourceFacts,
    pub output: PreparedOutputResourceFacts,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralModalResponseSpectrumAssemblyResourceDemand {
    pub scope_digest: Digest,
    pub direction_count: Count,
    pub direction_encoded_bytes: ByteCount,
    pub physical_dof_count: Count,
    pub inputs: PreparedInputResourceFacts,
    pub ordering_workspace_bytes_upper_bound: ByteCount,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "demand")]
pub enum FeaResourceDemand {
    StructuralLinearPipeline(StructuralOperatorResourceDemand),
    StructuralModalExtraction(StructuralModalResourceDemand),
    StructuralBuckling(StructuralBucklingResourceDemand),
    StructuralFrequencyEntry(StructuralFrequencyOperatorResourceDemand),
    StructuralFrequencyFieldRecovery(StructuralFrequencyOperatorResourceDemand),
    StructuralObservationPartition(ObservationPartitionResourceDemand),
    StructuralObservationReduction(ObservationReductionResourceDemand),
    StructuralObservationHistoryCheckpoint(ObservationHistoryCheckpointResourceDemand),
    StructuralObservationAssembly(ObservationAssemblyResourceDemand),
    StructuralFrequencyResultAssembly(StructuralFrequencyResultAssemblyResourceDemand),
    StructuralModalResponseSpectrumDirection(
        StructuralModalResponseSpectrumDirectionResourceDemand,
    ),
    StructuralModalResponseSpectrumAssembly(StructuralModalResponseSpectrumAssemblyResourceDemand),
}

impl FeaResourceDemand {
    /// Returns the admitted semantic envelope for one output kind governed by this phase.
    /// Legacy whole-pipeline plans predate independently prepared output envelopes.
    pub const fn output_facts(
        &self,
        kind: AnalysisNumericalObjectKind,
    ) -> Option<&PreparedOutputResourceFacts> {
        match (self, kind) {
            (
                Self::StructuralModalExtraction(demand),
                AnalysisNumericalObjectKind::SolverResult,
            ) => Some(&demand.result_output),
            (
                Self::StructuralModalExtraction(demand),
                AnalysisNumericalObjectKind::NumericalReceipt,
            ) => Some(&demand.receipt_output),
            (Self::StructuralBuckling(demand), AnalysisNumericalObjectKind::SolverResult) => {
                Some(&demand.result_output)
            }
            (Self::StructuralBuckling(demand), AnalysisNumericalObjectKind::NumericalReceipt) => {
                Some(&demand.receipt_output)
            }
            (
                Self::StructuralFrequencyEntry(demand),
                AnalysisNumericalObjectKind::StructuralFrequencyEntryResult,
            )
            | (
                Self::StructuralFrequencyFieldRecovery(demand),
                AnalysisNumericalObjectKind::Field,
            ) => Some(&demand.output),
            (
                Self::StructuralObservationPartition(demand),
                AnalysisNumericalObjectKind::ObservationPartitionContribution,
            ) => Some(&demand.output),
            (
                Self::StructuralObservationReduction(demand),
                AnalysisNumericalObjectKind::ObservationResult,
            ) => Some(&demand.output),
            (
                Self::StructuralObservationHistoryCheckpoint(demand),
                AnalysisNumericalObjectKind::ObservationHistoryCheckpoint,
            ) => Some(&demand.output),
            (
                Self::StructuralObservationAssembly(demand),
                AnalysisNumericalObjectKind::Observation,
            ) => Some(&demand.output),
            (
                Self::StructuralFrequencyResultAssembly(demand),
                AnalysisNumericalObjectKind::SolverResult,
            ) => Some(&demand.result_output),
            (
                Self::StructuralFrequencyResultAssembly(demand),
                AnalysisNumericalObjectKind::NumericalReceipt,
            ) => Some(&demand.receipt_output),
            (
                Self::StructuralModalResponseSpectrumDirection(demand),
                AnalysisNumericalObjectKind::StructuralModalResponseSpectrumDirectionResult,
            ) => Some(&demand.output),
            (
                Self::StructuralModalResponseSpectrumAssembly(demand),
                AnalysisNumericalObjectKind::SolverResult,
            ) => Some(&demand.result_output),
            (
                Self::StructuralModalResponseSpectrumAssembly(demand),
                AnalysisNumericalObjectKind::NumericalReceipt,
            ) => Some(&demand.receipt_output),
            _ => None,
        }
    }

    pub const fn governs_prepared_outputs(&self) -> bool {
        !matches!(self, Self::StructuralLinearPipeline(_))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaResourcePlanningInput {
    /// Version of the deterministic projection and numerical accounting model.
    pub estimation_model_revision: u16,
    pub phase: FeaResourcePhase,
    pub demand: FeaResourceDemand,
    pub output_set: Option<PreparedOutputSetResourceFacts>,
    pub worker: WorkerResourcePolicy,
    pub accuracy_ceiling: AccuracyResourceCeiling,
}

impl FeaResourcePlanningInput {
    pub const fn topology_digest(&self) -> Digest {
        match &self.demand {
            FeaResourceDemand::StructuralLinearPipeline(demand) => demand.topology_digest,
            FeaResourceDemand::StructuralModalExtraction(demand) => demand.operator.topology_digest,
            FeaResourceDemand::StructuralBuckling(demand) => demand.operator.topology_digest,
            FeaResourceDemand::StructuralFrequencyEntry(demand)
            | FeaResourceDemand::StructuralFrequencyFieldRecovery(demand) => {
                demand.operator.topology_digest
            }
            FeaResourceDemand::StructuralObservationPartition(demand) => demand.topology_digest,
            FeaResourceDemand::StructuralObservationReduction(demand) => demand.topology_digest,
            FeaResourceDemand::StructuralObservationHistoryCheckpoint(demand) => {
                demand.topology_digest
            }
            FeaResourceDemand::StructuralObservationAssembly(demand) => demand.topology_digest,
            FeaResourceDemand::StructuralFrequencyResultAssembly(demand) => demand.scope_digest,
            FeaResourceDemand::StructuralModalResponseSpectrumDirection(demand) => {
                demand.operator.topology_digest
            }
            FeaResourceDemand::StructuralModalResponseSpectrumAssembly(demand) => {
                demand.scope_digest
            }
        }
    }

    pub const fn observations(&self) -> Option<&ObservationResourceFacts> {
        match &self.demand {
            FeaResourceDemand::StructuralLinearPipeline(demand) => Some(&demand.observations),
            FeaResourceDemand::StructuralModalExtraction(demand) => {
                Some(&demand.operator.observations)
            }
            FeaResourceDemand::StructuralBuckling(demand) => Some(&demand.operator.observations),
            FeaResourceDemand::StructuralFrequencyEntry(demand)
            | FeaResourceDemand::StructuralFrequencyFieldRecovery(demand) => {
                Some(&demand.operator.observations)
            }
            FeaResourceDemand::StructuralObservationPartition(_)
            | FeaResourceDemand::StructuralObservationReduction(_)
            | FeaResourceDemand::StructuralObservationHistoryCheckpoint(_)
            | FeaResourceDemand::StructuralObservationAssembly(_)
            | FeaResourceDemand::StructuralFrequencyResultAssembly(_) => None,
            FeaResourceDemand::StructuralModalResponseSpectrumDirection(demand) => {
                Some(&demand.operator.observations)
            }
            FeaResourceDemand::StructuralModalResponseSpectrumAssembly(_) => None,
        }
    }

    pub const fn partitioning(&self) -> &PartitioningPolicy {
        match &self.demand {
            FeaResourceDemand::StructuralLinearPipeline(demand) => &demand.partitioning,
            FeaResourceDemand::StructuralModalExtraction(demand) => &demand.operator.partitioning,
            FeaResourceDemand::StructuralBuckling(demand) => &demand.operator.partitioning,
            FeaResourceDemand::StructuralFrequencyEntry(demand)
            | FeaResourceDemand::StructuralFrequencyFieldRecovery(demand) => {
                &demand.operator.partitioning
            }
            FeaResourceDemand::StructuralObservationPartition(_)
            | FeaResourceDemand::StructuralObservationReduction(_)
            | FeaResourceDemand::StructuralObservationHistoryCheckpoint(_)
            | FeaResourceDemand::StructuralObservationAssembly(_)
            | FeaResourceDemand::StructuralFrequencyResultAssembly(_) => {
                static MONOLITHIC: PartitioningPolicy = PartitioningPolicy::Monolithic;
                &MONOLITHIC
            }
            FeaResourceDemand::StructuralModalResponseSpectrumDirection(demand) => {
                &demand.operator.partitioning
            }
            FeaResourceDemand::StructuralModalResponseSpectrumAssembly(_) => {
                static MONOLITHIC: PartitioningPolicy = PartitioningPolicy::Monolithic;
                &MONOLITHIC
            }
        }
    }

    pub const fn structural_operator(&self) -> Option<&StructuralOperatorResourceDemand> {
        match &self.demand {
            FeaResourceDemand::StructuralLinearPipeline(demand) => Some(demand),
            FeaResourceDemand::StructuralModalExtraction(demand) => Some(&demand.operator),
            FeaResourceDemand::StructuralBuckling(demand) => Some(&demand.operator),
            FeaResourceDemand::StructuralFrequencyEntry(demand)
            | FeaResourceDemand::StructuralFrequencyFieldRecovery(demand) => Some(&demand.operator),
            FeaResourceDemand::StructuralObservationPartition(_)
            | FeaResourceDemand::StructuralObservationReduction(_)
            | FeaResourceDemand::StructuralObservationHistoryCheckpoint(_)
            | FeaResourceDemand::StructuralObservationAssembly(_)
            | FeaResourceDemand::StructuralFrequencyResultAssembly(_) => None,
            FeaResourceDemand::StructuralModalResponseSpectrumDirection(demand) => {
                Some(&demand.operator)
            }
            FeaResourceDemand::StructuralModalResponseSpectrumAssembly(_) => None,
        }
    }

    pub fn structural_operator_mut(&mut self) -> Option<&mut StructuralOperatorResourceDemand> {
        match &mut self.demand {
            FeaResourceDemand::StructuralLinearPipeline(demand) => Some(demand),
            FeaResourceDemand::StructuralModalExtraction(demand) => Some(&mut demand.operator),
            FeaResourceDemand::StructuralBuckling(demand) => Some(&mut demand.operator),
            FeaResourceDemand::StructuralFrequencyEntry(demand)
            | FeaResourceDemand::StructuralFrequencyFieldRecovery(demand) => {
                Some(&mut demand.operator)
            }
            FeaResourceDemand::StructuralObservationPartition(_)
            | FeaResourceDemand::StructuralObservationReduction(_)
            | FeaResourceDemand::StructuralObservationHistoryCheckpoint(_)
            | FeaResourceDemand::StructuralObservationAssembly(_)
            | FeaResourceDemand::StructuralFrequencyResultAssembly(_) => None,
            FeaResourceDemand::StructuralModalResponseSpectrumDirection(demand) => {
                Some(&mut demand.operator)
            }
            FeaResourceDemand::StructuralModalResponseSpectrumAssembly(_) => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceBreakdown {
    /// Prepared numerical inputs, excluding the resource-plan object authorizing this phase.
    pub input_encoded_payload_bytes: ByteCount,
    pub input_transport_closure_bytes: ByteCount,
    pub input_decoded_resident_bytes: ByteCount,
    /// Nonrecursive fixed envelope for the authorizing resource-plan object itself.
    pub resource_plan_transport_closure_bytes: ByteCount,
    pub resource_plan_decoded_resident_bytes: ByteCount,
    pub output_set_bytes: ByteCount,
    pub topology_bytes: ByteCount,
    pub csr_row_offset_bytes: ByteCount,
    pub csr_column_index_bytes: ByteCount,
    pub csr_value_bytes: ByteCount,
    pub assembly_contribution_bytes: ByteCount,
    pub assembly_scratch_bytes: ByteCount,
    pub workspace_bytes: ByteCount,
    pub preconditioner_bytes: ByteCount,
    pub factorization_bytes: ByteCount,
    pub constraint_bytes: ByteCount,
    pub phase_resident_bytes: ByteCount,
    pub phase_scratch_bytes: ByteCount,
    pub phase_retained_bytes: ByteCount,
    pub result_artifact_bytes: ByteCount,
    pub observation_maximum_samples_per_observation: Count,
    pub observation_produced_sample_count: Count,
    pub observation_retained_sample_count: Count,
    pub observation_operation_count: Count,
    pub observation_peak_bytes: ByteCount,
    pub observation_stored_bytes: ByteCount,
    pub observation_transfer_bytes: ByteCount,
    pub result_egress_bytes: ByteCount,
    pub checkpoint_egress_bytes: ByteCount,
    pub total_egress_bytes: ByteCount,
    pub observation_collectives: Count,
    pub checkpoint_bytes: ByteCount,
    pub artifact_overhead_bytes: ByteCount,
    pub worker_overhead_bytes: ByteCount,
    pub partition_replicated_memory_bytes: ByteCount,
    pub partition_halo_memory_bytes: ByteCount,
    pub partition_replicated_scratch_bytes: ByteCount,
    pub partition_halo_scratch_bytes: ByteCount,
    pub partitionable_memory_bytes: ByteCount,
    pub fixed_memory_bytes: ByteCount,
    pub partitionable_scratch_bytes: ByteCount,
    pub fixed_scratch_bytes: ByteCount,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FeaResourcePlanIdentity(pub(crate) Digest);

impl FeaResourcePlanIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "FeaResourcePlanData", into = "FeaResourcePlanData")]
#[serde(deny_unknown_fields)]
pub struct FeaResourcePlan {
    pub schema_version: u16,
    pub identity: FeaResourcePlanIdentity,
    pub input: FeaResourcePlanningInput,
    pub breakdown: ResourceBreakdown,
    pub request: ResourceRequest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeaResourcePlanData {
    schema_version: u16,
    identity: FeaResourcePlanIdentity,
    input: FeaResourcePlanningInput,
    breakdown: ResourceBreakdown,
    request: ResourceRequest,
}

impl From<FeaResourcePlan> for FeaResourcePlanData {
    fn from(value: FeaResourcePlan) -> Self {
        Self {
            schema_version: value.schema_version,
            identity: value.identity,
            input: value.input,
            breakdown: value.breakdown,
            request: value.request,
        }
    }
}

impl TryFrom<FeaResourcePlanData> for FeaResourcePlan {
    type Error = ResourcePlanningError;

    fn try_from(value: FeaResourcePlanData) -> Result<Self, Self::Error> {
        let expected = super::estimate::build_resource_plan(value.input.clone())?;
        if value.schema_version != expected.schema_version
            || value.identity != expected.identity
            || value.breakdown != expected.breakdown
            || value.request != expected.request
        {
            return Err(ResourcePlanningError::Identity);
        }
        Ok(expected)
    }
}

impl FeaResourcePlan {
    pub(crate) fn derive_identity(&self) -> Result<FeaResourcePlanIdentity, ResourcePlanningError> {
        let encoded = encode_contract(
            RESOURCE_PREFIX,
            "resource-plan",
            &(
                self.schema_version,
                &self.input,
                &self.breakdown,
                &self.request,
            ),
            RESOURCE_LIMITS,
        )?;
        Ok(FeaResourcePlanIdentity(Digest::sha256(encoded)))
    }

    pub fn validate_identity(&self) -> Result<(), ResourcePlanningError> {
        if self.schema_version != 1 || self.identity != self.derive_identity()? {
            return Err(ResourcePlanningError::Identity);
        }
        self.request
            .validate()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ResourcePlanningError> {
        self.validate_identity()?;
        Ok(encode_contract(
            RESOURCE_PREFIX,
            "resource-plan-artifact",
            self,
            RESOURCE_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ResourcePlanningError> {
        let plan: Self = decode_contract(
            RESOURCE_PREFIX,
            "resource-plan-artifact",
            bytes,
            RESOURCE_LIMITS,
        )?;
        plan.validate_identity()?;
        Ok(plan)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResourceAdmission {
    Admit {
        plan: FeaResourcePlan,
    },
    RequiresPartitioning {
        plan: FeaResourcePlan,
        minimum_partition_count: u32,
        per_partition_request: ResourceRequest,
    },
    Reject {
        reason: ResourcePlanningError,
    },
}

#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
pub enum ResourcePlanningError {
    #[error("invalid FEA resource fact: {0}")]
    Invalid(String),
    #[error("FEA resource arithmetic overflow: {0}")]
    Overflow(&'static str),
    #[error("accuracy resource ceiling exceeded: {0}")]
    AccuracyCeiling(&'static str),
    #[error("worker inventory is insufficient: {0}")]
    Inventory(&'static str),
    #[error("worker lacks required capability `{0:?}`")]
    Capability(Capability),
    #[error(
        "worker lacks accelerator `{class}` count {count} with {memory_bytes_each} bytes each"
    )]
    Accelerator {
        class: String,
        count: u16,
        memory_bytes_each: u64,
    },
    #[error("FEA resource plan identity does not converge")]
    Identity,
    #[error("FEA resource plan encoding failed: {0}")]
    Encoding(String),
}

impl From<runmat_canonical_codec::CanonicalCodecError> for ResourcePlanningError {
    fn from(value: runmat_canonical_codec::CanonicalCodecError) -> Self {
        Self::Encoding(value.to_string())
    }
}
