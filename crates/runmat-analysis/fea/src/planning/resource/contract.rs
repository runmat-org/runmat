use std::collections::BTreeSet;

use runmat_analysis_core::AccuracyResourceCeiling;
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::{
    resource::{AcceleratorRequest, Capability, ResourceRequest, ResourceUsage},
    Digest,
};
use serde::{Deserialize, Serialize};

const RESOURCE_PREFIX: &[u8] = b"runmat-fea-resource-plan/v1\0";
const RESOURCE_LIMITS: CanonicalLimits =
    CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);
pub const RESOURCE_ESTIMATION_MODEL_REVISION: u16 = 1;

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
    /// Caller-attested upper bound for the complete resident projection, including
    /// heap allocations, maps, provenance, and allocator overhead.
    pub resident_bytes_upper_bound: ByteCount,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseResourceFacts {
    pub rows: Count,
    pub columns: Count,
    pub nonzeros: Count,
    pub value_array_count: Count,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstraintResourceFacts {
    pub rows: Count,
    pub coefficients: Count,
    pub reference_points: Count,
    /// Caller-attested upper bound for the complete immutable constraint plan.
    pub resident_bytes_upper_bound: ByteCount,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationResourceFacts {
    pub sample_count: Count,
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
#[serde(rename_all = "snake_case")]
pub enum SolveResourcePolicy {
    Iterative {
        vector_count: Count,
        scalar: NumericalScalar,
        preconditioner: PreconditionerStorage,
    },
    DirectFactorization {
        vector_count: Count,
        scalar: NumericalScalar,
        factor_nonzero_upper_bound: Count,
        factor_bytes_upper_bound: ByteCount,
    },
}

impl SolveResourcePolicy {
    pub(crate) fn vector_count(&self) -> u64 {
        match self {
            Self::Iterative { vector_count, .. }
            | Self::DirectFactorization { vector_count, .. } => vector_count.0,
        }
    }

    pub(crate) fn scalar_bytes(&self) -> u64 {
        let scalar = match self {
            Self::Iterative { scalar, .. } | Self::DirectFactorization { scalar, .. } => scalar,
        };
        match scalar {
            NumericalScalar::RealF64 => 8,
            NumericalScalar::ComplexF64 => 16,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryResourcePolicy {
    pub resident_value_count: Count,
    pub resident_bytes_upper_bound: ByteCount,
    pub scratch_bytes: ByteCount,
    pub retained_bytes: ByteCount,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointArtifactPolicy {
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaResourcePlanningInput {
    /// Version of the evidence model used for caller-attested resident bounds.
    pub estimation_model_revision: u16,
    pub topology_digest: Digest,
    pub operator_pattern_digest: Digest,
    pub topology: TopologyResourceFacts,
    pub sparse: SparseResourceFacts,
    pub constraints: ConstraintResourceFacts,
    pub assembly: AssemblyResourcePolicy,
    pub solve: SolveResourcePolicy,
    pub recovery: RecoveryResourcePolicy,
    pub observations: ObservationResourceFacts,
    pub publication: CheckpointArtifactPolicy,
    pub worker: WorkerResourcePolicy,
    pub partitioning: PartitioningPolicy,
    pub accuracy_demand: AccuracyResourceDemand,
    pub accuracy_ceiling: AccuracyResourceCeiling,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceBreakdown {
    pub topology_bytes: ByteCount,
    pub csr_row_offset_bytes: ByteCount,
    pub csr_column_index_bytes: ByteCount,
    pub csr_value_bytes: ByteCount,
    pub assembly_contribution_bytes: ByteCount,
    pub assembly_scratch_bytes: ByteCount,
    pub solve_vector_bytes: ByteCount,
    pub preconditioner_bytes: ByteCount,
    pub factorization_bytes: ByteCount,
    pub constraint_bytes: ByteCount,
    pub recovery_resident_bytes: ByteCount,
    pub recovery_scratch_bytes: ByteCount,
    pub recovery_retained_bytes: ByteCount,
    pub observation_sample_count: Count,
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnalysisResourceUsage {
    pub scheduler: ResourceUsage,
    pub peak_cpu_millicores: u32,
    pub peak_memory_bytes: ByteCount,
    pub peak_scratch_bytes: ByteCount,
    pub artifact_bytes: ByteCount,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResourceOverrun {
    CpuPeak {
        planned: u32,
        actual: u32,
    },
    CpuTime {
        planned_core_millis: u64,
        actual_core_millis: u64,
    },
    Memory {
        planned: u64,
        actual: u64,
    },
    MemoryTime {
        planned_byte_millis: u128,
        actual_byte_millis: u128,
    },
    Scratch {
        planned: u64,
        actual: u64,
    },
    Artifact {
        planned: u64,
        actual: u64,
    },
    Egress {
        planned: u64,
        actual: u64,
    },
    Relay {
        planned: u64,
        actual: u64,
    },
    WallTime {
        planned: u64,
        actual: u64,
    },
    RetainedTime {
        planned_byte_millis: u128,
        actual_byte_millis: u128,
    },
    AcceleratorTime {
        class: String,
        planned_device_millis: u64,
        actual_device_millis: u64,
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
