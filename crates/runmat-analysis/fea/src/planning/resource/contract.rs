use std::collections::BTreeSet;

use runmat_analysis_core::AccuracyResourceCeiling;
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::{
    identity::{AttemptId, WorkerId},
    resource::{
        AcceleratorRequest, Capability, ResourceAccounting, ResourceAccountingSummary,
        ResourceRequest, ResourceUsage,
    },
    Digest,
};
use serde::{Deserialize, Serialize};

const RESOURCE_PREFIX: &[u8] = b"runmat-fea-resource-plan/v1\0";
const RESOURCE_USAGE_PREFIX: &[u8] = b"runmat-fea-resource-usage/v1\0";
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
    /// Version of the deterministic projection and numerical accounting model.
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisResourceUsage {
    pub scheduler: ResourceUsage,
    pub peak_cpu_millicores: u32,
    pub peak_memory_bytes: ByteCount,
    pub peak_scratch_bytes: ByteCount,
    pub artifact_bytes: ByteCount,
}

impl From<ResourceAccountingSummary> for AnalysisResourceUsage {
    fn from(summary: ResourceAccountingSummary) -> Self {
        Self {
            scheduler: summary.usage,
            peak_cpu_millicores: summary.peak_cpu_millicores,
            peak_memory_bytes: ByteCount(summary.peak_memory_bytes),
            peak_scratch_bytes: ByteCount(summary.peak_scratch_bytes),
            artifact_bytes: ByteCount(summary.artifact_bytes),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResourceUsageEvidenceIdentity(Digest);

impl ResourceUsageEvidenceIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceUsageAttempt {
    pub attempt_id: AttemptId,
    pub worker_id: WorkerId,
    pub resources: ResourceAccounting,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ResourceUsageEvidenceWire")]
pub struct ResourceUsageEvidence {
    pub schema_version: u16,
    identity: ResourceUsageEvidenceIdentity,
    pub plan: FeaResourcePlanIdentity,
    pub workload_digest: Digest,
    pub result_digest: Digest,
    pub budget: ResourceRequest,
    pub attempts: Vec<ResourceUsageAttempt>,
    pub actual: AnalysisResourceUsage,
    pub overruns: Vec<ResourceOverrun>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResourceUsageEvidenceWire {
    schema_version: u16,
    identity: ResourceUsageEvidenceIdentity,
    plan: FeaResourcePlanIdentity,
    workload_digest: Digest,
    result_digest: Digest,
    budget: ResourceRequest,
    attempts: Vec<ResourceUsageAttempt>,
    actual: AnalysisResourceUsage,
    overruns: Vec<ResourceOverrun>,
}

impl TryFrom<ResourceUsageEvidenceWire> for ResourceUsageEvidence {
    type Error = ResourcePlanningError;

    fn try_from(wire: ResourceUsageEvidenceWire) -> Result<Self, Self::Error> {
        let actual = wire.actual;
        let expected = Self::build(
            wire.plan,
            wire.workload_digest,
            wire.result_digest,
            wire.budget,
            wire.attempts,
            actual.scheduler.wall_millis,
        )?;
        if wire.schema_version != expected.schema_version
            || wire.identity != expected.identity
            || actual != expected.actual
            || wire.overruns != expected.overruns
        {
            return Err(ResourcePlanningError::Identity);
        }
        Ok(expected)
    }
}

impl ResourceUsageEvidence {
    pub(crate) fn build(
        plan: FeaResourcePlanIdentity,
        workload_digest: Digest,
        result_digest: Digest,
        budget: ResourceRequest,
        mut attempts: Vec<ResourceUsageAttempt>,
        graph_wall_millis: u64,
    ) -> Result<Self, ResourcePlanningError> {
        if workload_digest == Digest::sha256([]) || result_digest == Digest::sha256([]) {
            return Err(ResourcePlanningError::Invalid(
                "resource usage must bind nonempty sources".into(),
            ));
        }
        attempts.sort_by_key(|attempt| attempt.attempt_id);
        if attempts.is_empty()
            || attempts
                .windows(2)
                .any(|pair| pair[0].attempt_id == pair[1].attempt_id)
        {
            return Err(ResourcePlanningError::Invalid(
                "resource usage requires distinct measured attempts".into(),
            ));
        }
        for attempt in &attempts {
            attempt
                .resources
                .validate()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        }
        budget
            .validate()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        if attempts.iter().any(|attempt| {
            !super::admission::allocation_fits_budget(&attempt.resources.allocation, &budget)
        }) {
            return Err(ResourcePlanningError::Invalid(
                "resource usage attempt allocation exceeds its budget".into(),
            ));
        }
        let intervals = attempts
            .iter()
            .map(|attempt| attempt.resources.clone())
            .collect::<Vec<_>>();
        let summary = ResourceAccountingSummary::aggregate(&intervals, graph_wall_millis)
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        let actual = AnalysisResourceUsage::from(summary);
        let overruns = super::admission::reconcile_resource_request(&budget, &actual)?;
        let identity = ResourceUsageEvidenceIdentity(Digest::sha256(encode_contract(
            RESOURCE_USAGE_PREFIX,
            "resource-usage-identity",
            &(
                plan,
                workload_digest,
                result_digest,
                &budget,
                &attempts,
                &actual,
                &overruns,
            ),
            RESOURCE_LIMITS,
        )?));
        Ok(Self {
            schema_version: 1,
            identity,
            plan,
            workload_digest,
            result_digest,
            budget,
            attempts,
            actual,
            overruns,
        })
    }

    pub const fn identity(&self) -> ResourceUsageEvidenceIdentity {
        self.identity
    }

    pub fn validate_identity(&self) -> Result<(), ResourcePlanningError> {
        let expected = Self::build(
            self.plan,
            self.workload_digest,
            self.result_digest,
            self.budget.clone(),
            self.attempts.clone(),
            self.actual.scheduler.wall_millis,
        )?;
        if self != &expected {
            return Err(ResourcePlanningError::Identity);
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ResourcePlanningError> {
        self.validate_identity()?;
        Ok(encode_contract(
            RESOURCE_USAGE_PREFIX,
            "resource-usage-evidence",
            self,
            RESOURCE_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ResourcePlanningError> {
        Ok(decode_contract(
            RESOURCE_USAGE_PREFIX,
            "resource-usage-evidence",
            bytes,
            RESOURCE_LIMITS,
        )?)
    }
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
