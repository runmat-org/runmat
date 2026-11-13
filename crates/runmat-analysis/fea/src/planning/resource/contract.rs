use std::collections::BTreeSet;

use runmat_analysis_core::AccuracyResourceCeiling;
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::{
    resource::{AcceleratorRequest, Capability, ResourceRequest},
    Digest,
};
use serde::{Deserialize, Serialize};

use crate::execution::{
    AccuracyEvaluationRequestIdentity, AnalysisNumericalObjectKind, AnalysisWorkloadIdentity,
    FieldRecoveryRequestIdentity, ObservationAssemblyRequestIdentity,
    ObservationHistoryCheckpointRequestIdentity, ObservationHistoryFinalizationRequestIdentity,
    ObservationPartitionEvaluationRequestIdentity, ObservationPartitionReductionRequestIdentity,
    SolverResultIdentity, StructuralBucklingWorkloadIdentity,
    StructuralFrequencyEntryRequestIdentity, StructuralFrequencyResultAssemblyRequestIdentity,
    StructuralModalResponseSpectrumAssemblyRequestIdentity,
    StructuralModalResponseSpectrumDirectionRequestIdentity, StructuralModalWorkloadIdentity,
    StructuralRandomResponseAssemblyRequestIdentity,
    StructuralRandomResponseIntervalRequestIdentity, StructuralTransientBlockRequestIdentity,
    StructuralTransientResultAssemblyRequestIdentity,
};
mod accuracy_demand;
mod acoustic_demand;
mod electrical_static_demand;
mod fluid_demand;
pub use fluid_demand::{
    FluidFieldRecoveryResourceDemand, FluidSteadySolveResourceDemand,
    FluidTransientBlockResourceDemand,
};
mod electrical_recovery_demand;
mod magnetic_recovery_demand;
mod structural_recovery_demand;
pub use structural_recovery_demand::StructuralFieldRecoveryResourceDemand;
mod nonlinear_recovery_demand;
pub use nonlinear_recovery_demand::NonlinearFieldRecoveryResourceDemand;
mod maxwell_recovery_demand;
mod acoustic_response_demand;
pub use acoustic_response_demand::AcousticResponseResourceDemand;
pub use maxwell_recovery_demand::MaxwellFieldRecoveryResourceDemand;
mod acoustic_recovery_demand;
pub use acoustic_recovery_demand::AcousticFieldRecoveryResourceDemand;
mod magnetic_static_demand;
mod maxwell_harmonic_demand;
pub use acoustic_demand::AcousticHarmonicResourceDemand;
pub use electrical_recovery_demand::ElectricalFieldRecoveryResourceDemand;
pub use electrical_static_demand::ElectricalStaticResourceDemand;
pub use magnetic_recovery_demand::MagneticFieldRecoveryResourceDemand;
pub use magnetic_static_demand::{MagneticGmresResourceEnvelope, MagnetostaticResourceDemand};
pub use maxwell_harmonic_demand::{MaxwellDirectFactorEnvelope, MaxwellHarmonicResourceDemand};
mod assembly_demand;
pub use accuracy_demand::AccuracyEvaluationResourceDemand;
mod error;
mod modal_demand;
mod prepared;
mod random_response_demand;
mod result_publication_demand;
mod state_publication_demand;
pub use result_publication_demand::ResultPublicationResourceDemand;
mod static_demand;
pub use static_demand::StructuralLinearPipelineResourceDemand;
mod nonlinear_demand;
pub use nonlinear_demand::{
    NonlinearHistoryLimits, NonlinearSolidBlockTransportDemand,
    PreparedNonlinearSolidBlockResourceFacts, StructuralNonlinearPipelineResourceDemand,
    StructuralNonlinearRecoveryResourceDemand,
};
mod thermal_demand;
pub use state_publication_demand::SemanticStatePublicationResourceDemand;
mod transient_block_demand;
mod transient_recovery_demand;
mod transient_scope_demand;
pub use assembly_demand::{
    AcousticFrequencyAssemblyResourceDemand, StructuralFrequencyResultAssemblyResourceDemand,
    StructuralModalResponseSpectrumAssemblyResourceDemand,
    StructuralRandomResponseAssemblyResourceDemand,
    StructuralTransientResultAssemblyResourceDemand,
};
pub use error::{ResourcePlanningError, RESIDENT_MEMORY_RESOURCE};
pub use modal_demand::{
    StructuralModalResourceDemand, StructuralModalResponseSpectrumDirectionResourceDemand,
};
pub use prepared::{
    PreparedInputResourceFacts, PreparedOutputResourceFacts, PreparedOutputSetResourceFacts,
    PreparedPhaseResourceFacts, PreparedStructuralModalResourceFacts,
};
pub use random_response_demand::StructuralRandomResponseIntervalResourceDemand;
pub use thermal_demand::{ThermalFieldSourceResourceFacts, ThermalSolveResourceDemand};
pub use transient_block_demand::StructuralTransientBlockResourceDemand;
pub use transient_recovery_demand::StructuralTransientFieldRecoveryResourceDemand;
pub use transient_scope_demand::StructuralTransientScopeResourceDemand;

pub use super::solve_policy::{
    ModalSolveSelection, NumericalScalar, PreconditionerStorage, SolveResourcePolicy,
    SolverAdmissionResourcePolicy, StaticSolvePath, StaticSolveSelection, SubspaceSolvePath,
};

const RESOURCE_PREFIX: &[u8] = b"runmat-fea-resource-plan/v1\0";
pub const RESOURCE_PLAN_MAX_ENCODED_BYTES: u64 = 64 * 1024;
const RESOURCE_LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024, 8_192, 65_536, 64);
pub const RESOURCE_ESTIMATION_MODEL_REVISION: u16 = 111;

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

/// Exact family-owned field-recovery work admitted before observation evaluation.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldRecoveryResourceFacts {
    pub requirement_count: Count,
    pub source_entity_count: Count,
    pub output_entity_count: Count,
    pub topology_incidence_count: Count,
    pub arithmetic_operation_count: Count,
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
    pub class: runmat_execution::resource::AcceleratorClass,
    pub count: u16,
    pub memory_bytes_each: ByteCount,
}

impl From<&AcceleratorRequirement> for AcceleratorRequest {
    fn from(value: &AcceleratorRequirement) -> Self {
        Self {
            class: value.class.clone(),
            count: value.count,
            minimum_allocation_bytes: value.memory_bytes_each.0,
            provider: None,
            required_features: Default::default(),
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

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExcitationWorkspaceResourceFacts {
    pub active_field_count: Count,
    pub scalar_component_count: Count,
    pub resident_bytes_upper_bound: ByteCount,
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
    pub excitation_workspace: ExcitationWorkspaceResourceFacts,
    pub solve: SolveResourcePolicy,
    pub phase_resources: PhaseResourcePolicy,
    pub field_recovery: FieldRecoveryResourceFacts,
    pub observations: ObservationResourceFacts,
    pub publication: CheckpointArtifactPolicy,
    pub partitioning: PartitioningPolicy,
    pub accuracy_demand: AccuracyResourceDemand,
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

/// Exact prepared-object inputs imported by an independently scheduled structural-frequency
/// operator phase. Keeping these facts beside, rather than inside, the reusable numerical
/// operator demand prevents the legacy whole-pipeline plan from pretending it has an object
/// transport boundary that it does not yet expose.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralFrequencyOperatorResourceDemand {
    /// Fresh original constraint construction, rederived by workers.
    pub constraint_compilation_peak_bytes: ByteCount,
    pub operator: StructuralOperatorResourceDemand,
    pub inputs: PreparedInputResourceFacts,
    pub output: PreparedOutputResourceFacts,
}

mod observation_demand;
pub use observation_demand::{
    ObservationAssemblyResourceDemand, ObservationHistoryCheckpointResourceDemand,
    ObservationHistoryFinalizationResourceDemand, ObservationPartitionPayloadLayout,
    ObservationPartitionResourceDemand, ObservationReductionResourceDemand,
};

mod phases;
pub use phases::{FeaResourceDemand, FeaResourcePhase};

mod output_facts;

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

mod planning_input;

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
    pub field_recovery_requirement_count: Count,
    pub field_recovery_source_entity_count: Count,
    pub field_recovery_output_entity_count: Count,
    pub field_recovery_topology_incidence_count: Count,
    pub field_recovery_arithmetic_operation_count: Count,
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
