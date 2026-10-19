use super::*;

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
    pub execution_evidence_encoded_bytes_upper_bound: ByteCount,
    pub execution_evidence_workspace_bytes_upper_bound: ByteCount,
    pub output: PreparedOutputResourceFacts,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationHistoryFinalizationResourceDemand {
    pub request: ObservationHistoryFinalizationRequestIdentity,
    pub topology_digest: Digest,
    pub retained_sample_count: Count,
    pub retained_value_count: Count,
    pub inputs: PreparedInputResourceFacts,
    pub workspace_bytes_upper_bound: ByteCount,
    pub history_output: PreparedOutputResourceFacts,
    pub series_output: Option<PreparedOutputResourceFacts>,
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
