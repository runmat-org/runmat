//! Prepared transport and materialization resource facts shared by numerical phases.

use serde::{Deserialize, Serialize};

use super::ByteCount;

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

/// Prepared transport envelopes for the modal task outputs and their shared output-set root.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedStructuralModalResourceFacts {
    pub local_execution: super::super::LocalExecutionPlan,
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
    pub state_profiles_output: PreparedOutputResourceFacts,
    pub output_set: PreparedOutputSetResourceFacts,
}
