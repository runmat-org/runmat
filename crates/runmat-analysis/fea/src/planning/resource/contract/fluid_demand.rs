use super::*;
use crate::planning::resource::FluidSteadyResourceProjection;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidSteadySolveResourceDemand {
    pub projection: FluidSteadyResourceProjection,
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidFieldRecoveryResourceDemand {
    pub request: FieldRecoveryRequestIdentity,
    pub source_result: SolverResultIdentity,
    pub topology_digest: Digest,
    pub construction_workspace_bytes: ByteCount,
    pub source_validation_workspace_bytes: ByteCount,
    pub actual_decode_resident_bytes: ByteCount,
    pub inputs: PreparedInputResourceFacts,
    pub output: PreparedOutputResourceFacts,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidTransientBlockResourceDemand {
    pub projection: crate::planning::resource::FluidTransientBlockResourceProjection,
    pub inputs: PreparedInputResourceFacts,
    pub output: PreparedOutputResourceFacts,
}
