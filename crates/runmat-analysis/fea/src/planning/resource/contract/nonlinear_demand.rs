use super::*;

/// Nonlinear storage remains live during assembly, solve and field construction;
/// it cannot be hidden inside a sequential linear-recovery peak.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralNonlinearPipelineResourceDemand {
    /// Fresh original compiler phase; cold values are claims rederived by execution.
    pub constraint_compilation_peak_bytes: super::ByteCount,
    pub operator: StructuralOperatorResourceDemand,
    pub integration_identity: Digest,
    pub point_count: Count,
    pub maximum_accepted_increments: Count,
    /// Live outputs for this phase; global controller/work caps remain above.
    pub maximum_retained_output_increments: Count,
    pub tangent_evaluation_limit: Count,
    pub linear_solve_limit: Count,
    pub additional_persistent_bytes: ByteCount,
    pub state_profiles_output: PreparedOutputResourceFacts,
    pub block_transport: Option<NonlinearSolidBlockTransportDemand>,
}

/// Separate transport authority for an independently admitted original-history block.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonlinearSolidBlockTransportDemand {
    pub request: Digest,
    pub inputs: PreparedInputResourceFacts,
    pub output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedNonlinearSolidBlockResourceFacts {
    pub local_execution: crate::planning::resource::LocalExecutionPlan,
    pub inputs: PreparedInputResourceFacts,
    pub output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
    pub output_set: PreparedOutputSetResourceFacts,
}

/// Read-only revalidation and projection of an original accepted nonlinear history.
/// It neither advances load steps nor produces a new solver/state-profile result.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralNonlinearRecoveryResourceDemand {
    /// Fresh original compiler phase; cold values are claims rederived by execution.
    pub constraint_compilation_peak_bytes: super::ByteCount,
    pub operator: StructuralOperatorResourceDemand,
    pub integration_identity: Digest,
    pub point_count: Count,
    pub retained_output_increments: Count,
    pub physical_evaluation_limit: Count,
    pub additional_persistent_bytes: ByteCount,
}
