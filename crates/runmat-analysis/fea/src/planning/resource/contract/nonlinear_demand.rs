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
    pub history_limits: NonlinearHistoryLimits,
}

/// Canonical-codec bounds of one accepted nonlinear history, derived by the resource plan from
/// the DOF, element and retained-increment counts. They bound the numerical content digest and
/// the controller continuation; they are the plan's admitted inventory, not format constants.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonlinearHistoryLimits {
    pub encoded_bytes: super::ByteCount,
    pub collection_items: Count,
}

impl NonlinearHistoryLimits {
    /// String and nesting bounds are fixed by the history's format, not by its size.
    pub const MAXIMUM_STRING_BYTES: usize = 65_536;
    pub const MAXIMUM_NESTING_DEPTH: usize = 64;

    pub fn canonical(self) -> Result<runmat_canonical_codec::CanonicalLimits, String> {
        let bytes = usize::try_from(self.encoded_bytes.0)
            .map_err(|_| "nonlinear history byte bound exceeds this platform".to_string())?;
        let items = usize::try_from(self.collection_items.0)
            .map_err(|_| "nonlinear history item bound exceeds this platform".to_string())?;
        Ok(runmat_canonical_codec::CanonicalLimits::new(
            bytes,
            items,
            Self::MAXIMUM_STRING_BYTES,
            Self::MAXIMUM_NESTING_DEPTH,
        ))
    }
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
    pub history_limits: NonlinearHistoryLimits,
}
