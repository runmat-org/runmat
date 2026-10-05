use super::*;
impl NumericalPlanIdentity {
    /// Original transient block intent and source mesh; independent of placement/chunks.
    pub fn from_fluid_transient_block_workload(
        workload: AnalysisWorkloadIdentity,
        topology: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "fluid-transient-block-numerical-plan",
            &(workload, topology),
        )
    }
    /// Original steady finite-volume workload and mesh; independent of placement/chunks.
    pub fn from_fluid_steady_workload(
        workload: AnalysisWorkloadIdentity,
        topology: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive("fluid-steady-numerical-plan", &(workload, topology))
    }
}
