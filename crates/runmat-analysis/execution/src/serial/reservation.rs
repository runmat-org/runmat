//! Immutable execution capacity at the shared analysis dispatch boundary.
use super::*;
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution::resource::ResourceRequest;

pub struct ReservedAnalysisKernelInvocation<'a> {
    invocation: AnalysisKernelInvocation<'a>,
    resources: &'a ResourceRequest,
}
impl<'a> ReservedAnalysisKernelInvocation<'a> {
    pub(super) fn new(
        invocation: AnalysisKernelInvocation<'a>,
        resources: &'a ResourceRequest,
        plan: Option<&FeaResourcePlan>,
    ) -> AnalysisBridgeResult<Self> {
        let plan = plan.ok_or(AnalysisBridgeError::Identity(
            "reserved analysis requires a current governing resource plan",
        ))?;
        plan.request
            .require_within(resources)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        Ok(Self {
            invocation,
            resources,
        })
    }
    pub fn resources(&self) -> &ResourceRequest {
        self.resources
    }
    pub fn invocation(&self) -> &AnalysisKernelInvocation<'a> {
        &self.invocation
    }
    pub fn into_invocation(self) -> AnalysisKernelInvocation<'a> {
        self.invocation
    }
}
