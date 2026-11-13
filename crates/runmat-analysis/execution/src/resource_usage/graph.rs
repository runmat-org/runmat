//! Bind an independently admitted plan to complete portable execution accounting.
use crate::{AnalysisBridgeError, AnalysisBridgeResult};
use runmat_analysis_fea::planning::resource::{allocation_matches_plan, FeaResourcePlan};
use runmat_execution_runner::{
    AttemptResourceAccounting, AttemptResourceAggregate, TaskExecutionEvidence,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedResourceUsageAttempt {
    /// The analysis-domain plan admitted for this exact scheduler attempt.
    pub plan: FeaResourcePlan,
    pub accounting: AttemptResourceAccounting,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedResourceUsageGraph {
    pub attempts: Vec<PlannedResourceUsageAttempt>,
    /// Elapsed wall time of this subgraph, independent of summed reserved task time.
    pub wall_millis: u64,
}

impl PlannedResourceUsageGraph {
    pub fn new(
        attempts: Vec<PlannedResourceUsageAttempt>,
        wall_millis: u64,
    ) -> AnalysisBridgeResult<Self> {
        for attempt in &attempts {
            attempt
                .plan
                .validate_identity()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
            if !allocation_matches_plan(
                &attempt.accounting.resources.allocation,
                &attempt.plan.request,
            ) {
                return Err(AnalysisBridgeError::Identity(
                    "original attempt allocation differs from the admitted plan",
                ));
            }
        }
        AttemptResourceAggregate::new(
            attempts
                .iter()
                .map(|attempt| attempt.accounting.clone())
                .collect(),
            wall_millis,
        )
        .map_err(|error| {
            AnalysisBridgeError::Invalid(format!("invalid execution-graph accounting: {error}"))
        })?;
        Ok(Self {
            attempts,
            wall_millis,
        })
    }
}

impl PlannedResourceUsageGraph {
    /// Every original attempt is charged. Unknown consumption blocks complete usage;
    /// this validates internal lineage, not independent worker authorization or RSS.
    pub fn from_execution(
        plan: &FeaResourcePlan,
        execution: &TaskExecutionEvidence,
        wall_millis: u64,
    ) -> AnalysisBridgeResult<Self> {
        plan.validate_identity()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        execution
            .validate()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        if !allocation_matches_plan(&execution.task.submission.request.resources, &plan.request) {
            return Err(AnalysisBridgeError::Identity(
                "original task reservation differs from the admitted plan",
            ));
        }
        let aggregate = execution
            .resource_aggregate(wall_millis)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        let attempts = aggregate
            .attempts
            .into_iter()
            .map(|accounting| PlannedResourceUsageAttempt {
                plan: plan.clone(),
                accounting,
            })
            .collect();
        Self::new(attempts, wall_millis)
    }
}

#[cfg(test)]
mod tests;
