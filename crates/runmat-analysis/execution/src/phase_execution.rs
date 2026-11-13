//! Original portable task history plus its enclosing host-observed execution interval.
//! Construction validates internal lineage, not worker authorization, utilization or RSS.
use crate::{
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisHostWorkload, PlannedResourceUsageGraph,
};
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution::{state::TaskState, value::ValuePayload};
use runmat_execution_artifact::ProgramExecutionResponse;
use runmat_execution_runner::{AttemptResourceAccounting, TaskExecutionEvidence};
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AnalysisPhaseExecutionEvidence {
    task: TaskExecutionEvidence,
    /// Host-observed interval, independent of summed reserved attempt time.
    graph_wall_millis: u64,
}

impl<'de> Deserialize<'de> for AnalysisPhaseExecutionEvidence {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            task: TaskExecutionEvidence,
            graph_wall_millis: u64,
        }
        let wire = Wire::deserialize(deserializer)?;
        Self::new(wire.task, wire.graph_wall_millis).map_err(serde::de::Error::custom)
    }
}

impl AnalysisPhaseExecutionEvidence {
    pub fn new(task: TaskExecutionEvidence, graph_wall_millis: u64) -> AnalysisBridgeResult<Self> {
        task.validate().map_err(invalid)?;
        if task.task.state != TaskState::Succeeded
            || task
                .attempts
                .iter()
                .any(|attempt| attempt.terminal.is_none())
            || graph_wall_millis == 0
        {
            return Err(invalid(
                "phase execution requires a terminal successful task and observed interval",
            ));
        }
        Ok(Self {
            task,
            graph_wall_millis,
        })
    }

    pub fn task_evidence(&self) -> &TaskExecutionEvidence {
        &self.task
    }

    pub fn graph_wall_millis(&self) -> u64 {
        self.graph_wall_millis
    }

    /// Only for checking the delivered physical output. Complete usage comes from resource_graph.
    pub fn winning_accounting(&self) -> &AttemptResourceAccounting {
        self.task
            .task
            .committed
            .as_ref()
            .expect("sealed successful task has its original commit")
            .result
            .resources()
            .expect("sealed commit has original accounting")
    }

    pub fn resource_graph(
        &self,
        plan: &FeaResourcePlan,
    ) -> AnalysisBridgeResult<PlannedResourceUsageGraph> {
        PlannedResourceUsageGraph::from_execution(plan, &self.task, self.graph_wall_millis)
    }

    pub fn validate_for(
        &self,
        host: &AnalysisHostWorkload,
        plan: &FeaResourcePlan,
        response: &ProgramExecutionResponse,
    ) -> AnalysisBridgeResult<()> {
        host.validate()?;
        let request = &self.task.task.submission.request;
        let commit = self
            .task
            .task
            .committed
            .as_ref()
            .expect("sealed successful task has a commit");
        let ProgramExecutionResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } = response
        else {
            return Err(invalid("phase execution requires externalized completion"));
        };
        let Some((committed_outputs, committed_objects)) = commit.result.values() else {
            return Err(invalid("phase commit is not a value result"));
        };
        if request.callable.owner_identity != crate::task::ANALYSIS_EXECUTION_CALLABLE_OWNER
            || request.callable.program != crate::host::analysis_host_callable()
            || outputs.as_slice() != committed_outputs
            || result_objects.as_slice() != committed_objects
        {
            return Err(AnalysisBridgeError::Identity(
                "phase response or callable differs from its original task",
            ));
        }
        let roots = request
            .inputs
            .iter()
            .map(|input| match input {
                ValuePayload::Object(root) => Ok(root.as_ref().clone()),
                _ => Err(invalid("phase task has a non-object numerical input")),
            })
            .collect::<AnalysisBridgeResult<Vec<_>>>()?;
        let Some((invocation, inputs)) = roots.split_first() else {
            return Err(invalid("phase task omits its original host invocation"));
        };
        let (_, expected_invocation) = host.invocation_descriptor()?;
        if invocation != &expected_invocation {
            return Err(AnalysisBridgeError::Identity(
                "phase task invocation differs from its host",
            ));
        }
        host.validate_inputs(inputs)?;
        crate::phase_accounting::validate_winning_accounting(
            host,
            plan,
            response,
            self.winning_accounting(),
        )?;
        self.resource_graph(plan)?;
        Ok(())
    }
}

fn invalid(message: impl ToString) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.to_string())
}
