use runmat_execution::identity::{AttemptId, WorkerId};
use runmat_execution::state::AttemptState;
use runmat_execution::task::TaskRequest;
use runmat_execution::value::{GraphNode, PayloadGraph, ValuePayload, ValueRef};
use runmat_execution::{ExecutionScopeId, TaskId};
use serde::{Deserialize, Serialize};

use crate::cancellation::CancellationEscalation;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AttemptRequest {
    pub id: AttemptId,
    pub task_id: TaskId,
    pub scope_id: ExecutionScopeId,
    pub worker_id: WorkerId,
    pub ordinal: u16,
    pub driver_fence: u64,
    pub resource_assignment: runmat_execution::resource::ResourceAssignment,
    pub task: TaskRequest,
}

impl AttemptRequest {
    pub fn validate(&self) -> crate::RunnerResult<()> {
        if self.task_id != self.task.id || self.scope_id != self.task.scope_id || self.ordinal == 0
        {
            return Err(crate::RunnerError::Invalid(
                "attempt identity differs from its task or has a zero ordinal".into(),
            ));
        }
        self.task
            .resources
            .validate()
            .map_err(|error| crate::RunnerError::Invalid(error.to_string()))?;
        self.task
            .host
            .validate()
            .map_err(|error| crate::RunnerError::Invalid(error.to_string()))?;
        self.resource_assignment
            .validate_for_request(&self.task.resources)
            .map_err(|error| crate::RunnerError::Invalid(error.to_string()))?;
        for lease in &self.resource_assignment.accelerator_leases {
            lease
                .validate_for_attempt(self.id)
                .map_err(|error| crate::RunnerError::Invalid(error.to_string()))?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum AttemptSuccess {
    Values {
        outputs: Vec<ValuePayload>,
        result_objects: Vec<ValueRef>,
    },
    Graph {
        graph: PayloadGraph,
    },
    Spmd {
        outputs: Vec<Option<runmat_execution::SpmdOutputValue>>,
    },
}

impl AttemptSuccess {
    pub fn validate_for_portable_transport(&self) -> crate::RunnerResult<()> {
        use runmat_execution::value::{ValueLimits, ValuePayload, ValueTransportContext};

        match self {
            Self::Values {
                outputs,
                result_objects,
            } => {
                for output in outputs {
                    output
                        .validate_for_transport(
                            ValueLimits::default(),
                            ValueTransportContext::Portable,
                        )
                        .map_err(|error| crate::RunnerError::Invalid(error.to_string()))?;
                }
                for reference in result_objects {
                    ValuePayload::Object(Box::new(reference.clone()))
                        .validate_for_transport(
                            ValueLimits::default(),
                            ValueTransportContext::Portable,
                        )
                        .map_err(|error| crate::RunnerError::Invalid(error.to_string()))?;
                    if reference.kind != runmat_execution::value::ValueRefKind::ResultObject {
                        return Err(crate::RunnerError::Invalid(
                            "result object inventory contains a non-result reference".into(),
                        ));
                    }
                }
            }
            Self::Spmd { outputs } => {
                for output in outputs.iter().flatten() {
                    output
                        .validate_for_transport(ValueTransportContext::Portable)
                        .map_err(|error| crate::RunnerError::Invalid(error.to_string()))?;
                }
            }
            Self::Graph { graph } => {
                let limits = ValueLimits::default();
                if graph.nodes.len() as u64 > limits.max_nodes
                    || graph
                        .roots
                        .iter()
                        .any(|root| root.0 as usize >= graph.nodes.len())
                {
                    return Err(crate::RunnerError::Invalid(
                        "graph result exceeds its node bound or has an invalid root".into(),
                    ));
                }
                for node in &graph.nodes {
                    if let GraphNode::Leaf { value } = node {
                        value
                            .validate_for_transport(limits, ValueTransportContext::Portable)
                            .map_err(|error| crate::RunnerError::Invalid(error.to_string()))?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn values(&self) -> Option<(&[ValuePayload], &[ValueRef])> {
        match self {
            Self::Values {
                outputs,
                result_objects,
            } => Some((outputs, result_objects)),
            Self::Graph { .. } | Self::Spmd { .. } => None,
        }
    }

    pub fn into_values(self) -> Option<(Vec<ValuePayload>, Vec<ValueRef>)> {
        match self {
            Self::Values {
                outputs,
                result_objects,
            } => Some((outputs, result_objects)),
            Self::Graph { .. } | Self::Spmd { .. } => None,
        }
    }

    pub fn spmd_outputs(&self) -> Option<&[Option<runmat_execution::SpmdOutputValue>]> {
        match self {
            Self::Spmd { outputs } => Some(outputs),
            Self::Values { .. } | Self::Graph { .. } => None,
        }
    }

    pub fn graph(&self) -> Option<&PayloadGraph> {
        match self {
            Self::Graph { graph } => Some(graph),
            Self::Values { .. } | Self::Spmd { .. } => None,
        }
    }

    pub fn result_objects(&self) -> &[ValueRef] {
        match self {
            Self::Values { result_objects, .. } => result_objects,
            Self::Graph { .. } | Self::Spmd { .. } => &[],
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptFailureKind {
    Infrastructure,
    Execution,
    Rejected,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum AttemptReport {
    Started,
    Succeeded {
        result: AttemptSuccess,
    },
    Failed {
        kind: AttemptFailureKind,
        message: String,
    },
    RuntimeFailed {
        failure: runmat_execution::ProgramRuntimeFailure,
    },
    Lost {
        message: String,
    },
    /// A worker tree or its foreign resources could not be proven retired.
    /// This is terminal and non-retryable, but not ordinary cancellation.
    RetirementUnverified {
        message: String,
    },
    /// The backend that owns an active Graph worker has verified its process
    /// tree and foreign resources retired after cancellation. Ordinary Lost
    /// or Cancelled reports do not carry that cleanup proof.
    GraphRetiredCancelled,
    Cancelled,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AttemptRecord {
    pub request: AttemptRequest,
    pub state: AttemptState,
    pub assigned_at_millis: u64,
    pub cancellation_requested_at: Option<u64>,
    pub cancellation_escalation: Option<CancellationEscalation>,
}
