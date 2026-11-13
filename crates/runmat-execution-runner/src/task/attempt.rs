use std::collections::BTreeSet;

use runmat_execution::identity::{AttemptId, WorkerId};
use runmat_execution::resource::{ResourceAccounting, ResourceAccountingSummary};
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
            .validate_executable()
            .map_err(|error| crate::RunnerError::Invalid(error.to_string()))?;
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
        /// Present on scheduler attempts. Serial domain executors can prepare
        /// results without scheduler accounting; the generic driver requires
        /// worker-bound accounting before accepting an attempt report.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        resources: Option<Box<AttemptResourceAccounting>>,
    },
    Graph {
        graph: PayloadGraph,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        result_objects: Vec<ValueRef>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        resources: Option<Box<AttemptResourceAccounting>>,
    },
    Spmd {
        outputs: Vec<Option<runmat_execution::SpmdOutputValue>>,
    },
}

impl AttemptSuccess {
    /// Bind worker-metered accounting to a value result for one scheduler attempt.
    pub fn new(
        request: &AttemptRequest,
        outputs: Vec<ValuePayload>,
        result_objects: Vec<ValueRef>,
        wall_millis: u64,
        egress_bytes: u64,
        relay_bytes: u64,
    ) -> Result<Self, String> {
        let artifact_bytes = result_artifact_bytes(&result_objects)?;
        Ok(Self::Values {
            outputs,
            result_objects,
            resources: Some(Box::new(AttemptResourceAccounting::for_attempt(
                request,
                wall_millis,
                artifact_bytes,
                egress_bytes,
                relay_bytes,
            )?)),
        })
    }

    /// Bind graph result-object inventory and worker-measured interval to the
    /// original assignment. Repeated graph references retain one exact object.
    pub fn new_graph(
        request: &AttemptRequest,
        graph: PayloadGraph,
        wall_millis: u64,
        egress_bytes: u64,
        relay_bytes: u64,
    ) -> Result<Self, String> {
        Self::graph_accounted(
            request,
            graph,
            wall_millis,
            GraphEgress::Measured(egress_bytes),
            relay_bytes,
        )
    }

    /// Remote delivery downloads the derived immutable result inventory;
    /// inline Graph wire bytes are not object-transfer egress.
    pub fn new_graph_for_remote_transfer(
        request: &AttemptRequest,
        graph: PayloadGraph,
        wall_millis: u64,
    ) -> Result<Self, String> {
        Self::graph_accounted(request, graph, wall_millis, GraphEgress::ResultObjects, 0)
    }

    fn graph_accounted(
        request: &AttemptRequest,
        graph: PayloadGraph,
        wall_millis: u64,
        egress: GraphEgress,
        relay_bytes: u64,
    ) -> Result<Self, String> {
        let result_objects = graph_result_objects(&graph)?;
        let artifact_bytes = result_artifact_bytes(&result_objects)?;
        let egress_bytes = match egress {
            GraphEgress::Measured(bytes) => bytes,
            GraphEgress::ResultObjects => artifact_bytes,
        };
        Ok(Self::Graph {
            graph,
            result_objects,
            resources: Some(Box::new(AttemptResourceAccounting::for_attempt(
                request,
                wall_millis,
                artifact_bytes,
                egress_bytes,
                relay_bytes,
            )?)),
        })
    }

    /// Worker-bound accounting for value and graph scheduler results.
    pub fn resources(&self) -> Option<&AttemptResourceAccounting> {
        match self {
            Self::Values { resources, .. } | Self::Graph { resources, .. } => resources.as_deref(),
            Self::Spmd { .. } => None,
        }
    }

    pub fn validate_against(&self, request: &AttemptRequest) -> Result<(), String> {
        if matches!(self, Self::Spmd { .. }) {
            return Ok(());
        }
        self.validate_accounting()?;
        self.resources()
            .ok_or_else(|| "attempt success has no worker resource accounting".to_string())?
            .validate_against(request)
    }

    pub fn validate_accounting(&self) -> Result<(), String> {
        if let Self::Graph {
            graph,
            result_objects,
            ..
        } = self
        {
            if graph_result_objects(graph)? != *result_objects {
                return Err("graph result object inventory differs from its payload".into());
            }
        }
        let resources = self
            .resources()
            .ok_or_else(|| "attempt success has no worker resource accounting".to_string())?;
        let measured = &resources.resources;
        if measured.usage.wall_millis > measured.allocation.max_wall_millis
            || measured.artifact_bytes > measured.allocation.max_artifact_bytes
            || measured.usage.egress_bytes > measured.allocation.max_egress_bytes
            || measured.usage.relay_bytes > measured.allocation.max_relay_bytes
        {
            return Err("successful result consumption exceeds its original grant".into());
        }
        if result_artifact_bytes(self.result_objects())? != resources.resources.artifact_bytes {
            return Err("attempt resource accounting differs from its result objects".into());
        }
        resources.validate()
    }

    pub fn validate_for_portable_transport(&self, task: &TaskRequest) -> crate::RunnerResult<()> {
        use runmat_execution::value::{ValueLimits, ValuePayload, ValueTransportContext};

        let owned = matches!(
            &task.executable,
            runmat_execution::TaskExecutableIdentity::OwnedBuiltin(_)
        );
        let limits = if owned {
            runmat_execution::OWNED_BUILTIN_VALUE_LIMITS
        } else {
            ValueLimits::default()
        };

        match self {
            Self::Values {
                outputs,
                result_objects,
                ..
            } => {
                if owned
                    && (outputs.len() != usize::from(task.outputs.requested_outputs)
                        || !result_objects.is_empty()
                        || outputs
                            .iter()
                            .any(|value| !matches!(value, ValuePayload::Inline(_))))
                {
                    return Err(crate::RunnerError::Invalid(
                        "owned builtin result differs from its self-contained output contract"
                            .into(),
                    ));
                }
                if owned {
                    let contract = task.outputs.owned_work.as_ref().ok_or_else(|| {
                        crate::RunnerError::Invalid(
                            "owned builtin task lost its dense output contract".into(),
                        )
                    })?;
                    contract
                        .validate(task.outputs.requested_outputs)
                        .map_err(|error| crate::RunnerError::Invalid(error.to_string()))?;
                    ValuePayload::validate_all(
                        outputs,
                        runmat_execution::OWNED_BUILTIN_VALUE_LIMITS,
                    )
                    .map_err(|error| crate::RunnerError::Invalid(error.to_string()))?;
                    if !outputs
                        .iter()
                        .zip(&contract.dense)
                        .all(|(output, contract)| contract.accepts(output))
                    {
                        return Err(crate::RunnerError::Invalid(
                            "owned builtin result differs from its dense output contract".into(),
                        ));
                    }
                    contract
                        .validate_json_wire(self)
                        .map_err(|error| crate::RunnerError::Invalid(error.to_string()))?;
                }
                for output in outputs {
                    output
                        .validate_for_transport(limits, ValueTransportContext::Portable)
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
                if owned {
                    return Err(crate::RunnerError::Invalid(
                        "owned builtin cannot return an SPMD result".into(),
                    ));
                }
                for output in outputs.iter().flatten() {
                    output
                        .validate_for_transport(ValueTransportContext::Portable)
                        .map_err(|error| crate::RunnerError::Invalid(error.to_string()))?;
                }
            }
            Self::Graph { graph, .. } => {
                if owned {
                    return Err(crate::RunnerError::Invalid(
                        "owned builtin cannot return a graph result".into(),
                    ));
                }
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
                ..
            } => Some((outputs, result_objects)),
            Self::Graph { .. } | Self::Spmd { .. } => None,
        }
    }

    pub fn into_values(self) -> Option<(Vec<ValuePayload>, Vec<ValueRef>)> {
        match self {
            Self::Values {
                outputs,
                result_objects,
                ..
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
            Self::Graph { graph, .. } => Some(graph),
            Self::Values { .. } | Self::Spmd { .. } => None,
        }
    }

    /// Append a result object to a value result (e.g. stage evidence published
    /// beside the root). Graph inventories derive only from authenticated
    /// payload references and cannot acquire unrelated side objects.
    pub fn push_result_object(&mut self, object: ValueRef) -> Result<(), String> {
        match self {
            Self::Values { result_objects, .. } => {
                result_objects.push(object);
                Ok(())
            }
            Self::Graph { .. } | Self::Spmd { .. } => {
                Err("only value results carry result objects".into())
            }
        }
    }

    pub fn result_objects(&self) -> &[ValueRef] {
        match self {
            Self::Values { result_objects, .. } | Self::Graph { result_objects, .. } => {
                result_objects
            }
            Self::Spmd { .. } => &[],
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AttemptResourceAccounting {
    pub attempt_id: AttemptId,
    pub worker_id: WorkerId,
    pub resources: ResourceAccounting,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttemptResourceAggregate {
    pub attempts: Vec<AttemptResourceAccounting>,
    pub summary: ResourceAccountingSummary,
}

impl AttemptResourceAggregate {
    pub fn new(
        mut attempts: Vec<AttemptResourceAccounting>,
        graph_wall_millis: u64,
    ) -> Result<Self, String> {
        if attempts.is_empty() {
            return Err("resource aggregate requires at least one attempt".into());
        }
        for attempt in &attempts {
            attempt.validate()?;
        }
        attempts.sort_by_key(|attempt| attempt.attempt_id);
        if attempts
            .iter()
            .map(|attempt| attempt.attempt_id)
            .collect::<BTreeSet<_>>()
            .len()
            != attempts.len()
        {
            return Err("resource aggregate contains duplicate attempts".into());
        }
        let intervals = attempts
            .iter()
            .map(|attempt| attempt.resources.clone())
            .collect::<Vec<_>>();
        let summary = ResourceAccountingSummary::aggregate(&intervals, graph_wall_millis)
            .map_err(|error| error.to_string())?;
        Ok(Self { attempts, summary })
    }

    pub fn validate(&self) -> Result<(), String> {
        let expected = Self::new(self.attempts.clone(), self.summary.usage.wall_millis)?;
        if self != &expected {
            return Err("attempt resource aggregate is internally inconsistent".into());
        }
        Ok(())
    }
}

impl AttemptResourceAccounting {
    pub fn new(
        attempt_id: AttemptId,
        worker_id: WorkerId,
        resources: ResourceAccounting,
    ) -> Result<Self, String> {
        resources.validate().map_err(|error| error.to_string())?;
        Ok(Self {
            attempt_id,
            worker_id,
            resources,
        })
    }

    pub fn for_attempt(
        request: &AttemptRequest,
        wall_millis: u64,
        artifact_bytes: u64,
        egress_bytes: u64,
        relay_bytes: u64,
    ) -> Result<Self, String> {
        Self::new(
            request.id,
            request.worker_id,
            ResourceAccounting::for_assigned_interval(
                request.task.resources.clone(),
                &request.resource_assignment,
                wall_millis,
                artifact_bytes,
                egress_bytes,
                relay_bytes,
            )
            .map_err(|error| error.to_string())?,
        )
    }

    pub fn validate(&self) -> Result<(), String> {
        self.resources.validate().map_err(|error| error.to_string())
    }

    pub fn validate_against(&self, request: &AttemptRequest) -> Result<(), String> {
        if self.attempt_id != request.id
            || self.worker_id != request.worker_id
            || self.resources.allocation != request.task.resources
        {
            return Err(
                "attempt resource accounting differs from its allocation or identity".into(),
            );
        }
        self.validate()
    }
}

enum GraphEgress {
    Measured(u64),
    ResultObjects,
}

fn result_artifact_bytes(result_objects: &[ValueRef]) -> Result<u64, String> {
    result_objects.iter().try_fold(0_u64, |total, reference| {
        total
            .checked_add(reference.encoded_length)
            .ok_or_else(|| "attempt result artifact accounting overflowed".to_string())
    })
}

fn graph_result_objects(graph: &PayloadGraph) -> Result<Vec<ValueRef>, String> {
    let mut objects = std::collections::BTreeMap::new();
    for node in &graph.nodes {
        if let GraphNode::Leaf {
            value: ValuePayload::Object(reference),
        } = node
        {
            if reference.kind == runmat_execution::value::ValueRefKind::ResultObject {
                reference
                    .validate(runmat_execution::value::ValueLimits::default())
                    .map_err(|error| error.to_string())?;
                if let Some(previous) = objects.insert(reference.id, reference.as_ref().clone()) {
                    if previous != **reference {
                        return Err(
                            "graph repeats an object identity with conflicting metadata".into()
                        );
                    }
                }
            }
        }
    }
    Ok(objects.into_values().collect())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptFailureKind {
    Infrastructure,
    Execution,
    Rejected,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
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
    /// The logical task becomes indeterminate; its attempt and reservation
    /// remain owned until a later exact retirement certificate.
    RetirementUnverified {
        message: String,
    },
    /// The backend owning the original attempt certifies all its work and
    /// resources retired. Accepted only after cancellation or recorded
    /// uncertainty; this cannot create a successful operation result.
    Retired,
    /// Backend-authorized proof that the original assignment never began
    /// execution. The scheduler retains its ordinal but no consumption claim.
    CertifiedNotStarted {},
    Cancelled,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AttemptRecord {
    pub request: AttemptRequest,
    pub state: AttemptState,
    pub assigned_at_millis: u64,
    /// Driver closure and original worker evidence; absence means still active.
    pub terminal: Option<super::AttemptTerminal>,
    /// Original uncertainty survives logical failure and later physical proof.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unverified_retirement: Option<Box<super::AttemptRetirementUnverified>>,
    pub cancellation_requested_at: Option<u64>,
    pub cancellation_escalation: Option<CancellationEscalation>,
}

impl AttemptRecord {
    /// Original provider evidence that the assignment never started. This does not erase
    /// preflight resource charges, and logical rejection alone is not proof.
    pub fn certifies_not_started(&self) -> bool {
        let Some(terminal) = &self.terminal else {
            return false;
        };
        match terminal.disposition {
            super::AttemptDisposition::Succeeded => false,
            super::AttemptDisposition::CertifiedNotStarted => {
                terminal.validate_against(self).is_ok()
            }
            _ => {
                self.unverified_retirement.is_none()
                    && terminal.execution.as_ref().is_some_and(|observation| {
                        observation.validate_against(&self.request).is_ok()
                            && observation.termination
                                == super::AttemptExecutionTermination::NotStarted
                    })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::resource::{ResourceAccounting, ResourceRequest};

    fn accounting(name: &[u8], wall_millis: u64, artifact_bytes: u64) -> AttemptResourceAccounting {
        AttemptResourceAccounting::new(
            AttemptId::derive(&[name, b"attempt"]),
            WorkerId::derive(&[name, b"worker"]),
            ResourceAccounting::for_reserved_interval(
                ResourceRequest {
                    cpu_millicores: 1_000,
                    memory_bytes: 2_048,
                    scratch_bytes: 1_024,
                    max_wall_millis: 1_000,
                    max_artifact_bytes: 4_096,
                    max_egress_bytes: 100,
                    max_relay_bytes: 100,
                    accelerators: Vec::new(),
                    required_capabilities: BTreeSet::new(),
                },
                wall_millis,
                artifact_bytes,
                0,
                0,
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn attempt_aggregate_sums_work_and_preserves_graph_wall_time() {
        let first = accounting(b"first", 10, 20);
        let second = accounting(b"second", 15, 30);
        let aggregate =
            AttemptResourceAggregate::new(vec![second.clone(), first.clone()], 18).unwrap();
        assert_eq!(aggregate.summary.usage.cpu_core_millis, 25);
        assert_eq!(aggregate.summary.usage.memory_byte_millis, 51_200);
        assert_eq!(aggregate.summary.usage.wall_millis, 18);
        assert_eq!(aggregate.summary.artifact_bytes, 50);
        let mut expected_attempts = vec![first, second];
        expected_attempts.sort_by_key(|attempt| attempt.attempt_id);
        assert_eq!(aggregate.attempts, expected_attempts);
        aggregate.validate().unwrap();
    }

    #[test]
    fn graph_accounting_is_worker_bound_and_preserves_the_original_grant() {
        let graph = PayloadGraph {
            nodes: Vec::new(),
            roots: Vec::new(),
        };
        let mut result = AttemptSuccess::Graph {
            graph,
            result_objects: Vec::new(),
            resources: None,
        };
        assert!(result.validate_accounting().is_err());
        let original = accounting(b"graph", 15, 0);
        if let AttemptSuccess::Graph { resources, .. } = &mut result {
            *resources = Some(Box::new(original.clone()));
        }
        result.validate_accounting().unwrap();
        assert_eq!(result.resources(), Some(&original));
        assert_eq!(result.resources().unwrap().resources.usage.wall_millis, 15);
        let digest = runmat_execution::Digest::sha256(b"graph-result");
        let encryption = runmat_execution::Digest::sha256(b"graph-scope");
        let reference = ValueRef {
            schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
            id: runmat_execution::value::artifact_object_value_id(
                digest,
                encryption,
                "graph-scope",
            ),
            logical_digest: digest,
            encoded_length: 12,
            media_type: "application/vnd.runmat.test".into(),
            value_schema: "runmat.test".into(),
            encryption_context: encryption,
            kind: runmat_execution::value::ValueRefKind::ResultObject,
            authorization_scope: "graph-scope".into(),
            resident_fence: None,
        };
        let mut graph = PayloadGraph {
            roots: Vec::new(),
            nodes: vec![
                GraphNode::Leaf {
                    value: ValuePayload::Object(Box::new(reference.clone())),
                },
                GraphNode::Leaf {
                    value: ValuePayload::Object(Box::new(reference.clone())),
                },
            ],
        };
        assert_eq!(graph_result_objects(&graph).unwrap(), vec![reference]);
        if let GraphNode::Leaf {
            value: ValuePayload::Object(object),
        } = &mut graph.nodes[1]
        {
            object.encoded_length += 1;
        }
        assert!(graph_result_objects(&graph).is_err());
        if let AttemptSuccess::Graph {
            resources: Some(resources),
            ..
        } = &mut result
        {
            resources.resources.usage.egress_bytes =
                resources.resources.allocation.max_egress_bytes + 1;
        }
        assert!(result.validate_accounting().is_err());
    }

    #[test]
    fn attempt_aggregate_rejects_duplicates_short_wall_time_and_substitution() {
        let attempt = accounting(b"only", 10, 20);
        assert!(AttemptResourceAggregate::new(vec![attempt.clone(), attempt.clone()], 10).is_err());
        assert!(AttemptResourceAggregate::new(vec![attempt.clone()], 9).is_err());
        let mut aggregate = AttemptResourceAggregate::new(vec![attempt], 10).unwrap();
        aggregate.summary.artifact_bytes += 1;
        assert!(aggregate.validate().is_err());
    }
}
