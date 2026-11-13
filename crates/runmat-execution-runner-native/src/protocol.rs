use serde::{Deserialize, Serialize};

pub use runmat_execution_artifact::{
    ProgramExecutionResponse as WorkerResponse, ReservedProgramExecutionRequest,
    WorkerProgramRequest as WorkerRequest, PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};

/// The Direct arm serializes as the existing WorkerRequest with no new fields.
/// A Graph child receives only a reference to its parent's owned artifact
/// store; it must verify and deserialize those exact bytes before effects.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub(crate) enum WorkerLaunchRequest {
    ReservedFeaProxy {
        request: Box<ReservedProgramExecutionRequest>,
        console_budget: Option<runmat_execution::console::ConsoleBudget>,
        fea_parent_admission: bool,
    },
    ReservedFea(Box<ReservedFeaLaunch>),
    Direct(Box<WorkerRequest>),
    /// A Direct program with the scheduler's reserved resources, so the worker
    /// admits CPU parallelism within its own reservation.
    Reserved(Box<ReservedProgramExecutionRequest>),
    ReservedConsole {
        request: Box<ReservedProgramExecutionRequest>,
        console_budget: runmat_execution::console::ConsoleBudget,
    },
    StagedGraph(Box<StagedGraphWorkerLaunch>),
}

/// Replaces the existing request allocation; it does not nest another request
/// box. Serde preserves the original request/console_budget/fea_grant object.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReservedFeaLaunch {
    pub(crate) request: ReservedProgramExecutionRequest,
    pub(crate) console_budget: Option<runmat_execution::console::ConsoleBudget>,
    pub(crate) fea_grant: crate::fea_delegation::WorkerFeaGrant,
}

impl ReservedFeaLaunch {
    pub(crate) fn from_reserved(
        request: Box<ReservedProgramExecutionRequest>,
        console_budget: Option<runmat_execution::console::ConsoleBudget>,
        fea_grant: crate::fea_delegation::WorkerFeaGrant,
    ) -> Self {
        // The old Box is deallocated on this function's return, before the
        // caller allocates the replacement envelope from this moved value.
        Self {
            request: *request,
            console_budget,
            fea_grant,
        }
    }

    pub(crate) fn into_parts(
        self: Box<Self>,
    ) -> (
        ReservedProgramExecutionRequest,
        Option<runmat_execution::console::ConsoleBudget>,
        crate::fea_delegation::WorkerFeaGrant,
    ) {
        let Self {
            request,
            console_budget,
            fea_grant,
        } = *self;
        // Likewise free the envelope before a caller restores its request Box.
        (request, console_budget, fea_grant)
    }
}

/// One envelope per process/attempt. The borrowed encoder makes no envelope
/// copy; the receiver unwraps this block before restoring the request box.
pub(crate) const FEA_LAUNCH_LAYOUT_GROWTH: usize = std::mem::size_of::<ReservedFeaLaunch>()
    - std::mem::size_of::<ReservedProgramExecutionRequest>();

impl WorkerLaunchRequest {
    pub(crate) fn fea_proxy(&self) -> bool {
        match self {
            Self::ReservedFeaProxy {
                fea_parent_admission,
                ..
            } => *fea_parent_admission,
            Self::StagedGraph(graph) => graph.staged_graph_v10.fea_parent_admission,
            _ => false,
        }
    }
    pub(crate) fn fea_grant(&self) -> Option<&crate::fea_delegation::WorkerFeaGrant> {
        match self {
            Self::ReservedFea(launch) => Some(&launch.fea_grant),
            Self::StagedGraph(graph) => graph.staged_graph_v10.fea_grant.as_ref(),
            _ => None,
        }
    }
    pub(crate) fn assignment(&self) -> Option<&runmat_execution::ProgramExecutionAssignment> {
        match self {
            Self::Reserved(request)
            | Self::ReservedFeaProxy { request, .. }
            | Self::ReservedConsole { request, .. } => request.request().assignment.as_ref(),
            Self::ReservedFea(launch) => launch.request.request().assignment.as_ref(),
            Self::StagedGraph(graph) => Some(&graph.staged_graph_v10.assignment),
            _ => None,
        }
    }
    pub(crate) fn resources(&self) -> Option<&runmat_execution::resource::ResourceRequest> {
        match self {
            Self::Reserved(request)
            | Self::ReservedFeaProxy { request, .. }
            | Self::ReservedConsole { request, .. } => Some(request.resources()),
            Self::ReservedFea(launch) => Some(launch.request.resources()),
            Self::StagedGraph(graph) => Some(&graph.staged_graph_v10.resources),
            _ => None,
        }
    }
    pub(crate) fn console_budget(&self) -> Option<runmat_execution::console::ConsoleBudget> {
        match self {
            Self::ReservedFeaProxy { console_budget, .. } => *console_budget,
            Self::ReservedFea(launch) => launch.console_budget,
            Self::ReservedConsole { console_budget, .. } => Some(*console_budget),
            Self::StagedGraph(graph) => graph.staged_graph_v10.console_budget,
            Self::Direct(_) | Self::Reserved(_) => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StagedGraphWorkerLaunch {
    pub staged_graph_v10: StagedGraphWorkerReference,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StagedGraphWorkerReference {
    pub fea_parent_admission: bool,
    pub artifact_id: runmat_execution::identity::ArtifactId,
    pub stored_digest: runmat_execution::Digest,
    pub stored_length: u64,
    pub assignment: runmat_execution::ProgramExecutionAssignment,
    pub context: runmat_execution::ProgramInvocationContext,
    pub requested_outputs: u16,
    /// The original scheduler reservation, never inferred from project policy.
    pub resources: runmat_execution::resource::ResourceRequest,
    #[serde(deserialize_with = "required_option")]
    pub console_budget: Option<runmat_execution::console::ConsoleBudget>,
    #[serde(deserialize_with = "required_option")]
    pub fea_grant: Option<crate::fea_delegation::WorkerFeaGrant>,
}

pub const NATIVE_WORKER_MESSAGE_SCHEMA_V2: u16 = 2;
const MAX_PROGRESS_PAYLOAD_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramProgress {
    pub schema_version: u16,
    pub sequence: u64,
    pub media_type: String,
    pub value_schema: String,
    pub payload: Vec<u8>,
}

impl ProgramProgress {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != NATIVE_WORKER_MESSAGE_SCHEMA_V2
            || self.sequence == 0
            || self.media_type.is_empty()
            || self.media_type.len() > 128
            || !self.media_type.is_ascii()
            || self.media_type.chars().any(char::is_whitespace)
            || self.value_schema.is_empty()
            || self.value_schema.len() > 128
            || !self.value_schema.is_ascii()
            || self.value_schema.chars().any(char::is_whitespace)
            || self.payload.is_empty()
            || self.payload.len() > MAX_PROGRESS_PAYLOAD_BYTES
        {
            return Err("native program progress is malformed or exceeds its bound".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "message", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkerProcessMessage {
    FeaAdmissionCall {
        call: crate::fea_proxy::Call,
    },
    Progress {
        progress: ProgramProgress,
    },
    Console {
        record: runmat_execution::console::ConsoleRecord,
    },
    CollectiveRequest {
        request: runmat_execution::CollectiveRequest,
    },
    Completed {
        response: WorkerResponse,
        #[serde(deserialize_with = "required_option")]
        console: Option<runmat_execution::console::ConsoleReceipt>,
        #[serde(deserialize_with = "runmat_execution::console::deserialize_scopes")]
        console_scopes: Vec<runmat_execution::console::ScopedConsoleTranscript>,
    },
    /// Worker-side admission/transport failure before a Program response
    /// exists. It cannot be mistaken for a user-code execution failure.
    InfrastructureFailure {
        #[serde(rename = "detail")]
        message: String,
    },
    /// The selected foreign host could not retire gracefully. The child
    /// retains its host and private roots until the parent kills and reaps
    /// this one-shot worker; this is never a user-code Graph failure.
    RetirementUnverified {
        #[serde(rename = "detail")]
        message: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum CollectiveProcessResult {
    Completed {
        response: runmat_execution::CollectiveResponse,
    },
    Failed {
        message: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "message", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkerDriverMessage {
    FeaAdmissionReply {
        sequence: u64,
        reply: crate::fea_proxy::Reply,
    },
    FeaLeaseUpdate {
        run_id: String,
        attempt_id: runmat_execution::identity::AttemptId,
        update: crate::fea_delegation::FeaLeaseUpdate,
    },
    CollectiveCompletion {
        context: runmat_execution::SpmdTaskContext,
        id: runmat_types::CollectiveId,
        sequence: runmat_execution::CollectiveSequence,
        result: CollectiveProcessResult,
    },
    /// Graph-only cooperative cancellation for the exact running attempt.
    /// Direct keeps its previous immediate process-retirement behavior.
    CancelAttempt {
        attempt_id: runmat_execution::identity::AttemptId,
        reason: runmat_execution::CancellationReason,
    },
}

/// The Direct variant keeps the existing stored shape for admitted tasks.
/// A Graph task owns its one payload in the artifact store; scheduler inputs
/// remain empty and must never become a second numerical payload authority.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StoredProgram {
    Direct(Box<StoredDirectProgram>),
    Graph(Box<StoredGraphProgram>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredDirectProgram {
    pub recipe: runmat_execution_artifact::ProgramBuildRecipe,
    pub artifact: runmat_execution_artifact::ProgramArtifact,
}

fn required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    serde::Deserialize::deserialize(deserializer)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredGraphProgram {
    pub request: runmat_execution_artifact::GraphExecutionRequest,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reserved_request(
        resources: runmat_execution::resource::ResourceRequest,
    ) -> ReservedProgramExecutionRequest {
        use runmat_execution::{
            Digest, OutputContract, ProgramCallable, ProgramFunctionId, ProgramInvocationContext,
            ProgramRevision,
        };
        use runmat_execution_artifact::{
            ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionRequest,
            ProgramTarget,
        };
        let revision = ProgramRevision::new(
            Digest::sha256(b"layout-graph"),
            Digest::sha256(b"layout-source"),
            runmat_core::program_environment(runmat_core::CompatMode::Matlab),
        )
        .unwrap();
        let recipe = ProgramBuildRecipe {
            schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: revision,
            entrypoint: "0".into(),
            outputs: OutputContract {
                requested_outputs: 1,
                owned_work: None,
            },
            execution_mode: "interpreter".into(),
            target: ProgramTarget::portable("launch-layout-test"),
            interop: runmat_types::InteropManifest::empty(),
            accelerators: Vec::new(),
            features: Default::default(),
            compile_options: Default::default(),
            source_objects: Vec::new(),
            foreign_policy: None,
            expected_artifact_id: None,
        };
        let artifact = ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::InterpreterBytecodeV2,
            b"not-executed".to_vec(),
        )
        .unwrap();
        ReservedProgramExecutionRequest::new(
            ProgramExecutionRequest {
                schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
                recipe,
                artifact,
                callable: ProgramCallable::semantic(ProgramFunctionId(0), None),
                context: ProgramInvocationContext::Direct,
                assignment: None,
                job_id: None,
                arguments: Vec::new(),
                requested_outputs: 1,
                environment: Default::default(),
                console: None,
            },
            resources,
        )
        .unwrap()
    }

    #[test]
    fn reserved_fea_envelope_keeps_original_wire_and_inline_request_storage() {
        let grant = crate::fea_delegation::test_grant();
        let resources = grant.resources.clone();
        let request = reserved_request(resources.clone());
        let old =
            serde_json::json!({"request": &request, "console_budget": null, "fea_grant": &grant});
        let launch = WorkerLaunchRequest::ReservedFea(Box::new(ReservedFeaLaunch::from_reserved(
            Box::new(request),
            None,
            grant,
        )));
        assert_eq!(serde_json::to_value(&launch).unwrap(), old);
        let decoded: WorkerLaunchRequest = serde_json::from_value(old).unwrap();
        let WorkerLaunchRequest::ReservedFea(launch) = decoded else {
            panic!("wrong wire variant")
        };
        let (request, console, grant) = launch.into_parts();
        assert_eq!(request.resources(), &resources);
        assert_eq!(grant.resources, resources);
        assert!(console.is_none());
        assert!(std::mem::size_of::<WorkerLaunchRequest>() <= 128);
        assert_eq!(
            std::mem::size_of::<ReservedFeaLaunch>(),
            std::mem::size_of::<ReservedProgramExecutionRequest>() + FEA_LAUNCH_LAYOUT_GROWTH
        );
    }

    #[test]
    fn progress_envelope_is_bounded_and_strictly_shaped() {
        let progress = ProgramProgress {
            schema_version: NATIVE_WORKER_MESSAGE_SCHEMA_V2,
            sequence: 1,
            media_type: "application/vnd.runmat.progress+cbor".into(),
            value_schema: "runmat.progress.v1".into(),
            payload: vec![1],
        };
        progress.validate().unwrap();

        let mut empty = progress.clone();
        empty.payload.clear();
        assert!(empty.validate().is_err());
        let mut oversized = progress;
        oversized.payload = vec![0; MAX_PROGRESS_PAYLOAD_BYTES + 1];
        assert!(oversized.validate().is_err());

        let unknown = br#"{"message":"progress","progress":{"schema_version":1,"sequence":1,"media_type":"application/x.test","value_schema":"test.v1","payload":[1],"extra":true}}"#;
        assert!(serde_json::from_slice::<WorkerProcessMessage>(unknown).is_err());
    }
}
