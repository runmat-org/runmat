//! Source capture uses the existing runtime host and private control writer.
use super::*;
use crate::protocol::{ReservedProgramExecutionRequest, WorkerProcessMessage};
use runmat_execution::console::ConsoleRecord;
use runmat_process_host::ipc::private_control::ControlFrameWriter;
use runmat_runtime::context::PortableConsoleSink;
use std::rc::Rc;

/// Console records share the ordered private writer with collective and terminal
/// frames, and are published ahead of the terminal receipt.
struct ControlSink {
    writer: ControlFrameWriter,
    limits: FrameLimits,
}
impl PortableConsoleSink for ControlSink {
    fn emit(&self, record: &ConsoleRecord) -> Result<(), String> {
        let message = WorkerProcessMessage::Console {
            record: record.clone(),
        };
        let bytes = runmat_canonical_codec::encode_json_bounded(
            &message,
            u64::from(self.limits.max_message_bytes),
        )
        .map_err(|error| error.to_string())?;
        self.writer
            .write_blocking(bytes)
            .map_err(|error| error.to_string())
    }
}

pub(super) fn graph_host(
    controls: Option<runmat_execution::console::ConsoleLimits>,
    budget: Option<runmat_execution::console::ConsoleBudget>,
    assignment: &runmat_execution::ProgramExecutionAssignment,
    writer: ControlFrameWriter,
) -> NativeExecutionResult<Option<Rc<runmat_runtime::context::PortableConsoleHost>>> {
    if controls.is_some() != budget.is_some() {
        return Err(NativeExecutionError::Protocol(
            "Graph console capture differs from its delegated subtree grant".into(),
        ));
    }
    let limits = FrameLimits {
        max_message_bytes: WORKER_MAX_MESSAGE_BYTES,
    };
    controls
        .map(|controls| {
            controls
                .validate_transport(u64::from(limits.max_message_bytes))
                .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
            let sink: Rc<dyn PortableConsoleSink> = Rc::new(ControlSink { writer, limits });
            let host = Rc::new(
                runmat_runtime::context::PortableConsoleHost::with_sink(controls, Some(sink))
                    .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?,
            );
            let owner = crate::console_retention::ConsoleRetentionOwner::from_subgrant(
                budget.expect("matched grant"),
            )?;
            host.install_console_admission(owner, identity(assignment))
                .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
            Ok(host)
        })
        .transpose()
}

/// Execute a reserved Direct program, streaming its requested console records.
pub(super) async fn execute_reserved(
    request: ReservedProgramExecutionRequest,
    budget: Option<runmat_execution::console::ConsoleBudget>,
    writer: ControlFrameWriter,
    collective: Option<Rc<dyn runmat_runtime::context::RuntimeCollectiveService>>,
) -> NativeExecutionResult<(
    WorkerResponse,
    Option<crate::console_bridge::NativeCapturedConsole>,
)> {
    request
        .validate()
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    let controls = request.request().console;
    if controls.is_some() != budget.is_some() {
        return Err(NativeExecutionError::Protocol(
            "worker console capture differs from its delegated subtree grant".into(),
        ));
    }
    let limits = FrameLimits {
        max_message_bytes: WORKER_MAX_MESSAGE_BYTES,
    };
    if let Some(controls) = controls {
        controls
            .validate_transport(u64::from(limits.max_message_bytes))
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    }
    let (program, resources) = request.into_parts();
    let console_owner = budget
        .map(|budget| {
            let assignment = program.assignment.as_ref().ok_or_else(|| {
                NativeExecutionError::Protocol("captured worker has no original assignment".into())
            })?;
            Ok::<_, NativeExecutionError>(crate::test_workload::HostConsoleOwner {
                owner: crate::console_retention::ConsoleRetentionOwner::from_subgrant(budget)?,
                identity: identity(assignment),
                budget,
            })
        })
        .transpose()?;
    let sink: Rc<dyn PortableConsoleSink> = Rc::new(ControlSink { writer, limits });
    Ok(
        crate::test_workload::execute_host_program_request_with_console(
            program,
            crate::test_workload::HostProgramContext {
                resources: Some(&resources),
                collective,
                console_owner,
                console_sink: controls.map(|_| sink),
                ..Default::default()
            },
        )
        .await,
    )
}

pub(super) fn identity(
    assignment: &runmat_execution::ProgramExecutionAssignment,
) -> runmat_runtime::context::RuntimeConsoleScopeIdentity {
    runmat_runtime::context::RuntimeConsoleScopeIdentity {
        scope_id: assignment.scope_id,
        pool_id: assignment.pool_id,
        task_id: assignment.task_id,
        attempt_id: assignment.attempt_id,
    }
}
