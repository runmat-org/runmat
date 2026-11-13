//! One authenticated, framed launch boundary for owned native worker roles.
use super::collective::WorkerControlChannel;
use crate::protocol::{WorkerLaunchRequest, WorkerProcessMessage};
use crate::{NativeExecutionError, NativeExecutionResult};
use runmat_process_host::ipc::{FrameLimits, WorkerParentLink};
use runmat_runtime::analysis::fea_run_admission::{FeaRunAdmissionContext, FeaRunScope};
use std::{cell::RefCell, sync::Arc};

#[derive(Clone, Copy)]
pub(crate) enum WorkerRole {
    Program,
    Meshing,
}

impl WorkerRole {
    fn trailing_limits(self, initial: FrameLimits) -> FrameLimits {
        match self {
            Self::Program => initial,
            Self::Meshing => {
                const {
                    assert!(
                        crate::fea_proxy::MAX_FRAME_BYTES == crate::fea_delegation::MAX_GRANT_BYTES
                    );
                }
                FrameLimits {
                    max_message_bytes: crate::fea_delegation::MAX_GRANT_BYTES as u32,
                }
            }
        }
    }
}

// Meshing trailing backing uses three MAX_GRANT_BYTES components: one frame
// being read, one queued ordinary frame, and the current decoded control. The
// fourth component in fea_delegation::STORAGE_BYTES retains the small bound
// observer/scope metadata. Paid binding precedes reader startup, so a pending
// decoded renewal cannot overlap grant installation. Proxy uses its existing
// four-frame component plus 4096 bytes of fixed mailbox metadata. The initial
// request wire buffer has already retired; its original cap is not duplicated.
// The reader's fixed stack is separately included by fea_proxy::storage_bytes.

pub(crate) struct PrivateWorkerBootstrap {
    parent: WorkerParentLink,
    pub(super) channel: WorkerControlChannel,
    limits: FrameLimits,
    initial_launch: RefCell<Option<NativeExecutionResult<AdmittedWorkerLaunch>>>,
}

pub(crate) struct AdmittedWorkerLaunch {
    pub request: WorkerLaunchRequest,
    pub scope: Option<FeaRunScope>,
    pub host: Option<FeaRunAdmissionContext>,
}

impl PrivateWorkerBootstrap {
    pub(crate) fn connect(limits: FrameLimits, role: WorkerRole) -> NativeExecutionResult<Self> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| protocol(error.to_string()))?;
        let started_unix_millis =
            u64::try_from(now.as_millis()).map_err(|error| protocol(error.to_string()))?;
        let control = runmat_process_host::ipc::private_control::connect_host_blocking()
            .map_err(runmat_process_host::ProcessHostError::Io)?;
        // No numerical work is active on this one-request process. Read on its
        // existing startup thread without another blocking-thread stack.
        let mut reader = control.reader;
        let payload = runmat_process_host::ipc::read_payload_blocking(&mut reader, limits)?;
        let initial_launch = decode_launch(&payload, role);
        drop(payload);
        // Retire the initial wire payload before any trailing-frame allocation
        // or nested scope/provider. Meshing has no collective-result role.
        let lease_observer = Arc::new(crate::fea_delegation::LeaseObserver::default());
        let proxy_replies = Arc::new(crate::fea_proxy::Replies::default());
        let writer = runmat_process_host::ipc::private_control::ControlFrameWriter::new(
            control.writer,
            limits,
        )?;
        let initial_launch = initial_launch.and_then(|request| {
            admit_launch(
                request,
                &writer,
                &lease_observer,
                &proxy_replies,
                started_unix_millis,
            )
        });
        // Bind a paid scope before the first trailing frame can arrive. No
        // pending decoded lease update overlaps grant installation in this role.
        let replies = proxy_replies.clone();
        let observer = lease_observer.clone();
        let (parent, frames) = WorkerParentLink::start_bounded_with_observer(
            reader,
            role.trailing_limits(limits),
            Some(Arc::new(move |frame| {
                replies.observe(frame) || observer.observe(frame)
            })),
        )?;
        proxy_replies.activate_reader();
        Ok(Self {
            parent,
            channel: WorkerControlChannel::new(frames, writer),
            limits,
            initial_launch: RefCell::new(Some(initial_launch)),
        })
    }

    pub(crate) fn writer(&self) -> &runmat_process_host::ipc::private_control::ControlFrameWriter {
        self.channel.writer()
    }

    pub(crate) async fn read_control(&self) -> NativeExecutionResult<Vec<u8>> {
        Ok(self.channel.read_payload().await?)
    }

    pub(crate) fn take_launch(&self) -> NativeExecutionResult<AdmittedWorkerLaunch> {
        self.initial_launch
            .borrow_mut()
            .take()
            .ok_or_else(|| protocol("private worker launch was consumed twice"))?
    }

    pub(crate) fn failure_payload(
        &self,
        error: impl std::fmt::Display,
    ) -> NativeExecutionResult<Vec<u8>> {
        runmat_canonical_codec::encode_json_bounded(
            &WorkerProcessMessage::InfrastructureFailure {
                message: error.to_string(),
            },
            u64::from(self.limits.max_message_bytes),
        )
        .map_err(Into::into)
    }

    pub(crate) fn disarm(&self) {
        self.parent.disarm();
    }

    pub(crate) async fn publish(&self, payload: Vec<u8>) -> NativeExecutionResult<()> {
        // A parent may close the request pipe immediately after this frame.
        self.disarm();
        self.channel.write_payload(payload).await?;
        Ok(())
    }
}

fn admit_launch(
    request: WorkerLaunchRequest,
    writer: &runmat_process_host::ipc::private_control::ControlFrameWriter,
    lease_observer: &crate::fea_delegation::LeaseObserver,
    proxy_replies: &Arc<crate::fea_proxy::Replies>,
    started_unix_millis: u64,
) -> NativeExecutionResult<AdmittedWorkerLaunch> {
    let proxy_host = if request.fea_proxy() {
        if request.fea_grant().is_some() {
            return Err(protocol(
                "worker received both paid FEA delegation and first-root proxy",
            ));
        }
        let assignment = request
            .assignment()
            .ok_or_else(|| protocol("FEA proxy lacks original worker assignment"))?
            .clone();
        let resources = request
            .resources()
            .ok_or_else(|| protocol("FEA proxy lacks original worker reservation"))?;
        if resources.memory_bytes
            < crate::fea_proxy::storage_bytes(0, crate::fea_proxy::StorageRole::Proxy)
                .map_err(protocol)?
        {
            return Err(protocol("FEA proxy metadata exceeds worker reservation"));
        }
        let client = crate::fea_proxy::Client::new(
            assignment,
            u64::from(resources.cpu_millicores),
            writer.clone(),
            proxy_replies.clone(),
        );
        Some(FeaRunAdmissionContext::new(Arc::new(
            crate::fea_proxy::WorkerAdmission(client),
        )))
    } else {
        None
    };
    let (request, grant) = match request {
        WorkerLaunchRequest::ReservedFeaProxy {
            request,
            console_budget,
            fea_parent_admission: true,
        } => (reserved(request, console_budget), None),
        WorkerLaunchRequest::ReservedFea(launch) => {
            let (request, budget, grant) = launch.into_parts();
            (reserved(Box::new(request), budget), Some(grant))
        }
        WorkerLaunchRequest::StagedGraph(mut graph) => {
            let grant = graph.staged_graph_v10.fea_grant.take();
            (WorkerLaunchRequest::StagedGraph(graph), grant)
        }
        other => (other, None),
    };
    let scope = grant
        .as_ref()
        .map(|grant| {
            let assignment = request
                .assignment()
                .ok_or_else(|| protocol("FEA grant lacks an assigned child"))?;
            let resources = request
                .resources()
                .ok_or_else(|| protocol("FEA grant lacks reserved child resources"))?;
            let client = crate::fea_proxy::Client::new(
                assignment.clone(),
                u64::from(resources.cpu_millicores),
                writer.clone(),
                Arc::clone(proxy_replies),
            );
            let scope = grant
                .install_with_parent(assignment, resources, &client)
                .map_err(protocol)?;
            lease_observer.install(grant, &scope).map_err(protocol)?;
            Ok::<_, NativeExecutionError>(scope)
        })
        .transpose()?;
    let host = if scope.is_none() {
        request
            .resources()
            .map(|resources| {
                let deadline = started_unix_millis
                    .checked_add(resources.max_wall_millis)
                    .ok_or_else(|| protocol("worker FEA deadline overflow"))?;
                let make = || {
                    FeaRunAdmissionContext::with_current_resource_bounds(
                        u64::from(resources.cpu_millicores),
                        deadline,
                    )
                };
                match &proxy_host {
                    Some(host) => host.run_sync(make),
                    None => make(),
                }
                .map_err(|error| protocol(error.to_string()))
            })
            .transpose()?
    } else {
        None
    };
    Ok(AdmittedWorkerLaunch {
        request,
        scope,
        host,
    })
}

fn reserved(
    request: Box<crate::protocol::ReservedProgramExecutionRequest>,
    budget: Option<runmat_execution::console::ConsoleBudget>,
) -> WorkerLaunchRequest {
    match budget {
        Some(console_budget) => WorkerLaunchRequest::ReservedConsole {
            request,
            console_budget,
        },
        None => WorkerLaunchRequest::Reserved(request),
    }
}

fn decode_launch(payload: &[u8], role: WorkerRole) -> NativeExecutionResult<WorkerLaunchRequest> {
    let request = serde_json::from_slice(payload).map_err(|error| protocol(error.to_string()))?;
    validate_launch(&request, role)?;
    Ok(request)
}

fn validate_launch(request: &WorkerLaunchRequest, role: WorkerRole) -> NativeExecutionResult<()> {
    let direct = match request {
        WorkerLaunchRequest::Reserved(request)
        | WorkerLaunchRequest::ReservedConsole { request, .. }
        | WorkerLaunchRequest::ReservedFeaProxy { request, .. } => Some(request.as_ref()),
        WorkerLaunchRequest::ReservedFea(launch) => Some(&launch.request),
        WorkerLaunchRequest::Direct(_) | WorkerLaunchRequest::StagedGraph(_) => None,
    };
    if let Some(reserved) = direct {
        reserved
            .validate()
            .map_err(|error| protocol(error.to_string()))?;
        if let Some(assignment) = &reserved.request().assignment {
            assignment
                .validate()
                .map_err(|error| protocol(error.to_string()))?;
            assignment
                .resources
                .validate_for_request(reserved.resources())
                .map_err(|error| protocol(error.to_string()))?;
        }
        if reserved.request().console.is_some() != request.console_budget().is_some() {
            return Err(protocol(
                "worker console capture differs from its delegated subtree grant",
            ));
        }
    }
    if matches!(role, WorkerRole::Meshing) {
        let reserved =
            direct.ok_or_else(|| protocol("meshing worker requires a reserved meshing launch"))?;
        if reserved.request().artifact.form
            != runmat_execution_artifact::ExecutableForm::MeshingWorkload
            || !matches!(
                reserved.request().context,
                runmat_execution::ProgramInvocationContext::Direct
            )
            || reserved.request().assignment.is_none()
        {
            return Err(protocol(
                "meshing worker requires its original assigned Direct meshing workload",
            ));
        }
    }
    Ok(())
}

fn protocol(message: impl Into<String>) -> NativeExecutionError {
    NativeExecutionError::Protocol(message.into())
}

#[cfg(test)]
mod tests;
