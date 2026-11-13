//! The real permit remains with the process owner until settlement or retirement.
use super::*;
use runmat_process_host::ipc::{read_payload, write_payload, FrameLimits};

pub(super) type ActiveAdmission = Arc<std::sync::Mutex<Option<Arc<fea_proxy::Parent>>>>;

/// The canonical process owner drops this lease only after its retirement proof,
/// or calls persist_unverified before dropping an unproved child. The session's
/// dynamic per-test slot is the same slot retained by that physical owner.
#[derive(Default)]
pub(super) struct TestAdmissionLease {
    pub(super) active: ActiveAdmission,
}

impl runmat_process_host::child::ProcessRetirementLease for TestAdmissionLease {
    fn persist_unverified(&mut self, process_id: Option<u32>, reason: &str) -> Result<(), String> {
        if let Some(parent) = self
            .active
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        {
            // No physical certificate exists. Keep the original permit alive;
            // neither a close timestamp nor a final usage claim is justified.
            // This is the existing exceptional-retirement lease contract, not
            // a successful settlement or a fabricated durable recovery record.
            std::mem::forget(parent);
            eprintln!("test worker {process_id:?} retirement unverified; original FEA permit retained without settlement: {reason}");
        }
        Ok(())
    }
}

impl Drop for TestAdmissionLease {
    fn drop(&mut self) {
        if let Some(parent) = self
            .active
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        {
            parent.close();
        }
    }
}

fn active(session: &ProcessSession) -> Option<Arc<fea_proxy::Parent>> {
    session
        .state
        .fea_admission
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

pub(super) fn close(session: &ProcessSession) {
    if let Some(parent) = session
        .state
        .fea_admission
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
    {
        parent.close();
    }
}

pub(super) fn finish(session: &ProcessSession) -> Result<(), BackendError> {
    if let Some(parent) = active(session) {
        parent
            .terminal_check()
            .map_err(|error| BackendError::new(BackendErrorKind::Rejected, error.to_string()))?;
    }
    close(session);
    Ok(())
}

async fn check(session: &ProcessSession) -> Result<(), BackendError> {
    if let Some(parent) = active(session) {
        if let Err(error) = parent.check() {
            // Stop numerical work before the real permit can be retired.
            session
                .state
                .child
                .lock()
                .await
                .terminate_tree()
                .await
                .map_err(transport_error)?;
            session.state.retired.store(true, Ordering::Release);
            close(session);
            return Err(BackendError::new(
                BackendErrorKind::Rejected,
                error.to_string(),
            ));
        }
    }
    Ok(())
}

pub(super) async fn read_response(
    session: &ProcessSession,
    reader: &mut BufReader<ControlReader>,
) -> Result<WorkerResponse, BackendError> {
    let limits = FrameLimits {
        max_message_bytes: session.state.limits.max_message_bytes,
    };
    loop {
        // Keep partially received frames alive while checking the original wall/hold.
        let read = read_payload(reader, limits);
        tokio::pin!(read);
        let payload = loop {
            tokio::select! {
                payload = &mut read => break payload.map_err(transport_error)?,
                _ = tokio::time::sleep(std::time::Duration::from_millis(10)) => check(session).await?,
            }
        };
        let call = fea_proxy::decode_call(&payload)
            .map_err(|message| BackendError::new(BackendErrorKind::MalformedProtocol, message))?;
        if let Some(call) = call {
            let parent = active(session).ok_or_else(|| {
                BackendError::new(
                    BackendErrorKind::Rejected,
                    "FEA call has no original worker admission",
                )
            })?;
            let sequence = call.sequence;
            let mut reply = tokio::task::spawn_blocking(move || parent.call(call));
            let reply = loop {
                tokio::select! {
                    reply = &mut reply => break reply.map_err(|error| BackendError::new(BackendErrorKind::Transport, error.to_string()))?,
                    _ = tokio::time::sleep(std::time::Duration::from_millis(10)) => check(session).await?,
                }
            };
            let bytes = fea_proxy::encode_reply(sequence, reply).map_err(|message| {
                BackendError::new(BackendErrorKind::MalformedProtocol, message)
            })?;
            write_payload(&mut *session.state.writer.lock().await, &bytes, limits)
                .await
                .map_err(transport_error)?;
        } else {
            return runmat_test::protocol::decode_response(&payload, session.state.limits).map_err(
                |error| BackendError::new(BackendErrorKind::MalformedProtocol, error.to_string()),
            );
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use runmat_execution::{resource::ResourceRequest, ProgramExecutionAssignment};
    use runmat_runtime::analysis::fea_run_admission::*;

    struct Host {
        pid: u32,
        settled: Arc<AtomicBool>,
    }
    struct Permit {
        pid: u32,
        settled: Arc<AtomicBool>,
        budget: FeaRunBudget,
    }
    impl FeaRunAdmission for Host {
        fn admit(&self, _: &FeaRunRequest) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied> {
            Ok(Box::new(Permit {
                pid: self.pid,
                settled: self.settled.clone(),
                budget: FeaRunBudget {
                    budget_ops: runmat_simulation_usage::OperationCount::from_operations(u64::MAX),
                    granted_ops: runmat_simulation_usage::OperationCount::from_operations(u64::MAX),
                    expires_at_unix_millis: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64
                        + 60_000,
                    heartbeat_millis: 1000,
                },
            }))
        }
    }
    impl FeaRunPermit for Permit {
        fn budget(&self) -> FeaRunBudget {
            self.budget
        }
        fn heartbeat(&self, _: &FeaRunUsage) -> Result<FeaRunBudget, FeaRunDenied> {
            Ok(self.budget)
        }
        fn settle(
            self: Box<Self>,
            outcome: FeaRunOutcome,
            _: &FeaRunUsage,
        ) -> Result<(), FeaRunDenied> {
            assert_eq!(outcome, FeaRunOutcome::Failed);
            assert!(
                !runmat_process_host::is_process_alive(self.pid),
                "permit settled before physical worker retirement"
            );
            assert!(
                !self.settled.swap(true, Ordering::AcqRel),
                "duplicate settlement"
            );
            Ok(())
        }
    }

    #[tokio::test]
    async fn exceptional_owner_drop_retires_worker_before_settling_dynamic_permit() {
        let mut command = runmat_process_host::HostCommand::new("/bin/sh");
        command.arguments = vec!["-c".into(), "sleep 30".into()];
        let child = command.spawn().await.unwrap();
        let pid = child.id().unwrap();
        let settled = Arc::new(AtomicBool::new(false));
        let assignment = ProgramExecutionAssignment {
            scope_id: runmat_execution::ExecutionScopeId::derive(&[b"test-scope"]),
            pool_id: runmat_execution::PoolId::derive(&[b"test-pool"]),
            task_id: runmat_execution::TaskId::derive(&[b"test-task"]),
            attempt_id: runmat_execution::identity::AttemptId::derive(&[b"test-attempt"]),
            worker_id: runmat_execution::identity::WorkerId::derive(&[b"test-worker"]),
            backend: runmat_execution::PoolBackend::LocalProcesses,
            resources: Default::default(),
        };
        let resources = ResourceRequest {
            cpu_millicores: 1000,
            memory_bytes: 64 * 1024 * 1024,
            scratch_bytes: 0,
            max_wall_millis: 60_000,
            max_artifact_bytes: 0,
            max_egress_bytes: 0,
            max_relay_bytes: 0,
            accelerators: vec![],
            required_capabilities: Default::default(),
        };
        let parent = fea_proxy::Parent::new(
            Ok(Arc::new(Host {
                pid,
                settled: settled.clone(),
            })),
            assignment.clone(),
            resources,
            None,
            0,
        )
        .unwrap();
        assert!(matches!(
            parent.call(fea_proxy::Call {
                assignment,
                sequence: 1,
                operation: fea_proxy::Operation::Start {
                    request: FeaRunRequest::local(
                        FeaRunKind::Solve,
                        1000,
                        runmat_execution::Digest::sha256(b"test-study")
                    )
                    .unwrap()
                }
            }),
            fea_proxy::Reply::Budget { .. }
        ));
        let lease = TestAdmissionLease::default();
        let session_slot = lease.active.clone();
        *session_slot.lock().unwrap() = Some(parent);
        let owned = runmat_process_host::child::RetirementBoundChild::new(child, lease)
            .unwrap_or_else(|_| panic!("test worker must own its tree"));
        // Model exceptional ProcessState drop: no cancel/shutdown/close call.
        drop(owned);
        assert!(session_slot.lock().unwrap().is_none());
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while !settled.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }
}
