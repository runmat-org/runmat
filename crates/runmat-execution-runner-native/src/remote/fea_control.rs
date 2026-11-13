//! Worker-side transport adapter for the existing parent FEA admission protocol.
//! Only typed calls and replies cross the authenticated remote attempt route.
use crate::fea_proxy::{parent_reply_timeout, Call, Operation, Reply};
use runmat_execution::cancellation::CancellationToken;
use runmat_execution::ProgramExecutionAssignment;
use runmat_runtime::analysis::fea_run_admission::{
    FeaRunBudget, FeaRunDenied, FeaRunOutcome, FeaRunPermit, FeaRunRequest, FeaRunUsage,
};
use runmat_simulation_usage::OperationCount;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

const CANCELLATION_POLL: Duration = Duration::from_millis(100);

struct Pending {
    sequence: u64,
    reply: mpsc::SyncSender<Reply>,
}
#[derive(Default)]
struct Calls {
    sequence: u64,
    active: Option<u64>,
}

pub(super) struct WorkerControl {
    assignment: ProgramExecutionAssignment,
    cpu_millicores: u64,
    calls: Mutex<Calls>,
    pending: Mutex<Option<Pending>>,
    outgoing: tokio::sync::mpsc::Sender<Call>,
    cancellation: CancellationToken,
}

impl WorkerControl {
    pub(super) fn new(
        assignment: ProgramExecutionAssignment,
        cpu_millicores: u64,
        cancellation: CancellationToken,
    ) -> (Arc<Self>, tokio::sync::mpsc::Receiver<Call>) {
        let (outgoing, receiver) = tokio::sync::mpsc::channel(1);
        (
            Arc::new(Self {
                assignment,
                cpu_millicores,
                calls: Mutex::new(Calls::default()),
                pending: Mutex::new(None),
                outgoing,
                cancellation,
            }),
            receiver,
        )
    }

    fn call(
        &self,
        calls: &mut Calls,
        operation: Operation,
        timeout: Duration,
    ) -> Result<Reply, FeaRunDenied> {
        let settling = matches!(&operation, Operation::Settle { .. });
        if !settling && self.cancellation.is_cancelled() {
            return Err(FeaRunDenied::LeaseExpired);
        }
        calls.sequence = calls
            .sequence
            .checked_add(1)
            .ok_or(FeaRunDenied::LicenseInvalid)?;
        let sequence = calls.sequence;
        let (reply, received) = mpsc::sync_channel(1);
        {
            let mut pending = self
                .pending
                .lock()
                .map_err(|_| FeaRunDenied::LicenseInvalid)?;
            if pending.is_some() {
                return Err(FeaRunDenied::LicenseInvalid);
            }
            *pending = Some(Pending { sequence, reply });
        }
        let sent = self.outgoing.try_send(Call {
            assignment: self.assignment.clone(),
            sequence,
            operation,
        });
        if sent.is_err() {
            self.pending
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take();
            return Err(FeaRunDenied::ServerUnreachable);
        }
        let deadline = std::time::Instant::now() + timeout;
        let result = loop {
            if !settling && self.cancellation.is_cancelled() {
                break Err(FeaRunDenied::LeaseExpired);
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                break Err(FeaRunDenied::ServerUnreachable);
            }
            match received.recv_timeout(remaining.min(CANCELLATION_POLL)) {
                Ok(reply) => break Ok(reply),
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    break Err(FeaRunDenied::ServerUnreachable)
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
        };
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        match result? {
            Reply::Denied { denied } => Err(denied),
            other => Ok(other),
        }
    }

    pub(super) fn complete(&self, sequence: u64, reply: Reply) -> Result<(), &'static str> {
        let mut pending = self.pending.lock().map_err(|_| "FEA reply lock poisoned")?;
        let Some(waiting) = pending.as_ref() else {
            return Err("FEA reply has no live parent request");
        };
        if waiting.sequence != sequence {
            return Err("FEA reply sequence differs from its parent request");
        }
        let waiting = pending.take().expect("validated live FEA parent request");
        waiting
            .reply
            .send(reply)
            .map_err(|_| "FEA numerical owner stopped before its parent reply")
    }

    pub(super) fn admit(
        self: &Arc<Self>,
        request: &FeaRunRequest,
    ) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied> {
        request.validate()?;
        if request.site != runmat_runtime::analysis::fea_run_admission::FeaRunSite::Dispatched
            || request.allocated_cpu_millicores > self.cpu_millicores
        {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        let mut calls = self
            .calls
            .lock()
            .map_err(|_| FeaRunDenied::LicenseInvalid)?;
        if calls.active.is_some() {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        let Reply::Budget { budget } = self.call(
            &mut calls,
            Operation::Start {
                request: request.clone(),
            },
            parent_reply_timeout(None),
        )?
        else {
            return Err(FeaRunDenied::LicenseInvalid);
        };
        budget.validate()?;
        let root = calls.sequence;
        calls.active = Some(root);
        Ok(Box::new(WorkerPermit {
            control: Arc::clone(self),
            root,
            budget,
        }))
    }
}

struct WorkerPermit {
    control: Arc<WorkerControl>,
    root: u64,
    budget: FeaRunBudget,
}
impl WorkerPermit {
    fn renew(&self, operation: Operation) -> Result<FeaRunBudget, FeaRunDenied> {
        let mut calls = self
            .control
            .calls
            .lock()
            .map_err(|_| FeaRunDenied::LicenseInvalid)?;
        if calls.active != Some(self.root) {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        match self.control.call(
            &mut calls,
            operation,
            parent_reply_timeout(Some(&self.budget)),
        )? {
            Reply::Budget { budget } => Ok(budget),
            _ => Err(FeaRunDenied::LicenseInvalid),
        }
    }
}
impl FeaRunPermit for WorkerPermit {
    fn budget(&self) -> FeaRunBudget {
        self.budget
    }
    fn heartbeat(&self, usage: &FeaRunUsage) -> Result<FeaRunBudget, FeaRunDenied> {
        self.renew(Operation::Heartbeat {
            root: self.root,
            usage: usage.clone(),
        })
    }
    fn request_grant(
        &self,
        usage: &FeaRunUsage,
        required: OperationCount,
    ) -> Result<FeaRunBudget, FeaRunDenied> {
        self.renew(Operation::RequestGrant {
            root: self.root,
            usage: usage.clone(),
            required,
        })
    }
    fn settle(
        self: Box<Self>,
        outcome: FeaRunOutcome,
        usage: &FeaRunUsage,
    ) -> Result<(), FeaRunDenied> {
        let mut calls = self
            .control
            .calls
            .lock()
            .map_err(|_| FeaRunDenied::LicenseInvalid)?;
        if calls.active != Some(self.root) {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        let result = self.control.call(
            &mut calls,
            Operation::Settle {
                root: self.root,
                outcome,
                usage: usage.clone(),
            },
            parent_reply_timeout(Some(&self.budget)),
        );
        calls.active = None;
        match result? {
            Reply::Settled => Ok(()),
            _ => Err(FeaRunDenied::LicenseInvalid),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::resource::ResourceUsage;
    use runmat_execution::{CancellationReason, Digest, PoolBackend};
    use runmat_runtime::analysis::fea_run_admission::{FeaRunKind, FeaRunSite};
    use runmat_simulation_usage::CompletedUsage;

    fn assignment() -> ProgramExecutionAssignment {
        ProgramExecutionAssignment {
            scope_id: runmat_execution::ExecutionScopeId::derive(&[b"fea-control-scope"]),
            pool_id: runmat_execution::PoolId::derive(&[b"fea-control-pool"]),
            task_id: runmat_execution::TaskId::derive(&[b"fea-control-task"]),
            attempt_id: runmat_execution::identity::AttemptId::derive(&[b"fea-control-attempt"]),
            worker_id: runmat_execution::identity::WorkerId::derive(&[b"fea-control-worker"]),
            backend: PoolBackend::Remote,
            resources: Default::default(),
        }
    }

    fn budget(granted: u64) -> FeaRunBudget {
        FeaRunBudget {
            budget_ops: OperationCount::from_operations(1_000),
            granted_ops: OperationCount::from_operations(granted),
            expires_at_unix_millis: super::super::fea_admission::now() + 60_000,
            heartbeat_millis: 30_000,
        }
    }

    async fn next(outgoing: &mut tokio::sync::mpsc::Receiver<Call>) -> Call {
        tokio::time::timeout(Duration::from_secs(2), outgoing.recv())
            .await
            .expect("bounded parent call arrives")
            .expect("numerical owner retains control channel")
    }

    #[tokio::test(flavor = "current_thread")]
    async fn exact_parent_exchange_preserves_pending_sequence_and_cancelled_settlement() {
        let assignment = assignment();
        let cancellation = CancellationSource::new();
        let (control, mut outgoing) =
            WorkerControl::new(assignment.clone(), 1_000, cancellation.token());
        let mut request = FeaRunRequest::local(FeaRunKind::Solve, 1_000, Digest::sha256(b"study"))
            .expect("valid synthetic FEA root");
        request.site = FeaRunSite::Dispatched;
        let usage = FeaRunUsage {
            completed_ops: CompletedUsage {
                solve: OperationCount::from_operations(7),
                ..CompletedUsage::default()
            },
            resources: ResourceUsage::default(),
        };
        let numerical = std::thread::spawn({
            let control = Arc::clone(&control);
            let request = request.clone();
            let usage = usage.clone();
            move || {
                let permit = control.admit(&request).expect("parent admits exact root");
                let first = permit
                    .request_grant(&usage, OperationCount::from_operations(30))
                    .expect("parent grants first numerical batch");
                let renewed = permit.heartbeat(&usage).expect("parent accepts usage");
                let cancelled = permit.request_grant(&usage, OperationCount::from_operations(80));
                let settled = permit.settle(FeaRunOutcome::Cancelled, &usage);
                (first, renewed, cancelled, settled)
            }
        });

        let start = next(&mut outgoing).await;
        assert_eq!(start.assignment, assignment);
        assert_eq!(start.sequence, 1);
        assert!(
            matches!(start.operation, Operation::Start { request: actual } if actual == request)
        );
        control
            .complete(1, Reply::Budget { budget: budget(20) })
            .unwrap();

        let grant = next(&mut outgoing).await;
        assert_eq!(grant.assignment, assignment);
        assert_eq!(grant.sequence, 2);
        assert!(
            matches!(grant.operation, Operation::RequestGrant { root: 1, required, .. } if required == OperationCount::from_operations(30))
        );
        assert!(control
            .complete(99, Reply::Budget { budget: budget(50) })
            .is_err());
        control
            .complete(2, Reply::Budget { budget: budget(50) })
            .unwrap();

        let heartbeat = next(&mut outgoing).await;
        assert_eq!(heartbeat.sequence, 3);
        assert!(
            matches!(heartbeat.operation, Operation::Heartbeat { root: 1, usage: actual } if actual == usage)
        );
        control
            .complete(3, Reply::Budget { budget: budget(50) })
            .unwrap();

        let pending_grant = next(&mut outgoing).await;
        assert_eq!(pending_grant.sequence, 4);
        assert!(
            matches!(pending_grant.operation, Operation::RequestGrant { root: 1, required, .. } if required == OperationCount::from_operations(80))
        );
        cancellation.cancel(CancellationReason::User);
        let settlement = next(&mut outgoing).await;
        assert_eq!(settlement.sequence, 5);
        assert!(
            matches!(settlement.operation, Operation::Settle { root: 1, outcome: FeaRunOutcome::Cancelled, usage: actual } if actual == usage)
        );
        assert!(
            !numerical.is_finished(),
            "no physical retirement before settlement reply and thread join"
        );
        assert!(control
            .complete(4, Reply::Budget { budget: budget(80) })
            .is_err());
        control.complete(5, Reply::Settled).unwrap();

        let (first, renewed, cancelled, settled) =
            numerical.join().expect("numerical thread retired");
        assert_eq!(first.granted_ops, OperationCount::from_operations(50));
        assert_eq!(renewed.granted_ops, OperationCount::from_operations(50));
        assert_eq!(cancelled, Err(FeaRunDenied::LeaseExpired));
        assert_eq!(settled, Ok(()));
    }
}
