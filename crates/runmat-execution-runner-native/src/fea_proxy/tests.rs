use super::*;
use std::sync::mpsc;
use std::time::Duration;

#[test]
fn discarded_or_completed_cleanup_closure_settles_original_usage_once() {
    for discarded in [true, false] {
        let events = Arc::new(Mutex::new(Events::default()));
        let observed = usage(2000, 17);
        let mut settlement = FailedSettlement {
            permit: Some(Box::new(Permit(events.clone()))),
            observed: observed.clone(),
        };
        let mut cleanup = move || settlement.settle();
        if discarded {
            drop(cleanup);
        } else {
            cleanup();
        }
        let events = events.lock().unwrap();
        assert_eq!(events.settles.len(), 1);
        assert_eq!(events.settles[0], (FeaRunOutcome::Failed, observed));
    }
}
#[derive(Default)]
struct Events {
    starts: Vec<FeaRunRequest>,
    usage: Vec<ResourceUsage>,
    settles: Vec<(FeaRunOutcome, ResourceUsage)>,
}
struct Host(Arc<Mutex<Events>>);
struct Permit(Arc<Mutex<Events>>);
impl FeaRunAdmission for Host {
    fn admit(&self, request: &FeaRunRequest) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied> {
        self.0.lock().unwrap().starts.push(request.clone());
        Ok(Box::new(Permit(self.0.clone())))
    }
}
impl FeaRunPermit for Permit {
    fn budget(&self) -> FeaRunBudget {
        FeaRunBudget {
            held_core_millis: 1_000_000,
            expires_at_unix_millis: now() + 120_000,
            heartbeat_millis: 30_000,
        }
    }
    fn heartbeat(&self, usage: &ResourceUsage) -> Result<FeaRunBudget, FeaRunDenied> {
        self.0.lock().unwrap().usage.push(usage.clone());
        Ok(self.budget())
    }
    fn settle(
        self: Box<Self>,
        outcome: FeaRunOutcome,
        usage: &ResourceUsage,
    ) -> Result<(), FeaRunDenied> {
        self.0
            .lock()
            .unwrap()
            .settles
            .push((outcome, usage.clone()));
        Ok(())
    }
}
fn fixture() -> (Arc<Parent>, Arc<Mutex<Events>>) {
    let events = Arc::new(Mutex::new(Events::default()));
    let assignment = ProgramExecutionAssignment {
        scope_id: runmat_execution::ExecutionScopeId::derive(&[b"desktop-scope"]),
        pool_id: runmat_execution::PoolId::derive(&[b"desktop-pool"]),
        task_id: runmat_execution::TaskId::derive(&[b"desktop-task"]),
        attempt_id: runmat_execution::identity::AttemptId::derive(&[b"desktop-attempt"]),
        worker_id: runmat_execution::identity::WorkerId::derive(&[b"desktop-worker"]),
        backend: runmat_execution::PoolBackend::LocalProcesses,
        resources: Default::default(),
    };
    let resources = ResourceRequest {
        cpu_millicores: 1000,
        memory_bytes: 64 * 1024 * 1024,
        scratch_bytes: 0,
        max_wall_millis: 60_000,
        max_artifact_bytes: 64 * 1024 * 1024,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators: vec![],
        required_capabilities: Default::default(),
    };
    let parent = Parent::new(
        Ok(Arc::new(Host(events.clone()))),
        assignment,
        resources,
        Some(now() + 50_000),
        0,
    )
    .unwrap();
    (parent, events)
}
fn start(parent: &Parent) -> Call {
    Call {
        assignment: parent.assignment.clone(),
        sequence: 1,
        operation: Operation::Start {
            request: FeaRunRequest::local(
                FeaRunKind::Solve,
                1000,
                runmat_execution::Digest::sha256(b"desktop-study"),
            )
            .unwrap(),
        },
    }
}

#[test]
fn first_root_uses_explicit_originating_host_and_settles_once_with_original_bounds() {
    let (mut parent, events) = fixture();
    Arc::get_mut(&mut parent).unwrap().started = Instant::now() - Duration::from_secs(5);
    assert!(
        events.lock().unwrap().starts.is_empty(),
        "plain dispatch never calls admission"
    );
    let Reply::Budget { budget } = parent.call(start(&parent)) else {
        panic!("trusted Desktop-style host must admit without any CLI auth store")
    };
    assert_eq!(budget.expires_at_unix_millis, parent.deadline);
    assert!(budget.held_core_millis <= 50_000);
    let reply = parent.call(Call {
        assignment: parent.assignment.clone(),
        sequence: 2,
        operation: Operation::Heartbeat {
            root: 1,
            usage: ResourceUsage::default(),
        },
    });
    assert!(matches!(reply, Reply::Budget { .. }));
    let reply = parent.call(Call {
        assignment: parent.assignment.clone(),
        sequence: 3,
        operation: Operation::Settle {
            root: 1,
            outcome: FeaRunOutcome::Completed,
            usage: ResourceUsage::default(),
        },
    });
    assert_eq!(reply, Reply::Settled);
    assert!(parent.terminal_check().is_ok());
    drop(parent);
    let events = events.lock().unwrap();
    assert_eq!(events.starts.len(), 1);
    assert_eq!(events.usage.len(), 1);
    assert_eq!(events.settles.len(), 1);
    assert_eq!(events.settles[0].0, FeaRunOutcome::Completed);
    assert!(events.settles[0].1.accelerator_millis.is_empty());
    assert!(
        events.settles[0].1.wall_millis < 5_000,
        "ordinary script time before first FEA is not billed"
    );
}

#[test]
fn wrong_assignment_or_excess_cpu_never_reaches_originating_provider() {
    for excess_cpu in [false, true] {
        let (parent, events) = fixture();
        let mut call = start(&parent);
        if excess_cpu {
            let Operation::Start { request } = &mut call.operation else {
                unreachable!()
            };
            request.allocated_cpu_millicores = 2000;
        } else {
            call.assignment.attempt_id =
                runmat_execution::identity::AttemptId::derive(&[b"other-worker"]);
        }
        assert_eq!(
            parent.call(call),
            Reply::Denied {
                denied: FeaRunDenied::LicenseInvalid
            }
        );
        assert!(events.lock().unwrap().starts.is_empty());
    }
}

#[test]
fn automatic_reservations_quote_first_root_but_explicit_grants_stay_fixed() {
    let (original, events) = fixture();
    let context = FeaRunAdmissionContext::new(original.provider.as_ref().unwrap().clone());
    let quoted = context.run_sync(automatic_storage_bytes).unwrap();
    assert_eq!(quoted, storage_bytes(0, StorageRole::FirstRoot).unwrap());
    assert!(
        events.lock().unwrap().starts.is_empty(),
        "quoting does not call paid admission"
    );
    let invocation = 4096;
    let base = 1024 * 1024 + invocation;
    let mut explicit = original.resources.clone();
    explicit.memory_bytes = base;
    let parent = Parent::new(
        original.provider.clone(),
        original.assignment.clone(),
        explicit.clone(),
        Some(original.deadline),
        invocation,
    )
    .unwrap();
    assert_eq!(
        parent.call(start(&parent)),
        Reply::Denied {
            denied: FeaRunDenied::LicenseInvalid
        }
    );
    assert!(events.lock().unwrap().starts.is_empty());
    assert_eq!(parent.resources, explicit, "explicit authority never grows");
    let mut automatic = explicit;
    automatic.memory_bytes = base + quoted;
    let parent = Parent::new(
        original.provider.clone(),
        original.assignment.clone(),
        automatic,
        Some(original.deadline),
        invocation,
    )
    .unwrap();
    assert!(matches!(parent.call(start(&parent)), Reply::Budget { .. }));
    assert_eq!(events.lock().unwrap().starts.len(), 1);

    let Operation::Start { request } = start(&parent).operation else {
        unreachable!()
    };
    let delegated =
        FeaRunScope::from_delegated_budget(request, Permit(events.clone()).budget(), 0).unwrap();
    let inherited = delegated.run_sync(automatic_storage_bytes).unwrap();
    assert_eq!(inherited, storage_bytes(0, StorageRole::Delegated).unwrap());
    assert!(
        inherited < quoted,
        "inherited work has no second reporter stack"
    );
}

#[test]
fn forged_usage_or_wrong_root_cannot_spend_or_settle_another_permit() {
    for wrong_root in [false, true] {
        let (parent, events) = fixture();
        assert!(matches!(parent.call(start(&parent)), Reply::Budget { .. }));
        let operation = if wrong_root {
            Operation::Settle {
                root: 999,
                outcome: FeaRunOutcome::Completed,
                usage: ResourceUsage::default(),
            }
        } else {
            Operation::Heartbeat {
                root: 1,
                usage: usage(1000, u64::MAX),
            }
        };
        assert_eq!(
            parent.call(Call {
                assignment: parent.assignment.clone(),
                sequence: 2,
                operation
            }),
            Reply::Denied {
                denied: FeaRunDenied::LicenseInvalid
            }
        );
        assert!(events.lock().unwrap().usage.is_empty());
        assert!(events.lock().unwrap().settles.is_empty());
        parent.close();
        drop(parent);
        let events = events.lock().unwrap();
        assert_eq!(events.settles.len(), 1);
        assert_eq!(events.settles[0].0, FeaRunOutcome::Failed);
        assert!(events.settles[0].1.wall_millis < 60_000);
    }
}

#[test]
fn worker_loss_remains_observable_during_blocked_admission_and_late_permit_settles_once() {
    struct BlockingHost {
        events: Arc<Mutex<Events>>,
        entered: mpsc::SyncSender<()>,
        release: Mutex<mpsc::Receiver<()>>,
    }
    impl FeaRunAdmission for BlockingHost {
        fn admit(&self, request: &FeaRunRequest) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied> {
            self.events.lock().unwrap().starts.push(request.clone());
            self.entered.send(()).unwrap();
            self.release
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(2))
                .unwrap();
            Ok(Box::new(Permit(self.events.clone())))
        }
    }
    let (original, events) = fixture();
    let (entered, pending) = mpsc::sync_channel(1);
    let (release, blocked) = mpsc::sync_channel(1);
    let parent = Parent::new(
        Ok(Arc::new(BlockingHost {
            events: events.clone(),
            entered,
            release: Mutex::new(blocked),
        })),
        original.assignment.clone(),
        original.resources.clone(),
        Some(original.deadline),
        0,
    )
    .unwrap();
    drop(original);
    let worker = parent.clone();
    let call = start(&parent);
    let operation = std::thread::spawn(move || worker.call(call));
    pending.recv_timeout(Duration::from_secs(1)).unwrap();
    parent.close();
    assert_eq!(parent.check(), Err(FeaRunDenied::LeaseExpired));
    release.send(()).unwrap();
    assert_eq!(
        operation.join().unwrap(),
        Reply::Denied {
            denied: FeaRunDenied::LeaseExpired
        }
    );
    drop(parent);
    let events = events.lock().unwrap();
    assert_eq!(events.settles.len(), 1);
    assert_eq!(events.settles[0].0, FeaRunOutcome::Failed);
    assert_eq!(events.settles[0].1.wall_millis, 0);
}
