//! The host route has its own VM checked-publication boundary. No decision is
//! pinned: each trial executes the canonical placement policy with real costs.

use super::*;
use cpu_placement::CpuCapturingPlacement;
use runmat_accelerate::placement::{ExplorationMode, PlacementSessionConfig};
use runmat_execution::{ExecutionCandidateKind, PlacementExecutionAuthorityRevision};

#[derive(Clone, Copy, Debug)]
enum FaultCase {
    Reject,
    Cancel,
    DropPending,
}

fn one_host_fault(case: FaultCase) {
    let session = Rc::new(PlacementSession::new(PlacementSessionConfig {
        exploration: ExplorationMode::TransactionalEvery { interval: 2 },
        exploration_max_overhead_basis_points: u32::MAX,
        ..PlacementSessionConfig::default()
    }));
    let capture = Rc::new(CapturingPlacement {
        session,
        reduction_selection: RefCell::new(Vec::new()),
        delivered: Arc::new(Mutex::new(Vec::new())),
        invalidate_selected_provider: Cell::new(false),
        revoked_before_publication: Cell::new(0),
    });
    let placement = Rc::new(CpuCapturingPlacement {
        inner: capture.clone(),
        selected: RefCell::new(Vec::new()),
    });
    let bytecode = compile_source(HOST_FOLDED_MIN);

    // The sibling cohort needed more than twelve authentic placement calls:
    // physical provider recipes and Cold/Warm facts can occupy earlier
    // exploration turns. Keep the search bounded without pinning a route.
    for _ in 0..24 {
        let cancellation = CancellationSource::new();
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(runmat_runtime::execution::RuntimeExecutionService::new()),
            cancellation.token(),
        )
        .with_service_ports(RuntimeServicePorts::default().with_placement(placement.clone()));
        let vars = crate::InterpreterVars::from_owned(vec![Value::Num(0.0); bytecode.var_count]);
        let reached = Rc::new(Cell::new(false));
        let fault = match case {
            FaultCase::Reject => VmFusionPublicationFault::Reject(
                VmFusionPublicationFaultRoute::SharedRuntime,
                reached.clone(),
            ),
            FaultCase::Cancel => VmFusionPublicationFault::Cancel(
                VmFusionPublicationFaultRoute::SharedRuntime,
                cancellation.clone(),
                reached.clone(),
            ),
            FaultCase::DropPending => VmFusionPublicationFault::Suspend(
                VmFusionPublicationFaultRoute::SharedRuntime,
                reached.clone(),
            ),
        };
        let selected_before = placement.selected.borrow().len();
        let delivered_before = capture.delivered.lock().unwrap().len();
        let result = if matches!(case, FaultCase::DropPending) {
            let mut pending = Box::pin(scoped_faultable_run(
                bytecode.clone(),
                vars,
                runtime.clone(),
                Some(fault),
            ));
            let result = block_on(std::future::poll_fn(|context| {
                match pending.as_mut().poll(context) {
                    Poll::Ready(result) => Poll::Ready(Some(result)),
                    Poll::Pending if reached.get() => Poll::Ready(None),
                    Poll::Pending => Poll::Pending,
                }
            }));
            drop(pending);
            result
        } else {
            Some(block_on(scoped_faultable_run(
                bytecode.clone(),
                vars,
                runtime.clone(),
                Some(fault),
            )))
        };
        block_on(runtime.scope(async {}));
        let selections = placement.selected.borrow();
        assert_eq!(
            selections.len(),
            selected_before + 1,
            "one authentic reduction decision"
        );
        let selected = &selections[selected_before];
        assert!(matches!(
            selected.signature.revision.authority,
            PlacementExecutionAuthorityRevision::RuntimeBindingSession(_)
        ));
        // Keep assertions outside the capture mutex. An authentic placement
        // failure must report its first cause rather than poison feedback
        // delivery during the session's destructor.
        let new_receipts = || {
            let delivered = capture.delivered.lock().unwrap();
            delivered[delivered_before..]
                .iter()
                .filter(|feedback| feedback.signature.operation == "fusion.reduction")
                .cloned()
                .collect::<Vec<_>>()
        };
        if selected.kind == ExecutionCandidateKind::SharedRuntime {
            let receipts = new_receipts();
            assert!(
                reached.get(),
                "host fault must reach the exact checked StoreVar"
            );
            match case {
                FaultCase::Reject => {
                    let error = match result.expect("reject executes to an error") {
                        Ok(_) => panic!("ordinary StoreVar rejects transient OutputList"),
                        Err(error) => error,
                    };
                    assert_eq!(
                        error.identifier(),
                        Some("RunMat:TransientSequenceNotStorable")
                    );
                }
                FaultCase::Cancel => {
                    let error = match result.expect("cancel executes to an error") {
                        Ok(_) => panic!("cancellation terminates selected host call"),
                        Err(error) => error,
                    };
                    assert_eq!(error.identifier(), Some("RunMat:ExecutionCancelled"));
                    assert!(cancellation.token().is_cancelled());
                }
                FaultCase::DropPending => assert!(result.is_none(), "host future suspended"),
            }
            assert!(
                receipts.is_empty(),
                "failed, cancelled, or dropped host publication deposits no receipt"
            );
            return;
        }
        assert_eq!(selected.kind, ExecutionCandidateKind::ProviderFusion);
        assert!(
            !reached.get(),
            "host-only hook does not alter provider publication"
        );
        result
            .expect("provider trial completes rather than host-suspending")
            .expect("normal provider trial succeeds");
        // Provider completion is asynchronous after the VM has published the
        // selected output. The scope exit performs only one nonblocking drain;
        // wait for this exact receipt rather than racing the device callback.
        // Each prior provider trial is drained before starting the next one.
        let started_waiting = std::time::Instant::now();
        let receipts = loop {
            runmat_runtime::context::RuntimePlacementService::drain_feedback(placement.as_ref());
            let receipts = new_receipts();
            if receipts.iter().any(|receipt| {
                receipt.signature == selected.signature
                    && receipt.candidate == selected.candidate
                    && receipt.kind == selected.kind
            }) {
                break receipts;
            }
            assert!(
                started_waiting.elapsed() < std::time::Duration::from_secs(5),
                "selected provider receipt did not complete: {receipts:?}"
            );
            std::thread::yield_now();
        };
        assert_eq!(receipts.len(), 1, "normal provider trial has one receipt");
        assert_eq!(receipts[0].signature, selected.signature);
        assert_eq!(receipts[0].candidate, selected.candidate);
        assert_eq!(receipts[0].kind, selected.kind);
        assert!(receipts[0].succeeded);
    }
    let choices = placement
        .selected
        .borrow()
        .iter()
        .enumerate()
        .map(|(index, selected)| {
            format!(
                "{}: {:?} {} {:?}",
                index + 1,
                selected.kind,
                selected.candidate,
                selected.signature.cache_key()
            )
        })
        .collect::<Vec<_>>();
    panic!(
        "canonical exploration did not select an authentic SharedRuntime reduction: {choices:?}"
    );
}

#[test]
#[ignore = "requires actual WGPU adapter and an authentic SharedRuntime choice"]
fn checked_host_store_rejection_cancellation_and_drop_have_no_feedback() {
    runmat_gc::gc_test_context(|| {
        let _provider_scope = actual_provider_scope();
        for case in [FaultCase::Reject, FaultCase::Cancel, FaultCase::DropPending] {
            one_host_fault(case);
        }
    });
}
