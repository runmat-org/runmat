//! Host-composed admission to one bounded native CPU-worker lane. Requests
//! retain their own inputs and managed reservations through physical exit.

#[cfg(not(target_arch = "wasm32"))]
use std::rc::Rc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, OnceLock};

#[cfg(not(target_arch = "wasm32"))]
use futures::channel::mpsc;
#[cfg(not(target_arch = "wasm32"))]
use futures::{SinkExt, StreamExt};
#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::Mutex;

use super::RuntimeContinuationFuture;
use crate::context::RuntimeContext;

mod job;
pub use job::OwnedCpuJob;
#[cfg(not(target_arch = "wasm32"))]
pub(super) mod completion;

use super::ExecutionServiceError;

pub type OwnedCpuSubmission =
    Result<RuntimeContinuationFuture<'static, (), ExecutionServiceError>, ExecutionServiceError>;

pub trait RuntimeBlockingCpuService {
    /// Wait for bounded queue capacity, then transfer the entire owned job.
    /// Dropping the admission future before acceptance drops the job and its
    /// reservation. Once accepted, caller cancellation cannot retire it.
    fn submit(&self, runtime: Option<&RuntimeContext>, job: OwnedCpuJob) -> OwnedCpuSubmission;
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeBlockingCpuService;

#[cfg(not(target_arch = "wasm32"))]
static NATIVE_OWNER: OnceLock<Result<Arc<Mutex<mpsc::Sender<OwnedCpuJob>>>, String>> =
    OnceLock::new();

#[cfg(not(target_arch = "wasm32"))]
fn native_owner() -> Result<Arc<Mutex<mpsc::Sender<OwnedCpuJob>>>, ExecutionServiceError> {
    let owner = NATIVE_OWNER.get_or_init(|| {
        // One executing and one queued job; other callers await admission
        // without blocking an executor thread or spuriously failing busy.
        // futures::mpsc reserves one sender slot in addition to the explicit
        // buffer. With the single mutex-held sender, zero buffer means at
        // most one queued job behind the one physically executing.
        let (send, mut receive) = mpsc::channel::<OwnedCpuJob>(0);
        std::thread::Builder::new()
            .name("runmat-blocking-cpu".into())
            .spawn(move || {
                futures::executor::block_on(async move {
                    while let Some(job) = receive.next().await {
                        // Each job owns its inputs, response and reservations.
                        // Unwinding retires that invocation; it must not close
                        // the persistent lane or replay any of the failed work.
                        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job.run()))
                            .is_err()
                        {
                            log::error!("owned CPU job panicked before publication");
                        }
                    }
                });
            })
            .map_err(|error| error.to_string())?;
        Ok(Arc::new(Mutex::new(send)))
    });
    owner.as_ref().map(Arc::clone).map_err(|message| {
        ExecutionServiceError::Infrastructure(format!(
            "owned CPU worker could not start: {message}"
        ))
    })
}

#[cfg(not(target_arch = "wasm32"))]
impl RuntimeBlockingCpuService for NativeBlockingCpuService {
    fn submit(&self, runtime: Option<&RuntimeContext>, job: OwnedCpuJob) -> OwnedCpuSubmission {
        super::admit_optional_hostcall_with(
            runtime,
            ExecutionServiceError::HostAllocation,
            || async move {
                // Persistent lane initialization is host infrastructure, distinct
                // from this admitted request's concrete queue-wait future.
                let sender = native_owner()?;
                let mut sender = sender.lock().await;
                sender
                    .send(job)
                    .await
                    .map_err(|_| ExecutionServiceError::WorkerLost("owned CPU worker ended".into()))
            },
        )
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn native_default_blocking_cpu_service() -> Rc<dyn RuntimeBlockingCpuService> {
    Rc::new(NativeBlockingCpuService)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::execution::{CpuCooperationBudget, RuntimeExecutionService};
    use runmat_execution::cancellation::CancellationSource;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Retained(Arc<AtomicUsize>);

    impl Drop for Retained {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn failed_job_retires_without_replay_and_the_lane_accepts_more_work() {
        let source = CancellationSource::new();
        let budget =
            CpuCooperationBudget::new(Rc::new(RuntimeExecutionService::new()), source.token());
        let service = native_default_blocking_cpu_service();
        let retired = Arc::new(AtomicUsize::new(0));
        let executions = Arc::new(AtomicUsize::new(0));
        let executed = Arc::clone(&executions);
        let failed: Result<(), ExecutionServiceError> =
            futures::executor::block_on(budget.run_owned_blocking(
                Some(service.as_ref()),
                Retained(Arc::clone(&retired)),
                move || {
                    executed.fetch_add(1, Ordering::SeqCst);
                    panic!("owned CPU fixture failed");
                },
            ));
        assert!(matches!(failed, Err(ExecutionServiceError::WorkerLost(_))));
        let result = futures::executor::block_on(budget.run_owned_blocking(
            Some(service.as_ref()),
            (),
            || 17,
        ));
        assert_eq!(result.unwrap(), 17);
        assert_eq!(executions.load(Ordering::SeqCst), 1);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }

    struct Parallel(crate::context::host_effect::ManagedHostEffectLedger);
    impl crate::context::RuntimeParallelService for Parallel {
        fn supports(&self, _: crate::context::ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> crate::context::RuntimeParallelResources {
            let state = self.0.snapshot();
            crate::context::RuntimeParallelResources {
                memory_available_bytes: Some(state.free_bytes()),
                epoch: state.epoch,
                ..Default::default()
            }
        }
        fn managed_host_effect_ledger(
            &self,
        ) -> Option<crate::context::host_effect::ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    #[derive(Default)]
    struct Held(std::cell::RefCell<Option<OwnedCpuJob>>);
    impl RuntimeBlockingCpuService for Held {
        fn submit(&self, runtime: Option<&RuntimeContext>, job: OwnedCpuJob) -> OwnedCpuSubmission {
            let submitted = super::super::admit_optional_hostcall_with(
                runtime,
                ExecutionServiceError::HostAllocation,
                || async { Ok(()) },
            )?;
            assert!(self.0.borrow_mut().replace(job).is_none());
            Ok(submitted)
        }
    }
    fn managed(
        capacity: u64,
        cancel: &CancellationSource,
    ) -> (
        RuntimeContext,
        crate::context::host_effect::ManagedHostEffectLedger,
    ) {
        let ledger = crate::context::host_effect::ManagedHostEffectLedger::new(capacity);
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(RuntimeExecutionService::new()),
            cancel.token(),
        )
        .with_service_ports(
            crate::context::RuntimeServicePorts::default()
                .with_parallel(Rc::new(Parallel(ledger.clone()))),
        );
        (runtime, ledger)
    }

    #[test]
    fn exact_cpu_control_denies_before_submit_and_survives_cancel_drop_and_panic() {
        use std::future::Future;
        use std::task::{Context, Poll};
        let cancel = CancellationSource::new();
        let (runtime, ledger) = managed(0, &cancel);
        let held = Held::default();
        let ran = Arc::new(AtomicUsize::new(0));
        let run = ran.clone();
        let result =
            futures::executor::block_on(runtime.cpu_cooperation().run_owned_blocking_in_context(
                &runtime,
                Some(&held),
                (),
                move || run.fetch_add(1, Ordering::SeqCst),
            ));
        assert!(matches!(
            result,
            Err(ExecutionServiceError::HostAllocation(_))
        ));
        assert!(held.0.borrow().is_none());
        assert_eq!(ran.load(Ordering::SeqCst), 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

        for mode in 0..3 {
            let cancel = CancellationSource::new();
            let (runtime, ledger) = managed(1_000_000, &cancel);
            let held = Held::default();
            let retired = Arc::new(AtomicUsize::new(0));
            let budget = runtime.cpu_cooperation();
            let mut future = Box::pin(budget.run_owned_blocking_in_context(
                &runtime,
                Some(&held),
                Retained(retired.clone()),
                move || {
                    if mode == 2 {
                        panic!("owned CPU controlled panic");
                    }
                    17usize
                },
            ));
            let mut context = Context::from_waker(futures::task::noop_waker_ref());
            assert!(future.as_mut().poll(&mut context).is_pending());
            assert!(ledger.snapshot().live_claimed_bytes > 0);
            if mode == 1 {
                cancel.cancel(runmat_execution::CancellationReason::User);
                assert!(matches!(
                    future.as_mut().poll(&mut context),
                    Poll::Ready(Err(ExecutionServiceError::Cancelled))
                ));
            }
            drop(future);
            assert_eq!(retired.load(Ordering::SeqCst), 0);
            assert!(ledger.snapshot().live_claimed_bytes > 0);
            let job = held.0.borrow_mut().take().expect("accepted physical job");
            let exit = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job.run()));
            assert_eq!(exit.is_err(), mode == 2);
            assert_eq!(retired.load(Ordering::SeqCst), 1);
            assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        }
    }

    #[test]
    fn queued_cpu_job_checks_original_authority_before_work_and_publication() {
        use std::future::Future;
        use std::task::{Context, Poll};
        for revoke_inside in [false, true] {
            let cancel = CancellationSource::new();
            let (runtime, ledger) = managed(1_000_000, &cancel);
            let held = Held::default();
            let ran = Arc::new(AtomicUsize::new(0));
            let run = ran.clone();
            let worker_ledger = ledger.clone();
            let retired = Arc::new(AtomicUsize::new(0));
            let budget = runtime.cpu_cooperation();
            let mut future = Box::pin(budget.run_owned_blocking_in_context(
                &runtime,
                Some(&held),
                Retained(retired.clone()),
                move || {
                    run.fetch_add(1, Ordering::SeqCst);
                    if revoke_inside {
                        worker_ledger.revoke();
                    }
                    17usize
                },
            ));
            let mut context = Context::from_waker(futures::task::noop_waker_ref());
            assert!(future.as_mut().poll(&mut context).is_pending());
            if !revoke_inside {
                ledger.revoke();
            }
            held.0.borrow_mut().take().unwrap().run();
            assert_eq!(ran.load(Ordering::SeqCst), usize::from(revoke_inside));
            assert!(matches!(
                future.as_mut().poll(&mut context),
                Poll::Ready(Err(ExecutionServiceError::HostAllocation(
                    crate::context::host_allocation::HostAllocationError::StaleAuthority
                )))
            ));
            drop(future);
            assert_eq!(retired.load(Ordering::SeqCst), 1);
            assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        }
    }

    #[test]
    fn cpu_completion_drop_clears_receiver_waker_and_sender_loss_is_terminal() {
        use std::future::Future;
        use std::pin::Pin;
        use std::task::Context;
        struct Wake;
        impl futures::task::ArcWake for Wake {
            fn wake_by_ref(_: &Arc<Self>) {}
        }
        let cancel = CancellationSource::new();
        let (runtime, ledger) = managed(1_000_000, &cancel);
        let (send, mut receive) = completion::channel::<usize, _>(Some(&runtime), ()).unwrap();
        let wake = Arc::new(Wake);
        let waker = futures::task::waker(wake.clone());
        let baseline = Arc::strong_count(&wake);
        assert!(Pin::new(&mut receive)
            .poll(&mut Context::from_waker(&waker))
            .is_pending());
        assert!(Arc::strong_count(&wake) > baseline);
        drop(receive);
        assert_eq!(Arc::strong_count(&wake), baseline);
        assert!(ledger.snapshot().live_claimed_bytes > 0);
        drop(send);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        let (send, mut receive) = completion::channel::<usize, _>(Some(&runtime), ()).unwrap();
        drop(send);
        assert!(matches!(
            Pin::new(&mut receive).poll(&mut Context::from_waker(&waker)),
            std::task::Poll::Ready(Err(ExecutionServiceError::WorkerLost(_)))
        ));
        drop(receive);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

        struct Buffered {
            reservation_retired: Arc<AtomicUsize>,
            retired: Arc<AtomicUsize>,
        }
        impl Drop for Buffered {
            fn drop(&mut self) {
                assert_eq!(self.reservation_retired.load(Ordering::SeqCst), 0);
                self.retired.fetch_add(1, Ordering::SeqCst);
            }
        }
        let reservation_retired = Arc::new(AtomicUsize::new(0));
        let result_retired = Arc::new(AtomicUsize::new(0));
        let (send, receive) =
            completion::channel(Some(&runtime), Retained(reservation_retired.clone())).unwrap();
        send.send(Ok(Buffered {
            reservation_retired: reservation_retired.clone(),
            retired: result_retired.clone(),
        }));
        assert_eq!(reservation_retired.load(Ordering::SeqCst), 0);
        drop(receive);
        assert_eq!(result_retired.load(Ordering::SeqCst), 1);
        assert_eq!(reservation_retired.load(Ordering::SeqCst), 1);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    }
}
