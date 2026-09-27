use super::*;
use crate::execute::call_suspension::{CallCompletion, PendingCall};
use crate::execute::region::observation::test_support::observed_state;
use crate::region::placement_test_support::Probe;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_runtime::native::{
    NativeCall, NativeExit, NativeHostStatus, NativeSitePhase, NativeSiteRequest,
};
use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

#[derive(Default)]
struct Wakes(AtomicUsize);
impl Wake for Wakes {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

struct PendingProbe {
    polls: Rc<Cell<usize>>,
    drops: Rc<Cell<usize>>,
    ready: Rc<Cell<bool>>,
    on_drop: Option<Box<dyn Fn()>>,
}
impl Future for PendingProbe {
    type Output = NativeExecutorResult<CallCompletion>;
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        self.polls.set(self.polls.get() + 1);
        if self.ready.get() {
            Poll::Ready(Ok(CallCompletion::Values(vec![])))
        } else {
            Poll::Pending
        }
    }
}
impl Drop for PendingProbe {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
        if let Some(on_drop) = &self.on_drop {
            on_drop();
        }
    }
}

unsafe extern "C" fn completed_entry(
    _: *mut NativeCall,
    exit: *mut NativeExit,
) -> NativeHostStatus {
    // SAFETY: this fixture is called through the executor's checked ABI record.
    unsafe {
        *exit = NativeExit::completed(0);
    }
    NativeHostStatus::OK
}

fn invocation(source: &CancellationSource, work: PendingProbe) -> NativeInvocation {
    let mut state = observed_state(&Rc::new(Probe::default()));
    state.runtime = state
        .runtime
        .clone()
        .with_cancellation_token(source.token());
    state
        .function
        .expected_sites
        .push(state.function.blocks[0].terminator.site.clone());
    state.pending_call = Some(PendingCall {
        continuation: 1,
        generation: 1,
        request: NativeSiteRequest {
            function: 0,
            block: 0,
            position: 0,
            phase: NativeSitePhase::TERMINATOR,
            ordinal: 0,
            reserved: 0,
        },
        embedded: None,
        future: Box::pin(work),
    });
    NativeInvocation::new(
        state,
        completed_entry,
        vec![],
        0,
        NativeResumeState::default(),
        None,
    )
}

fn work() -> (
    PendingProbe,
    Rc<Cell<usize>>,
    Rc<Cell<usize>>,
    Rc<Cell<bool>>,
) {
    let polls = Rc::new(Cell::new(0));
    let drops = Rc::new(Cell::new(0));
    let ready = Rc::new(Cell::new(false));
    (
        PendingProbe {
            polls: polls.clone(),
            drops: drops.clone(),
            ready: ready.clone(),
            on_drop: None,
        },
        polls,
        drops,
        ready,
    )
}

#[test]
fn cancellation_wakes_pending_resume_and_terminalizes_retained_invocation() {
    let source = CancellationSource::new();
    let (work, polls, drops, _) = work();
    let mut invocation = invocation(&source, work);
    let wakes = Arc::new(Wakes::default());
    let waker = Waker::from(wakes.clone());
    let mut cx = Context::from_waker(&waker);
    let mut resume = Box::pin(invocation.resume_suspension(1, 1));
    assert!(resume.as_mut().poll(&mut cx).is_pending());
    assert_eq!(polls.get(), 1);
    source.cancel(CancellationReason::User);
    assert_eq!(wakes.0.load(Ordering::Relaxed), 1);
    assert!(matches!(
        resume.as_mut().poll(&mut cx),
        Poll::Ready(Err(NativeExecutorError::Cancelled))
    ));
    drop(resume);
    assert_eq!(polls.get(), 1, "cancellation must not poll the body again");
    assert_eq!(drops.get(), 1);
    assert!(invocation.active.is_none());
    assert!(invocation.advance().is_err());
    assert!(futures::executor::block_on(invocation.resume_suspension(1, 1)).is_err());
    assert!(invocation.resume_deoptimization().is_err());
    assert!(invocation.resume_optimization().is_err());
    drop(invocation);
    assert_eq!(drops.get(), 1);
}

#[test]
fn unpolled_drop_and_invalid_requests_preserve_retry_but_polled_drop_abandons() {
    let source = CancellationSource::new();
    let (work, polls, drops, _) = work();
    let mut invocation = invocation(&source, work);
    drop(invocation.resume_suspension(1, 1));
    assert!(futures::executor::block_on(invocation.resume_suspension(2, 1)).is_err());
    assert!(invocation.resume_deoptimization().is_err());
    assert!(invocation.resume_optimization().is_err());
    assert!(invocation.active.is_some());
    assert_eq!(polls.get(), 0);
    assert_eq!(drops.get(), 0);
    let waker = Waker::from(Arc::new(Wakes::default()));
    let mut resume = Box::pin(invocation.resume_suspension(1, 1));
    assert!(resume
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    drop(resume);
    assert!(invocation.active.is_none());
    assert_eq!(drops.get(), 1);
    assert!(invocation.advance().is_err());
}

#[test]
fn correct_resume_after_stale_request_restores_active_execution() {
    let source = CancellationSource::new();
    let (work, polls, drops, ready) = work();
    let mut invocation = invocation(&source, work);
    assert!(futures::executor::block_on(invocation.resume_suspension(1, 2)).is_err());
    ready.set(true);
    futures::executor::block_on(invocation.resume_suspension(1, 1)).unwrap();
    assert_eq!(polls.get(), 1);
    assert_eq!(drops.get(), 1);
    assert!(invocation.active.is_some());
}

#[test]
fn observed_cancellation_releases_captured_arena_roots_before_invocation_drop() {
    runmat_gc::gc_test_context(|| {
        let source = CancellationSource::new();
        let allocation =
            runmat_gc::gc_allocate_rooted(runmat_value::Value::String("capture".into())).unwrap();
        let handle = allocation.handle();
        let (mut work, _, drops, _) = work();
        work.on_drop = Some(Box::new(move || {
            let runtime = runmat_runtime::context::legacy::active()
                .expect("originating runtime during pending destructor");
            assert!(runtime.is_cancelled());
            runmat_gc::gc_collect_major().unwrap();
            assert!(
                runmat_gc::gc_clone_value(&handle).is_ok(),
                "arena roots outlive pending destructor"
            );
        }));
        let mut invocation = invocation(&source, work);
        let state = &mut invocation.active.as_mut().unwrap().state;
        let value =
            state
                .arena
                .insert(runmat_value::Value::HandleObject(runmat_value::HandleRef {
                    class_name: "CancelledCapture".into(),
                    target: handle,
                    valid: true,
                }));
        state
            .captured_sequences
            .insert(runmat_mir::MirSequenceLocalId(0), vec![value]);
        allocation.unroot().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());
        source.cancel(CancellationReason::User);
        assert!(matches!(
            invocation.advance(),
            Err(NativeExecutorError::Cancelled)
        ));
        assert_eq!(drops.get(), 1);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
        assert!(invocation.active.is_none());
    });
}

#[test]
fn polled_resume_drop_and_cancellation_keep_roots_and_context_during_work_destruction() {
    for cancel in [false, true] {
        runmat_gc::gc_test_context(|| {
            let source = CancellationSource::new();
            let allocation =
                runmat_gc::gc_allocate_rooted(runmat_value::Value::String("polled capture".into()))
                    .unwrap();
            let handle = allocation.handle();
            let (mut work, _, drops, _) = work();
            work.on_drop = Some(Box::new(move || {
                let runtime =
                    runmat_runtime::context::legacy::active().expect("resume destructor context");
                assert_eq!(runtime.is_cancelled(), cancel);
                runmat_gc::gc_collect_major().unwrap();
                assert!(runmat_gc::gc_clone_value(&handle).is_ok());
            }));
            let mut invocation = invocation(&source, work);
            invocation.active.as_mut().unwrap().state.arena.insert(
                runmat_value::Value::HandleObject(runmat_value::HandleRef {
                    class_name: "PolledCapture".into(),
                    target: handle,
                    valid: true,
                }),
            );
            allocation.unroot().unwrap();
            let waker = Waker::from(Arc::new(Wakes::default()));
            let mut cx = Context::from_waker(&waker);
            let mut resume = Box::pin(invocation.resume_suspension(1, 1));
            assert!(resume.as_mut().poll(&mut cx).is_pending());
            assert!(runmat_runtime::context::legacy::active().is_none());
            if cancel {
                source.cancel(CancellationReason::User);
                assert!(matches!(
                    resume.as_mut().poll(&mut cx),
                    Poll::Ready(Err(NativeExecutorError::Cancelled))
                ));
            }
            drop(resume);
            assert_eq!(drops.get(), 1);
            assert!(invocation.active.is_none());
            runmat_gc::gc_collect_major().unwrap();
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
            assert!(runmat_runtime::context::legacy::active().is_none());
        });
    }
}
