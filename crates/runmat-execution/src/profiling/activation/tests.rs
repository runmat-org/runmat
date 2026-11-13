use super::*;
use crate::profiling::{CommandProfilingRoot, ProviderObservationTerminal};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Wake, Waker};

struct NoopWake;
impl Wake for NoopWake {
    fn wake(self: Arc<Self>) {}
}

struct StepClock(AtomicU64);

impl CommandThreadCpuSampler for StepClock {
    fn sample_current_thread(&self) -> Option<Duration> {
        let next = self.0.fetch_add(1, Ordering::Relaxed).checked_add(1)?;
        Some(Duration::from_nanos(next * 10))
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn nested_same_owner_reads_once_per_real_transition() {
    let clock = Arc::new(StepClock(AtomicU64::new(0)));
    let root = CommandProfilingRoot::new_with_thread_cpu_sampler(Some(clock.clone()));
    with_command_profiling(Some(root.receipt()), || {
        assert_eq!(clock.0.load(Ordering::Relaxed), 1);
        with_command_profiling(Some(root.receipt()), || {
            assert_eq!(clock.0.load(Ordering::Relaxed), 1);
        });
        assert_eq!(clock.0.load(Ordering::Relaxed), 1);
    });
    assert_eq!(clock.0.load(Ordering::Relaxed), 2);
    assert_eq!(
        root.seal().cpu_thread_attributed,
        Some(Duration::from_nanos(10))
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn different_owners_and_masked_extent_are_exclusive() {
    let clock = Arc::new(StepClock(AtomicU64::new(0)));
    let first = CommandProfilingRoot::new_with_thread_cpu_sampler(Some(clock.clone()));
    let second = CommandProfilingRoot::new_with_thread_cpu_sampler(Some(clock.clone()));
    with_command_profiling(Some(first.receipt()), || {
        with_command_profiling(Some(second.receipt()), || {
            with_command_profiling(None, || assert!(current_command_profiling().is_none()));
        });
    });
    assert_eq!(clock.0.load(Ordering::Relaxed), 6);
    assert_eq!(
        first.seal().cpu_thread_attributed,
        Some(Duration::from_nanos(20))
    );
    assert_eq!(
        second.seal().cpu_thread_attributed,
        Some(Duration::from_nanos(20))
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn different_clock_sources_are_never_cross_subtracted() {
    let left_clock = Arc::new(StepClock(AtomicU64::new(100)));
    let right_clock = Arc::new(StepClock(AtomicU64::new(0)));
    let left = CommandProfilingRoot::new_with_thread_cpu_sampler(Some(left_clock.clone()));
    let right = CommandProfilingRoot::new_with_thread_cpu_sampler(Some(right_clock.clone()));
    with_command_profiling(Some(left.receipt()), || {
        with_command_profiling(Some(right.receipt()), || {});
    });
    assert_eq!(left_clock.0.load(Ordering::Relaxed), 104);
    assert_eq!(right_clock.0.load(Ordering::Relaxed), 2);
    assert_eq!(
        left.seal().cpu_thread_attributed,
        Some(Duration::from_nanos(20))
    );
    assert_eq!(
        right.seal().cpu_thread_attributed,
        Some(Duration::from_nanos(10))
    );
}

struct BackwardClock(AtomicU64);

impl CommandThreadCpuSampler for BackwardClock {
    fn sample_current_thread(&self) -> Option<Duration> {
        let read = self.0.fetch_add(1, Ordering::Relaxed);
        Some(Duration::from_nanos(if read == 0 { 20 } else { 10 }))
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn backward_clock_fails_closed_without_fabricating_zero() {
    let root = CommandProfilingRoot::new_with_thread_cpu_sampler(Some(Arc::new(BackwardClock(
        AtomicU64::new(0),
    ))));
    with_command_profiling(Some(root.receipt()), || {});
    assert_eq!(root.seal().cpu_thread_attributed, None);
}

struct UnavailableClock(AtomicU64);

impl CommandThreadCpuSampler for UnavailableClock {
    fn sample_current_thread(&self) -> Option<Duration> {
        self.0.fetch_add(1, Ordering::Relaxed);
        None
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn unavailable_sampler_remains_unknown_and_no_owner_makes_no_reads() {
    let clock = Arc::new(UnavailableClock(AtomicU64::new(0)));
    with_command_profiling(None, || assert!(current_command_profiling().is_none()));
    assert_eq!(clock.0.load(Ordering::Relaxed), 0);
    let root = CommandProfilingRoot::new_with_thread_cpu_sampler(Some(clock.clone()));
    with_command_profiling(Some(root.receipt()), || {});
    assert_eq!(clock.0.load(Ordering::Relaxed), 1);
    assert_eq!(root.seal().cpu_thread_attributed, None);
}

struct ReentrantSampler;

impl CommandThreadCpuSampler for ReentrantSampler {
    fn sample_current_thread(&self) -> Option<Duration> {
        let _guard = select_command_profiling(None);
        Some(Duration::from_nanos(10))
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn reentrant_sampler_does_not_change_selection_or_publish_cpu() {
    let root = CommandProfilingRoot::new_with_thread_cpu_sampler(Some(Arc::new(ReentrantSampler)));
    with_command_profiling(Some(root.receipt()), || {
        assert!(current_command_profiling().is_some());
    });
    assert!(current_command_profiling().is_none());
    assert_eq!(root.seal().cpu_thread_attributed, None);
}

struct PanickingClock {
    reads: AtomicU64,
    panic_on: u64,
}

impl CommandThreadCpuSampler for PanickingClock {
    fn sample_current_thread(&self) -> Option<Duration> {
        let read = self.reads.fetch_add(1, Ordering::Relaxed) + 1;
        if read == self.panic_on {
            panic!("clock failure fixture");
        }
        Some(Duration::from_nanos(read * 10))
    }
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn sampler_entry_and_exit_panics_cannot_leak_selection_or_abort_unwind() {
    for (panic_on, user_panics) in [(1, false), (2, false), (2, true)] {
        let failed =
            CommandProfilingRoot::new_with_thread_cpu_sampler(Some(Arc::new(PanickingClock {
                reads: AtomicU64::new(0),
                panic_on,
            })));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_command_profiling(Some(failed.receipt()), || {
                assert!(current_command_profiling().is_some());
                if user_panics {
                    panic!("user unwind fixture");
                }
            });
        }));
        assert_eq!(result.is_err(), user_panics);
        assert!(current_command_profiling().is_none());
        assert_eq!(failed.seal().cpu_thread_attributed, None);

        let independent = CommandProfilingRoot::new_with_thread_cpu_sampler(Some(Arc::new(
            StepClock(AtomicU64::new(0)),
        )));
        with_command_profiling(Some(independent.receipt()), || {});
        assert_eq!(
            independent.seal().cpu_thread_attributed,
            Some(Duration::from_nanos(10))
        );
    }

    let parent = CommandProfilingRoot::new_with_thread_cpu_sampler(Some(Arc::new(StepClock(
        AtomicU64::new(0),
    ))));
    let failing_child =
        CommandProfilingRoot::new_with_thread_cpu_sampler(Some(Arc::new(PanickingClock {
            reads: AtomicU64::new(0),
            panic_on: 2,
        })));
    with_command_profiling(Some(parent.receipt()), || {
        with_command_profiling(Some(failing_child.receipt()), || {});
        let selected = current_command_profiling();
        let expected_parent = parent.receipt();
        assert!(same_owner(selected.as_ref(), Some(&expected_parent)));
    });
    assert!(current_command_profiling().is_none());
    assert_eq!(failing_child.seal().cpu_thread_attributed, None);
    assert!(parent.seal().cpu_thread_attributed.is_some());

    let later = CommandProfilingRoot::new_with_thread_cpu_sampler(Some(Arc::new(StepClock(
        AtomicU64::new(0),
    ))));
    with_command_profiling(Some(later.receipt()), || {});
    assert!(later.seal().cpu_thread_attributed.is_some());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn ready_future_retires_inner_without_reselecting_during_drop() {
    let clock = Arc::new(StepClock(AtomicU64::new(0)));
    let root = CommandProfilingRoot::new_with_thread_cpu_sampler(Some(clock.clone()));
    let mut future = Box::pin(observe_command_future(
        root.receipt(),
        std::future::ready(()),
    ));
    let waker = Waker::from(Arc::new(NoopWake));
    let mut cx = Context::from_waker(&waker);
    assert!(future.as_mut().poll(&mut cx).is_ready());
    assert_eq!(clock.0.load(Ordering::Relaxed), 2);
    drop(future);
    assert_eq!(clock.0.load(Ordering::Relaxed), 2);
    assert_eq!(
        root.seal().cpu_thread_attributed,
        Some(Duration::from_nanos(10))
    );
}

#[test]
fn nested_selection_masks_and_restores_parent() {
    let parent = CommandProfilingRoot::new();
    with_command_profiling(Some(parent.receipt()), || {
        assert!(current_command_profiling().is_some());
        with_command_profiling(None, || assert!(current_command_profiling().is_none()));
        current_command_profiling()
            .expect("parent restored")
            .begin_provider_operation()
            .expect("parent contribution")
            .complete(ProviderObservationTerminal::Missing);
    });
    assert!(current_command_profiling().is_none());
    assert_eq!(parent.seal().observations.expect("valid").missing, 1);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn unwind_restores_the_previous_poll_owner() {
    let parent = CommandProfilingRoot::new();
    let child = CommandProfilingRoot::new();
    with_command_profiling(Some(parent.receipt()), || {
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_command_profiling(Some(child.receipt()), || panic!("fixture unwind"));
        }));
        assert!(unwound.is_err());
        current_command_profiling()
            .expect("parent restored")
            .begin_provider_operation()
            .expect("parent contribution")
            .complete(ProviderObservationTerminal::Missing);
    });
    assert_eq!(parent.seal().observations.expect("valid").missing, 1);
    assert!(current_command_profiling().is_none());
}

#[test]
fn interleaved_pending_futures_restore_their_own_command() {
    let clock = Arc::new(StepClock(AtomicU64::new(0)));
    let first = CommandProfilingRoot::new_with_thread_cpu_sampler(Some(clock.clone()));
    let second = CommandProfilingRoot::new_with_thread_cpu_sampler(Some(clock));
    let mut first_future = Box::pin(observe_command_future(first.receipt(), async {
        std::future::poll_fn(|_| {
            current_command_profiling()
                .expect("first selected")
                .begin_provider_operation()
                .expect("first contribution")
                .complete(ProviderObservationTerminal::Missing);
            Poll::<()>::Pending
        })
        .await
    }));
    let mut second_future = Box::pin(observe_command_future(second.receipt(), async {
        std::future::poll_fn(|_| {
            current_command_profiling()
                .expect("second selected")
                .begin_provider_operation()
                .expect("second contribution")
                .complete(ProviderObservationTerminal::Failed);
            Poll::<()>::Pending
        })
        .await
    }));
    let waker = Waker::from(Arc::new(NoopWake));
    let mut cx = Context::from_waker(&waker);
    assert!(first_future.as_mut().poll(&mut cx).is_pending());
    assert!(current_command_profiling().is_none());
    assert!(second_future.as_mut().poll(&mut cx).is_pending());
    assert!(current_command_profiling().is_none());
    drop(first_future);
    drop(second_future);
    let first = first.seal();
    let second = second.seal();
    assert_eq!(first.observations.expect("first valid").missing, 1);
    assert_eq!(second.observations.expect("second valid").failed, 1);
    assert!(first.cpu_thread_attributed.is_some());
    assert!(second.cpu_thread_attributed.is_some());
}
