//! Hardware-free tests exercise the same gate and scope-stack mechanics as the
//! real device. They do not fabricate a WGPU device, buffer, or allocation.
use super::*;
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
use std::{
    future::{ready, Ready},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    task::{Context, Poll, Wake, Waker},
};

#[derive(Default)]
struct Driver {
    stack: Mutex<Vec<Vec<u32>>>,
    unscoped: Mutex<Vec<u32>>,
    popped: AtomicUsize,
}
impl Driver {
    fn emit(&self, value: u32) {
        if let Some(scope) = self.stack.lock().unwrap().last_mut() {
            scope.push(value);
        } else {
            self.unscoped.lock().unwrap().push(value);
        }
    }
}
impl ScopeDriver for Driver {
    type Pending = Ready<Option<Vec<u32>>>;
    fn push(&self, _: wgpu::ErrorFilter) {
        self.stack.lock().unwrap().push(Vec::new());
    }
    fn pop(&self) -> Self::Pending {
        let values = self.stack.lock().unwrap().pop().expect("balanced scopes");
        self.popped.fetch_add(1, Ordering::SeqCst);
        ready((!values.is_empty()).then_some(values))
    }
}

struct NoWake;
impl Wake for NoWake {
    fn wake(self: Arc<Self>) {}
}
fn poll_once<F: Future>(future: Pin<&mut F>) -> Poll<F::Output> {
    let waker = Waker::from(Arc::new(NoWake));
    future.poll(&mut Context::from_waker(&waker))
}

#[test]
fn authority_identity_requires_the_same_owner_not_equal_payloads() {
    let original = Shared::new(7u32);
    let shared = original.clone();
    let unrelated = Shared::new(7u32);
    assert!(same_authority(&original, &shared));
    assert!(!same_authority(&original, &unrelated));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn poisoned_admission_allows_only_cleanup_after_scope_unwind() {
    let fenced = gate::Gate::default();
    fenced.fence_emission();
    fenced.cleanup(|| ()).unwrap();
    assert!(fenced.is_fenced());
    assert!(matches!(
        fenced.shared(|| ()),
        Err(WgpuAccessError::Poisoned)
    ));
    assert!(matches!(
        fenced.exclusive(|| ()),
        Err(WgpuAccessError::Poisoned)
    ));
    let gate = gate::Gate::default();
    let driver = Driver::default();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = gate.exclusive(|| {
            let _scopes = ActiveScopes::push(&driver);
            panic!("injected encoder failure");
        });
    }));
    assert!(panic.is_err());
    assert!(driver.stack.lock().unwrap().is_empty());
    assert!(matches!(gate.shared(|| ()), Err(WgpuAccessError::Poisoned)));
    let cleanup_count = AtomicUsize::new(0);
    gate.cleanup(|| {
        cleanup_count.fetch_add(1, Ordering::SeqCst);
    })
    .unwrap();
    assert_eq!(cleanup_count.load(Ordering::SeqCst), 1);
    assert!(matches!(gate.shared(|| ()), Err(WgpuAccessError::Poisoned)));
    assert!(matches!(
        gate.exclusive(|| ()),
        Err(WgpuAccessError::Poisoned)
    ));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn scope_error_preserves_original_cause_and_recovers_inspection_poison() {
    #[derive(Debug)]
    struct OriginalCause;
    impl fmt::Display for OriginalCause {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("original driver cause")
        }
    }
    impl Error for OriginalCause {}
    fn require_send_sync<T: Send + Sync>() {}
    require_send_sync::<WgpuScopeErrors>();
    let errors = WgpuScopeErrors::new(vec![wgpu::Error::Validation {
        source: Box::new(OriginalCause),
        description: "injected validation".into(),
    }]);
    assert_eq!(errors.kinds(), &[WgpuScopeErrorKind::Validation]);
    errors.with_errors(|original| {
        assert!(original[0].source().unwrap().is::<OriginalCause>());
    });
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        errors.with_errors(|_| panic!("observer failed"));
    }));
    assert!(panic.is_err());
    assert!(errors.to_string().contains("Validation"));
    errors.with_errors(|original| {
        assert!(original[0].source().unwrap().is::<OriginalCause>());
    });
    let retained = crate::ProviderExecutionError::from_source(errors);
    assert!(retained.source().unwrap().is::<WgpuScopeErrors>());
}

#[test]
fn one_error_does_not_skip_later_scope_results_or_release_work_early() {
    use std::sync::atomic::AtomicBool;
    struct Delayed {
        ready: Arc<AtomicBool>,
        index: usize,
    }
    impl Future for Delayed {
        type Output = Option<usize>;
        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            if self.index == 0 || self.ready.load(Ordering::SeqCst) {
                Poll::Ready(Some(self.index))
            } else {
                Poll::Pending
            }
        }
    }
    struct DelayedDriver {
        ready: Arc<AtomicBool>,
        pops: AtomicUsize,
    }
    impl ScopeDriver for DelayedDriver {
        type Pending = Delayed;
        fn push(&self, _: wgpu::ErrorFilter) {}
        fn pop(&self) -> Delayed {
            Delayed {
                ready: self.ready.clone(),
                index: self.pops.fetch_add(1, Ordering::SeqCst),
            }
        }
    }
    let ready = Arc::new(AtomicBool::new(false));
    let driver = DelayedDriver {
        ready: ready.clone(),
        pops: AtomicUsize::new(0),
    };
    let drops = Arc::new(AtomicUsize::new(0));
    let receipt = ActiveScopes::push(&driver).detach(Retained(drops.clone()));
    let mut future = Box::pin(receipt.resolve());
    assert!(poll_once(future.as_mut()).is_pending());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    ready.store(true, Ordering::SeqCst);
    let Poll::Ready(Err(errors)) = poll_once(future.as_mut()) else {
        panic!("all scopes resolved")
    };
    assert_eq!(
        errors,
        (0..driver.pops.load(Ordering::SeqCst)).collect::<Vec<_>>()
    );
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn detach_pops_every_scope_before_the_receipt_is_polled() {
    let driver = Driver::default();
    let gate = gate::Gate::default();
    let receipt = gate
        .exclusive(|| {
            let scopes = ActiveScopes::push(&driver);
            driver.emit(31);
            scopes.detach(9u32)
        })
        .unwrap();
    assert!(driver.stack.lock().unwrap().is_empty());
    assert_eq!(driver.popped.load(Ordering::SeqCst), 3);
    gate.shared(|| driver.emit(42)).unwrap();
    let mut result = Box::pin(receipt.resolve());
    assert_eq!(poll_once(result.as_mut()), Poll::Ready(Err(vec![vec![31]])));
    assert_eq!(*driver.unscoped.lock().unwrap(), vec![42]);
}

struct Retained(Arc<AtomicUsize>);

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn scope_error_kinds_preserve_every_original_backend_cause() {
    #[derive(Debug)]
    struct OriginalCause;
    impl fmt::Display for OriginalCause {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("original backend cause")
        }
    }
    impl Error for OriginalCause {}

    let errors = WgpuScopeErrors::new(vec![
        wgpu::Error::Validation {
            source: Box::new(OriginalCause),
            description: "validation".into(),
        },
        wgpu::Error::OutOfMemory {
            source: Box::new(OriginalCause),
        },
        wgpu::Error::Internal {
            source: Box::new(OriginalCause),
            description: "internal".into(),
        },
    ]);
    assert_eq!(
        errors.kinds(),
        &[
            WgpuScopeErrorKind::Validation,
            WgpuScopeErrorKind::OutOfMemory,
            WgpuScopeErrorKind::Internal,
        ]
    );
    errors.with_errors(|original| {
        assert_eq!(original.len(), 3);
        for error in original {
            assert!(error.source().unwrap().is::<OriginalCause>());
        }
    });
}

impl Drop for Retained {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn cancelling_detached_error_wait_releases_unpublished_work_without_scopes() {
    struct PendingDriver(AtomicUsize);
    impl ScopeDriver for PendingDriver {
        type Pending = std::future::Pending<Option<()>>;
        fn push(&self, _: wgpu::ErrorFilter) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
        fn pop(&self) -> Self::Pending {
            assert_ne!(self.0.fetch_sub(1, Ordering::SeqCst), 0);
            std::future::pending()
        }
    }
    let driver = PendingDriver(AtomicUsize::new(0));
    let gate = gate::Gate::default();
    let drops = Arc::new(AtomicUsize::new(0));
    let receipt = gate
        .exclusive(|| ActiveScopes::push(&driver).detach(Retained(drops.clone())))
        .unwrap();
    assert_eq!(driver.0.load(Ordering::SeqCst), 0);
    let mut future = Box::pin(receipt.resolve());
    assert!(poll_once(future.as_mut()).is_pending());
    gate.exclusive(|| ()).unwrap(); // no guard survives in the pending future
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(future);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(driver.0.load(Ordering::SeqCst), 0);
}

#[test]
fn successful_receipt_releases_work_only_after_every_pop_result() {
    let driver = Driver::default();
    let drops = Arc::new(AtomicUsize::new(0));
    let receipt = ActiveScopes::push(&driver).detach(Retained(drops.clone()));
    let mut future = Box::pin(receipt.resolve());
    let Poll::Ready(Ok(value)) = poll_once(future.as_mut()) else {
        panic!("ready scopes")
    };
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(value);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn two_shared_readers_overlap_before_either_is_released() {
    use std::{sync::mpsc, time::Duration};
    let gate = Arc::new(gate::Gate::default());
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let (entered, entries) = mpsc::channel();
    let mut releases = Vec::new();
    let mut readers = Vec::new();
    for index in 0..2 {
        let (release, released) = mpsc::channel();
        releases.push(release);
        let (gate, active, peak, entered) =
            (gate.clone(), active.clone(), peak.clone(), entered.clone());
        readers.push(std::thread::spawn(move || {
            gate.shared(|| {
                let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(count, Ordering::SeqCst);
                entered.send(index).unwrap();
                // Bounded even if a regression serializes the readers. A timeout
                // is a failure, never evidence of successful concurrency.
                let result = released.recv_timeout(Duration::from_secs(30));
                active.fetch_sub(1, Ordering::SeqCst);
                result
            })
        }));
    }
    let first = entries.recv_timeout(Duration::from_secs(5));
    let second = entries.recv_timeout(Duration::from_secs(5));
    let overlapped = active.load(Ordering::SeqCst);
    // Always release/join before asserting, including a failed entry handshake.
    for release in releases {
        let _ = release.send(());
    }
    for reader in readers {
        reader.join().unwrap().unwrap().unwrap();
    }
    assert!(first.is_ok() && second.is_ok(), "both readers must enter");
    assert_ne!(first.unwrap(), second.unwrap());
    assert_eq!(overlapped, 2, "both readers held the gate simultaneously");
    assert_eq!(peak.load(Ordering::SeqCst), 2);
    assert_eq!(active.load(Ordering::SeqCst), 0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn exclusive_transaction_isolates_scope_errors_from_concurrent_shared_emission() {
    use std::sync::{mpsc, Barrier};
    let gate = Arc::new(gate::Gate::default());
    let driver = Arc::new(Driver::default());
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let writer = {
        let (gate, driver, entered, release) = (
            gate.clone(),
            driver.clone(),
            entered.clone(),
            release.clone(),
        );
        std::thread::spawn(move || {
            gate.exclusive(|| {
                let scopes = ActiveScopes::push(driver.as_ref());
                driver.emit(11);
                entered.wait();
                release.wait();
                scopes.detach(())
            })
            .unwrap()
        })
    };
    entered.wait();
    let (started, ready) = mpsc::channel();
    let (finished, done) = mpsc::channel();
    let reader = {
        let (gate, driver) = (gate.clone(), driver.clone());
        std::thread::spawn(move || {
            started.send(()).unwrap();
            gate.shared(|| driver.emit(22)).unwrap();
            finished.send(()).unwrap();
        })
    };
    ready.recv().unwrap();
    assert!(matches!(done.try_recv(), Err(mpsc::TryRecvError::Empty)));
    release.wait();
    let receipt = writer.join().unwrap();
    reader.join().unwrap();
    let mut future = Box::pin(receipt.resolve());
    assert_eq!(poll_once(future.as_mut()), Poll::Ready(Err(vec![vec![11]])));
    assert_eq!(*driver.unscoped.lock().unwrap(), vec![22]);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn unwind_balances_scopes_before_poisoned_gate_is_observed() {
    let gate = gate::Gate::default();
    let driver = Driver::default();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = gate.exclusive(|| {
            let _scopes = ActiveScopes::push(&driver);
            panic!("driver creation unwound");
        });
    }));
    assert!(result.is_err());
    assert!(driver.stack.lock().unwrap().is_empty());
    assert_eq!(gate.shared(|| ()), Err(WgpuAccessError::Poisoned));
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
fn browser_reentrancy_is_nonblocking_and_borrow_is_released() {
    let gate = gate::Gate::default();
    gate.shared(|| assert_eq!(gate.exclusive(|| ()), Err(WgpuAccessError::Busy)))
        .unwrap();
    gate.exclusive(|| assert_eq!(gate.shared(|| ()), Err(WgpuAccessError::Busy)))
        .unwrap();
    gate.shared(|| ()).unwrap();
}
