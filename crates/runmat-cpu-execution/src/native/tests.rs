use super::*;
use std::{collections::BTreeSet, sync::Mutex};

static TEST_OWNER: Mutex<()> = Mutex::new(());

// witness: FEA-SHARED-RESOURCES-001/machine_cpu_execution @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Four persistent worker identities survive repeated operations, indexed outputs match
//   serial computation, and invalid or nested pool budgets reject. This component check does not
//   establish native machine discovery, task admission, mesh/solve equivalence, cancellation,
//   measured throughput or whole-study scaling.
#[test]
fn persistent_workers_ordered_results_and_explicit_budgets() {
    let _owner = TEST_OWNER.lock().unwrap();
    let ids = || {
        with_cpu_pool(4, || {
            let workers = Mutex::new(BTreeSet::new());
            rayon::broadcast(|_| {
                workers
                    .lock()
                    .unwrap()
                    .insert(format!("{:?}", std::thread::current().id()));
            });
            let input = (0..4096).collect::<Vec<_>>();
            assert_eq!(
                ordered_map(&input, |value| value * value),
                input.iter().map(|value| value * value).collect::<Vec<_>>()
            );
            assert_eq!(current_threads(), 4);
            workers.into_inner().unwrap()
        })
        .unwrap()
    };
    assert_eq!(ids(), ids());
    assert_eq!(current_threads(), 1);
    assert_eq!(
        with_cpu_pool(1, || with_cpu_pool(1, || ())).unwrap(),
        Err(CpuExecutionError::NestedExecution)
    );
    assert_eq!(
        with_cpu_pool(0, || ()),
        Err(CpuExecutionError::InvalidThreadCount)
    );
    assert_eq!(
        with_cpu_pool(4, || with_cpu_pool(2, || ())).unwrap(),
        Err(CpuExecutionError::NestedExecution)
    );
    let foreign = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    foreign.install(|| {
        assert_eq!(current_threads(), 1);
        let caller = std::thread::current().id();
        assert!(ordered_map(&[0; 128], |_| std::thread::current().id())
            .iter()
            .all(|thread| *thread == caller));
        assert_eq!(
            with_cpu_pool(2, || ()),
            Err(CpuExecutionError::NestedExecution)
        );
    });
}

#[test]
fn borrowed_disjoint_outputs_and_panics_release_owner() {
    let _owner = TEST_OWNER.lock().unwrap();
    let mut output = vec![0; 4096];
    with_cpu_pool(3, || for_each_mut(&mut output, |value| *value = 7)).unwrap();
    assert!(output.iter().all(|value| *value == 7));
    let panic = std::panic::catch_unwind(|| with_cpu_pool(3, || panic!("test panic")));
    assert!(panic.is_err());
    assert_eq!(with_cpu_pool(3, current_threads).unwrap(), 3);
}

// witness: FEA-SHARED-RESOURCES-001/machine_cpu_execution @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Two concurrent owners enter before either is released at admitted capacities one and
//   two, and ordered kernels retain the declared capacity. This checks lease concurrency and pool
//   ownership, not whole-machine saturation, native throughput, or measured stack residency.
#[test]
fn same_capacity_owners_overlap_without_allocating_competing_pools() {
    use std::sync::mpsc;
    use std::time::Duration;
    let _owner = TEST_OWNER.lock().unwrap();
    for threads in [1, 2] {
        let (started, observed) = mpsc::channel();
        std::thread::scope(|scope| {
            let mut releases = Vec::new();
            let mut handles = Vec::new();
            for _ in 0..2 {
                let started = started.clone();
                let (release, wait) = mpsc::channel();
                releases.push(release);
                handles.push(scope.spawn(move || {
                    with_cpu_pool(threads, move || {
                        started.send(current_threads()).unwrap();
                        wait.recv().unwrap();
                        let values = ordered_map(&[0; 128], |_| current_threads());
                        assert!(values.iter().all(|value| *value == threads));
                    })
                    .unwrap();
                }));
            }
            let first = observed.recv_timeout(Duration::from_secs(5));
            let second = observed.recv_timeout(Duration::from_secs(5));
            // Release and join even on failure, so a serialization regression cannot hang tests.
            for release in releases {
                release.send(()).unwrap();
            }
            for handle in handles {
                handle.join().unwrap();
            }
            assert_eq!(first.unwrap(), threads);
            assert_eq!(second.unwrap(), threads);
        });
    }
}

// witness: FEA-SHARED-RESOURCES-001/machine_cpu_execution @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: An incompatible one-thread request cancels while a two-thread owner remains active;
//   its body never executes, and the existing pool remains usable. This is controlled acquisition
//   evidence, not cancellation latency for arbitrary native kernels.
#[test]
fn incompatible_capacity_wait_observes_cancellation_before_the_owner_finishes() {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    };
    use std::time::Duration;
    let _owner = TEST_OWNER.lock().unwrap();
    let cancelled = AtomicBool::new(false);
    let checked = AtomicBool::new(false);
    let (started, active) = mpsc::channel();
    let (release, wait) = mpsc::channel();
    let (probed, observed) = mpsc::channel();
    let (finished, result) = mpsc::channel();
    std::thread::scope(|scope| {
        let owner = scope.spawn(move || {
            with_cpu_pool(2, move || {
                started.send(()).unwrap();
                wait.recv().unwrap();
            })
            .unwrap()
        });
        active.recv_timeout(Duration::from_secs(5)).unwrap();
        let contender = scope.spawn(|| {
            let outcome = with_cpu_pool_cancellable(
                1,
                || {
                    if !checked.swap(true, Ordering::Relaxed) {
                        probed.send(()).unwrap();
                    }
                    cancelled.load(Ordering::Relaxed)
                },
                || panic!("cancelled capacity must never execute"),
            );
            finished.send(outcome).unwrap();
        });
        let check = observed.recv_timeout(Duration::from_secs(5));
        cancelled.store(true, Ordering::Relaxed);
        let outcome = result.recv_timeout(Duration::from_secs(5));
        // Always release and join, including a regression where acquisition ignores cancellation.
        release.send(()).unwrap();
        owner.join().unwrap();
        contender.join().unwrap();
        check.unwrap();
        assert_eq!(outcome.unwrap(), Err(CpuExecutionError::Cancelled));
    });
    assert_eq!(with_cpu_pool(2, current_threads).unwrap(), 2);
}
