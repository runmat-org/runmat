//! A caught late protocol error must retire the throwing site's mutation cursor.

use super::*;

#[test]
fn caught_late_nested_write_then_mixed_assignment_uses_fresh_site_state() {
    let _lock = runmat_filesystem::provider_override_lock();
    let memory = MemoryFsProvider::with_current_dir("/mapped-core");
    memory
        .write_project_path("/mapped-core/data.bin", &[1, 2, 3, 4])
        .unwrap();
    let (release, receiver) = oneshot::channel();
    let evidence = Arc::new(Evidence::default());
    let provider = Arc::new(GatedProvider {
        memory,
        gate: Arc::new(Mutex::new(None)),
        write_gate: Arc::new(Mutex::new(Some(receiver))),
        fail_write: true,
        evidence: evidence.clone(),
    });
    let _owner = replace_provider(provider.clone());
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new(
                "mapped-core-caught-write@1",
                "catchNestedWriteThenMulti.m",
                SOURCE,
            ),
            None,
        ))
        .unwrap();
        let control = InvocationControl::default().force_generic_native();
        let mut pending = Box::pin(session.invoke_executable(
            &unit,
            invocation("catchNestedWriteThenMulti", 1),
            &control,
        ));
        for _ in 0..8 {
            let mut task = Context::from_waker(noop_waker_ref());
            if let Poll::Ready(outcome) = pending.as_mut().poll(&mut task) {
                panic!("caught write completed before backing suspension: {outcome:?}");
            }
            if evidence.writes.load(Ordering::SeqCst) == 1 {
                break;
            }
        }
        assert_eq!(evidence.writes.load(Ordering::SeqCst), 1);
        release.send(()).unwrap();
        let completed = block_on(pending).expect("caught late write and fresh mixed assignment");
        assert_eq!(completed.as_slice(), &[Value::Num(9.0)]);
        assert_eq!(evidence.opens.load(Ordering::SeqCst), 1);
        assert_eq!(evidence.writes.load(Ordering::SeqCst), 1);
        assert_eq!(evidence.completed_writes.load(Ordering::SeqCst), 0);
        assert_eq!(evidence.retired_writes.load(Ordering::SeqCst), 1);
        assert_eq!(
            provider
                .memory
                .read_project_path("/mapped-core/data.bin")
                .unwrap(),
            [1, 2, 3, 4]
        );
    });
}

#[test]
fn mixed_place_and_sequence_target_retains_pending_rhs_without_replay() {
    let _lock = runmat_filesystem::provider_override_lock();
    let memory = MemoryFsProvider::with_current_dir("/mapped-core");
    memory
        .write_project_path("/mapped-core/data.bin", &[1, 2, 3, 4])
        .unwrap();
    let (release, receiver) = oneshot::channel();
    let evidence = Arc::new(Evidence::default());
    let provider = Arc::new(GatedProvider {
        memory,
        gate: Arc::new(Mutex::new(Some(receiver))),
        write_gate: Arc::new(Mutex::new(None)),
        fail_write: false,
        evidence: evidence.clone(),
    });
    let _owner = replace_provider(provider);
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new("mapped-core-mixed@1", "mixedMappedRead.m", SOURCE),
            None,
        ))
        .unwrap();
        let control = InvocationControl::default().force_generic_native();
        let mut pending =
            Box::pin(session.invoke_executable(&unit, invocation("mixedMappedRead", 1), &control));
        for _ in 0..8 {
            let mut task = Context::from_waker(noop_waker_ref());
            if let Poll::Ready(outcome) = pending.as_mut().poll(&mut task) {
                panic!("mixed RHS completed before backing suspension: {outcome:?}");
            }
            if evidence.reads.load(Ordering::SeqCst) == 1 {
                break;
            }
        }
        assert_eq!(evidence.opens.load(Ordering::SeqCst), 1);
        assert_eq!(evidence.reads.load(Ordering::SeqCst), 1);
        assert_eq!(evidence.completed_reads.load(Ordering::SeqCst), 0);
        release.send(()).unwrap();
        let completed = block_on(pending).expect("mixed target RHS resumed");
        assert_eq!(completed.as_slice(), &[Value::Num(5.0)]);
        assert_eq!(evidence.opens.load(Ordering::SeqCst), 1);
        assert_eq!(evidence.reads.load(Ordering::SeqCst), 1);
        assert_eq!(evidence.completed_reads.load(Ordering::SeqCst), 1);
        drop(completed);
        drop(session);
        assert_eq!(evidence.retired_backings.load(Ordering::SeqCst), 1);
    });
}
