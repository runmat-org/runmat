use super::*;

#[test]
fn caught_nested_pop_failure_prevents_outer_success_and_preserves_own_error() {
    for original_error in [false, true] {
        let (state, driver) = owner();
        let result = state.with_context(|| {
            driver.fail_pop.store(true, Ordering::SeqCst);
            assert!(state.with_context(|| Ok(())).is_err());
            if original_error {
                Err(anyhow!("original operation error"))
            } else {
                Ok(())
            }
        });
        let error = result.unwrap_err();
        if original_error {
            assert_eq!(error.to_string(), "original operation error");
        } else {
            assert!(error
                .downcast_ref::<context::ContextUnavailable>()
                .is_some());
        }
        assert_eq!(
            *driver.events.lock().unwrap(),
            ["push", "push", "pop", "pop"]
        );
    }
}

#[test]
fn caught_nested_quarantine_does_not_publish_initialized_allocation() {
    let (state, driver) = owner();
    let result = state.allocate(
        vec![1],
        NativeDeviceClass::Numeric(NumericElementType::F64),
        GpuTensorStorage::Real,
        8,
        || Ok(1201),
        |_| {
            driver.fail_pop.store(true, Ordering::SeqCst);
            assert!(state.with_context(|| Ok(())).is_err());
            Ok(())
        },
    );
    assert!(result
        .unwrap_err()
        .downcast_ref::<context::ContextUnavailable>()
        .is_some());
    assert_eq!(state.records()[&1].allocation.unwrap().pointer, 1201);
    assert!(
        driver.frees().is_empty(),
        "quarantine forbids unsafe cleanup"
    );
    drop(state);
    assert!(driver.frees().is_empty());
    assert_eq!(driver.release_count(), 1);
}

#[test]
fn other_lane_quarantine_before_completion_prevents_allocation_publication() {
    let (state, driver) = owner();
    let entered = Arc::new(std::sync::Barrier::new(2));
    let quarantined = Arc::new(std::sync::Barrier::new(2));
    let worker = {
        let state = state.clone();
        let entered = entered.clone();
        let quarantined = quarantined.clone();
        std::thread::spawn(move || {
            state.allocate(
                vec![1],
                NativeDeviceClass::Numeric(NumericElementType::F64),
                GpuTensorStorage::Real,
                8,
                || Ok(1202),
                |_| {
                    entered.wait();
                    quarantined.wait();
                    Ok(())
                },
            )
        })
    };
    entered.wait();
    driver.fail_pop.store(true, Ordering::SeqCst);
    let failure = state.with_context(|| Ok(()));
    // Release the worker before asserting, so an unexpected test failure does
    // not leave the fixture's execution lane blocked forever.
    quarantined.wait();
    let result = worker.join().unwrap();
    assert!(failure.is_err());
    assert!(result
        .unwrap_err()
        .downcast_ref::<context::ContextUnavailable>()
        .is_some());
    assert_eq!(state.records()[&1].allocation.unwrap().pointer, 1202);
    assert!(driver.frees().is_empty());
    assert_eq!(
        *driver.events.lock().unwrap(),
        ["push", "push", "pop", "pop"]
    );
}
