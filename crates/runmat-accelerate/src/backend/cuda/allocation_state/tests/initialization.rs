//! The same initializer seam used by upload and single/two-component copies.
//! Injected failures exercise ownership, not CUDA numerical execution.
use super::*;

fn initialize_failure(stages: usize, unwind: bool, free_fails: bool) {
    let (state, driver) = owner();
    let borrowed = allocate(&state, 1301);
    driver.events.lock().unwrap().clear();
    driver.fail_free.store(free_fails, Ordering::SeqCst);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state.allocate(
            vec![1, 2],
            NativeDeviceClass::Numeric(NumericElementType::U64),
            GpuTensorStorage::Real,
            16,
            || Ok(1302),
            |_| {
                for stage in 0..stages {
                    assert!(
                        state.allocations.try_lock().is_ok(),
                        "initializer must execute outside allocation lock"
                    );
                    driver.events.lock().unwrap().push(if stage == 0 {
                        "transfer-first"
                    } else {
                        "transfer-second"
                    });
                }
                if unwind {
                    panic!("transfer initializer unwind");
                }
                Err(anyhow!("transfer initializer error"))
            },
        )
    }));
    if unwind {
        assert!(result.is_err());
    } else {
        assert_eq!(
            result.unwrap().unwrap_err().to_string(),
            "transfer initializer error"
        );
    }
    assert_eq!(driver.frees(), [1302]);
    assert!(
        state.allocation(&borrowed).is_ok(),
        "borrowed component remains owned and live"
    );
    if free_fails {
        assert!(matches!(state.records()[&2].phase, Phase::Uncertain(_)));
        assert!(state.retire_record(2).is_err());
    } else {
        assert!(!state.records().contains_key(&2));
    }
    let events = driver.events.lock().unwrap().clone();
    assert_eq!(events.first(), Some(&"push"));
    assert_eq!(&events[events.len() - 4..], ["pop", "push", "free", "pop"]);
    assert_eq!(
        events
            .iter()
            .filter(|event| event.starts_with("transfer-"))
            .count(),
        stages
    );
    state.retire_record(borrowed.buffer_id()).unwrap();
    drop(state);
    assert_eq!(
        driver.frees(),
        [1302, 1301],
        "failed output is never blindly retried at teardown"
    );
}

#[test]
fn unpublished_transfer_and_second_component_errors_cleanup_once_preserving_original_error() {
    for stages in [1, 2] {
        for free_fails in [false, true] {
            initialize_failure(stages, false, free_fails);
        }
    }
}

#[test]
fn independent_copy_initializer_publishes_distinct_allocation_without_retiring_source() {
    let (state, driver) = owner();
    let source = allocate(&state, 1401);
    let original = state.allocation(&source).unwrap();
    let output = state
        .allocate(
            source.shape().to_vec(),
            NativeDeviceClass::Numeric(original.descriptor.element_type.unwrap()),
            original.descriptor.storage.unwrap(),
            original.byte_length,
            || Ok(1402),
            |destination| {
                assert_ne!(destination, original.pointer);
                assert!(state.allocation(&source).is_ok());
                assert!(state.allocations.try_lock().is_ok());
                Ok(())
            },
        )
        .unwrap();
    assert_ne!(source.buffer_id(), output.buffer_id());
    state.retire_record(output.buffer_id()).unwrap();
    assert_eq!(driver.frees(), [1402]);
    assert!(state.allocation(&source).is_ok());
    state.retire_record(source.buffer_id()).unwrap();
    assert_eq!(driver.frees(), [1402, 1401]);
}

#[test]
#[cfg(panic = "unwind")]
fn unpublished_transfer_and_second_component_unwind_cleanup_once_preserving_borrowed_inputs() {
    for stages in [1, 2] {
        for free_fails in [false, true] {
            initialize_failure(stages, true, free_fails);
        }
    }
}
