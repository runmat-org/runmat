//! Request-local completion ownership across error, retry, and cancellation.
use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn completed_cleanup_error_is_false_but_unpolled_call_is_inert() {
    let PreparedRuntime {
        runtime,
        published,
        prepared,
        checkpoints,
        ..
    } = prepared_runtime("sum", false, true, false, None);
    drop(crate::call_builtin_sequence_async_with_outputs_in_context(
        &runtime,
        "sum",
        &arguments(),
        1,
    ));
    assert_eq!(prepared.get(), 0);
    assert!(published.borrow().is_empty());
    let result =
        futures::executor::block_on(crate::call_builtin_sequence_async_with_outputs_in_context(
            &runtime,
            "sum",
            &arguments(),
            1,
        ));
    assert!(result.is_err());
    assert_eq!(prepared.get(), 1);
    assert_eq!(checkpoints.get(), 1);
    assert_eq!(*published.borrow(), [PreparedCallStatus::CompletedFailure]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn prepared_right_is_one_shot_and_retry_invalidation_discards_it() {
    let binding = reduction_binding("sum");
    let (runtime, scoped, _placement) = harness(binding);
    let published = Rc::new(RefCell::new(Vec::new()));
    let observation = CallObservation::capture(Some(&runtime)).unwrap();
    let authority = RegistrationAuthority::Scoped {
        generation: scoped.binding_registration_revision(binding).unwrap(),
        service: scoped.clone(),
    };
    let right = observation
        .sink(binding, authority, Some(1))
        .unwrap()
        .prepared_right(binding.identity, Some(1))
        .unwrap();
    let duplicate = observation
        .sink(
            binding,
            RegistrationAuthority::Scoped {
                generation: scoped.binding_registration_revision(binding).unwrap(),
                service: scoped,
            },
            Some(1),
        )
        .unwrap()
        .prepared_right(binding.identity, Some(1));
    assert!(
        duplicate.is_none(),
        "one call cannot claim two prepared owners"
    );
    right.deposit(Box::new(Ticket {
        binding: binding.identity,
        status: PreparedCallStatus::Succeeded,
        published: published.clone(),
    }));
    observation.invalidate_retry();
    observation.finish_completed(PostCallObservationOutcome::CompletedSuccess);
    assert!(published.borrow().is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cancellation_after_cleanup_drops_the_prepared_ticket() {
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    let binding = reduction_binding("sum");
    let (runtime, scoped, _placement) = harness(binding);
    let source = CancellationSource::new();
    let runtime = runtime.with_cancellation_token(source.token());
    let published = Rc::new(RefCell::new(Vec::new()));
    let observation = CallObservation::capture(Some(&runtime)).unwrap();
    let authority = RegistrationAuthority::Scoped {
        generation: scoped.binding_registration_revision(binding).unwrap(),
        service: scoped,
    };
    let right = observation
        .sink(binding, authority, Some(1))
        .unwrap()
        .prepared_right(binding.identity, Some(1))
        .unwrap();
    right.deposit(Box::new(Ticket {
        binding: binding.identity,
        status: PreparedCallStatus::Succeeded,
        published: published.clone(),
    }));
    source.cancel(CancellationReason::User);
    observation.finish_completed(PostCallObservationOutcome::CompletedSuccess);
    assert!(published.borrow().is_empty());
}
