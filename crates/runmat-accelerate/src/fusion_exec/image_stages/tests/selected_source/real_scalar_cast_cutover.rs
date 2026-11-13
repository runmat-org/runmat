//! Held compiled-source scalar/cast receipt and alias lifetime witnesses.
use super::*;
use crate::fusion_exec::image_stages::tests::real_elementwise_provider::RealFault;
use futures::FutureExt;

use super::real_binary_cutover::bound;

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn selected_program_rejects_interleaved_unplanned_binary_before_effects() {
    let provider = provider_with_precision(None, None, ProviderPrecision::F64);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input, _probe, invocation) = bound([2, 4, 5], provider).await;
    let source = invocation.selected_real_output_calls().unwrap();
    let mut changed = session.program.clone();
    let mut extra = changed.output_program.as_ref().unwrap().calls[0].clone();
    extra.source_output = 10_000;
    extra.hook = runmat_accelerate_api::ProviderImageOutputHook::ElemMul;
    extra.checkpoint = false;
    changed
        .output_program
        .as_mut()
        .unwrap()
        .calls
        .insert(2, extra);
    assert!(changed.validate());
    assert!(selected_real_output::plan(
        provider,
        &source,
        session.selected_binary_plans.as_ref().unwrap(),
        session.selected_power_plans.as_ref().unwrap(),
        &changed,
    )
    .unwrap()
    .is_none());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);

    // A post-admission internal program mutation is terminal too: the
    // selected executor must never fall through to the legacy binary hook.
    session.program = changed;
    let mean = session.mean().await.unwrap().0;
    let sigma = session.sigma().await.unwrap().0;
    let error = session.next_output_checkpoint().await.err().unwrap();
    assert!(error.is_after_effect());
    assert!(matches!(session.state, State::Failed));
    assert!(!provider.trace().contains(&StageHook::Mul));
    assert!(session.next_output_checkpoint().await.is_err());
    for handle in [&input, &mean, &sigma] {
        handle.allocation_lease().validate_live().unwrap();
    }
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn compiled_suffix_chains_owned_mul_add_max_and_exact_alias_casts() {
    for shape in [[2, 3, 4], [2, 4, 5]] {
        let provider = provider_with_precision(None, None, ProviderPrecision::F64);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let (mut session, input, probe, invocation) = bound(shape, provider).await;
        let source = invocation.selected_real_output_calls().unwrap();
        let ids: [u64; 5] = std::array::from_fn(|index| source.calls()[index].output());
        let expected = session
            .selected_real_plans
            .as_ref()
            .unwrap()
            .recipe_identities();
        let _mean = session.mean().await.unwrap().0;
        let _sigma = session.sigma().await.unwrap().0;
        let first = session.next_output_checkpoint().await.unwrap();
        assert_eq!(first.source_output, ids[2]);
        let first_receipts = session.selected_real_plans.as_ref().unwrap().receipts();
        assert_eq!(first_receipts.len(), 3);
        for (index, (receipt, identity)) in first_receipts.iter().zip(expected).enumerate() {
            assert_eq!(receipt.selected_identity, invocation.semantic_identity());
            assert_eq!(receipt.recipe_identity, identity);
            assert_eq!(receipt.source_input, source.calls()[index].input());
            assert_eq!(
                receipt.source_control,
                source.calls()[index].control_source()
            );
            assert_eq!(receipt.selected_binding, source.calls()[index].binding());
            assert!(receipt.observed.output_buffer_bytes.is_some());
        }
        let State::OutputAwaitAck { progress, .. } = &session.state else {
            panic!("selected first checkpoint must await its VM store");
        };
        assert!(first
            .handle
            .shares_allocation_with(progress.get(ids[1]).unwrap()));
        assert!(first.handle.allocation_lease().validate_live().is_ok());
        session.acknowledge_output_checkpoint(ids[2]).unwrap();
        let second = session.next_output_checkpoint().await.unwrap();
        assert_eq!(second.source_output, ids[3]);
        let receipts = session.selected_real_plans.as_ref().unwrap().receipts();
        assert_eq!(receipts.len(), 4);
        assert!(!second.handle.shares_allocation_with(&first.handle));
        session.acknowledge_output_checkpoint(ids[3]).unwrap();
        let final_checkpoint = session.next_output_checkpoint().await.unwrap();
        let final_id = session
            .program
            .output_program
            .as_ref()
            .unwrap()
            .calls
            .last()
            .unwrap()
            .source_output;
        assert_eq!(final_checkpoint.source_output, final_id);
        let receipts = session.selected_real_plans.as_ref().unwrap().receipts();
        assert_eq!(receipts.len(), 5);
        assert_eq!(
            receipts
                .iter()
                .map(|receipt| receipt.source_output)
                .collect::<Vec<_>>(),
            ids
        );
        let State::OutputAwaitAck { progress, .. } = &session.state else {
            panic!("selected final checkpoint must await its VM store");
        };
        assert!(progress
            .get(ids[4])
            .unwrap()
            .shares_allocation_with(progress.get(ids[3]).unwrap()));
        session.acknowledge_output_checkpoint(final_id).unwrap();
        for checkpoint in [&first.handle, &second.handle, &final_checkpoint.handle] {
            checkpoint.allocation_lease().validate_live().unwrap();
        }
        input.allocation_lease().validate_live().unwrap();
        assert_eq!(probe.retirements(), 0);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn selected_scalar_cast_source_control_binding_and_wire_mutations_fail_closed() {
    let provider = provider_with_precision(None, None, ProviderPrecision::F64);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (_session, _input, _probe, invocation) = bound([2, 4, 5], provider).await;
    let calls = invocation.selected_real_output_calls().unwrap();
    let ids: [u64; 5] = std::array::from_fn(|index| calls.calls()[index].output());
    let changed_control_source = invocation.with_real_source_input_for_test(
        ValueId::try_from(ids[1]).unwrap(),
        ValueId::try_from(ids[0]).unwrap(),
    );
    assert!(changed_control_source
        .selected_real_output_calls()
        .is_none());
    let changed_binding = invocation.with_real_source_binding_for_test(
        ValueId::try_from(ids[2]).unwrap(),
        runmat_builtins::BuiltinBindingIdentity {
            builtin: runmat_builtins::MAX_CATALOG_ENTRY.identity,
            variant: "default",
        },
    );
    assert!(changed_binding.selected_real_output_calls().is_none());
    let changed_bits = invocation.with_real_wire_control_for_test(
        ids[0],
        runmat_accelerate_api::ProviderImageOutputOperand::ScalarF64Bits(1.0123_f64.to_bits()),
    );
    assert!(changed_bits.selected_real_output_calls().is_none());
    assert!(invocation.selected_real_output_calls().is_some());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn scalar_add_uses_selected_dense_predecessor_without_promoting_unknown_static_storage() {
    let provider = provider_with_precision(None, None, ProviderPrecision::F64);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (session, _input, _probe, invocation) = bound([2, 4, 5], provider).await;
    let mul =
        ValueId::try_from(invocation.selected_real_output_calls().unwrap().calls()[0].output())
            .unwrap();
    let unknown =
        invocation.with_real_output_storage_for_test(mul, runmat_types::StorageFact::Unknown);
    let calls = unknown.selected_real_output_calls().unwrap();
    assert!(selected_real_output::plan(
        provider,
        &calls,
        session.selected_binary_plans.as_ref().unwrap(),
        session.selected_power_plans.as_ref().unwrap(),
        &session.program,
    )
    .unwrap()
    .is_some());
    for storage in [
        runmat_types::StorageFact::Sparse,
        runmat_types::StorageFact::Opaque,
    ] {
        assert!(invocation
            .with_real_output_storage_for_test(mul, storage)
            .selected_real_output_calls()
            .is_none());
    }
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn selected_scalar_observation_failure_is_terminal_and_keeps_public_borrows() {
    let provider = provider_with_precision(None, None, ProviderPrecision::F64);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input, probe, _) = bound([2, 4, 5], provider).await;
    let mean = session.mean().await.unwrap().0;
    let sigma = session.sigma().await.unwrap().0;
    *provider.real_fault.lock().unwrap() = Some((false, RealFault::ObservedOutput));
    let error = session.next_output_checkpoint().await.err().unwrap();
    assert!(error.is_after_effect());
    assert!(matches!(session.state, State::Failed));
    assert_eq!(
        provider
            .trace()
            .iter()
            .filter(|hook| **hook == StageHook::Mul)
            .count(),
        1
    );
    assert!(session.next_output_checkpoint().await.is_err());
    for handle in [&input, &mean, &sigma] {
        handle.allocation_lease().validate_live().unwrap();
    }
    assert_eq!(probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn selected_scalar_cannot_return_borrowed_public_mean_as_owned_result() {
    let provider = provider_with_precision(None, None, ProviderPrecision::F64);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input, probe, _) = bound([2, 4, 5], provider).await;
    let mean = session.mean().await.unwrap().0;
    let sigma = session.sigma().await.unwrap().0;
    *provider.real_fault.lock().unwrap() = Some((false, RealFault::Alias(mean.clone())));
    let error = session.next_output_checkpoint().await.err().unwrap();
    assert!(error.is_after_effect());
    assert!(matches!(session.state, State::Failed));
    for handle in [&input, &mean, &sigma] {
        handle.allocation_lease().validate_live().unwrap();
    }
    assert_eq!(probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn dropped_pending_scalar_keeps_borrowed_moments_and_cannot_replay() {
    let provider = provider_with_precision(None, None, ProviderPrecision::F64);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input, probe, _) = bound([2, 3, 4], provider).await;
    let mean = session.mean().await.unwrap().0;
    let sigma = session.sigma().await.unwrap().0;
    *provider.real_fault.lock().unwrap() = Some((false, RealFault::Pending));
    let mut future = Box::pin(session.next_output_checkpoint());
    assert!(future.as_mut().now_or_never().is_none());
    drop(future);
    assert!(matches!(session.state, State::InFlight));
    assert!(session.next_output_checkpoint().await.is_err());
    for handle in [&input, &mean, &sigma] {
        handle.allocation_lease().validate_live().unwrap();
    }
    assert_eq!(probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn pending_max_after_first_store_keeps_published_alias_and_retires_only_new_work() {
    let provider = provider_with_precision(None, None, ProviderPrecision::F64);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input, probe, invocation) = bound([2, 4, 5], provider).await;
    let source = invocation.selected_real_output_calls().unwrap();
    let ids: [u64; 5] = std::array::from_fn(|index| source.calls()[index].output());
    let mean = session.mean().await.unwrap().0;
    let sigma = session.sigma().await.unwrap().0;
    let first = session.next_output_checkpoint().await.unwrap();
    assert_eq!(first.source_output, ids[2]);
    session.acknowledge_output_checkpoint(ids[2]).unwrap();
    let prior_probes = provider.probes.lock().unwrap().len();
    *provider.real_fault.lock().unwrap() = Some((false, RealFault::Pending));
    let mut future = Box::pin(session.next_output_checkpoint());
    assert!(future.as_mut().now_or_never().is_none());
    let max_probe = provider.probes.lock().unwrap()[prior_probes].clone();
    assert_eq!(max_probe.retirements(), 0);
    drop(future);
    assert_eq!(max_probe.retirements(), 1);
    assert!(matches!(session.state, State::InFlight));
    assert!(session.next_output_checkpoint().await.is_err());
    for handle in [&input, &mean, &sigma, &first.handle] {
        handle.allocation_lease().validate_live().unwrap();
    }
    assert_eq!(probe.retirements(), 0);
}
