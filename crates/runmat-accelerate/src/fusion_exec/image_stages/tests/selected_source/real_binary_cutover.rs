//! Compiled-source, request-bound selected Sub/Div fixture. These mock
//! receipts exercise authority/lifecycle, not device arithmetic or timing.
use super::*;
use crate::fusion_exec::image_stages::tests::real_binary_provider::BinaryFault;
use futures::FutureExt;

pub(super) async fn bound(
    shape: [usize; 3],
    provider: &'static StagedProvider,
) -> (
    ImageNormalizeStageSession,
    GpuTensorHandle,
    RetirementProbe,
    ImageSourceInvocation,
) {
    *provider.alias_same_class_cast.lock().unwrap() = true;
    let graph = compiled_image_graph(shape);
    let plans = FusionPlan::from_graph(&graph, &detect_fusion_groups(&graph));
    let plan = plans
        .groups
        .iter()
        .find(|plan| plan.group.kind == FusionKind::ImageNormalize)
        .expect("actual compiled image group");
    let FusionPattern::ImageNormalize(pattern) = plan.group.pattern.as_ref().unwrap() else {
        panic!("actual image pattern");
    };
    let input = provider
        .allocations
        .allocate_zeroed(
            shape.to_vec(),
            NumericElementType::F32,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let values = plan
        .inputs
        .iter()
        .map(|id| {
            if *id == pattern.input {
                Value::GpuTensor(input.handle.clone())
            } else {
                exact_source_input(&graph, *id, plan.group.span.start)
            }
        })
        .collect::<Vec<_>>();
    let invocation =
        ImageSourceInvocation::bind(&graph, plan, &values, ProviderPrecision::F64, None)
            .await
            .unwrap();
    assert!(
        ImageArithmeticProof::for_selected(&invocation).is_none(),
        "a graph-only provisional checkpoint cannot authorize selected execution"
    );
    let session = ImageNormalizeStageSession::begin_selected(
        provider,
        &input.handle,
        &invocation,
        ImageArithmeticProof::fixture(),
    )
    .unwrap()
    .unwrap();
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert!(provider.trace().is_empty());
    (session, input.handle, input.probe, invocation)
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn compiled_binary_calls_have_exact_source_order_and_observed_receipts() {
    for shape in [[2, 3, 4], [2, 4, 5]] {
        let provider = provider_with_precision(None, None, ProviderPrecision::F64);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let (mut session, input, probe, invocation) = bound(shape, provider).await;
        let source = invocation.selected_binary_calls().unwrap();
        let mean = session.mean().await.unwrap().0;
        let sigma = session.sigma().await.unwrap().0;
        let selected = session.selected_binary_plans.as_ref().unwrap();
        let expected_ids = selected.recipe_identities();
        let expected_physical = selected.proposals();
        let receipts = selected.receipts();
        assert_eq!(receipts.len(), 1);
        assert_eq!(
            receipts[0].selected_identity,
            invocation.semantic_identity()
        );
        assert_eq!(receipts[0].source_output, source.sigma_center().output());
        assert_eq!(receipts[0].source_inputs, source.sigma_center().inputs());
        assert_eq!(receipts[0].recipe_identity, expected_ids[0]);
        assert_eq!(
            receipts[0].observed.output_buffer_bytes,
            Some(expected_physical[0].output_buffer_bytes)
        );
        let checkpoint_result = session.next_output_checkpoint().await;
        let checkpoint = checkpoint_result.unwrap_or_else(|error| {
            panic!(
                "selected suffix failed after provider trace {:?}; source calls {:?}; error {error:?}",
                provider.trace(),
                session.program.output_program.as_ref().unwrap().calls.iter()
                    .map(|call| (call.source_output, call.hook, call.output_precision, call.checkpoint))
                    .collect::<Vec<_>>(),
            )
        });
        let receipts = session.selected_binary_plans.as_ref().unwrap().receipts();
        assert_eq!(receipts.len(), 3);
        assert_eq!(
            receipts
                .iter()
                .map(|receipt| receipt.source_output)
                .collect::<Vec<_>>(),
            [
                source.sigma_center().output(),
                source.output_sub().output(),
                source.output_div().output(),
            ]
        );
        for ((receipt, identity), physical) in
            receipts.iter().zip(expected_ids).zip(expected_physical)
        {
            assert_eq!(receipt.selected_identity, invocation.semantic_identity());
            assert_eq!(receipt.recipe_identity, identity);
            assert_eq!(
                receipt.observed.output_buffer_bytes,
                Some(physical.output_buffer_bytes)
            );
            assert_eq!(
                receipt.observed.uniform_allocations,
                physical.uniform_allocations
            );
            assert_eq!(
                receipt.observed.uniform_buffer_bytes,
                physical.uniform_buffer_bytes
            );
            assert_eq!(receipt.observed.dispatches, physical.dispatches);
            assert_eq!(
                receipt.observed.pre_output_submissions,
                physical.pre_output_submissions
            );
            assert_eq!(
                receipt.observed.queue_submissions,
                physical.queue_submissions
            );
        }
        assert_eq!(receipts[1].source_inputs, source.output_sub().inputs());
        assert_eq!(receipts[2].source_inputs, source.output_div().inputs());
        assert!(
            session.next_output_checkpoint().await.is_err(),
            "publication requires exact VM acknowledgement"
        );
        assert!(session
            .acknowledge_output_checkpoint(checkpoint.source_output + 1)
            .is_err());
        assert_eq!(
            session
                .selected_binary_plans
                .as_ref()
                .unwrap()
                .receipts()
                .len(),
            3
        );
        session
            .acknowledge_output_checkpoint(checkpoint.source_output)
            .unwrap();
        drop(session);
        for handle in [&input, &mean, &sigma, &checkpoint.handle] {
            handle.allocation_lease().validate_live().unwrap();
        }
        assert_eq!(probe.retirements(), 0);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn binary_plan_refusal_is_pre_effect_and_observed_or_alias_failure_is_terminal() {
    for fault in [
        BinaryFault::ObservedOutput,
        BinaryFault::ObservedSubmission,
        BinaryFault::WrongShape,
        BinaryFault::WrongClass,
        BinaryFault::WrongOwner,
        BinaryFault::HookError,
    ] {
        let provider = provider_with_precision(None, None, ProviderPrecision::F64);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let (mut session, input, probe, _) = bound([2, 3, 4], provider).await;
        let mean = session.mean().await.unwrap().0;
        let original = matches!(&fault, BinaryFault::HookError);
        *provider.binary_fault.lock().unwrap() = Some(fault);
        let error = session.sigma().await.unwrap_err();
        assert!(error.is_after_effect());
        if original {
            assert!(std::error::Error::source(&error)
                .unwrap()
                .downcast_ref::<std::io::Error>()
                .is_some());
        }
        assert!(matches!(session.state, State::Failed));
        assert!(session.sigma().await.is_err());
        for handle in [&input, &mean] {
            handle.allocation_lease().validate_live().unwrap();
        }
        assert_eq!(probe.retirements(), 0);
    }
    let provider = provider_with_precision(None, None, ProviderPrecision::F64);
    let _selected = ThreadProviderGuard::set(Some(provider));
    *provider.binary_fault.lock().unwrap() = Some(BinaryFault::RejectPlan);
    let graph = compiled_image_graph([2, 3, 4]);
    let plans = FusionPlan::from_graph(&graph, &detect_fusion_groups(&graph));
    let plan = plans
        .groups
        .iter()
        .find(|plan| plan.group.kind == FusionKind::ImageNormalize)
        .unwrap();
    let FusionPattern::ImageNormalize(pattern) = plan.group.pattern.as_ref().unwrap() else {
        panic!("image pattern");
    };
    let input = provider
        .allocations
        .allocate_zeroed(
            vec![2, 3, 4],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        )
        .unwrap();
    let values = plan
        .inputs
        .iter()
        .map(|id| {
            if *id == pattern.input {
                Value::GpuTensor(input.handle.clone())
            } else {
                exact_source_input(&graph, *id, plan.group.span.start)
            }
        })
        .collect::<Vec<_>>();
    let invocation =
        ImageSourceInvocation::bind(&graph, plan, &values, ProviderPrecision::F64, None)
            .await
            .unwrap();
    assert!(ImageNormalizeStageSession::begin_selected(
        provider,
        &input.handle,
        &invocation,
        ImageArithmeticProof::fixture()
    )
    .unwrap()
    .is_none());
    // A missing provider proposal cannot be substituted by the legacy F32 hook.
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn output_binary_observation_failure_after_sigma_cannot_replay() {
    let provider = provider_with_precision(None, None, ProviderPrecision::F64);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input, probe, _) = bound([2, 3, 4], provider).await;
    let mean = session.mean().await.unwrap().0;
    let sigma = session.sigma().await.unwrap().0;
    let before = provider.trace();
    *provider.binary_fault.lock().unwrap() = Some(BinaryFault::ObservedSubmission);
    let error = session.next_output_checkpoint().await.unwrap_err();
    assert!(error.is_after_effect());
    assert!(matches!(session.state, State::Failed));
    let after = provider.trace();
    assert!(after.len() > before.len());
    assert!(session.next_output_checkpoint().await.is_err());
    assert_eq!(provider.trace(), after);
    for handle in [&input, &mean, &sigma] {
        handle.allocation_lease().validate_live().unwrap();
    }
    assert_eq!(probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn changed_public_mean_role_is_rejected_before_suffix_sub_effect() {
    let provider = provider_with_precision(None, None, ProviderPrecision::F64);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input, probe, invocation) = bound([2, 4, 5], provider).await;
    let mean = session.mean().await.unwrap().0;
    let sigma = session.sigma().await.unwrap().0;
    let sub = invocation
        .selected_binary_calls()
        .unwrap()
        .output_sub()
        .output();
    let call = session
        .program
        .output_program
        .as_mut()
        .unwrap()
        .calls
        .iter_mut()
        .find(|call| call.source_output == sub)
        .unwrap();
    assert_eq!(
        call.operands[1],
        runmat_accelerate_api::ProviderImageOutputOperand::Mean
    );
    call.operands[1] = runmat_accelerate_api::ProviderImageOutputOperand::Sigma;
    let before = provider.trace();
    let error = session.next_output_checkpoint().await.unwrap_err();
    assert!(error.is_after_effect());
    assert!(matches!(session.state, State::Failed));
    assert_eq!(
        provider.trace(),
        before,
        "swapped Mean/Sigma performs no suffix Sub"
    );
    for handle in [&input, &mean, &sigma] {
        handle.allocation_lease().validate_live().unwrap();
    }
    assert_eq!(probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn missing_selected_binary_receipt_never_reverts_to_legacy_hooks() {
    let provider = provider_with_precision(None, None, ProviderPrecision::F64);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input, probe, _) = bound([2, 4, 5], provider).await;
    let mean = session.mean().await.unwrap().0;
    let before = provider.trace();
    session.selected_binary_plans = None;
    let error = session.sigma().await.unwrap_err();
    assert!(error.is_after_effect());
    assert!(matches!(session.state, State::Failed));
    assert_eq!(
        provider.trace(),
        before,
        "no legacy Sub or later selected hook"
    );
    input.allocation_lease().validate_live().unwrap();
    mean.allocation_lease().validate_live().unwrap();
    assert_eq!(probe.retirements(), 0);

    let provider = provider_with_precision(None, None, ProviderPrecision::F64);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, _, _, _) = bound([2, 3, 4], provider).await;
    session.mean().await.unwrap();
    session.sigma().await.unwrap();
    let before = provider.trace();
    session.selected_binary_plans = None;
    let error = session.next_output_checkpoint().await.unwrap_err();
    assert!(error.is_after_effect());
    assert!(matches!(session.state, State::Failed));
    assert_eq!(provider.trace(), before, "no legacy suffix Sub/Div");

    let provider = provider_with_precision(None, None, ProviderPrecision::F64);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, _, _, _) = bound([2, 3, 4], provider).await;
    session.mean().await.unwrap();
    let before = provider.trace();
    session
        .program
        .selected_powers
        .as_mut()
        .unwrap()
        .square
        .base += 1;
    let error = session.sigma().await.unwrap_err();
    assert!(error.is_after_effect());
    assert!(matches!(session.state, State::Failed));
    assert_eq!(
        provider.trace(),
        before,
        "changed square source cannot issue Sub"
    );
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn borrowed_alias_and_cancelled_binary_keep_published_leases() {
    let provider = provider_with_precision(None, None, ProviderPrecision::F64);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input, probe, _) = bound([2, 4, 5], provider).await;
    let mean = session.mean().await.unwrap().0;
    *provider.binary_fault.lock().unwrap() = Some(BinaryFault::Alias(mean.clone()));
    let error = session.sigma().await.unwrap_err();
    assert!(error.is_after_effect());
    assert!(matches!(session.state, State::Failed));
    assert_eq!(probe.retirements(), 0);
    mean.allocation_lease().validate_live().unwrap();
    input.allocation_lease().validate_live().unwrap();

    let provider = provider_with_precision(None, None, ProviderPrecision::F64);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input, probe, _) = bound([2, 3, 4], provider).await;
    let mean = session.mean().await.unwrap().0;
    *provider.binary_fault.lock().unwrap() = Some(BinaryFault::Pending);
    assert!(Box::pin(session.sigma()).now_or_never().is_none());
    assert!(matches!(session.state, State::InFlight));
    assert!(session.sigma().await.is_err());
    input.allocation_lease().validate_live().unwrap();
    mean.allocation_lease().validate_live().unwrap();
    assert_eq!(probe.retirements(), 0);
}
